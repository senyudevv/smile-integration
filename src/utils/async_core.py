# src/utils/async_core.py
import threading
import queue
import torch
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from facenet_pytorch import MTCNN, InceptionResnetV1

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(2)
print(f"✅ async_core: using device {DEVICE}")

# ==========================================================
# 🧠 MODELLI GLOBALI (UNICA ISTANZA)
# ==========================================================

# 🔧 FIX: Un solo ResNet condiviso tra tutti i worker
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# 🔧 FIX: MTCNN globale per detection (creata nel worker con warm-up)
mtcnn_global = None

print("⚙️ Chargement des modèles...")

# ==========================================================
# 🎧 EXECUTOR PER TTS E OLLAMA
# ==========================================================

_tts_executor = None
_ollama_executor = None

# Queues
detect_request_q = queue.Queue(maxsize=2)
detect_result_q  = queue.Queue(maxsize=4)
embed_request_q  = queue.Queue(maxsize=4)
embed_result_q   = queue.Queue(maxsize=8)
tts_q            = queue.Queue(maxsize=8)
exit_event       = threading.Event()
embed_semaphore  = threading.Semaphore(1)

# ==========================================================
# 🔥 WARM-UP COMPLETO GPU
# ==========================================================

print("🔥 Préchauffage GPU...")

# 🔧 FIX: Warm-up ResNet con tensore realistico
with torch.no_grad():
    dummy_tensor = torch.randn((1, 3, 160, 160), device=DEVICE)  # Random invece di zeros
    _ = resnet(dummy_tensor)

print(f"✅ ResNet prêt sur {DEVICE}")

# ==========================================================
# 🧩 EVENTI DI SINCRONIZZAZIONE
# ==========================================================

worker_ready_event = threading.Event()
embedding_ready_event = threading.Event()

# ==========================================================
# 📸 DETECTION WORKER (MTCNN)
# ==========================================================

def detection_worker():
    """Worker per detection volti con MTCNN."""
    global mtcnn_global
    
    print("📸 Worker détection démarré...")
    
    # 🔧 FIX: MTCNN con parametri ottimizzati
    mtcnn_global = MTCNN(
        keep_all=True,
        device=DEVICE,
        min_face_size=60,  # 🔧 Aumentato da 40 (meno falsi positivi)
        thresholds=[0.6, 0.7, 0.8],  # 🔧 Più restrittivo (era [0.5, 0.6, 0.7])
        post_process=True  # Allineamento automatico
    )
    
    # 🔧 FIX: Warm-up MTCNN completo con più passaggi
    print("🔥 Préchauffage MTCNN (3 passes)...")
    dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    for i in range(3):
        _ = mtcnn_global.detect(dummy_frame)
    
    print("✅ MTCNN prêt.")
    worker_ready_event.set()
    
    while not exit_event.is_set():
        try:
            fid, frame_rgb = detect_request_q.get(timeout=0.1)
        except queue.Empty:
            continue
        
        # Copia per sicurezza
        frame_rgb = np.ascontiguousarray(frame_rgb.copy())
        
        try:
            boxes, probs = mtcnn_global.detect(frame_rgb)
            
            # 🔧 FIX: Filtra box troppo piccole o con bassa confidence
            if boxes is not None and probs is not None:
                filtered_boxes = []
                for box, prob in zip(boxes, probs):
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    
                    # Filtra: min 80x80px e confidence > 0.9
                    if w >= 80 and h >= 80 and prob > 0.9:
                        filtered_boxes.append(box)
                
                boxes = filtered_boxes if filtered_boxes else None
        
        except Exception as e:
            print(f"[DETECT] Errore su frame {fid}: {e}")
            boxes = None
        
        detect_result_q.put((fid, boxes))
        detect_request_q.task_done()

# ==========================================================
# 🧬 EMBEDDING WORKER (ResNet)
# ==========================================================

def embedding_worker():
    """Worker per generazione embedding con preprocessing corretto."""
    print("🧬 Worker embedding démarré...")
    
    # 🔧 FIX: Warm-up embedding worker
    print("🔥 Préchauffage embedding...")
    dummy_face = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
    dummy_tensor = torch.tensor(dummy_face).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    dummy_tensor = dummy_tensor.to(DEVICE)
    
    with torch.no_grad():
        _ = resnet(dummy_tensor)
    
    print("✅ Worker embedding prêt.")
    embedding_ready_event.set()
    
    while not exit_event.is_set():
        try:
            face_id, frame_rgb, box = embed_request_q.get(timeout=0.5)
        except queue.Empty:
            continue
        
        with embed_semaphore:
            try:
                x1, y1, x2, y2 = [int(v) for v in box]
                
                # 🔧 FIX: Aggiungi margine per migliorare allineamento
                h, w, _ = frame_rgb.shape
                margin = 10
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)
                
                face = frame_rgb[y1:y2, x1:x2]
                
                if face.size == 0:
                    continue
                
                # 🔧 FIX: Preprocessing migliorato (simile a MTCNN)
                face_tensor = torch.tensor(face).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                face_tensor = torch.nn.functional.interpolate(
                    face_tensor, 
                    size=(160, 160), 
                    mode='bilinear',  # 🔧 Migliorato da default
                    align_corners=False
                ).to(DEVICE)
                
                # 🔧 FIX: Normalizzazione come FaceNet
                mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(DEVICE)
                std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(DEVICE)
                face_tensor = (face_tensor - mean) / std
                
                with torch.no_grad():
                    emb = resnet(face_tensor).cpu().numpy()
                
                embed_result_q.put((face_id, emb))
                
            except Exception as e:
                print(f"[EMBED] Errore: {e}")
            finally:
                embed_request_q.task_done()

# ==========================================================
# 🔊 TTS WORKER
# ==========================================================

def tts_worker(speak_func):
    """Worker per Text-To-Speech."""
    while not exit_event.is_set():
        try:
            text = tts_q.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            speak_func(text)
        except Exception as e:
            print(f"[TTS] Errore: {e}")
        finally:
            tts_q.task_done()

# ==========================================================
# 🎯 EXECUTOR MANAGEMENT
# ==========================================================

def start_executors():
    """Inizializza gli executor per TTS e Ollama."""
    global _tts_executor, _ollama_executor
    if _tts_executor is None:
        _tts_executor = ThreadPoolExecutor(max_workers=1)
        print("🔊 Executor TTS démarré")
    if _ollama_executor is None:
        _ollama_executor = ThreadPoolExecutor(max_workers=1)
        print("🧠 Executor Ollama démarré")

def shutdown_executors():
    """Chiude gli executor in modo pulito."""
    global _tts_executor, _ollama_executor
    if _tts_executor:
        _tts_executor.shutdown(wait=True)
        print("🔊 Executor TTS arrêté")
        _tts_executor = None
    if _ollama_executor:
        _ollama_executor.shutdown(wait=True)
        print("🧠 Executor Ollama arrêté")
        _ollama_executor = None

def speak_async(func, *args, **kwargs):
    """Esegue la funzione TTS in background."""
    if _tts_executor is None:
        start_executors()
    return _tts_executor.submit(func, *args, **kwargs)

def ask_ollama_async(func, *args, **kwargs):
    """Esegue la chiamata a Ollama in background."""
    if _ollama_executor is None:
        start_executors()
    return _ollama_executor.submit(func, *args, **kwargs)

# ==========================================================
# 🚀 START WORKERS
# ==========================================================

def start_workers(speak_func=None):
    """
    Avvia tutti i thread asincroni e gli executor necessari:
      - detection_worker (MTCNN)
      - embedding_worker (ResNet)
      - executor TTS e Ollama
    """
    threading.Thread(target=detection_worker, daemon=True).start()
    threading.Thread(target=embedding_worker, daemon=True).start()
    start_executors()
    
    print("✅ Tous les workers démarrés")