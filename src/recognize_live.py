# ==========================================
# 🎥 FACE RECOGNITION LIVE (ASYNCHRONOUS)
# ==========================================

import os
import cv2
import pickle
import time
import threading
import msvcrt
import queue
import traceback

# === UTILS ===

from src.config import EMBEDDINGS_FILE
from src.utils.facenet_utils import compare_embeddings
from src.utils.speech_utils import speak, transcribe_audio, extract_name_from_text
from src.utils.dialog_manager import ask_ollama_with_context, summarize_conversation
from src.utils.text_post import clean_llm_reply
from src.utils.profile_manager import load_recent_history
from src.utils.memory_manager import log_full_conversation, save_new_face
from src.utils.async_core import (
    detect_request_q, detect_result_q,
    embed_request_q, embed_result_q,
    start_workers, exit_event,
    speak_async, shutdown_executors,
    worker_ready_event, ask_ollama_async,
    embedding_ready_event
)

# ==========================================
# ⚙️ CONFIGURAZIONE
# ==========================================

# EMB_FILE = "../data/embeddings.pkl"

# 🔧 Configurazione ottimizzata
TRACKER_MAX_LOST = 15  # 🔧 Augmenté de 8 (plus tolérant)
EMBED_INTERVAL = 20.0   # 🔧 Secondes entre embeddings du même tracker
RESEEN_THRESHOLD = 30  # 🔧 Secondes avant de re-saluer
IOU_THRESHOLD = 0.3    # 🔧 Seuil IoU pour le matching

# ==========================================
# 🧮 UTILITY FUNCTIONS
# ==========================================

def iou(box1, box2):
    """
    Calcola Intersection over Union tra due box.
    box format: (x, y, w, h)
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0

# ==========================================
# ⌨️ ÉCOUTE TOUCHE Q
# ==========================================

def key_listener():
    """Thread per intercettare il tasto Q in qualsiasi momento."""
    while not exit_event.is_set():
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key in [b'q', b'Q']:
                print("\n👋 Fermeture demandée (touche Q)...")
                exit_event.set()
                break

# ==========================================
# 🔊 INTERAZIONE (TTS + STT + LLM)
# ==========================================
last_interaction_time = 0
conversation_lock = threading.Lock()

def handle_interaction(name: str, embedding=None):
    try:
        # === 1. Salutation initiale / reconnaissance utilisateur ===
        # Nouvel utilisateur -> demander le nom et enregistrer
        if name == "Visage détecté" and embedding is not None:
            speak_async(speak, "Bonjour ! Je ne crois pas t'avoir déjà rencontré. Comment tu t'appelles ?").result()
            time.sleep(1.2)

            user_name = transcribe_audio(
                duration=12,
                stop_on_silence=True,
                silence_limit=3.5
            ).strip()

            # Extrait le prénom
            name = extract_name_from_text(user_name)

            speak_async(speak, f"Enchanté {name} ! Je me souviendrai de toi. Comment tu vas aujourd'hui ?").result()
            save_new_face(name, embedding)
            time.sleep(1.2)

        else:
            # Utilisateur déjà connu
            speak_async(speak, f"Bonjour {name} !").result()
            time.sleep(1.2)

        # === 2. État conversationnel ===
        # GREETING : premiers tours après la reconnaissance
        # FREE_TALK : conversation libre
        # FAREWELL : clôture
        state = "GREETING"
        if state == "GREETING":
            print("👋 État initial : GREETING")

        silence_counter = 0
        max_silence_rounds = 5

        # mots indiquant un salut initial
        greeting_keywords = [
            "bonjour",
            "salut",
            "bonsoir",
            "coucou",
            "hey",
            "allô"
        ]
        farewell_keywords = [
            "au revoir",
            "à bientôt",
            "à plus",
            "bonne journée",
            "bonne soirée",
            "je dois y aller",
            "je m'en vais",
            "ciao"
        ]

        print("\n🟢 Conversation active — tu peux parler maintenant !\n")

        # === 3. Boucle conversationnelle ===
        # first_turn : True uniquement pour la PREMIÈRE réponse générée par le LLM
        first_turn = True

        while not exit_event.is_set():
            # 🎤 écoute l'utilisateur
            user_text = transcribe_audio(
                duration=20,
                stop_on_silence=True,
                silence_limit=3.2
            ).strip()

            # gestion silence / inactivité
            if not user_text:
                silence_counter += 1
                print(f"🤫 Silence détecté ({silence_counter}/{max_silence_rounds})")

                if silence_counter >= max_silence_rounds:
                    print("🕓 Aucune réponse depuis trop longtemps, fin de la conversation.")
                    break
                continue

            # reset compteur silences car l'utilisateur a parlé
            silence_counter = 0
            print(f"🗣️ [STT] Tu as dit : \"{user_text}\"")

            lower_text = user_text.lower()

            # === 3a. Gestion état GREETING ===
            if state == "GREETING":
                # dès que l'utilisateur dit plus qu'un simple bonjour, on passe en FREE_TALK
                if (
                    len(lower_text.split()) > 1
                    or "come" in lower_text
                    or "sto" in lower_text
                    or "bene" in lower_text
                    or "male" in lower_text
                ):
                    state = "FREE_TALK"
                else:
                    # Encore un salut léger, répondre et continuer
                    reply_future = ask_ollama_async(
                        lambda prompt: ask_ollama_with_context(
                            name,
                            prompt,
                            is_first_turn=first_turn,
                            state=state
                        ),
                        user_text
                    )

                    reply_raw = reply_future.result(timeout=120)
                    reply = clean_llm_reply(reply_raw, state=state, is_first_turn=first_turn)
                    first_turn = False

                    speak_async(speak, reply).result()
                    log_full_conversation(name, user_text, reply)
                    print("🟢 Prêt à écouter !")
                    time.sleep(1.0)
                    continue

            # === 3b. Fin de conversation ? (FREE_TALK ou GREETING)
            goodbye_phrases = [
                "je dois y aller",
                "je m'en vais",
                "à tout à l'heure",
                "à bientôt",
                "au revoir",
                "bonne journée",
                "bonne soirée",
                "c'est tout pour moi",
                "on se revoit",
                "à plus tard",
                "à plus",
            ]

            is_goodbye = (
                any(kw in lower_text for kw in goodbye_phrases)
                or any(kw in lower_text for kw in ["au revoir", "à bientôt", "à plus"])
            )

            # escludi solo i "ciao" di saluto iniziale
            is_pure_greeting = (
                len(lower_text.split()) <= 2
                and any(kw in lower_text for kw in greeting_keywords)
                and not any(kw in lower_text for kw in goodbye_phrases)
            )

            if is_goodbye and not is_pure_greeting:
                print("👋 Au revoir détecté.")

                farewell_prompt = (
                    f"L'utilisateur {name} a dit : '{user_text}'. "
                    "Réponds avec un au revoir chaleureux et amical. "
                    "Maximum deux phrases. Ne pose pas de questions."
                )

                reply_future = ask_ollama_async(
                    lambda prompt: ask_ollama_with_context(
                        name,
                        prompt,
                        is_first_turn=False,
                        state="FAREWELL"
                    ),
                    farewell_prompt
                )
                farewell_raw = reply_future.result(timeout=120)
                farewell_reply = clean_llm_reply(
                    farewell_raw,
                    state="FAREWELL",
                    is_first_turn=False
                )

                speak_async(speak, farewell_reply).result()
                log_full_conversation(name, user_text, farewell_reply)
                print(f"🔊 [TTS] Robot a dit : \"{farewell_reply}\"")

                break  # ⛔ esci dal ciclo dopo il saluto


            # === 3c. Conversation normale (FREE_TALK)
            state = "FREE_TALK"

            reply_future = ask_ollama_async(
                lambda prompt: ask_ollama_with_context(
                    name,
                    prompt,
                    is_first_turn=first_turn,
                    state=state
                ),
                user_text
            )

            reply_raw = reply_future.result(timeout=120)
            reply = clean_llm_reply(
                reply_raw,
                state=state,
                is_first_turn=first_turn
            )

            # après la première réponse, ce n'est plus le premier tour
            first_turn = False

            speak_async(speak, reply).result()
            log_full_conversation(name, user_text, reply)
            #update_profile_notes(
            #    name,
            #    f"L'utente ha detto: \"{user_text}\". Il robot ha risposto: \"{reply}\"."
            #)

            print(f"🔊 [TTS] Robot a dit : \"{reply}\"\n")

            time.sleep(1.0)
            print("🟢 Prêt à écouter !")

        # === 4. Fine conversazione ===
        print(f"✅ Conversation avec {name} terminée.\n")

        try:
            # 1) récupérer les derniers tours depuis le JSON des conversations
            recent = load_recent_history(name, window=10)  # qui ci sono "user" e "bot"

            # 2) demander à Ollama de les résumer et les écrire dans le profil
            summarize_conversation(name, recent)

            print(f"🧠 Profil de {name} mis à jour avec le résumé de la conversation.")
        except Exception as e:
            print(f"[MEMORY] Erreur lors de la mise à jour du profil : {e}")

    except Exception as e:
        print("[INTERACT] Erreur :", repr(e))
        traceback.print_exc()


def handle_interaction_threadsafe(name, embedding=None):
    global last_interaction_time
    with conversation_lock:
        last_interaction_time = time.time()
        handle_interaction(name, embedding)
        last_interaction_time = time.time()

# ==========================================
# 📦 DATABASE VOLTI
# ==========================================

def load_known_faces():
    """Charge la base de données d'embeddings connus."""
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE, "rb") as f:
            known = pickle.load(f)
        print(f"✅ Chargé {len(known)} visage(s) connu(s).")
        return known
    else:
        print("⚠️ Aucun visage enregistré. Démarrage en mode détection.")
        return {}

# ==========================================
# 🧠 MAIN LOOP
# ==========================================

def main():
    # === DÉMARRAGE WORKERS ET TRACKERS ===
    start_workers(speak_func=speak)

    print("🔊 Préchauffage TTS...")
    speak(" ")
    print("✅ TTS prêt.")

    print("🕐 Attente fin du préchauffage des workers...")
    worker_ready_event.wait()
    embedding_ready_event.wait()
    print("✅ Tous les workers prêts. Démarrage webcam.")

    global _cap, _active_interactions

    # Carica database
    known_faces = load_known_faces()

    # Démarrage webcam
    _cap = cv2.VideoCapture(0)
    cap = _cap
    if not cap.isOpened():
        print("❌ Erreur : impossible d'ouvrir la webcam.")
        return

    print("\n🎬 Démarrage reconnaissance en direct...")
    print("Appuie sur 'q' pour quitter.\n")

    # seen_names : dict nom -> timestamp dernier salut
    seen_names = {}  # nom -> timestamp dernier salut
    _active_interactions = {}
    active_interactions = _active_interactions
    
    trackers = {}            # id → tracker
    track_lost = {}          # id → compteur frames perdues
    tracker_boxes = {}       # id → (x, y, w, h) ultima box nota
    last_embed_time = {}     # id → timestamp dernier embedding
    next_face_id = 0
    frame_id = 0

    # Démarrage thread écoute touche Q
    threading.Thread(target=key_listener, daemon=True).start()

    print("\n🎬 Sistema pronto. Avvio stream video...\n")

    while not exit_event.is_set():
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame non lu correctement.")
            break

        frame = cv2.resize(frame, (640, 480))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_id += 1
        current_time = time.time()

        # --- 🔹 Invia frame al detection worker (max 1 alla volta)
        if detect_request_q.qsize() < 1:
            try:
                detect_request_q.put_nowait((frame_id, rgb.copy()))
            except queue.Full:
                pass

        # --- 🔹 Recupera eventuali risultati del detection worker
        boxes = None
        try:
            while not detect_result_q.empty():
                det_fid, result = detect_result_q.get_nowait()
                detect_result_q.task_done()
                boxes = result
        except queue.Empty:
            pass

        # --- 🔹 Pas de détection : mise à jour des trackers existants
        if boxes is None and trackers:
            boxes = []
            for tid, tr in list(trackers.items()):
                ok, box = tr.update(frame)
                if ok:
                    x, y, w, h = map(int, box)
                    boxes.append([x, y, x + w, y + h])
                    track_lost[tid] = 0
                    tracker_boxes[tid] = (x, y, w, h)
                else:
                    track_lost[tid] += 1
                    # 🔧 Plus tolérant avant de supprimer
                    if track_lost[tid] > TRACKER_MAX_LOST:
                        print(f"❌ Tracker {tid} perdu définitivement")
                        del trackers[tid]
                        del track_lost[tid]
                        tracker_boxes.pop(tid, None)
                        last_embed_time.pop(tid, None)

        # --- 🔹 Gestion des trackers (créer, mettre à jour, supprimer)
        if boxes is not None:
            boxes = [b for b in boxes if b is not None]

            updated_trackers = {}
            matched_ids = set()

            for b in boxes:
                x1, y1, x2, y2 = map(int, b)
                w, h = x2 - x1, y2 - y1
                new_box = (x1, y1, w, h)

                # 🔧 Matching basé sur IoU
                best_iou = IOU_THRESHOLD
                matched_id = None
                
                for tid in list(trackers.keys()):
                    if tid in matched_ids:
                        continue
                    
                    if tid in tracker_boxes:
                        current_iou = iou(new_box, tracker_boxes[tid])
                        if current_iou > best_iou:
                            best_iou = current_iou
                            matched_id = tid

                # 🔹 Match trouvé : réinitialiser le tracker
                if matched_id is not None:
                    trackers[matched_id].init(frame, new_box)
                    updated_trackers[matched_id] = trackers[matched_id]
                    tracker_boxes[matched_id] = new_box
                    track_lost[matched_id] = 0
                    matched_ids.add(matched_id)
                else:
                    # 🔹 Créer un nouveau tracker
                    tracker = (
                        cv2.legacy.TrackerCSRT_create()
                        if hasattr(cv2.legacy, "TrackerCSRT_create")
                        else cv2.TrackerCSRT_create()
                    )
                    tid = f"t{next_face_id}"
                    tracker.init(frame, new_box)
                    updated_trackers[tid] = tracker
                    tracker_boxes[tid] = new_box
                    track_lost[tid] = 0
                    print(f"🆕 Nouveau tracker {tid} créé {new_box}")
                    next_face_id += 1
                    matched_ids.add(tid)

            trackers = updated_trackers

        # --- 🔹 Dessiner les boîtes et envoyer les requêtes d'embedding
        for tid, tr in list(trackers.items()):
            try:
                ok, box = tr.update(frame)
            except Exception:
                ok, box = False, None
            
            if not ok or box is None:
                track_lost[tid] += 1
                if track_lost[tid] > TRACKER_MAX_LOST:
                    print(f"❌ Tracker {tid} perdu, supprimé")
                    trackers.pop(tid, None)
                    track_lost.pop(tid, None)
                    tracker_boxes.pop(tid, None)
                    last_embed_time.pop(tid, None)
                continue
            
            x, y, w, h = map(int, box)
            
            if w <= 0 or h <= 0:
                track_lost[tid] += 1
                continue
            
            track_lost[tid] = 0
            tracker_boxes[tid] = (x, y, w, h)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 🔧 Limitation du débit des requêtes d'embedding
            if tid not in last_embed_time or current_time - last_embed_time[tid] > EMBED_INTERVAL:
                if not embed_request_q.full():
                    try:
                        embed_request_q.put_nowait((tid, rgb.copy(), (x, y, x + w, y + h)))
                        last_embed_time[tid] = current_time
                    except queue.Full:
                        pass

        # --- 🔹 Lire les embeddings disponibles
        try:
            while not embed_result_q.empty():
                emb_fid, embedding = embed_result_q.get_nowait()
                embed_result_q.task_done()

                name = "Visage détecté"

                # 🔍 Comparaison avec la base de visages connus
                if known_faces:
                    for person, emb_db in known_faces.items():
                        match, dist = compare_embeddings(embedding, emb_db)
                        if match:
                            name = person
                            break

                # === Éviter les interactions en double ===
                current_time = time.time()
                # inconnu → utiliser l'id tracker comme clé unique
                display_key = name if name != "Visage détecté" else emb_fid

                # Controlla se è già attiva un’interazione per questo volto
                existing = active_interactions.get(display_key)
                if existing and getattr(existing, "is_alive", lambda: False)():
                    # déjà en conversation, mettre à jour le timestamp et ignorer
                    seen_names[name] = current_time
                    continue

                # Vérifier le cooldown avant de re-saluer
                if name not in seen_names or current_time - seen_names[name] > RESEEN_THRESHOLD:
                    if conversation_lock.locked():
                        print(f"⏳ En attente de fin de conversation avant d'interagir avec {name}.")
                    else:
                        seen_names[name] = current_time
                        print(f"👁️  Nouveau visage détecté : {name}")

                        # Démarrer une nouvelle interaction dans un thread dédié
                        th = threading.Thread(target=handle_interaction_threadsafe, args=(name, embedding), daemon=False)
                        active_interactions[display_key] = th
                        th.start()

                    # Thread de surveillance qui nettoie l'entrée en fin d'interaction
                    def _cleanup_thread(t, key):
                        t.join()
                        active_interactions.pop(key, None)

                    threading.Thread(target=_cleanup_thread, args=(th, display_key), daemon=True).start()

        except queue.Empty:
            pass

        # --- 🔹 Afficher le frame
        cv2.imshow("Face Recognition Live", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            exit_event.set()
            break

    _cleanup()


_cap = None
_active_interactions = {}


def _cleanup():
    exit_event.set()
    if _cap is not None:
        _cap.release()
    cv2.destroyAllWindows()

    active_threads = [t for t in _active_interactions.values() if isinstance(t, threading.Thread) and t.is_alive()]
    if active_threads:
        print(f"⏳ Sauvegarde du profil en cours ({len(active_threads)} conversation(s))...")
        for t in active_threads:
            t.join(timeout=30)

    shutdown_executors()
    print("\n✅ Fermeture terminée.")


# ==========================================
# 🚀 AVVIO
# ==========================================
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n✅ Fermeture demandée (Ctrl+C).")
    finally:
        _cleanup()