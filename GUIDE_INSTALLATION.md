# Guide d'installation et de configuration — SMILE

**SMILE** (Social Memory Integrated Learning Environment) est un agent conversationnel embarqué qui :
- reconnaît les visages en temps réel (OpenCV + Facenet),
- transcrit la parole hors ligne (Vosk STT),
- génère des réponses via un LLM local (Ollama),
- mémorise chaque utilisateur entre les sessions (profils JSON + résumés LLM).

Ce guide explique comment faire tourner SMILE sur Linux (robot ou écran avec caméra + micro), et comment le passer en **français**.

---

## 1. Prérequis système

| Composant | Version recommandée |
|-----------|---------------------|
| Python | 3.11 |
| OS | Linux (Ubuntu 22.04 / Debian 12 ou équivalent) |
| Webcam | accessible via `/dev/video0` |
| Micro | accessible via ALSA / PulseAudio |
| RAM | ≥ 8 Go (16 Go recommandé pour le LLM) |
| GPU (optionnel) | CUDA si disponible, accélère Ollama |

Paquets système nécessaires :

```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3-pip \
    libportaudio2 portaudio19-dev \
    libopencv-dev \
    espeak espeak-ng \
    cmake build-essential
```

---

## 2. Installer Ollama (LLM local)

Ollama fait tourner le modèle de langage en local. Il doit être démarré avant SMILE.

```bash
curl -fsSL https://ollama.com/install.sh | sh

# Démarrer le serveur Ollama
ollama serve &

# Télécharger le modèle (choisir selon la RAM disponible)
ollama pull llama3:8b       # ~5 Go RAM — recommandé
# ollama pull mistral        # alternative légère
# ollama pull llama3:70b     # haute qualité, ~40 Go RAM
```

Vérifier qu'Ollama tourne :
```bash
curl http://localhost:11434/api/tags
```

---

## 3. Télécharger le modèle Vosk en français

Le modèle Vosk fourni par défaut est en **italien**. Pour le français, télécharger :

```bash
# Modèle léger (~40 Mo) — adapté robot/embarqué
wget https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip
unzip vosk-model-small-fr-0.22.zip -d ~/models/

# Modèle complet (~1.4 Go) — meilleure précision
# wget https://alphacephei.com/vosk/models/vosk-model-fr-0.22.zip
# unzip vosk-model-fr-0.22.zip -d ~/models/
```

Retenir le chemin, par exemple : `~/models/vosk-model-small-fr-0.22`

---

## 4. Installer les dépendances Python

```bash
cd /chemin/vers/LUTIN/SMILE

python3.11 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

> **Note** : Si `deepface` ou `tensorflow` pose problème sur ARM (Raspberry Pi, etc.),
> remplacer par `facenet-pytorch` et adapter `facenet_utils.py`.

---

## 5. Créer le fichier de configuration

```bash
cp src/config_example.py src/config.py
```

Éditer `src/config.py` avec les chemins Linux corrects :

```python
import os

# --- Chemins de base ---
BASE_DIR = "/chemin/vers/LUTIN/SMILE"          # ← adapter
DATA_DIR = os.path.join(BASE_DIR, "data")
KNOWN_FACES_DIR = os.path.join(DATA_DIR, "known_faces")
CONVERSATIONS_DIR = os.path.join(DATA_DIR, "conversations")
PROFILES_DIR = os.path.join(DATA_DIR, "profiles")
EMBEDDINGS_FILE = os.path.join(DATA_DIR, "embeddings.pkl")

# --- Vosk (modèle français) ---
VOSK_MODEL_PATH = os.path.expanduser("~/models/vosk-model-small-fr-0.22")

# --- Audio ---
MIC_SAMPLE_RATE = 16000
SILENCE_LIMIT = 3.2
SILENCE_HANGOVER = 0.8
SPEECH_MAX_DURATION = 20

# --- Voix TTS ---
VOICE_RATE = 160
VOICE_VOLUME = 1.0
DEFAULT_VOICE_INDEX = 0     # tester 0 ou 1 selon les voix installées

# --- Ollama ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:8b"

# --- Système ---
DEBUG_MODE = False
```

Créer les dossiers data manquants :
```bash
mkdir -p data/known_faces data/conversations data/profiles
```

---

## 6. Corrections à apporter pour Linux + français

Le code original a été développé sous Windows en italien. Deux catégories de modifications sont nécessaires.

### 6.1 Corriger le bug Linux critique (`msvcrt`)

Le fichier `src/recognize_live.py` importe `msvcrt` (module **Windows uniquement**).
Sur Linux, le remplacer par une écoute clavier compatible :

Trouver et remplacer dans `recognize_live.py` :

```python
# ❌ SUPPRIMER ces lignes :
import msvcrt

def key_listener():
    while not exit_event.is_set():
        if msvcrt.kbhit():
            key = msvcrt.getch()
            if key in [b'q', b'Q']:
                exit_event.set()
                break
```

```python
# ✅ REMPLACER PAR (compatible Linux) :
import sys
import select

def key_listener():
    while not exit_event.is_set():
        if select.select([sys.stdin], [], [], 0.1)[0]:
            key = sys.stdin.read(1)
            if key.lower() == 'q':
                print("\n👋 Fermeture demandée (touche Q)...")
                exit_event.set()
                break
```

### 6.2 Passer SMILE en français

#### A) Prompt LLM — `src/utils/dialog_manager.py`

Changer le prompt système (fonction `build_llm_prompt`) pour que le robot réponde en français :

```python
# ❌ AVANT (en bas du prompt) :
Tu sei "Robot", un assistente robotico che parla in italiano...

# ✅ APRÈS :
Tu es "Robot", un assistant robotique qui parle en français, avec un ton chaleureux et naturel.
Tu parles avec {user_name}, que tu connais.

ÉTAT ACTUEL : {state}
{stage}

⚠️ RÈGLES ⚠️
- NE commence PAS chaque réponse par "Bonjour" ou le prénom, sauf au tout premier tour.
- Si l'état est FREE_TALK, traite les "bonjour" comme une partie normale de la conversation.
- Si l'état est FAREWELL, fais seulement un salut final, sans poser de questions.
- Réponses courtes : 2-3 phrases.
```

Changer aussi les labels d'état :

```python
if state == "GREETING":
    stage = "C'est le début de la conversation. Tu peux saluer brièvement et te présenter naturellement."
elif state == "FAREWELL":
    stage = "La conversation se termine. Réponds avec un au revoir chaleureux, ne relance pas."
else:
    stage = "La conversation est déjà en cours. NE salue PAS à nouveau."
```

Et les labels dans le prompt :

```python
📘 MÉMOIRE LONG TERME :
{profile_txt}

🧠 MÉMOIRE ÉPISODIQUE (résumés précédents) :
{notes_summary}  # remplacer la valeur par défaut "(aucune mémoire épisodique disponible)"

💬 MÉMOIRE COURT TERME (7 derniers échanges) :
{history_txt}

🗣️ L'UTILISATEUR DIT :
{user_text}

Réponds en tant que "Robot" :
Robot:
```

#### B) Prompt de résumé — `src/utils/dialog_manager.py` (fonction `summarize_conversation`)

Traduire le prompt de résumé :

```python
prompt = f"""
Tu es un système de mémoire conversationnelle. Tu vas recevoir :
1. Le profil actuel de l'utilisateur (potentiellement incomplet)
2. La transcription de la dernière conversation

Ton rôle est de mettre à jour le profil de façon cohérente,
en ne déduisant que ce qui ressort clairement.

=== PROFIL ACTUEL ===
{json.dumps(profile, ensure_ascii=False, indent=2)}

=== CONVERSATION ===
{dialogue_text}

Retourne un JSON avec :
- summary: bref résumé de l'interaction (3-4 phrases)
- gender: "homme", "femme" ou null si non déductible
- age: tranche d'âge estimée (ex. "20-30") ou null
- occupation: profession ou domaine si mentionné
- interests: liste de sujets ou loisirs cités
- personality: traits comportementaux (ex. curieux, empathique, analytique)
- goals: objectifs personnels ou professionnels si mentionnés
"""
```

#### C) Phrases parlées — `src/recognize_live.py`

Remplacer les phrases hardcodées italiennes dans `handle_interaction` :

```python
# ❌ AVANT :
speak_async(speak, "Ciao! Non credo di averti mai conosciuto prima, come ti chiami?").result()
speak_async(speak, f"Piacere {name}! D'ora in poi ti riconoscerò. Dimmi pure, come va oggi?").result()
speak_async(speak, f"Ciao {name}!").result()

# ✅ APRÈS :
speak_async(speak, "Bonjour ! Je ne crois pas t'avoir déjà rencontré. Comment tu t'appelles ?").result()
speak_async(speak, f"Enchanté {name} ! Je me souviendrai de toi. Comment tu vas aujourd'hui ?").result()
speak_async(speak, f"Bonjour {name} !").result()
```

#### D) Mots-clés de salutation/aurevoir — `src/recognize_live.py`

```python
# ✅ En français :
greeting_keywords = [
    "bonjour", "salut", "bonsoir", "coucou", "hey", "allô"
]

farewell_keywords = [
    "au revoir", "à bientôt", "à plus", "bonne journée",
    "bonne soirée", "je dois y aller", "je m'en vais", "ciao"
]

goodbye_phrases = [
    "je dois y aller", "je m'en vais", "à tout à l'heure",
    "à bientôt", "au revoir", "bonne journée", "bonne soirée",
    "c'est tout pour moi", "on se revoit", "à plus tard"
]
```

#### E) Détection du prénom — `src/utils/speech_utils.py`

La fonction `extract_name_from_text` utilise des patterns italiens ("mi chiamo", "sono"). La passer en français :

```python
def extract_name_from_text(text: str) -> str:
    text = text.lower().strip()
    blacklist = {
        "bonjour", "salut", "je", "m'appelle", "appelle",
        "suis", "moi", "c'est", "le", "la", "un", "une", "oui", "non"
    }

    text = re.sub(r"[^a-zàâäéèêëîïôùûüœç\s]", "", text)

    # patterns français
    m = re.search(r"(?:je m'appelle|je suis|m'appelle|c'est)\s+([a-zàâäéèêëîïôùûüœç]+)", text)
    if m:
        name = m.group(1).capitalize()
    else:
        words = [w for w in text.split() if w not in blacklist]
        if not words:
            return f"Utilisateur_{int(time.time())}"
        candidates = [w for w in words if w not in ["merci", "voilà", "bien"]]
        name = candidates[0].capitalize() if candidates else f"Utilisateur_{int(time.time())}"

    if len(name) < 2 or name.lower() in ["bien", "merci", "oui", "non", "ok"]:
        name = f"Utilisateur_{int(time.time())}"

    return name
```

#### F) Labels dans `profile_manager.py`

```python
# Dans format_history_for_prompt :
return "Aucune conversation précédente avec cette personne."
# ...
lines.append(f"[{ts}] Utilisateur: {user_line}")
lines.append(f"[{ts}] Robot: {bot_line}")
```

---

## 7. Lancer SMILE

```bash
# Dans le répertoire SMILE, avec le venv activé
cd /chemin/vers/LUTIN/SMILE
source .venv/bin/activate

# S'assurer qu'Ollama tourne
ollama serve &

# Lancer
python -m src.recognize_live
```

Au démarrage :
- La webcam s'ouvre
- Un nouveau visage déclenche la question du prénom
- Les sessions suivantes : le robot reconnaît et salue directement
- Appuyer sur `q` pour quitter

---

## 8. Enregistrer un visage manuellement

Il est possible d'enregistrer des visages sans passer par la reconnaissance automatique :

```bash
python -m src.register_face
```

---

## 9. Tester la voix TTS en français

```bash
python3 -c "
import pyttsx3
engine = pyttsx3.init()
voices = engine.getProperty('voices')
for i, v in enumerate(voices):
    print(i, v.name, v.languages)
engine.setProperty('voice', voices[0].id)
engine.say('Bonjour, je suis SMILE, ton assistant robotique.')
engine.runAndWait()
"
```

Si aucune voix française n'est disponible, installer `espeak-ng` avec le paquet de langues :

```bash
sudo apt install espeak-ng-data
# ou pour une voix plus naturelle :
pip install TTS   # Coqui TTS — remplace pyttsx3 pour de meilleures voix
```

---

## 10. Structure des données persistantes

```
data/
├── embeddings.pkl          ← base de données des visages (binaire)
├── conversations/
│   └── Alice.json          ← historique des échanges par utilisateur
└── profiles/
    └── Alice.json          ← profil long terme (âge, intérêts, résumé, etc.)
```

Ces fichiers sont générés automatiquement. Ne pas les committer dans git (déjà dans `.gitignore`).

---

