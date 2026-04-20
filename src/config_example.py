"""
Example configuration file for SMILE conversational system.
Copy this file to `config.py` and adjust paths for your environment.
"""

# --- Base paths ---
BASE_DIR = "C:/path/to/FaceRecognition"
DATA_DIR = f"{BASE_DIR}/data"
KNOWN_FACES_DIR = f"{DATA_DIR}/known_faces"
CONVERSATIONS_DIR = f"{DATA_DIR}/conversations"
PROFILES_DIR = f"{DATA_DIR}/profiles"
EMBEDDINGS_PATH = f"{DATA_DIR}/embeddings.pkl"

# --- Vosk STT model (French) ---
# Download: https://alphacephei.com/vosk/models/vosk-model-small-fr-0.22.zip
VOSK_MODEL_PATH = "C:/Users/LUTIN/models/vosk-model-small-fr-0.22"

# --- Audio settings ---
MIC_SAMPLE_RATE = 16000
SILENCE_LIMIT = 3.2
SILENCE_HANGOVER = 0.8
SPEECH_MAX_DURATION = 20

# --- Ollama API ---
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:8b"

# --- System ---
DEBUG_MODE = False
