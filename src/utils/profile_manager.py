import os
import json
import time
from datetime import datetime
from src.config import CONVERSATIONS_DIR, PROFILES_DIR

'''
# cartelle persistenti
BASE_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
CONV_DIR = os.path.join(BASE_DATA_DIR, "conversations")
PROFILE_DIR = os.path.join(BASE_DATA_DIR, "profiles")

os.makedirs(CONV_DIR, exist_ok=True)
os.makedirs(PROFILE_DIR, exist_ok=True)
'''

def _conv_path(name: str) -> str:
    safe = name.replace(" ", "_")
    return os.path.join(CONVERSATIONS_DIR, f"{safe}.json")

def _profile_path(name: str) -> str:
    safe = name.replace(" ", "_")
    return os.path.join(PROFILES_DIR, f"{safe}.json")

# -------------------------
# 1. Conversazione (short-term memory)
# -------------------------

def load_recent_history(name: str, window: int = 7) -> list[dict]:
    """
    Ritorna gli ultimi `window` turni di conversazione salvati per questo utente.
    Ogni turno è: { "timestamp": ..., "user": "...", "bot": "..." }
    Se non c'è ancora storia, ritorna [].
    """
    path = _conv_path(name)
    if not os.path.exists(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    # prendi SOLO gli ultimi `window`
    return data[-window:]

def format_history_for_prompt(history: list[dict]) -> str:
    """
    Converte gli ultimi turni in un testo da dare al modello.
    """
    if not history:
        return "Aucune conversation précédente avec cette personne."
    lines = []
    for turn in history:
        user_line = turn.get("user", "").strip()
        bot_line  = turn.get("bot", "").strip()
        ts        = turn.get("timestamp", "?")
        if user_line:
            lines.append(f"[{ts}] Utilisateur: {user_line}")
        if bot_line:
            lines.append(f"[{ts}] Robot: {bot_line}")
    return "\n".join(lines)

# -------------------------
# 2. Profilo (long-term memory)
# -------------------------

def load_profile(name: str) -> dict:
    """
    Carica il profilo long-term dell'utente.
    Se non esiste ancora, restituisce un profilo vuoto di default.
    """
    path = _profile_path(name)
    if not os.path.exists(path):
        # profilo vuoto iniziale
        return {
            "name": name,
            "known_since": datetime.now().strftime("%Y-%m-%d"),
            "age": None,
            "gender": None,
            "occupation": None,
            "interests": [],
            "personality": None,
            "goals": [],
            "notes_summary": "",
            "recent_conversations": [],
            "last_update": None
        }

    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        # se c'è un problema di parsing, fallback a base
        return {
            "name": name,
            "known_since": datetime.now().strftime("%Y-%m-%d"),
            "age": None,
            "gender": None,
            "occupation": None,
            "interests": [],
            "personality": None,
            "goals": [],
            "notes_summary": "",
            "recent_conversations": [],
            "last_update": None
        }

def save_profile(name: str, profile: dict):
    """
    Salva/aggiorna il profilo long-term dell'utente.
    """
    path = _profile_path(name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(profile, f, ensure_ascii=False, indent=2)

def format_profile_for_prompt(profile: dict) -> str:
    """
    Serializza il profilo in forma leggibile dal modello.
    """
    return json.dumps(profile, ensure_ascii=False, indent=2)

def update_profile_notes(name: str, new_note: str):
    """
    Aggiorna campo 'notes' del profilo aggiungendo un'annotazione libera.
    Per ora semplice append testuale.
    (Lo useremo più avanti per Step 2 e Step 3)
    """
    prof = load_profile(name)
    notes = prof.get("notes", "")
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    prof["notes"] = (notes + f"\n[{timestamp}] {new_note}").strip()
    save_profile(name, prof)
