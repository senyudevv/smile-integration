import os
import json
import pickle
from datetime import datetime

from src.config import CONVERSATIONS_DIR, EMBEDDINGS_FILE


def log_full_conversation(name: str, user_text: str, bot_reply: str) -> None:
    os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
    path = os.path.join(CONVERSATIONS_DIR, f"{name}.json")

    history = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                history = json.load(f)
        except (json.JSONDecodeError, ValueError):
            history = []

    history.append({
        "timestamp": datetime.now().isoformat(),
        "user": user_text,
        "bot": bot_reply,
    })

    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def save_new_face(name: str, embedding) -> None:
    known = {}
    if os.path.exists(EMBEDDINGS_FILE):
        try:
            with open(EMBEDDINGS_FILE, "rb") as f:
                known = pickle.load(f)
        except Exception:
            known = {}

    known[name] = embedding

    os.makedirs(os.path.dirname(EMBEDDINGS_FILE), exist_ok=True)
    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(known, f)

    print(f"[MEMORY] Visage de '{name}' sauvegardé.")
