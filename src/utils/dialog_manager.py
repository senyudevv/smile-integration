# utils/dialog_manager.py
import requests
import json
from datetime import datetime

# 🔴 IMPORT GIUSTI
# prendi il profilo e la history SOLO da profile_manager
from src.utils.profile_manager import (
    load_profile, save_profile,
    load_recent_history,
    format_profile_for_prompt,
    format_history_for_prompt,
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3:8b"


def build_llm_prompt(
    user_name: str,
    user_text: str,
    is_first_turn: bool = False,
    state: str = "FREE_TALK",
) -> str:
    profile = load_profile(user_name)
    history = load_recent_history(user_name, window=7)

    profile_txt = format_profile_for_prompt(profile)
    history_txt = format_history_for_prompt(history)
    notes_summary = profile.get("notes_summary", "").strip() or "(aucune mémoire épisodique disponible)"

    if state == "GREETING":
        stage = "C'est le début de la conversation. Tu peux saluer brièvement et te présenter naturellement."
    elif state == "FAREWELL":
        stage = "La conversation se termine. Réponds avec un au revoir chaleureux, ne relance pas."
    else:
        stage = "La conversation est déjà en cours. NE salue PAS à nouveau."

    prompt = f"""
Tu es "Robot", un assistant robotique qui parle en français, avec un ton chaleureux et naturel.
Tu parles avec {user_name}, que tu connais.

ÉTAT ACTUEL : {state.upper()}
{stage}

⚠️ RÈGLES ⚠️
- NE commence PAS chaque réponse par "Bonjour" ou le prénom, sauf au tout premier tour.
- Si l'état est FREE_TALK, traite les "bonjour" comme une partie normale de la conversation.
- Si l'état est FAREWELL, fais seulement un salut final, sans poser de questions.
- Réponses courtes : 2-3 phrases.

📘 MÉMOIRE LONG TERME :
{profile_txt}

🧠 MÉMOIRE ÉPISODIQUE (résumés précédents) :
{notes_summary}

💬 MÉMOIRE COURT TERME (7 derniers échanges) :
{history_txt}

🗣️ L'UTILISATEUR DIT :
{user_text}

Réponds en tant que "Robot" :
Robot:
""".strip()

    return prompt


def ask_ollama(prompt: str, model: str = MODEL_NAME) -> str:
    data = {"model": model, "prompt": prompt, "stream": False}
    resp = requests.post(OLLAMA_URL, json=data)
    resp.raise_for_status()
    return resp.json().get("response", "")


def ask_ollama_with_context(
    user_name: str,
    user_text: str,
    is_first_turn: bool = False,
    state: str = "FREE_TALK",
) -> str:
    prompt = build_llm_prompt(user_name, user_text, is_first_turn=is_first_turn, state=state)
    return ask_ollama(prompt)

def summarize_conversation(name, conversation):
    """
    Riassume la conversazione, deduce informazioni sull'utente e aggiorna il profilo.
    Ora integra sesso, età, interessi, tono, personalità e obiettivi.
    """
    try:
        profile = load_profile(name)

        # --- 1. Prepara testo conversazione ---
        dialogue_text = "\n".join(
            [f"Utilisateur: {x['user']}\nAssistant: {x['bot']}" for x in conversation]
        )

        # --- 2. Prepara prompt ---
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

        response = ask_ollama(prompt, model=MODEL_NAME)

        # --- 3. Parsa output LLM ---
        try:
            data = json.loads(response)
        except Exception:
            # fallback: tenta di isolare il blocco JSON
            start = response.find("{")
            end = response.rfind("}") + 1
            data = json.loads(response[start:end]) if start != -1 and end != -1 else {}

        # --- 4. Merge intelligente ---
        profile["notes_summary"] = data.get("summary", profile.get("notes_summary", ""))

        # aggiorna solo se mancante o migliorabile
        for key in ["gender", "age", "occupation", "personality"]:
            val = data.get(key)
            if val and (profile.get(key) in [None, ""]):
                profile[key] = val

        # merge di liste senza duplicati
        def merge_list(a, b):
            return list(set((a or []) + (b or [])))

        profile["interests"] = merge_list(profile.get("interests", []), data.get("interests", []))
        profile["goals"] = merge_list(profile.get("goals", []), data.get("goals", []))

        # Salva anche le ultime conversazioni recenti
        profile["recent_conversations"] = conversation[-5:]
        profile["last_update"] = datetime.now().isoformat()

        # --- 5. Salva ---
        save_profile(name, profile)
        print(f"[MEMORY] ✅ Profilo di {name} aggiornato correttamente con nuove informazioni.")
        return profile

    except Exception as e:
        print(f"[MEMORY] Errore durante il riassunto: {e}")
        return None


