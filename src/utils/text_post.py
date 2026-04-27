import re


def clean_llm_reply(raw: str, state: str = "FREE_TALK", is_first_turn: bool = False) -> str:
    text = raw.strip()

    # Remove leading greeting phrases on non-first turns
    if not is_first_turn:
        text = re.sub(
            r"^(bonjour[\s,!]*|salut[\s,!]*|bonsoir[\s,!]*|coucou[\s,!]*|hey[\s,!]*)+",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

    # Strip questions during FAREWELL
    if state == "FAREWELL":
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s for s in sentences if not s.strip().endswith("?")]
        text = " ".join(sentences).strip()

    # Remove repeated consecutive phrases (e.g. "bien sûr bien sûr")
    text = re.sub(r"\b(.{4,}?)\s+\1\b", r"\1", text, flags=re.IGNORECASE)

    return text if text else raw.strip()
