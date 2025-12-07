import re
from typing import Dict, List, Tuple


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d[\d\s\-\(\)]{8,}\d)")
SECRET_RE = re.compile(r"(?:api[_-]?key|token|secret|password)\s*[:=]\s*([A-Za-z0-9_\-]{6,})", re.IGNORECASE)
OPENAI_RE = re.compile(r"sk-[A-Za-z0-9]{12,}", re.IGNORECASE)
TOXIC_HINTS = ["hate", "kill you", "idiot", "stupid"]


def redact_text(text: str) -> Tuple[str, Dict[str, bool]]:
    flags = {"pii": False, "secret": False}
    new_text = text

    if EMAIL_RE.search(new_text) or PHONE_RE.search(new_text):
        new_text = EMAIL_RE.sub("[REDACTED_EMAIL]", new_text)
        new_text = PHONE_RE.sub("[REDACTED_PHONE]", new_text)
        flags["pii"] = True

    if SECRET_RE.search(new_text) or OPENAI_RE.search(new_text):
        new_text = SECRET_RE.sub("[REDACTED_SECRET]", new_text)
        new_text = OPENAI_RE.sub("[REDACTED_SECRET]", new_text)
        flags["secret"] = True

    return new_text, flags


def scrub_messages(messages: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], Dict[str, bool]]:
    """Redact PII/secrets and surface skip tags."""
    combined_flags = {"pii": False, "secret": False, "no_rag": False, "no_train": False, "toxic": False}
    cleaned: List[Dict[str, str]] = []

    for msg in messages:
        content = msg.get("content", "")
        lower = content.lower()

        if "#no-rag" in lower:
            combined_flags["no_rag"] = True
        if "#no-train" in lower:
            combined_flags["no_train"] = True
        if any(hint in lower for hint in TOXIC_HINTS):
            combined_flags["toxic"] = True

        redacted, flags = redact_text(content)
        combined_flags["pii"] = combined_flags["pii"] or flags["pii"]
        combined_flags["secret"] = combined_flags["secret"] or flags["secret"]

        cleaned.append({**msg, "content": redacted})

    return cleaned, combined_flags
