# backend/app/redact.py
from __future__ import annotations

import re
from typing import Tuple

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b")
SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
MRN_RE = re.compile(r"\b(?:MRN|Member\s*ID|Policy\s*ID|ID)\s*[:#]?\s*[A-Z0-9\-]{6,}\b", re.IGNORECASE)
DOB_RE = re.compile(r"\b(?:DOB|Date of Birth)\s*[:#]?\s*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", re.IGNORECASE)

# loose date patterns like 01/02/2024 or 1-2-24
DATE_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")

# month-name dates like "January 15, 2024" or "Feb 3rd 2024"
MONTH_DATE_RE = re.compile(
    r"\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?"
    r"|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
    r"\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s+\d{2,4})\b",
    re.IGNORECASE
)

# stricter address pattern (requires street suffix)
ADDRESS_RE = re.compile(
    r"\b\d{1,6}\s+(?:[A-Z0-9]+\s+){0,6}"
    r"(?:st|street|ave|avenue|rd|road|blvd|boulevard|ln|lane|dr|drive|ct|court)\b",
    re.IGNORECASE
)

# Name hints for demo (English + Spanish).
# This catches "My name is Carlos Ruiz", "I'm Carlos Ruiz", "Me llamo Carlos Ruiz", "Soy Carlos Ruiz", etc.
NAME_HINT_RE = re.compile(
    r"\b(?:my name is|this is|i am|i'm|me llamo|mi nombre es|soy)\s+"
    r"([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+){0,2})\b",
    re.IGNORECASE
)


def redact_phi(text: str) -> Tuple[str, list[str]]:
    """
    Returns (redacted_text, tags_found).
    Demo-level redaction. For production, use an NLP PHI detector (e.g., Presidio) + review.
    """
    tags: list[str] = []

    def sub(pattern: re.Pattern, tag: str, s: str) -> str:
        nonlocal tags
        if pattern.search(s):
            tags.append(tag)
        return pattern.sub(f"[{tag}]", s)

    s = text or ""

    # Name hints first
    if NAME_HINT_RE.search(s):
        tags.append("NAME")
        # Replace just the name portion with [NAME]
        s = NAME_HINT_RE.sub(lambda m: m.group(0).replace(m.group(1), "[NAME]"), s)

    s = sub(EMAIL_RE, "EMAIL", s)
    s = sub(PHONE_RE, "PHONE", s)
    s = sub(SSN_RE, "SSN", s)
    s = sub(MRN_RE, "MEMBER_ID", s)
    s = sub(DOB_RE, "DOB", s)
    s = sub(MONTH_DATE_RE, "DATE", s)
    s = sub(DATE_RE, "DATE", s)

    # Address last because it's broad
    s = sub(ADDRESS_RE, "ADDRESS", s)


    return s, sorted([f"REDACTED_{tag}" for tag in set(tags)])
