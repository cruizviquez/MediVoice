from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Tuple

from starlette.concurrency import run_in_threadpool

from .redact import redact_phi

# Models configurable via env vars
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")  
DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT = (
    "You are a clinical operations assistant supporting a Medication Therapy Management (MTM) team.\n"
    "You must NOT provide medical advice, diagnosis, or prescribing instructions.\n"
    "Do NOT tell the patient to start, stop, continue, or change any medication dose.\n"
    "Do NOT provide treatment recommendations.\n"
    "Your job is to triage the transcript into structured workflow artifacts and documentation for pharmacist review.\n\n"
    "Safety rules:\n"
    "- If transcript includes red-flag symptoms (e.g., chest pain, trouble breathing, severe allergic reaction), set:\n"
    "  risk_level='high', pharmacist_task.queue='urgent_escalation', pharmacist_task.priority='urgent', due_in_hours=1,\n"
    "  and safe_patient_reply must instruct urgent care / emergency services.\n"
    "- Otherwise safe_patient_reply MUST be operational only: acknowledgement + pharmacist follow-up + escalation guidance.\n\n"
    "Output rules:\n"
    "- Return ONLY JSON that matches the IntakeResult schema provided.\n"
    "- Do not wrap JSON in markdown.\n"
)

# Phrases that often cause unsafe "medical advice" in patient replies
_ADVICE_PATTERNS = [
    r"\bdon't stop\b", r"\bdo not stop\b", r"\bstop taking\b", r"\bstart taking\b",
    r"\bcontinue taking\b", r"\bincrease\b", r"\bdecrease\b",
    r"\bchange (?:your|the) dose\b", r"\byou should\b", r"\byou shouldn't\b",
    r"\btake (?:your|the)\b",
    r"\bnot the right medication\b", r"\balternatives?\b", r"\bswitch\b", r"\bchange to\b",
]

def sanitize_safe_reply(text: str, risk_level: str) -> str:
    """
    Ensure patient reply is operational and non-clinical.
    """
    t = (text or "").strip()
    if not t:
        t = "Thanks for letting us know. A pharmacist will follow up to review your concern."

    lower = t.lower()
    if any(re.search(p, lower) for p in _ADVICE_PATTERNS):
        # Replace with compliant templates
        if risk_level == "high":
            return (
                "Thanks for letting us know. A pharmacist will follow up as soon as possible. "
                "If you have severe symptoms such as chest pain, trouble breathing, fainting, "
                "or you feel unsafe, please seek urgent care or call emergency services right away."
            )
        return (
            "Thanks for letting us know. A pharmacist will follow up to review your medication concern and discuss next steps. "
            "If symptoms worsen or you feel unsafe, please seek urgent care."
        )

    # Add safety line if missing
    if risk_level == "high":
        if "emergency" not in lower and "urgent care" not in lower:
            t += " If you feel unsafe or have severe symptoms, please seek urgent care or call emergency services."
    else:
        if "urgent care" not in lower and "feel unsafe" not in lower:
            t += " If symptoms worsen or you feel unsafe, please seek urgent care."
    return t

def _build_user_prompt(redacted_transcript: str, schema_json: Dict[str, Any], redaction_tags: list[str]) -> str:
    redaction_note = (
        f"Note: identifiers were redacted before analysis. Redaction tags present: {redaction_tags}\n\n"
        if redaction_tags else ""
    )
    return (
        "Convert the following patient transcript into an IntakeResult JSON object.\n"
        "Follow the schema strictly.\n"
        "Do not include medical advice. Do not instruct the patient to start/stop/change medications.\n\n"
        f"{redaction_note}"
        f"Transcript:\n{redacted_transcript}\n\n"
        f"Schema:\n{json.dumps(schema_json)}\n\n"
        "Return ONLY JSON."
    )

def _fallback_result(transcript: str, redaction_tags: list[str]) -> Dict[str, Any]:
    """
    Safe deterministic fallback if no keys or API fails.
    """
    t = (transcript or "").strip()
    return {
        "intent": "unknown" if len(t.split()) < 4 else "general_question",
        "risk_level": "low",
        "key_facts": [t] if t else [],
        "medications": [],
        "recommended_next_step": "Route to pharmacist outreach queue for MTM follow-up; verify medication list and barriers.",
        "safe_patient_reply": sanitize_safe_reply(
            "Thanks for letting us know. A pharmacist will follow up to review your concern.",
            "low",
        ),
        "soap_note": {
            "subjective": t or "Patient left a message requesting assistance.",
            "objective": "N/A",
            "assessment": "Needs pharmacist review (fallback mode).",
            "plan": "Pharmacist follow-up to clarify medication details, timing, and barriers."
        },
        "pharmacist_task": {
            "queue": "mtm_outreach",
            "priority": "normal",
            "due_in_hours": 72,
            "summary": "Pharmacist follow-up needed (fallback mode).",
            "tags": ["fallback"] + (["redacted"] if redaction_tags else [])
        }
    }

def normalize_task_and_routing(data: Dict[str, Any], redacted_transcript: str) -> Dict[str, Any]:
    """
    Post-processing guardrails so ops routing makes sense even if LLM output is slightly off.
    """
    t = (redacted_transcript or "").lower()
    intent = data.get("intent", "unknown")
    risk = data.get("risk_level", "low")

    # detect critical red flags
    red_flags = any(k in t for k in [
        "chest pain", "trouble breathing", "can't breathe", "cannot breathe",
        "fainting", "passed out", "severe allergic", "swelling of face", "swelling", "anaphyl"
    ])

    # detect discontinuation / adherence interruption + side effect language
    stopped = any(k in t for k in ["stopped taking", "stop taking", "i stopped", "i stopped taking", "not taking", "quit taking"])
    side_effect_words = any(k in t for k in ["dizzy", "dizziness", "nausea", "rash", "side effect", "makes me", "headache"])

    task = data.get("pharmacist_task") or {}
    # Ensure required keys exist if missing
    task.setdefault("queue", "mtm_outreach")
    task.setdefault("priority", "normal")
    task.setdefault("due_in_hours", 72)
    task.setdefault("summary", "Pharmacist follow-up required.")
    task.setdefault("tags", [])

    # Strict red-flag escalation
    if red_flags:
        data["risk_level"] = "high"
        task["queue"] = "urgent_escalation"
        task["priority"] = "urgent"
        task["due_in_hours"] = 1
        if "red_flag" not in task["tags"]:
            task["tags"].append("red_flag")
        data["recommended_next_step"] = data.get("recommended_next_step") or "Escalate immediately per protocol."
    
    # Downgrade accidental urgent tasks if no red flags
    if not red_flags and task.get("queue") == "urgent_escalation":
        task["queue"] = "side_effect_followup" if side_effect_words or stopped else "mtm_outreach"
        task["priority"] = "high" if (stopped and side_effect_words) else "normal"
        task["due_in_hours"] = 24 if (stopped and side_effect_words) else 72
    
    elif stopped and side_effect_words:
        # Operationally: this is a side effect follow-up with adherence barrier
        task["queue"] = "side_effect_followup"
        task["priority"] = "high"
        task["due_in_hours"] = min(int(task.get("due_in_hours", 24)), 24)
        if intent == "general_question":
            data["intent"] = "side_effects"
        # Keep adherence_issue if you want; both are defensible.
        data["recommended_next_step"] = (
            "Pharmacist outreach within 24 hours to address side-effect concern and adherence barrier."
        )
        if "side_effects" not in task["tags"]:
            task["tags"].append("side_effects")
        if "adherence" not in task["tags"]:
            task["tags"].append("adherence")
    else:
        # If intent indicates meaningful follow-up, tighten SLA a bit
        if intent in ("adherence_issue", "side_effects"):
            task["priority"] = "high" if task.get("priority") in ("normal", "low") else task.get("priority")
            task["due_in_hours"] = min(int(task.get("due_in_hours", 24)), 24)

    data["pharmacist_task"] = task
    return data

def _call_openai(redacted_transcript: str, schema_json: Dict[str, Any], redaction_tags: list[str]) -> Dict[str, Any]:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    resp = client.chat.completions.create(
        model=DEFAULT_OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(redacted_transcript, schema_json, redaction_tags)},
        ],
    )
    content = (resp.choices[0].message.content or "").strip()
    return json.loads(content)

def _call_groq(redacted_transcript: str, schema_json: Dict[str, Any], redaction_tags: list[str]) -> Dict[str, Any]:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    resp = client.chat.completions.create(
        model=DEFAULT_GROQ_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(redacted_transcript, schema_json, redaction_tags)},
        ],
    )
    content = (resp.choices[0].message.content or "").strip()
    return json.loads(content)

def _analyze_sync(raw_transcript: str, schema_json: Dict[str, Any]) -> Dict[str, Any]:
    # 1) Redact PHI BEFORE any external call
    redacted_transcript, redaction_tags = redact_phi(raw_transcript or "")

    # 2) Choose backend
    try:
        if os.getenv("GROQ_API_KEY"):
            print("LLM backend: groq")
            data = _call_groq(redacted_transcript, schema_json, redaction_tags)
        elif os.getenv("OPENAI_API_KEY"):
            print("LLM backend: openai")
            data = _call_openai(redacted_transcript, schema_json, redaction_tags)
        else:
            print("LLM backend: fallback (no keys found)")
            return _fallback_result(redacted_transcript, redaction_tags)
    except Exception as e:
        print("LLM_CALL_FAILED:", repr(e))
        return _fallback_result(redacted_transcript, redaction_tags)

    # 3) Post-process safety & ops routing
    risk_level = data.get("risk_level", "low")
    data["safe_patient_reply"] = sanitize_safe_reply(data.get("safe_patient_reply", ""), risk_level)
    data = normalize_task_and_routing(data, redacted_transcript)

    # Add a redaction tag to task tags (no PHI, just metadata)
    try:
        if redaction_tags:
            tags = data.setdefault("pharmacist_task", {}).setdefault("tags", [])
            if "redacted" not in tags:
                tags.append("redacted")
    except Exception:
        pass

    return data

async def analyze_transcript(transcript: str, schema_json: Dict[str, Any]) -> Dict[str, Any]:
    # Run sync calls in a threadpool to avoid blocking FastAPI event loop
    return await run_in_threadpool(_analyze_sync, transcript, schema_json)
