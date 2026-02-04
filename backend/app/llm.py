from __future__ import annotations

import json
import os
from typing import Any, Dict

from starlette.concurrency import run_in_threadpool

from .redact import redact_phi

SYSTEM_PROMPT = (
    "You are a clinical operations assistant supporting a Medication Therapy Management (MTM) team.\n"
    "You must NOT provide medical advice, diagnosis, or prescribing instructions.\n"
    "Your job is to triage the transcript into structured workflow artifacts and documentation.\n\n"
    "Safety rules:\n"
    "- If transcript includes red-flag symptoms (e.g., chest pain, trouble breathing, severe allergic reaction), set:\n"
    "  risk_level='high', pharmacist_task.queue='urgent_escalation', pharmacist_task.priority='urgent', due_in_hours=1,\n"
    "  and safe_patient_reply must instruct urgent care / emergency services.\n\n"
    "Output rules:\n"
    "- Return ONLY JSON that matches the IntakeResult schema provided.\n"
    "- Do not wrap JSON in markdown.\n"
)

DEFAULT_OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama3-70b-8192")

def _fallback_result(transcript: str, redaction_tags: list[str]) -> Dict[str, Any]:
    t = (transcript or "").strip()
    return {
        "agent_backend_used": "fallback",
        "intent": "unknown" if len(t.split()) < 4 else "general_question",
        "risk_level": "low",
        "key_facts": [t] if t else [],
        "medications": [],
        "recommended_next_step": "Route to pharmacist outreach queue for MTM follow-up; verify medication list and barriers.",
        "safe_patient_reply": (
            "Thanks for sharing that. I’ll flag this for a pharmacist to review and follow up. "
            "If symptoms worsen or you feel unsafe, please seek urgent care."
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

def _build_user_prompt(redacted_transcript: str, schema_json: Dict[str, Any], redaction_tags: list[str]) -> str:
    redaction_note = (
        f"Note: identifiers were redacted. Tags present: {redaction_tags}\n\n"
        if redaction_tags else ""
    )
    return (
        "Convert the following patient transcript into an IntakeResult JSON object.\n"
        "Follow the schema strictly and do not include medical advice.\n\n"
        f"{redaction_note}"
        f"Transcript:\n{redacted_transcript}\n\n"
        f"Schema:\n{json.dumps(schema_json)}\n\n"
        "Return ONLY JSON."
    )

def _call_openai(transcript: str, schema_json: Dict[str, Any], redaction_tags: list[str]) -> Dict[str, Any]:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=DEFAULT_OPENAI_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(transcript, schema_json, redaction_tags)},
        ],
    )
    content = (resp.choices[0].message.content or "").strip()
    data = json.loads(content)
    data["agent_backend_used"] = "openai"
    return data

def _call_groq(transcript: str, schema_json: Dict[str, Any], redaction_tags: list[str]) -> Dict[str, Any]:
    from groq import Groq

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model=DEFAULT_GROQ_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(transcript, schema_json, redaction_tags)},
        ],
    )
    content = (resp.choices[0].message.content or "").strip()
    data = json.loads(content)
    data["agent_backend_used"] = "groq"
    return data

def _analyze_sync(raw_transcript: str, schema_json: Dict[str, Any]) -> Dict[str, Any]:
    redacted, tags = redact_phi(raw_transcript or "")

    # Prefer Groq if present; else OpenAI; else fallback
    try:
        if os.getenv("GROQ_API_KEY"):
            print("LLM backend: groq")
            return _call_groq(redacted, schema_json, tags)
        if os.getenv("OPENAI_API_KEY"):
            print("LLM backend: openai")
            return _call_openai(redacted, schema_json, tags)
        print("LLM backend: fallback (no keys)")
        return _fallback_result(redacted, tags)
    except Exception as e:
        print("LLM_CALL_FAILED:", repr(e))
        return _fallback_result(redacted, tags)

async def analyze_transcript(transcript: str, schema_json: Dict[str, Any]) -> Dict[str, Any]:
    return await run_in_threadpool(_analyze_sync, transcript, schema_json)
