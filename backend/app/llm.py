from __future__ import annotations

import json
import os
from typing import Any, Dict

from starlette.concurrency import run_in_threadpool
from openai import OpenAI

from .redact import redact_phi

# Model can be configured via env var; defaults to a good demo choice.
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


SYSTEM_PROMPT = (
    "You are a clinical operations assistant supporting a Medication Therapy Management (MTM) team.\n"
    "You must NOT provide medical advice, diagnosis, or prescribing instructions.\n"
    "Your job is to triage the transcript into structured workflow artifacts and documentation.\n\n"
    "Safety rules:\n"
    "- If transcript includes red-flag symptoms (e.g., chest pain, trouble breathing, severe allergic reaction), set:\n"
    "  risk_level='high', pharmacist_task.queue='urgent_escalation', pharmacist_task.priority='urgent', due_in_hours=1,\n"
    "  and safe_patient_reply must instruct urgent care / emergency services.\n"
    "- Otherwise, provide a cautious, non-medical safe_patient_reply.\n\n"
    "Output rules:\n"
    "- Return ONLY JSON that matches the IntakeResult schema provided.\n"
    "- Do not wrap JSON in markdown.\n"
)

def _fallback_result(transcript: str, redaction_tags: list[str]) -> Dict[str, Any]:
    """Safe deterministic fallback if OPENAI_API_KEY is missing or the API call fails."""
    t = (transcript or "").strip()
    return {
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
    # We do NOT include raw PHI. We optionally tell the model that some identifiers were redacted.
    redaction_note = (
        f"Note: Identifiers were redacted before analysis. Redaction tags present: {redaction_tags}\n\n"
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

def _call_openai_sync(transcript: str, schema_json: Dict[str, Any]) -> Dict[str, Any]:
    # 1) Redact PHI BEFORE calling any external model
    redacted_transcript, redaction_tags = redact_phi(transcript or "")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _fallback_result(redacted_transcript, redaction_tags)

    client = OpenAI(api_key=api_key)
    model = DEFAULT_MODEL

    try:
        resp = client.chat.completions.create(
            model=model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(redacted_transcript, schema_json, redaction_tags)},
            ],
        )
        content = (resp.choices[0].message.content or "").strip()
        data = json.loads(content)

        # Optional: add a lightweight tag to task tags to show redaction happened (no PHI leaked).
        # Only if the schema allows it (it does in your PharmacistTask.tags list).
        try:
            if redaction_tags:
                data.setdefault("pharmacist_task", {}).setdefault("tags", [])
                if "redacted" not in data["pharmacist_task"]["tags"]:
                    data["pharmacist_task"]["tags"].append("redacted")
        except Exception:
            pass

        return data

    except Exception:
        # If OpenAI fails for any reason, keep the pipeline working safely.
        return _fallback_result(redacted_transcript, redaction_tags)

async def analyze_transcript(transcript: str, schema_json: Dict[str, Any]) -> Dict[str, Any]:
    # OpenAI SDK call is sync; run it in a threadpool so FastAPI stays responsive.
    return await run_in_threadpool(_call_openai_sync, transcript, schema_json)
