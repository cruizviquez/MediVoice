from __future__ import annotations
import json
import httpx
from typing import Any, Dict, Optional

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1"

SYSTEM_PROMPT = (
    "You are a clinical operations assistant for a Medication Therapy Management (MTM) team. "
    "You must be cautious: do not give medical advice or diagnosis. "
    "Your job is to triage the message and produce structured intake for pharmacist outreach. "
    "If red-flag symptoms are mentioned (e.g., chest pain, trouble breathing, severe allergic reaction), "
    "set risk_level=high, pharmacist_task.priority=urgent, pharmacist_task.queue=urgent_escalation, "
    "and instruct the patient to seek urgent care or call emergency services. "
    "Always produce JSON matching the provided schema."
)

def _make_soap(transcript: str, intent: str, risk: str) -> Dict[str, Any]:
    assessment_bits = []
    if intent == "adherence_issue":
        assessment_bits.append("Medication adherence barrier reported")
    if intent == "side_effects":
        assessment_bits.append("Possible medication intolerance or side-effect concern reported (non-diagnostic)")
    if intent == "refill_request":
        assessment_bits.append("Refill/medication access barrier reported")
    if intent == "appointment_request":
        assessment_bits.append("Care coordination / appointment request reported")
    if not assessment_bits:
        assessment_bits.append("Patient inquiry requires pharmacist review")

    plan_bits = ["Review medication list and context during pharmacist follow-up"]
    if risk == "high":
        plan_bits.append("Escalate per protocol and confirm patient has urgent care guidance")
    else:
        plan_bits.append("Contact patient for MTM follow-up and clarify details (dose, timing, barriers)")

    return {
        "subjective": transcript.strip() or "Patient left a message requesting assistance.",
        "objective": "N/A",
        "assessment": "; ".join(assessment_bits),
        "plan": ". ".join(plan_bits) + "."
    }

def _make_task(transcript: str, intent: str, risk: str) -> Dict[str, Any]:
    if risk == "high":
        return {
            "queue": "urgent_escalation",
            "priority": "urgent",
            "due_in_hours": 1,
            "summary": "Urgent symptom red-flag mentioned; follow escalation protocol and contact patient immediately.",
            "tags": ["red_flag", "urgent"]
        }

    mapping = {
        "adherence_issue": ("adherence_outreach", "high", 24, "Adherence barrier reported; contact patient within 24h."),
        "side_effects": ("side_effect_followup", "high", 24, "Side-effect concern reported; pharmacist follow-up within 24h."),
        "refill_request": ("refill_support", "normal", 48, "Refill/access issue; assist patient with refill process."),
        "appointment_request": ("provider_coordination", "normal", 48, "Appointment/care coordination request; route to coordination."),
        "general_question": ("mtm_outreach", "normal", 72, "General medication question; schedule pharmacist follow-up."),
        "unknown": ("mtm_outreach", "normal", 72, "Unclear request; pharmacist review needed.")
    }
    q, p, due, summ = mapping.get(intent, mapping["unknown"])
    return {"queue": q, "priority": p, "due_in_hours": due, "summary": summ, "tags": [intent]}

def _rule_based(transcript: str) -> Dict[str, Any]:
    t = transcript.lower()
    intent = "unknown"
    if any(k in t for k in ["missed", "forgot", "not taking", "stopped taking", "ran out"]):
        intent = "adherence_issue"
    elif any(k in t for k in ["dizzy", "nausea", "rash", "side effect", "makes me", "hurt"]):
        intent = "side_effects"
    elif any(k in t for k in ["refill", "renew", "pharmacy", "out of"]):
        intent = "refill_request"
    elif any(k in t for k in ["appointment", "schedule", "visit", "see my doctor"]):
        intent = "appointment_request"
    else:
        intent = "general_question" if len(t.split()) > 3 else "unknown"

    high_risk = any(k in t for k in ["chest pain", "can't breathe", "trouble breathing", "swelling of face", "passed out", "fainting"])
    risk = "high" if high_risk else ("medium" if intent in ["adherence_issue","side_effects"] else "low")

    safe_reply = (
        "Thanks for sharing that. I’m going to flag this for a pharmacist to review right away. "
        "If you’re experiencing severe symptoms or feel unsafe, please seek urgent care or call emergency services."
        if risk == "high"
        else
        "Thanks for sharing that. I’ll flag this for a pharmacist to review and follow up. "
        "If symptoms worsen or you feel unsafe, please seek urgent care."
    )

    soap_note = _make_soap(transcript, intent, risk)
    pharmacist_task = _make_task(transcript, intent, risk)

    return {
        "intent": intent,
        "risk_level": risk,
        "key_facts": [transcript.strip()] if transcript.strip() else [],
        "medications": [],
        "recommended_next_step": "Route to pharmacist outreach queue for MTM follow-up; verify medication list and adherence barriers.",
        "safe_patient_reply": safe_reply,
        "soap_note": soap_note,
        "pharmacist_task": pharmacist_task
    }

async def analyze_with_ollama(
    transcript: str,
    schema_json: Dict[str, Any],
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_MODEL,
    timeout_s: float = 30.0
) -> Optional[Dict[str, Any]]:
    """Try to call a local Ollama model. Returns dict if successful, else None."""
    user_prompt = (
        "Given this patient transcript, return ONLY valid JSON for the IntakeResult schema.\n"
        f"Schema (JSON Schema-like): {json.dumps(schema_json)}\n"
        f"Transcript: {transcript}\n"
        "Return JSON only."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False
    }

    try:
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            r = await client.post(f"{ollama_url}/api/chat", json=payload)
            r.raise_for_status()
            data = r.json()
            content = data.get("message", {}).get("content", "")
    except Exception:
        return None

    content = (content or "").strip()
    if not content:
        return None

    if content.startswith("```"):
        parts = content.split("```")
        if len(parts) >= 3:
            content = parts[1].strip()

    try:
        return json.loads(content)
    except Exception:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(content[start:end+1])
            except Exception:
                return None
        return None

async def analyze_transcript(transcript: str, schema_json: Dict[str, Any]) -> Dict[str, Any]:
    result = await analyze_with_ollama(transcript, schema_json)
    if result is None:
        result = _rule_based(transcript)
    return result
