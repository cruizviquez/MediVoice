from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Tuple

from starlette.concurrency import run_in_threadpool

from .redact import redact_phi

# Groq model configurable via env var
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

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
    "- Otherwise safe_patient_reply MUST be operational only: acknowledgement + pharmacist follow-up.\n\n"
    "Output rules:\n"
    "- Return ONLY JSON that matches the IntakeResult schema provided.\n"
    "- Do not wrap JSON in markdown.\n"
)

# Patterns that often create unsafe medical advice in a patient-facing reply
_ADVICE_PATTERNS = [
    r"\bdon't stop\b", r"\bdo not stop\b", r"\bstop taking\b", r"\bstart taking\b",
    r"\bcontinue taking\b", r"\bincrease\b", r"\bdecrease\b",
    r"\bchange (?:your|the) dose\b", r"\byou should\b", r"\byou shouldn't\b",
    r"\btake (?:your|the)\b",
    r"\bnot the right medication\b", r"\balternatives?\b", r"\bswitch\b", r"\bchange to\b",
    r"\bpossible alternatives\b", r"\btry a different\b",
]

# Very lightweight symptom heuristics (for demo)
_SYMPTOM_KEYWORDS = [
    "dizzy", "dizziness", "nausea", "vomit", "rash", "hives",
    "headache", "pain", "fever", "swelling", "short of breath",
    "trouble breathing", "can't breathe", "cannot breathe", "chest pain",
    "faint", "passed out", "bleeding"
]

_RED_FLAG_KEYWORDS = [
    "chest pain", "trouble breathing", "can't breathe", "cannot breathe",
    "fainting", "passed out", "severe allergic", "anaphyl", "swelling of face"
]


def _has_symptoms(t: str) -> bool:
    tl = (t or "").lower()
    return any(k in tl for k in _SYMPTOM_KEYWORDS)


def _has_red_flags(t: str) -> bool:
    tl = (t or "").lower()
    return any(k in tl for k in _RED_FLAG_KEYWORDS)


def sanitize_safe_reply(text: str, risk_level: str, has_symptoms: bool) -> str:
    """
    Ensure safe_patient_reply is operational and non-clinical.
    Only include urgent-care language if (a) high risk or (b) symptoms present.
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
        # Medium/low: no med advice; operational only
        out = (
            "Thanks for letting us know. A pharmacist will follow up to review your medication concern and discuss next steps."
        )
        if has_symptoms:
            out += " If symptoms worsen or you feel unsafe, please seek urgent care."
        return out

    # If it looks okay but is missing safety guidance, add it conditionally
    if risk_level == "high":
        if "emergency" not in lower and "urgent care" not in lower:
            t += " If you feel unsafe or have severe symptoms, please seek urgent care or call emergency services."
    else:
        # Only append this if symptoms exist (avoid weird messaging for admin-only calls)
        if has_symptoms and ("urgent care" not in lower and "feel unsafe" not in lower):
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
    Safe deterministic fallback if no GROQ key or API fails.
    """
    t = (transcript or "").strip()
    sym = _has_symptoms(t)
    return {
        "intent": "unknown" if len(t.split()) < 4 else "general_question",
        "risk_level": "low",
        "key_facts": [t] if t else [],
        "medications": [],
        "recommended_next_step": "Route to pharmacist outreach queue for MTM follow-up; verify medication list and barriers.",
        "safe_patient_reply": sanitize_safe_reply(
            "Thanks for letting us know. A pharmacist will follow up to review your concern.",
            "low",
            sym,
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

    red_flags = _has_red_flags(t)

    # discontinuation / adherence interruption + side effect language
    stopped = any(k in t for k in ["stopped taking", "i stopped", "not taking", "quit taking"])
    side_effect_words = any(k in t for k in ["dizzy", "dizziness", "nausea", "rash", "side effect", "makes me", "headache"])

    task = data.get("pharmacist_task") or {}
    task.setdefault("queue", "mtm_outreach")
    task.setdefault("priority", "normal")
    task.setdefault("due_in_hours", 72)
    task.setdefault("summary", "Pharmacist follow-up required.")
    task.setdefault("tags", [])

    # 1) Strict red-flag escalation ALWAYS wins
    if red_flags:
        data["risk_level"] = "high"
        task["queue"] = "urgent_escalation"
        task["priority"] = "urgent"
        task["due_in_hours"] = 1
        if "red_flag" not in task["tags"]:
            task["tags"].append("red_flag")
        data["recommended_next_step"] = data.get("recommended_next_step") or "Escalate immediately per protocol."
        data["pharmacist_task"] = task
        return data

    # 2) If model accidentally set urgent_escalation without red flags, downgrade,
    #    but still apply side-effect/adherence logic if relevant.
    if task.get("queue") == "urgent_escalation":
        if side_effect_words or stopped:
            task["queue"] = "side_effect_followup"
            task["priority"] = "high"
            task["due_in_hours"] = 24
            if "side_effects" not in task["tags"]:
                task["tags"].append("side_effects")
            if stopped and "adherence" not in task["tags"]:
                task["tags"].append("adherence")
            if intent == "general_question":
                data["intent"] = "side_effects"
            data["risk_level"] = "medium"
            data["recommended_next_step"] = (
                "Pharmacist outreach within 24 hours to address side-effect concern and adherence barrier."
            )
        else:
            task["queue"] = "mtm_outreach"
            task["priority"] = "normal"
            task["due_in_hours"] = 72

        data["pharmacist_task"] = task
        return data

    # 3) Side effects + stopped meds: high-value workflow
    if stopped and side_effect_words:
        task["queue"] = "side_effect_followup"
        task["priority"] = "high"
        task["due_in_hours"] = min(int(task.get("due_in_hours", 24)), 24)
        if intent == "general_question":
            data["intent"] = "side_effects"
        data["risk_level"] = "medium"
        data["recommended_next_step"] = (
            "Pharmacist outreach within 24 hours to address side-effect concern and adherence barrier."
        )
        if "side_effects" not in task["tags"]:
            task["tags"].append("side_effects")
        if "adherence" not in task["tags"]:
            task["tags"].append("adherence")

    # 4) If intent indicates meaningful follow-up, tighten SLA
    elif intent in ("adherence_issue", "side_effects"):
        task["priority"] = "high" if task.get("priority") in ("normal", "low") else task.get("priority")
        task["due_in_hours"] = min(int(task.get("due_in_hours", 24)), 24)

    data["pharmacist_task"] = task
    return data


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



def _analyze_sync(raw_transcript: str, schema_json: Dict[str, Any]) -> Tuple[Dict[str, Any], list[str]]:
    start_time = time.time()
    
    # 1) Redact PHI BEFORE any external call
    redaction_start = time.time()
    redacted_transcript, redaction_tags = redact_phi(raw_transcript or "")
    redaction_time = time.time() - redaction_start
    print(f"[llm] Redaction complete: {redaction_time:.2f}s")

    # 2) Groq-only backend
    if not os.getenv("GROQ_API_KEY"):
        print("[llm] Using fallback (no GROQ_API_KEY found)")
        result = _fallback_result(redacted_transcript, redaction_tags)
        total_time = time.time() - start_time
        print(f"[llm] Total analysis time: {total_time:.2f}s")
        return result, redaction_tags

    try:
        print("[llm] Calling Groq API...")
        groq_start = time.time()
        data = _call_groq(redacted_transcript, schema_json, redaction_tags)
        groq_time = time.time() - groq_start
        print(f"[llm] Groq API response: {groq_time:.2f}s")
    except Exception as e:
        error_time = time.time() - start_time
        print(f"[llm] Groq API failed after {error_time:.2f}s: {repr(e)}")
        result = _fallback_result(redacted_transcript, redaction_tags)
        return result, redaction_tags

    # 3) Post-process safety & ops routing
    post_start = time.time()
    sym = _has_symptoms(redacted_transcript)
    risk_level = data.get("risk_level", "low")
    data["safe_patient_reply"] = sanitize_safe_reply(
        data.get("safe_patient_reply", ""),
        risk_level,
        sym,
    )
    data = normalize_task_and_routing(data, redacted_transcript)

    # Normalize common Spanish medication spellings for demo consistency
    meds = data.get("medications") or []
    if meds:
        tl = (redacted_transcript or "").lower()
        if "metformina" in tl or "metamorfina" in tl:
            for med in meds:
                name = (med.get("name") or "").lower().strip()
                if name in {"meth", "metamorfina", "metformin", "metformine", "metformina"}:
                    med["name"] = "metformina"

    # Add redaction tag to task tags (no PHI, just metadata)
    if redaction_tags:
        tags = data.setdefault("pharmacist_task", {}).setdefault("tags", [])
        if "redacted" not in tags:
            tags.append("redacted")

    post_time = time.time() - post_start
    print(f"[llm] Post-processing: {post_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"[llm] Total analysis time: {total_time:.2f}s")
    
    return data, redaction_tags


async def analyze_transcript(transcript: str, schema_json: Dict[str, Any]) -> Tuple[Dict[str, Any], list[str]]:
    return await run_in_threadpool(_analyze_sync, transcript, schema_json)
