# backend/app/schemas.py
from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


class MedicationIssue(BaseModel):
    name: str = Field(..., description="Medication name as mentioned (may be unverified)")
    issue: str = Field(..., description="Side effect or issue reported")


class SoapNote(BaseModel):
    subjective: str
    objective: str = Field(..., description="Objective data; may be 'Not available (voice-only intake).'")
    assessment: str
    plan: str


class PharmacistTask(BaseModel):
    queue: str = Field(..., description="Workflow queue name (e.g., mtm_outreach, side_effect_followup, urgent_escalation)")
    priority: Literal["low", "normal", "high", "urgent"]
    due_in_hours: int = Field(..., ge=1, le=168)
    summary: str
    tags: List[str] = Field(default_factory=list)


class SafetyResult(BaseModel):
    red_flag_detected: bool = Field(..., description="True if urgent escalation is warranted")
    red_flag_signals: List[str] = Field(default_factory=list, description="Short reasons/signals supporting escalation")
    advice_violation: bool = Field(..., description="True if draft contained medical advice and required sanitization")


class IntakeResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: str = Field(..., description="e.g., adherence_issue, side_effects, refill_request, general_question, unknown")
    risk_level: Literal["low", "medium", "high"]
    key_facts: List[str] = Field(default_factory=list)

    medications: List[MedicationIssue] = Field(default_factory=list)

    recommended_next_step: str
    safe_patient_reply: str

    soap_note: SoapNote
    pharmacist_task: PharmacistTask

    # NEW: safety signals from the LLM + enforcement layer
    safety: SafetyResult


class VoiceIntakeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transcript: str
    language: Optional[str] = None
    result: IntakeResult
    redaction_tags: List[str] = Field(default_factory=list)
