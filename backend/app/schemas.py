from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class MedicationMention(BaseModel):
    name: str = Field(..., description="Medication name as mentioned by the patient")
    issue: Optional[str] = Field(None, description="e.g., missed doses, side effects, cost, confusion")

class SoapNote(BaseModel):
    subjective: str = Field(..., description="Patient-reported information (symptoms, concerns, barriers).")
    objective: str = Field(..., description="Objective or context info available. If none, say 'N/A'.")
    assessment: str = Field(..., description="Non-diagnostic assessment of the situation (e.g., adherence barrier, possible intolerance). No medical advice.")
    plan: str = Field(..., description="Operational plan for pharmacist/care team (follow-up, verify med list, provider outreach). No prescribing guidance.")

class PharmacistTask(BaseModel):
    queue: Literal["mtm_outreach","adherence_outreach","side_effect_followup","refill_support","provider_coordination","urgent_escalation"] = "mtm_outreach"
    priority: Literal["low","normal","high","urgent"] = "normal"
    due_in_hours: int = Field(24, ge=1, le=168, description="When this task should be acted on.")
    summary: str = Field(..., description="Short task summary for a pharmacist.")
    tags: List[str] = Field(default_factory=list)

class IntakeResult(BaseModel):
    intent: Literal["adherence_issue","side_effects","refill_request","appointment_request","general_question","unknown"] = "unknown"
    risk_level: Literal["low","medium","high"] = "low"
    key_facts: List[str] = Field(default_factory=list)
    medications: List[MedicationMention] = Field(default_factory=list)

    recommended_next_step: str = Field(..., description="What should happen next operationally (e.g., pharmacist outreach, schedule consult).")
    safe_patient_reply: str = Field(..., description="Patient-facing response with safety language (no medical advice).")

    soap_note: SoapNote
    pharmacist_task: PharmacistTask

class VoiceIntakeResponse(BaseModel):
    transcript: str
    language: Optional[str] = None
    result: IntakeResult
