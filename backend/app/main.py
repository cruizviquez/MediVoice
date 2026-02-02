from __future__ import annotations

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import TypeAdapter
from typing import Any, Dict
import tempfile
import os

from .schemas import VoiceIntakeResponse, IntakeResult
from .stt import transcribe_file
from .llm import analyze_transcript

app = FastAPI(title="MedVoice Intake Agent", version="0.1.0")

# Relaxed CORS for demo/dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Provide the IntakeResult schema to the LLM as a structured guide
intake_schema: Dict[str, Any] = TypeAdapter(IntakeResult).json_schema()

@app.get("/", response_class=HTMLResponse)
def home():
    # Simple redirect-style landing
    return """<html><body>
    <h2>MedVoice Intake Agent</h2>
    <p>Open <a href="/demo">/demo</a> for the browser recorder.</p>
    <p>Or POST audio to <code>/voice-intake</code>.</p>
    </body></html>"""

@app.get("/demo", response_class=HTMLResponse)
def demo_page():
    # Serves static HTML (keeps the repo dead simple)
    here = os.path.dirname(__file__)
    demo_path = os.path.join(here, "..", "..", "frontend", "recorder.html")
    with open(demo_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/voice-intake", response_model=VoiceIntakeResponse)
async def voice_intake(audio: UploadFile = File(...)):
    # Save upload to temp file
    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        content = await audio.read()
        tmp.write(content)

    try:
        transcript, language = await transcribe_file(tmp_path)
        analysis = await analyze_transcript(transcript, intake_schema)

        # Validate against our schema; if LLM output is invalid, this will raise.
        result = IntakeResult.model_validate(analysis)
        return VoiceIntakeResponse(transcript=transcript, language=language, result=result)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass

@app.get("/health")
def health():
    return {"ok": True}
