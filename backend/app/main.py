from __future__ import annotations

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import TypeAdapter, BaseModel
from typing import Any, Dict, List
import tempfile
import os
import json
import uuid
from datetime import datetime, timezone

from .schemas import VoiceIntakeResponse, IntakeResult
from .stt import transcribe_file
from .llm import analyze_transcript


class ExampleTranscriptPayload(BaseModel):
    transcript: str


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

HISTORY_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "history")
)


def _ensure_history_dir() -> str:
    os.makedirs(HISTORY_DIR, exist_ok=True)
    return HISTORY_DIR


def _history_path(entry_id: str) -> str:
    return os.path.join(_ensure_history_dir(), f"{entry_id}.json")


def _save_history_entry(response: VoiceIntakeResponse) -> str:
    entry_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    entry_id = f"{entry_id}_{uuid.uuid4().hex[:8]}"
    payload = response.model_dump()
    payload["created_at"] = datetime.now(timezone.utc).isoformat()
    try:
        with open(_history_path(entry_id), "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)
        print(f"[history] Saved entry {entry_id}")
        return entry_id
    except Exception as e:
        print(f"[history] ERROR saving entry {entry_id}: {e}")
        raise


def _list_history_entries() -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not os.path.isdir(_ensure_history_dir()):
        return entries
    for name in sorted(os.listdir(HISTORY_DIR), reverse=True):
        if not name.endswith(".json"):
            continue
        entry_id = name[:-5]
        path = os.path.join(HISTORY_DIR, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            entries.append(
                {
                    "id": entry_id,
                    "transcript": data.get("transcript", ""),
                    "intent": (data.get("result") or {}).get("intent", "unknown"),
                    "created_at": data.get("created_at"),
                }
            )
        except Exception:
            continue
    return entries


def _read_history_entry(entry_id: str) -> Dict[str, Any]:
    path = _history_path(entry_id)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="History entry not found")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/", response_class=HTMLResponse)
def home():
    return """<html><body>
    <h2>MedVoice Intake Agent</h2>
    <p>Open <a href="/demo">/demo</a> for the browser recorder.</p>
    <p>Or POST audio to <code>/voice-intake</code>.</p>
    </body></html>"""


@app.get("/demo", response_class=HTMLResponse)
def demo_page():
    here = os.path.dirname(__file__)
    demo_path = os.path.join(here, "..", "..", "frontend", "recorder.html")
    with open(demo_path, "r", encoding="utf-8") as f:
        return f.read()

@app.post("/voice-intake", response_model=VoiceIntakeResponse)
async def voice_intake(audio: UploadFile = File(...)):
    suffix = os.path.splitext(audio.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        content = await audio.read()
        tmp.write(content)

    try:
        print(f"[voice_intake] Starting transcription")
        transcript, language = await transcribe_file(tmp_path)
        print(f"[voice_intake] Transcription complete: {len(transcript)} chars")
        
        print(f"[voice_intake] Starting LLM analysis")
        analysis, redaction_tags = await analyze_transcript(transcript, intake_schema)
        print(f"[voice_intake] LLM analysis complete")

        # Validate against our schema; if LLM output is invalid, this will raise.
        print(f"[voice_intake] Validating response against schema")
        result = IntakeResult.model_validate(analysis)
        response = VoiceIntakeResponse(
            transcript=transcript,
            language=language,
            result=result,
            redaction_tags=redaction_tags,
        )
        
        print(f"[voice_intake] Saving to history")
        _save_history_entry(response)
        print(f"[voice_intake] Response complete, returning to client")
        return response
    except ValueError as exc:
        print(f"[voice_intake] ValueError: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        print(f"[voice_intake] RuntimeError: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        print(f"[voice_intake] Unexpected error: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(exc)}") from exc
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@app.post("/voice-intake-example", response_model=VoiceIntakeResponse)
async def voice_intake_example(payload: ExampleTranscriptPayload):
    """Process a pre-recorded example transcript (for demo purposes)."""
    transcript = payload.transcript.strip()

    if not transcript or len(transcript) < 5:
        raise HTTPException(status_code=400, detail="Transcript is too short")

    try:
        print(f"[voice_intake_example] Starting LLM analysis")
        analysis, redaction_tags = await analyze_transcript(transcript, intake_schema)
        print(f"[voice_intake_example] LLM analysis complete")
        
        print(f"[voice_intake_example] Validating response against schema")
        result = IntakeResult.model_validate(analysis)
        response = VoiceIntakeResponse(
            transcript=transcript,
            language="en",
            result=result,
            redaction_tags=redaction_tags,
        )
        
        print(f"[voice_intake_example] Saving to history")
        _save_history_entry(response)
        print(f"[voice_intake_example] Response complete, returning to client")
        return response
    except ValueError as exc:
        print(f"[voice_intake_example] ValueError: {exc}")
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        print(f"[voice_intake_example] RuntimeError: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        print(f"[voice_intake_example] Unexpected error: {type(exc).__name__}: {exc}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(exc)}") from exc


@app.get("/history")
def list_history():
    return {"items": _list_history_entries()}


@app.get("/history/{entry_id}")
def get_history_entry(entry_id: str):
    return _read_history_entry(entry_id)


@app.delete("/history")
def clear_history():
    deleted = 0
    if os.path.isdir(_ensure_history_dir()):
        for name in os.listdir(HISTORY_DIR):
            if not name.endswith(".json"):
                continue
            try:
                os.remove(os.path.join(HISTORY_DIR, name))
                deleted += 1
            except Exception:
                continue
    return {"ok": True, "deleted": deleted}


@app.get("/health")
def health():
    return {"ok": True}
