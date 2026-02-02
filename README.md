# MedVoice Intake Agent (Demo)

A small, GitHub-friendly prototype aligned to MTM workflows:

**Voice → Speech-to-Text → LLM triage → pharmacist-ready structured intake**

This demo turns an unstructured patient voice message into:
- a transcript
- structured triage (intent + risk)
- a **SOAP-style note** for documentation
- a **pharmacist task object** suitable for routing into an outreach queue

> ⚠️ Safety: This project is for demonstration only. It does **not** provide medical advice.

---

## Quickstart (Windows / macOS / Linux)

### 1) Create a virtual environment
```bash
python -m venv .venv
# Windows PowerShell:
#   .\.venv\Scripts\Activate.ps1
# macOS/Linux:
#   source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

### 3) Enable Speech-to-Text (choose one)

**Option A: faster-whisper (recommended)**
```bash
pip install faster-whisper
```

**Option B: openai-whisper**
```bash
pip install openai-whisper
```

> If you skip this, the app will run but `/voice-intake` will error with “No STT backend installed”.

### 4) (Optional) Enable a local LLM with Ollama
Install Ollama and pull a model:
```bash
ollama pull llama3.1
```
The backend will try Ollama first; if not available, it falls back to a rule-based triage (still returns SOAP + task).

### 5) Run the server
```bash
uvicorn backend.app.main:app --reload
```

Open:
- http://127.0.0.1:8000/demo

---

## API

### POST /voice-intake
Upload audio (`multipart/form-data` field name: `audio`).

Example with curl:
```bash
curl -X POST http://127.0.0.1:8000/voice-intake \
  -F "audio=@./examples/sample_audio.webm"
```

Response includes:
- `transcript`
- `language` (if detected)
- `result` with:
  - `intent`, `risk_level`
  - `soap_note` (S/O/A/P fields)
  - `pharmacist_task` (queue/priority/due time)

---

## Suggested demo script (what you say)

> “This is a minimal MTM intake assistant. A member leaves a voicemail-like message.  
> The system transcribes it, classifies intent and risk, generates a SOAP note for documentation,  
> and creates a pharmacist task for routing into an outreach queue.  
> It includes safety guardrails and escalation logic for red-flag symptoms.”

