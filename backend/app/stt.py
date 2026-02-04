from __future__ import annotations
from typing import Tuple, Optional
import tempfile
import os

# We keep STT optional so the repo runs even without Whisper installed.
# You can enable local STT by installing either:
#   pip install faster-whisper
# or
#   pip install openai-whisper

async def transcribe_file(filepath: str) -> Tuple[str, Optional[str]]:
    """Return (transcript, language). Raises RuntimeError if no STT backend is installed."""
    # Try faster-whisper first
    try:
        from faster_whisper import WhisperModel  # type: ignore
        model = WhisperModel("small", device="cpu", compute_type="int8")
        segments, info = model.transcribe(filepath, beam_size=5, vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        return text, getattr(info, "language", None)
    except Exception:
        pass

    # Try openai-whisper (the open-source package)
    try:
        import whisper  # type: ignore
        model = whisper.load_model("medium")
        result = model.transcribe(filepath)
        text = (result.get("text") or "").strip()
        return text, result.get("language")
    except Exception:
        pass

    raise RuntimeError(
        "No STT backend installed. Install 'faster-whisper' or 'openai-whisper' in requirements.txt."
    )
