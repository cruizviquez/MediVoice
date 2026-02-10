from __future__ import annotations

from typing import Tuple, Optional
import os
import time

# You can enable local STT by installing either:
#   pip install faster-whisper
# or
#   pip install openai-whisper

# Lowered to reduce false failures for short utterances in browser recordings.
MIN_AUDIO_BYTES = 5_000


WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
WHISPER_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
_WHISPER_MODEL_CACHE = None


def _get_faster_whisper_model():
    global _WHISPER_MODEL_CACHE
    if _WHISPER_MODEL_CACHE is None:
        print("[stt] Loading faster-whisper model...")
        model_start = time.time()
        _WHISPER_MODEL_CACHE = WhisperModel(
            WHISPER_MODEL_NAME,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE,
        )
        model_time = time.time() - model_start
        print(f"[stt] Model loaded: {model_time:.2f}s")
    return _WHISPER_MODEL_CACHE


async def transcribe_file(filepath: str) -> Tuple[str, Optional[str]]:
    """Return (transcript, language). Raises ValueError or RuntimeError on failure."""
    start_time = time.time()
    last_error: str | None = None
    try:
        file_size = os.path.getsize(filepath)
    except Exception:
        file_size = -1

    print(f"[stt] input: {filepath} ({file_size} bytes)")
    if 0 <= file_size < MIN_AUDIO_BYTES:
        raise ValueError(
            f"Audio file too small ({file_size} bytes). "
            "Check microphone permissions and ensure recording is not empty."
        )

    # Try faster-whisper first
    try:
        from faster_whisper import WhisperModel  # type: ignore

        model = _get_faster_whisper_model()
        
        transcribe_start = time.time()
        segments, info = model.transcribe(filepath, beam_size=5, vad_filter=True)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        transcribe_time = time.time() - transcribe_start
        print(f"[stt] Transcribed with VAD: {transcribe_time:.2f}s ({len(text)} chars)")

        if not text:
            print("[stt] Empty transcript with VAD, retrying without...")
            transcribe_start = time.time()
            segments, info = model.transcribe(filepath, beam_size=5, vad_filter=False)
            text = " ".join(seg.text.strip() for seg in segments).strip()
            transcribe_time = time.time() - transcribe_start
            print(f"[stt] Transcribed without VAD: {transcribe_time:.2f}s ({len(text)} chars)")

        if text:
            total_time = time.time() - start_time
            print(f"[stt] faster-whisper complete: {total_time:.2f}s total")
            return text, getattr(info, "language", None)

        print("[stt] faster-whisper returned empty transcript")
    except Exception as exc:
        elapsed = time.time() - start_time
        print(f"[stt] faster-whisper failed after {elapsed:.2f}s: {exc!r}")
        last_error = f"faster-whisper: {exc!r}"

    # Try openai-whisper (open-source local package)
    try:
        import whisper  # type: ignore

        print("[stt] Loading openai-whisper model...")
        model_start = time.time()
        model = whisper.load_model("medium")
        model_time = time.time() - model_start
        print(f"[stt] Model loaded: {model_time:.2f}s")
        
        transcribe_start = time.time()
        result = model.transcribe(filepath)
        text = (result.get("text") or "").strip()
        transcribe_time = time.time() - transcribe_start
        print(f"[stt] Transcribed: {transcribe_time:.2f}s ({len(text)} chars)")

        if text:
            total_time = time.time() - start_time
            print(f"[stt] openai-whisper complete: {total_time:.2f}s total")
            return text, result.get("language")

        print("[stt] openai-whisper returned empty transcript")
    except Exception as exc:
        elapsed = time.time() - start_time
        print(f"[stt] openai-whisper failed after {elapsed:.2f}s: {exc!r}")
        last_error = f"openai-whisper: {exc!r}"

    detail = f" ({last_error})" if last_error else ""
    raise RuntimeError("STT failed to transcribe audio. Check microphone input and audio format." + detail)
