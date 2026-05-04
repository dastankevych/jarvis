"""Speech-to-text using mlx-whisper (Apple Silicon) with sounddevice for capture."""
from __future__ import annotations

import os
import tempfile
from typing import Optional

import numpy as np

# Minimum RMS energy to bother transcribing
_MIN_AUDIO_RMS = 0.005

# If a single word or 2-gram takes up more than this fraction of text → hallucination
_REPETITION_RATIO = 0.4


def _is_hallucination(text: str) -> bool:
    """Detect Whisper hallucinations: repetitive tokens or known garbage phrases."""
    if not text:
        return True
    words = text.lower().split()
    if len(words) < 4:
        return False
    # Check if any single word dominates
    for word in set(words):
        if words.count(word) / len(words) > _REPETITION_RATIO:
            return True
    # Check bigram repetition
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
    for bg in set(bigrams):
        if bigrams.count(bg) / len(bigrams) > _REPETITION_RATIO:
            return True
    return False


class SpeechToText:
    def __init__(
        self,
        model_size: str = "base",
        sample_rate: int = 16000,
        language: str = "",
    ) -> None:
        self.model_size = model_size
        self.sample_rate = sample_rate
        self.language: Optional[str] = language or None  # None = auto-detect
        self._model = None
        self._backend: Optional[str] = None

    def _ensure_model(self) -> None:
        if self._backend is not None:
            return
        try:
            import mlx_whisper  # noqa: F401
            self._backend = "mlx"
        except ImportError:
            try:
                import whisper
                self._model = whisper.load_model(self.model_size)
                self._backend = "whisper"
            except ImportError:
                self._backend = "none"

    def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe a float32 numpy array recorded at self.sample_rate."""
        self._ensure_model()

        if self._backend == "none":
            return ""

        # Skip transcription if audio is too quiet (just noise)
        rms = float(np.sqrt(np.mean(audio ** 2)))
        if rms < _MIN_AUDIO_RMS:
            return ""

        import scipy.io.wavfile as wav  # type: ignore

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        try:
            pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
            wav.write(tmp, self.sample_rate, pcm)

            if self._backend == "mlx":
                import mlx_whisper
                kwargs = {
                    "path_or_hf_repo": f"mlx-community/whisper-{self.model_size}-mlx",
                }
                if self.language:
                    kwargs["language"] = self.language
                result = mlx_whisper.transcribe(tmp, **kwargs)
            else:
                kwargs = {}
                if self.language:
                    kwargs["language"] = self.language
                result = self._model.transcribe(tmp, **kwargs)  # type: ignore

            text = result["text"].strip()
            return text if not _is_hallucination(text) else ""
        finally:
            os.unlink(tmp)

    def record_until_silence(
        self,
        silence_threshold: float = 0.015,
        speech_threshold: float = 0.03,
        min_seconds: float = 1.0,
        max_seconds: float = 30.0,
        silence_seconds: float = 1.5,
    ) -> Optional[np.ndarray]:
        """Wait for speech, then record until sustained silence or max_seconds."""
        try:
            import sounddevice as sd
        except ImportError:
            print("[STT] sounddevice not available — cannot record audio")
            return None

        chunk = int(self.sample_rate * 0.1)  # 100 ms frames
        max_chunks = int(max_seconds / 0.1)
        min_chunks = int(min_seconds / 0.1)
        silence_needed = int(silence_seconds / 0.1)

        stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype=np.float32
        )
        stream.start()
        try:
            # Phase 1: wait until speech detected (max 10 seconds)
            wait_chunks = int(10.0 / 0.1)
            speech_found = False
            for _ in range(wait_chunks):
                frame, _ = stream.read(chunk)
                rms = float(np.sqrt(np.mean(frame.flatten() ** 2)))
                if rms >= speech_threshold:
                    chunks: list[np.ndarray] = [frame.flatten()]
                    speech_found = True
                    break
            if not speech_found:
                return None

            # Phase 2: record until silence
            silent_streak = 0
            for i in range(max_chunks):
                frame, _ = stream.read(chunk)
                frame = frame.flatten()
                chunks.append(frame)
                rms = float(np.sqrt(np.mean(frame ** 2)))
                if rms < silence_threshold:
                    silent_streak += 1
                else:
                    silent_streak = 0
                if i >= min_chunks and silent_streak >= silence_needed:
                    break
        finally:
            stream.stop()
            stream.close()

        return np.concatenate(chunks) if chunks else None
