"""Text-to-speech with tiered fallbacks:
1. edge-tts  (high quality, free Microsoft Azure voices)
2. macOS `say` command (offline fallback)
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile


class TextToSpeech:
    def __init__(self, voice: str = "en-US-AriaNeural") -> None:
        self.voice = voice
        self._backend: str | None = None

    def _detect_backend(self) -> str:
        try:
            import edge_tts  # noqa: F401
            return "edge_tts"
        except ImportError:
            pass
        # macOS fallback
        if os.path.exists("/usr/bin/say"):
            return "say"
        return "none"

    def speak(self, text: str) -> None:
        if not text.strip():
            return
        if self._backend is None:
            self._backend = self._detect_backend()

        if self._backend == "edge_tts":
            try:
                asyncio.run(self._speak_edge(text))
                return
            except Exception as exc:
                print(f"[TTS] edge-tts error: {exc} — falling back to say")
                self._backend = "say"

        if self._backend == "say":
            subprocess.run(["say", text], check=False)
        else:
            print(f"[TTS] {text}")

    async def _speak_edge(self, text: str) -> None:
        import edge_tts

        communicate = edge_tts.Communicate(text, self.voice)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            tmp = f.name
        try:
            await communicate.save(tmp)
            # Play with afplay (macOS) or ffplay
            if os.path.exists("/usr/bin/afplay"):
                subprocess.run(["afplay", tmp], check=False)
            else:
                subprocess.run(["ffplay", "-nodisp", "-autoexit", tmp],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               check=False)
        finally:
            os.unlink(tmp)
