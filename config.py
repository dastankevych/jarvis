import os

# LLM (Ollama)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
MODEL = os.getenv("JARVIS_MODEL", "qwen3:14b")

# STT
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "")
SAMPLE_RATE = 16000

# TTS
TTS_VOICE = "en-US-AriaNeural"

# Memory
DB_PATH = os.getenv("JARVIS_DB", "data/jarvis_memory.db")
MAX_CONVERSATION_TURNS = 20

# Runtime
USE_VOICE = os.getenv("JARVIS_VOICE", "0") == "1"
SILENCE_THRESHOLD = 0.015
