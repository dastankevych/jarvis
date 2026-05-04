"""Jarvis – a voice/text AI assistant.

Usage:
    python main.py                    # text mode
    JARVIS_VOICE=1 python main.py     # voice mode (mic + speakers)

Special commands:
    exit / quit / bye   – end session
    goals               – list goals
    clear               – clear short-term memory
"""
from __future__ import annotations

import select
import sys
import threading

import config
from core.dialogue import DialogueManager
from core.llm import LLMClient
from core.router import Router
from memory.memory import LongTermMemory, ShortTermMemory
from tools.registry import ToolRegistry
from voice.stt import SpeechToText
from voice.tts import TextToSpeech

BANNER = """\
Mode: {mode}  |  Model: {model}
Type 'exit' to quit, 'goals' to see goals.
"""


def build_components():
    llm = LLMClient(model=config.MODEL, base_url=config.OLLAMA_BASE_URL)
    short_mem = ShortTermMemory(max_turns=config.MAX_CONVERSATION_TURNS)
    long_mem = LongTermMemory(db_path=config.DB_PATH)
    tools = ToolRegistry()
    router = Router(llm, tools)
    dialogue = DialogueManager(llm, short_mem, long_mem, tools, router)
    stt = SpeechToText(
        model_size=config.WHISPER_MODEL,
        sample_rate=config.SAMPLE_RATE,
        language=config.WHISPER_LANGUAGE,
    )
    tts = TextToSpeech(voice=config.TTS_VOICE)
    return dialogue, stt, tts


def _process_interruptible(dialogue: DialogueManager, user_input: str) -> str | None:
    """Run dialogue.process() in a thread; return None if user pressed Enter to cancel."""
    result: list[str | None] = [None]
    done = threading.Event()

    def worker() -> None:
        result[0] = dialogue.process(user_input)
        done.set()

    threading.Thread(target=worker, daemon=True).start()
    print("[Processing... press Enter to cancel]")

    while not done.wait(timeout=0.1):
        if select.select([sys.stdin], [], [], 0)[0]:
            sys.stdin.readline()
            return None

    return result[0]


def text_loop(dialogue: DialogueManager, tts: TextToSpeech) -> None:
    greeting = "Hello! I'm Jarvis. How can I help you today?"
    print(f"\nJarvis: {greeting}")
    tts.speak(greeting)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input:
            continue

        low = user_input.lower()
        if low in ("exit", "quit", "bye"):
            farewell = "Goodbye! Have a great day."
            print(f"\nJarvis: {farewell}")
            tts.speak(farewell)
            break
        if low == "clear":
            dialogue.short.clear()
            print("Jarvis: Short-term memory cleared.")
            continue
        if low == "goals":
            user_input = "List all my goals"

        response = dialogue.process(user_input)
        print(f"\nJarvis: {response}")
        tts.speak(response)


def voice_loop(dialogue: DialogueManager, stt: SpeechToText, tts: TextToSpeech) -> None:
    greeting = "Hello! I'm Jarvis. Speak your command or say 'exit' to quit."
    print(f"\nJarvis: {greeting}")
    tts.speak(greeting)

    while True:
        print("\n[Listening — speak now, pause to stop]")
        try:
            audio = stt.record_until_silence(silence_threshold=config.SILENCE_THRESHOLD)
        except KeyboardInterrupt:
            break

        if audio is None:
            continue

        print("[Transcribing...]")
        user_input = stt.transcribe(audio)
        if not user_input:
            continue

        print(f"\nYou: {user_input}")
        if user_input.lower().strip(".,!? ") in ("exit", "quit", "bye", "goodbye"):
            farewell = "Goodbye!"
            print(f"\nJarvis: {farewell}")
            tts.speak(farewell)
            break

        response = _process_interruptible(dialogue, user_input)
        if response is None:
            print("[Cancelled — speak again]")
            continue
        print(f"\nJarvis: {response}")
        tts.speak(response)


def main() -> None:
    dialogue, stt, tts = build_components()
    mode = "voice" if config.USE_VOICE else "text"
    print(BANNER.format(mode=mode, model=config.MODEL))

    try:
        if config.USE_VOICE:
            voice_loop(dialogue, stt, tts)
        else:
            text_loop(dialogue, tts)
    finally:
        print("\n[Saving session summary…]")
        dialogue.summarise_session()
        print("Session saved. Bye!")


if __name__ == "__main__":
    main()
