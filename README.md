# Jarvis — Personal AI Assistant

A modular, voice-enabled AI assistant powered by a local LLM (Ollama). Supports natural conversation, goal tracking with step-by-step planning, persistent memory across sessions, and a built-in tool system.

## Features

- **Conversational AI** — multi-turn dialogue with short-term context window
- **Voice I/O** — speech recognition via mlx-whisper (Apple Silicon) and TTS via edge-tts
- **Intent routing** — LLM-based router dispatches multiple actions from a single message
- **Goal management** — create, complete, and list goals; persist across sessions
- **Planning** — break any goal into actionable steps, track progress per step
- **Long-term memory** — facts about the user and session summaries stored in SQLite
- **Built-in tools** — datetime, calculator, web search, shell commands

## Project Structure

```
jarvis/
├── core/           # LLM client, intent router, dialogue manager
├── memory/         # Short-term (in-session) and long-term (SQLite) memory
├── voice/          # Speech-to-text (STT) and text-to-speech (TTS)
├── tools/          # Extensible tool registry
├── prompts/        # All LLM prompt templates (.txt)
├── sql/            # All SQL queries (.sql)
├── data/           # SQLite database
├── config.py       # Configuration (env-overridable)
├── utils.py        # Prompt and SQL file loaders
└── main.py         # Entry point
```

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally with a pulled model
- macOS (for mlx-whisper and `afplay`); Linux works with fallback backends

## Installation

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd jarvis

# 2. Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Pull a model in Ollama (if not done already)
ollama pull qwen3:14b
```

## Usage

### Text mode
```bash
python main.py
```

### Voice mode (microphone + speakers)
```bash
JARVIS_VOICE=1 python main.py
```

### In-session commands
| Command | Action |
|---|---|
| `exit` / `quit` / `bye` | End session and save summary |
| `goals` | List all goals |
| `clear` | Clear short-term memory |
| Enter (voice mode) | Cancel current LLM request |

## Configuration

All settings can be overridden via environment variables:

| Variable | Default | Description |
|---|---|---|
| `JARVIS_MODEL` | `qwen3:14b` | Ollama model name |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL |
| `JARVIS_VOICE` | `0` | Set to `1` to enable voice mode |
| `WHISPER_MODEL` | `base` | Whisper model size (`tiny` / `base` / `small` / `medium`) |
| `WHISPER_LANGUAGE` | _(auto)_ | Force language, e.g. `ru` or `en` |
| `JARVIS_DB` | `data/jarvis_memory.db` | Path to SQLite database |

## Example Interactions

```
You: What time is it?
Jarvis: It's 7:42 PM on Saturday, May 3rd, 2026.

You: Create a plan to prepare for the ML exam
Jarvis: Plan for "Prepare for the ML exam":
  1. Review all lecture notes and slides
  2. Summarize key concepts per topic
  3. Practice coding exercises
  4. Do a timed mock exam
  5. Review mistakes and fill gaps

You: Mark goal 1 as done and add a new goal to go to the gym
Jarvis: Goal #1 marked as completed.
        Goal #3 added: "Go to the gym".

You: What do you remember about our previous conversations?
Jarvis: From past sessions: we discussed ML homework, set up grocery goals,
        and you mentioned London is the capital of Great Britain.
```

## Extending

**Add a new tool** — register it in `tools/registry.py`:
```python
self.register("my_tool", "Description. Args: x (str)", lambda x: f"result: {x}")
```

**Add a new prompt** — create `prompts/<name>.txt` and load it with:
```python
from utils import load_prompt
text = load_prompt("name", key="value")
```

**Add a new SQL query** — create `sql/<name>.sql` and load it with:
```python
from utils import load_sql
cursor.execute(load_sql("name"), (param,))
```
