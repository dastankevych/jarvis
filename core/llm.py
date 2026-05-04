"""LLM client using Ollama (local) with OpenAI-compatible API."""
import re
from typing import Dict, List, Optional

import httpx


def _sanitize(value: object) -> object:
    """Strip surrogate characters that break UTF-8 serialisation."""
    if isinstance(value, str):
        return value.encode("utf-8", errors="replace").decode("utf-8")
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    return value


class LLMClient:
    def __init__(self, model: str = "qwen3:14b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def chat(
        self,
        messages: List[Dict],
        system: Optional[str] = None,
        max_tokens: int = 1024,
    ) -> str:
        all_messages = []
        if system:
            all_messages.append({"role": "system", "content": system})
        all_messages.extend(messages)

        response = httpx.post(
            f"{self.base_url}/api/chat",
            json=_sanitize({
                "model": self.model,
                "messages": all_messages,
                "stream": False,
                "options": {"num_predict": max_tokens},
            }),
            timeout=120.0,
        )
        response.raise_for_status()
        content = response.json()["message"]["content"]
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content

    def extract_json(self, prompt: str, system: Optional[str] = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.chat(messages, system=system, max_tokens=512)
