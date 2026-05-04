"""Built-in tools that Jarvis can invoke."""
import datetime
import math
import os
import subprocess
import webbrowser
from typing import Any, Callable, Dict


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, Dict] = {}
        self._register_defaults()

    def register(self, name: str, description: str, func: Callable[..., str]) -> None:
        self._tools[name] = {"description": description, "func": func}

    def execute(self, name: str, **kwargs: Any) -> str:
        if name not in self._tools:
            return f"Unknown tool: {name}"
        try:
            return str(self._tools[name]["func"](**kwargs))
        except Exception as exc:
            return f"Tool error ({name}): {exc}"

    def descriptions(self) -> str:
        return "\n".join(
            f"  {n}: {info['description']}" for n, info in self._tools.items()
        )

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def _register_defaults(self) -> None:
        self.register(
            "get_datetime",
            "Return the current date and time (no args needed)",
            lambda: datetime.datetime.now().strftime("%A, %Y-%m-%d %H:%M:%S"),
        )
        self.register(
            "calculate",
            "Evaluate a safe math expression. Args: expr (str)",
            self._calculate,
        )
        self.register(
            "open_url",
            "Open a URL in the default browser. Args: url (str)",
            lambda url: (webbrowser.open(url), f"Opened {url}")[1],
        )
        self.register(
            "web_search",
            "Search Google for a query. Args: query (str)",
            lambda query: (
                webbrowser.open(f"https://www.google.com/search?q={query.replace(' ', '+')}"),
                f"Searched for: {query}",
            )[1],
        )
        self.register(
            "list_directory",
            "List files in a directory. Args: path (str, default '.')",
            lambda path=".": "\n".join(sorted(os.listdir(path))) or "(empty)",
        )
        self.register(
            "run_shell",
            "Run a safe, read-only shell command. Args: cmd (str)",
            self._run_shell,
        )

    @staticmethod
    def _calculate(expr: str) -> str:
        safe_names = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        safe_names["abs"] = abs
        result = eval(expr, {"__builtins__": {}}, safe_names)  # noqa: S307
        return str(result)

    @staticmethod
    def _run_shell(cmd: str) -> str:
        safe_prefixes = ("ls", "pwd", "date", "echo", "cat", "head", "tail", "wc", "grep")
        first_word = cmd.strip().split()[0] if cmd.strip() else ""
        if first_word not in safe_prefixes:
            return f"Shell command '{first_word}' is not in the allowed list."
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip() or result.stderr.strip() or "(no output)"
