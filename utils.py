from pathlib import Path

_ROOT = Path(__file__).parent
_PROMPTS = _ROOT / "prompts"
_SQL = _ROOT / "sql"


def load_prompt(name: str, **kwargs: str) -> str:
    """Load a prompt template from prompts/<name>.txt and format with kwargs."""
    text = (_PROMPTS / f"{name}.txt").read_text(encoding="utf-8")
    return text.format(**kwargs) if kwargs else text


def load_sql(name: str) -> str:
    """Load a SQL query from sql/<name>.sql."""
    return (_SQL / f"{name}.sql").read_text(encoding="utf-8").strip()
