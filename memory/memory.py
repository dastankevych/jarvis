"""Short-term (in-memory) and long-term (SQLite) memory."""
import sqlite3
from pathlib import Path
from typing import List, Dict

from utils import load_sql


class ShortTermMemory:
    def __init__(self, max_turns: int = 20):
        self._messages: List[Dict] = []
        self.max_turns = max_turns

    def add(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        if len(self._messages) > self.max_turns * 2:
            self._messages = self._messages[-(self.max_turns * 2):]

    def get(self) -> List[Dict]:
        return list(self._messages)

    def clear(self) -> None:
        self._messages = []

    def __len__(self) -> int:
        return len(self._messages)


class LongTermMemory:
    def __init__(self, db_path: str = "data/jarvis_memory.db"):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_schema()

    def _init_schema(self) -> None:
        self.conn.executescript(load_sql("schema"))
        self.conn.commit()

    # --- Facts ---

    def store_fact(self, category: str, content: str) -> None:
        self.conn.execute(load_sql("facts_insert"), (category, content))
        self.conn.commit()

    def get_facts(self, category: str | None = None, limit: int = 20) -> List[Dict]:
        if category:
            cur = self.conn.execute(load_sql("facts_select_by_category"), (category, limit))
        else:
            cur = self.conn.execute(load_sql("facts_select"), (limit,))
        return [{"category": r[0], "content": r[1], "ts": r[2]} for r in cur.fetchall()]

    # --- Summaries ---

    def store_summary(self, text: str) -> None:
        self.conn.execute(load_sql("summaries_insert"), (text,))
        self.conn.commit()

    def get_summaries(self, limit: int = 5) -> List[str]:
        cur = self.conn.execute(load_sql("summaries_select"), (limit,))
        return [r[0] for r in cur.fetchall()]

    # --- Goals ---

    def add_goal(self, goal: str) -> int:
        cur = self.conn.execute(load_sql("goals_insert"), (goal,))
        self.conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def complete_goal(self, goal_id: int) -> None:
        self.conn.execute(load_sql("goals_complete"), (goal_id,))
        self.conn.commit()

    def cancel_goal(self, goal_id: int) -> None:
        self.conn.execute(load_sql("goals_cancel"), (goal_id,))
        self.conn.commit()

    def get_active_goals(self) -> List[Dict]:
        cur = self.conn.execute(load_sql("goals_select_active"))
        return [{"id": r[0], "goal": r[1], "created_at": r[2]} for r in cur.fetchall()]

    def get_all_goals(self) -> List[Dict]:
        cur = self.conn.execute(load_sql("goals_select_all"))
        return [
            {"id": r[0], "goal": r[1], "status": r[2], "created_at": r[3], "completed_at": r[4]}
            for r in cur.fetchall()
        ]

    # --- Steps ---

    def add_steps(self, goal_id: int, descriptions: List[str]) -> None:
        for order, desc in enumerate(descriptions, start=1):
            self.conn.execute(load_sql("steps_insert"), (goal_id, order, desc))
        self.conn.commit()

    def get_steps(self, goal_id: int) -> List[Dict]:
        cur = self.conn.execute(load_sql("steps_select_by_goal"), (goal_id,))
        return [
            {"id": r[0], "order": r[1], "description": r[2], "status": r[3]}
            for r in cur.fetchall()
        ]

    def complete_step(self, step_id: int) -> None:
        self.conn.execute(load_sql("steps_complete"), (step_id,))
        self.conn.commit()

    def get_next_step(self, goal_id: int) -> Dict | None:
        cur = self.conn.execute(load_sql("steps_next_active"), (goal_id,))
        row = cur.fetchone()
        return {"id": row[0], "order": row[1], "description": row[2]} if row else None
