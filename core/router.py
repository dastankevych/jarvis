"""LLM-based router: classifies user intent and dispatches to tools or chat."""
import json
import re
from typing import List, Optional, Tuple

from core.llm import LLMClient
from tools.registry import ToolRegistry
from utils import load_prompt

ACTION_CHAT = "chat"
ACTION_TOOL = "tool"
ACTION_GOAL_ADD = "goal_add"
ACTION_GOAL_COMPLETE = "goal_complete"
ACTION_GOAL_LIST = "goal_list"
ACTION_GOAL_PLAN = "goal_plan"
ACTION_STEP_COMPLETE = "step_complete"
ACTION_STEP_LIST = "step_list"
ACTION_MEMORY = "memory_store"
ACTION_RECALL = "memory_recall"

Route = Tuple[str, Optional[str], dict]


class Router:
    def __init__(self, llm: LLMClient, tools: ToolRegistry) -> None:
        self.llm = llm
        self.tools = tools

    def route(self, user_input: str) -> List[Route]:
        system = load_prompt("router_system", tools=self.tools.descriptions())
        raw = self.llm.extract_json(user_input, system=system)

        try:
            cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not m:
                return [(ACTION_CHAT, None, {})]
            data = json.loads(m.group())
            actions = data.get("actions", [])
            if not actions:
                return [(ACTION_CHAT, None, {})]

            routes: List[Route] = []
            for item in actions:
                action = item.get("action", ACTION_CHAT)
                if action == ACTION_TOOL:
                    routes.append((ACTION_TOOL, item.get("tool"), item.get("args", {})))
                elif action == ACTION_GOAL_ADD:
                    routes.append((ACTION_GOAL_ADD, None, {"goal": item.get("goal", "")}))
                elif action == ACTION_GOAL_COMPLETE:
                    routes.append((ACTION_GOAL_COMPLETE, None, {
                        "goal_id": int(item.get("goal_id", 0)),
                        "goal_text": item.get("goal_text", ""),
                    }))
                elif action == ACTION_GOAL_LIST:
                    routes.append((ACTION_GOAL_LIST, None, {}))
                elif action == ACTION_MEMORY:
                    routes.append((ACTION_MEMORY, None, {
                        "category": item.get("category", "general"),
                        "content": item.get("content", ""),
                    }))
                elif action == ACTION_RECALL:
                    routes.append((ACTION_RECALL, None, {}))
                elif action == ACTION_GOAL_PLAN:
                    routes.append((ACTION_GOAL_PLAN, None, {
                        "goal_id": int(item.get("goal_id", 0)),
                        "goal_text": item.get("goal_text", ""),
                    }))
                elif action == ACTION_STEP_COMPLETE:
                    routes.append((ACTION_STEP_COMPLETE, None, {
                        "step_id": int(item.get("step_id", 0)),
                    }))
                elif action == ACTION_STEP_LIST:
                    routes.append((ACTION_STEP_LIST, None, {
                        "goal_id": int(item.get("goal_id", 0)),
                        "goal_text": item.get("goal_text", ""),
                    }))
                else:
                    routes.append((ACTION_CHAT, None, {}))

            return routes or [(ACTION_CHAT, None, {})]

        except Exception:
            return [(ACTION_CHAT, None, {})]
