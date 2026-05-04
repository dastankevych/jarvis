"""Dialogue manager: ties together LLM, memory, routing, and tools."""
from core.llm import LLMClient
import json
import re

from core.router import (
    ACTION_CHAT,
    ACTION_GOAL_ADD,
    ACTION_GOAL_COMPLETE,
    ACTION_GOAL_LIST,
    ACTION_GOAL_PLAN,
    ACTION_MEMORY,
    ACTION_RECALL,
    ACTION_STEP_COMPLETE,
    ACTION_STEP_LIST,
    ACTION_TOOL,
    Router,
)
from memory.memory import LongTermMemory, ShortTermMemory
from tools.registry import ToolRegistry
from utils import load_prompt


class DialogueManager:
    def __init__(
        self,
        llm: LLMClient,
        short_mem: ShortTermMemory,
        long_mem: LongTermMemory,
        tools: ToolRegistry,
        router: Router,
    ) -> None:
        self.llm = llm
        self.short = short_mem
        self.long = long_mem
        self.tools = tools
        self.router = router

    # ------------------------------------------------------------------

    def process(self, user_input: str) -> str:
        routes = self.router.route(user_input)
        print(f"[DEBUG] routes={routes}")

        results = []
        has_chat = False

        for action, tool_name, extra in routes:
            if action == ACTION_TOOL and tool_name:
                results.append(self._handle_tool(user_input, tool_name, extra))
            elif action == ACTION_GOAL_ADD:
                results.append(self._handle_goal_add(extra.get("goal", user_input)))
            elif action == ACTION_GOAL_COMPLETE:
                results.append(self._handle_goal_complete(
                    extra.get("goal_id", 0),
                    extra.get("goal_text", ""),
                ))
            elif action == ACTION_GOAL_LIST:
                results.append(self._handle_goal_list())
            elif action == ACTION_MEMORY:
                results.append(self._handle_memory_store(
                    extra.get("category", "general"),
                    extra.get("content", ""),
                ))
            elif action == ACTION_RECALL:
                results.append(self._handle_memory_recall(user_input))
            elif action == ACTION_GOAL_PLAN:
                results.append(self._handle_goal_plan(
                    extra.get("goal_id", 0),
                    extra.get("goal_text", ""),
                ))
            elif action == ACTION_STEP_COMPLETE:
                results.append(self._handle_step_complete(extra.get("step_id", 0)))
            elif action == ACTION_STEP_LIST:
                results.append(self._handle_step_list(
                    extra.get("goal_id", 0),
                    extra.get("goal_text", ""),
                ))
            else:
                has_chat = True

        if has_chat or not results:
            results.append(self._chat(user_input))

        return "\n".join(results)

    # ------------------------------------------------------------------

    def _chat(self, user_input: str) -> str:
        self.short.add("user", user_input)
        response = self.llm.chat(self.short.get(), system=self._build_system())
        self.short.add("assistant", response)
        return response

    def _handle_tool(self, user_input: str, tool_name: str, args: dict) -> str:
        result = self.tools.execute(tool_name, **args)
        return self._chat(f"{user_input}\n[Tool result from {tool_name}: {result}]")

    def _handle_goal_add(self, goal: str) -> str:
        if not goal:
            return "What goal would you like to set?"
        gid = self.long.add_goal(goal)
        response = f"Goal #{gid} added: \"{goal}\"."
        self.short.add("user", f"Add goal: {goal}")
        self.short.add("assistant", response)
        return response

    def _handle_goal_complete(self, goal_id: int, goal_text: str = "") -> str:
        if not goal_id and goal_text:
            goals = self.long.get_active_goals()
            needle = goal_text.lower()
            matches = [g for g in goals if needle in g["goal"].lower()]
            if len(matches) == 1:
                goal_id = matches[0]["id"]
            elif len(matches) > 1:
                lines = "\n".join(f"  #{g['id']}: {g['goal']}" for g in matches)
                return f"Multiple goals match \"{goal_text}\":\n{lines}\nPlease specify the number."

        if goal_id:
            self.long.complete_goal(goal_id)
            response = f"Goal #{goal_id} marked as completed."
        else:
            goals = self.long.get_active_goals()
            if not goals:
                response = "You have no active goals."
            else:
                lines = "\n".join(f"  #{g['id']}: {g['goal']}" for g in goals)
                response = f"Which goal to complete?\n{lines}"
        self.short.add("user", f"Complete goal {goal_id or goal_text}")
        self.short.add("assistant", response)
        return response

    def _handle_goal_list(self) -> str:
        goals = self.long.get_all_goals()
        if not goals:
            response = "You have no goals yet."
        else:
            icons = {"active": "⬜", "completed": "✅", "cancelled": "❌"}
            lines = [f"  {icons.get(g['status'], '?')} #{g['id']}: {g['goal']}" for g in goals]
            response = "Your goals:\n" + "\n".join(lines)
        self.short.add("user", "List goals")
        self.short.add("assistant", response)
        return response

    def _handle_memory_store(self, category: str, content: str) -> str:
        self.long.store_fact(category, content)
        response = f"Got it, I'll remember: {content}."
        self.short.add("user", content)
        self.short.add("assistant", response)
        return response

    def _handle_goal_plan(self, goal_id: int, goal_text: str) -> str:
        goal = self._resolve_goal(goal_id, goal_text)
        if not goal:
            return f"I couldn't find a goal matching \"{goal_text}\". Check your goals list."

        prompt = load_prompt("plan", goal=goal["goal"])
        raw = self.llm.extract_json(prompt)

        try:
            cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            m = re.search(r"\[.*\]", cleaned, re.DOTALL)
            steps = json.loads(m.group()) if m else []
        except Exception:
            steps = []

        if not steps:
            return "I couldn't generate a plan. Try rephrasing the goal."

        self.long.add_steps(goal["id"], steps)
        lines = "\n".join(f"  {i}. {s}" for i, s in enumerate(steps, 1))
        response = f"Plan for \"{goal['goal']}\":\n{lines}"
        self.short.add("user", f"Plan goal: {goal['goal']}")
        self.short.add("assistant", response)
        return response

    def _handle_step_complete(self, step_id: int) -> str:
        if not step_id:
            return "Please specify the step number to complete."
        self.long.complete_step(step_id)
        response = f"Step #{step_id} marked as completed."
        self.short.add("user", f"Complete step {step_id}")
        self.short.add("assistant", response)
        return response

    def _handle_step_list(self, goal_id: int, goal_text: str) -> str:
        goal = self._resolve_goal(goal_id, goal_text)
        if not goal:
            return f"I couldn't find a goal matching \"{goal_text}\"."

        steps = self.long.get_steps(goal["id"])
        if not steps:
            return f"No plan yet for \"{goal['goal']}\". Ask me to create one."

        icons = {"active": "⬜", "completed": "✅"}
        lines = [f"  {icons.get(s['status'], '?')} #{s['id']}: {s['description']}" for s in steps]
        next_step = self.long.get_next_step(goal["id"])
        response = f"Plan for \"{goal['goal']}\":\n" + "\n".join(lines)
        if next_step:
            response += f"\n\nNext up: {next_step['description']}"
        self.short.add("user", f"List steps for {goal['goal']}")
        self.short.add("assistant", response)
        return response

    def _resolve_goal(self, goal_id: int, goal_text: str) -> dict | None:
        """Find a goal by id or partial text match."""
        if goal_id:
            all_goals = self.long.get_all_goals()
            matches = [g for g in all_goals if g["id"] == goal_id]
            return matches[0] if matches else None
        if goal_text:
            needle = goal_text.lower()
            all_goals = self.long.get_all_goals()
            matches = [g for g in all_goals if needle in g["goal"].lower()]
            return matches[0] if matches else None
        return None

    def _handle_memory_recall(self, user_question: str) -> str:
        facts = self.long.get_facts(limit=50)
        summaries = self.long.get_summaries(limit=10)

        if not facts and not summaries:
            response = "I don't have anything stored in memory yet."
            self.short.add("user", user_question)
            self.short.add("assistant", response)
            return response

        sections = []
        if summaries:
            sections.append("Previous session summaries:\n" + "\n".join(f"- {s}" for s in summaries))
        if facts:
            sections.append("Stored facts:\n" + "\n".join(f"- {f['content']}" for f in facts))

        prompt = load_prompt(
            "recall",
            question=user_question,
            memory_sections="\n\n".join(sections),
        )
        response = self.llm.chat([{"role": "user", "content": prompt}])
        self.short.add("user", user_question)
        self.short.add("assistant", response)
        return response

    # ------------------------------------------------------------------

    def _build_system(self) -> str:
        parts = [load_prompt("base_system")]

        active_goals = self.long.get_active_goals()
        if active_goals:
            lines = "\n".join(f"  #{g['id']}: {g['goal']}" for g in active_goals)
            parts.append(f"User's active goals:\n{lines}")

        recent_facts = self.long.get_facts(limit=8)
        if recent_facts:
            lines = "\n".join(f"  - {f['content']}" for f in recent_facts)
            parts.append(f"Known facts about user:\n{lines}")

        summaries = self.long.get_summaries(limit=3)
        if summaries:
            parts.append("Recent session summaries:\n" + "\n".join(f"  - {s}" for s in summaries))

        return "\n\n".join(parts)

    def summarise_session(self) -> None:
        if len(self.short) < 4:
            return
        history = "\n".join(
            f"{m['role'].upper()}: {m['content']}" for m in self.short.get()
        )
        prompt = load_prompt("summarise", history=history)
        summary = self.llm.chat([{"role": "user", "content": prompt}])
        self.long.store_summary(summary)
