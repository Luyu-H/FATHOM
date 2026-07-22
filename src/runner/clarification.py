"""Clarification strategies — emit atomic questions for the simulated user (Oracle).

Every question is shaped to be parseable by Oracle's term extractor:
    "What does '<term>' mean?"  /  "How is '<term>' defined?"

Available strategies
--------------------
- ``spontaneous`` — no upfront detection or structured questioning. The
                    code-generation prompt is asked directly; if the LLM
                    responds with natural-language question(s) instead of
                    code, those are routed to the Oracle and the loop
                    continues, capped by ``max_turns``.
- ``direct``      — single LLM call generates the next batch of questions.
- ``candidate``   — generate K candidates, then pick N by information gain.
- ``planning``    — write a plan with [?] markers, ask about each [?].

Across every strategy at most ``questions_per_turn`` sub-questions are
emitted per turn and at most ``max_turns`` rounds are performed — both
values come straight from the config, no extra clamping.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .llm_client import LLMClient, Message, parse_json_block
from . import prompts

logger = logging.getLogger(__name__)


# =============================================================
# Turn record
# =============================================================

@dataclass
class ClarificationTurn:
    """One round of agent question(s) → oracle response(s)."""
    questions: List[str]
    oracle_responses: List[Dict[str, Any]] = field(default_factory=list)
    # Full LLM calls that produced this turn's ``questions``. Each entry:
    # {role, system_prompt, user_prompt, raw_response}. Captured so the
    # trace preserves every system/user prompt across the conversation.
    llm_calls: List[Dict[str, Any]] = field(default_factory=list)

    def render_for_history(self) -> str:
        """Render this turn as conversational text for downstream prompts."""
        lines: List[str] = []
        for q, r in zip(self.questions, self.oracle_responses):
            lines.append(f"Q: {q}")
            lines.append(f"A: {self._format_response(r)}")
        return "\n".join(lines)

    def render_clarifications_for_codegen(self) -> str:
        """Render the simulated user's natural-language answers for codegen.

        The simulated-user oracle now produces a complete natural-language
        reply per turn that already fully discloses any cited entry's
        mapped_params and reasoning_note. We forward those answers verbatim
        and skip turns where the oracle could not respond at all
        (``INVALID_QUERY`` / ``LLM_ERROR``).
        """
        parts: List[str] = []
        for resp in self.oracle_responses:
            if resp.get("status") in ("INVALID_QUERY", "LLM_ERROR"):
                continue
            text = (resp.get("answer_text") or "").strip()
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    @staticmethod
    def _format_response(resp: Dict[str, Any]) -> str:
        """Render the oracle's reply as conversational feedback.

        The oracle's ``answer_text`` is the simulated user's verbatim reply
        and is already a complete, minimally-disclosing natural-language
        answer. We surface it as-is; on the rare error states we fall back
        to the status ``message``.
        """
        text = (resp.get("answer_text") or "").strip()
        if text:
            return text
        return resp.get("message") or "(no answer)"


# =============================================================
# Base
# =============================================================

class ClarificationStrategy(ABC):
    """Generates clarification questions; returns None to stop the loop."""

    def __init__(self, llm: LLMClient, max_turns: int,
                 questions_per_turn: int):
        self.llm = llm
        self.max_turns = max_turns
        self.questions_per_turn = questions_per_turn
        self._plan: Optional[str] = None      # only set by planning strategy
        # Captures every LLM call made during the most recent
        # ``make_questions`` invocation. The agent reads + stores this on
        # the corresponding ``clarification_turns`` entry so the full
        # conversation (system + user prompts + raw replies) is preserved
        # in the trace. Each entry: {role, system_prompt, user_prompt, raw_response}.
        self._last_llm_calls: List[Dict[str, str]] = []

    def should_continue(self, history: List[ClarificationTurn]) -> bool:
        return len(history) < self.max_turns

    def reset(self) -> None:
        """Clear per-question state so it doesn't leak across ``answer()`` calls.

        The agent reuses one strategy instance for many questions; without this
        reset, a plan produced for an AMBIGUOUS question would still be exposed
        via ``self.plan`` on a later CLEAR question whose clarification loop
        never runs.
        """
        self._plan = None
        self._last_llm_calls = []

    @abstractmethod
    def make_questions(
        self,
        question: str,
        context: str,
        history: List[ClarificationTurn],
        ambiguous_terms: List[str],
    ) -> Optional[List[str]]: ...

    @property
    def plan(self) -> Optional[str]:
        return self._plan

    @property
    def last_llm_calls(self) -> List[Dict[str, str]]:
        """LLM calls made during the most recent ``make_questions`` invocation.

        The agent reads this AFTER each call and stores the entries on the
        corresponding ``clarification_turns`` record. Subclasses must reset
        and append into ``self._last_llm_calls`` themselves.
        """
        return list(self._last_llm_calls)

    @property
    def is_spontaneous(self) -> bool:
        """True if this strategy delegates question-asking to the codegen LLM."""
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _render_history(history: List[ClarificationTurn]) -> str:
        if not history:
            return "(none)"
        return "\n\n".join(
            f"[Turn {i+1}]\n{h.render_for_history()}"
            for i, h in enumerate(history)
        )


# =============================================================
# 1) Spontaneous — let the codegen LLM decide when to ask
# =============================================================

class SpontaneousClarification(ClarificationStrategy):
    """No upfront detection; no structured pre-codegen questioning.
    """

    def make_questions(self, question, context, history, ambiguous_terms):
        return None     # all interaction happens inside the codegen loop

    @property
    def is_spontaneous(self) -> bool:
        return True


# =============================================================
# 2) Direct — accumulate prior info and ask freely
# =============================================================

class DirectClarification(ClarificationStrategy):
    def __init__(self, llm, max_turns, questions_per_turn,
                 temperature: float = 0.0):
        super().__init__(llm, max_turns, questions_per_turn)
        self.temperature = temperature

    def make_questions(self, question, context, history, ambiguous_terms):
        self._last_llm_calls = []
        if not self.should_continue(history):
            return None
        sys = prompts.CLARIFY_DIRECT_PROMPT.format(
            n=self.questions_per_turn,
        )
        user = (
            f"Original question:\n{question}\n\n"
            f"Context:\n{context or '(none)'}\n\n"
            f"Prior clarification turns:\n{self._render_history(history)}"
        )
        # First turn only: signal that the upstream detector flagged this
        # question as ambiguous, WITHOUT leaking which specific terms it
        # picked — we want this LLM to identify them on its own. On later
        # turns the oracle responses already establish the conversation
        # context, so no extra signal is added.
        if ambiguous_terms and not history:
            user += "\n\nUpstream flagged this question as ambiguous."

        resp = self.llm.chat(
            messages=[Message("user", user)],
            system=sys, temperature=self.temperature,
        )[0]
        self._last_llm_calls.append({
            "role": "direct_generate",
            "system_prompt": sys,
            "user_prompt": user,
            "raw_response": resp.text,
        })
        data = parse_json_block(resp.text) or {}
        if not data.get("need_clarification"):
            return None
        questions = [
            q for q in (data.get("questions") or [])
            if isinstance(q, str) and q.strip()
        ]
        return questions[: self.questions_per_turn] or None


# =============================================================
# 3) Candidate — generate K candidates, pick N by information gain
# =============================================================

class CandidateClarification(ClarificationStrategy):
    def __init__(self, llm, max_turns, questions_per_turn,
                 n_candidates: int = 4,
                 gen_temperature: float = 0.4,
                 pick_temperature: float = 0.0):
        super().__init__(llm, max_turns, questions_per_turn)
        self.n_candidates = n_candidates
        self.gen_temperature = gen_temperature
        self.pick_temperature = pick_temperature

    def make_questions(self, question, context, history, ambiguous_terms):
        self._last_llm_calls = []
        if not self.should_continue(history):
            return None

        # Stage 1: generate candidates
        gen_sys = prompts.CLARIFY_CANDIDATE_GEN_PROMPT.format(
            k=self.n_candidates,
        )
        gen_user = (
            f"Original question:\n{question}\n\n"
            f"Context:\n{context or '(none)'}\n\n"
            f"Prior clarification turns:\n{self._render_history(history)}"
        )
        # First turn only — binary ambiguity hint, no term list. See
        # DirectClarification for the rationale.
        if ambiguous_terms and not history:
            gen_user += "\n\nUpstream flagged this question as ambiguous."

        resp = self.llm.chat(
            messages=[Message("user", gen_user)],
            system=gen_sys, temperature=self.gen_temperature,
        )[0]
        self._last_llm_calls.append({
            "role": "candidate_generate",
            "system_prompt": gen_sys,
            "user_prompt": gen_user,
            "raw_response": resp.text,
        })
        data = parse_json_block(resp.text) or {}
        if not data.get("need_clarification"):
            return None
        candidates = [
            q for q in (data.get("candidates") or [])
            if isinstance(q, str) and q.strip()
        ]
        if not candidates:
            return None

        # Stage 2: pick (LLM-based selection over the candidates).
        return self._pick(question, context, history, candidates) or None

    def _pick(self, question, context, history, candidates):
        sys = prompts.CLARIFY_CANDIDATE_PICK_PROMPT.format(n=self.questions_per_turn)
        cand_block = "\n".join(f"{i+1}. {c}" for i, c in enumerate(candidates))
        user = (
            f"Original question:\n{question}\n\n"
            f"Context:\n{context or '(none)'}\n\n"
            f"History:\n{self._render_history(history)}\n\n"
            f"Candidates:\n{cand_block}\n\n"
            f"Pick AT MOST {self.questions_per_turn} candidate(s) to ask this turn. "
            "Favor candidates whose answer would most reduce disagreement among "
            "reasonable analysts about how to operationalize the task and have "
            "the largest impact on the computation method or final numerical result. "
            "Deprioritize candidates that only affect minor implementation details; "
            "if none of the remaining candidates is important enough to be worth a "
            "turn of the user's attention, ask nothing this turn."
        )
        resp = self.llm.chat(
            messages=[Message("user", user)],
            system=sys, temperature=self.pick_temperature,
        )[0]
        self._last_llm_calls.append({
            "role": "candidate_pick",
            "system_prompt": sys,
            "user_prompt": user,
            "raw_response": resp.text,
        })
        data = parse_json_block(resp.text) or {}
        if data.get("need_clarification") is False:
            return []
        chosen: List[str] = [
            q for q in (data.get("questions") or [])
            if isinstance(q, str) and q.strip()
        ]
        return chosen[: self.questions_per_turn]


# =============================================================
# 4) Planning-based — write plan, identify [?], ask about each
# =============================================================

class PlanningClarification(ClarificationStrategy):
    def __init__(self, llm, max_turns, questions_per_turn,
                 plan_max_lines: int = 12,
                 temperature: float = 0.0):
        super().__init__(llm, max_turns, questions_per_turn)
        self.plan_max_lines = plan_max_lines
        self.temperature = temperature

    def make_questions(self, question, context, history, ambiguous_terms):
        self._last_llm_calls = []
        if not self.should_continue(history):
            return None
        sys = prompts.CLARIFY_PLANNING_PROMPT.format(
            n=self.questions_per_turn,
            max_lines=self.plan_max_lines,
        )
        user = (
            f"Original question:\n{question}\n\n"
            f"Context:\n{context or '(none)'}\n\n"
            f"Prior clarification turns:\n{self._render_history(history)}"
        )
        # First turn only — binary ambiguity hint, no term list. See
        # DirectClarification for the rationale.
        if ambiguous_terms and not history:
            user += "\n\nUpstream flagged this question as ambiguous."

        resp = self.llm.chat(
            messages=[Message("user", user)],
            system=sys, temperature=self.temperature,
        )[0]
        self._last_llm_calls.append({
            "role": "planning_generate",
            "system_prompt": sys,
            "user_prompt": user,
            "raw_response": resp.text,
        })
        data = parse_json_block(resp.text) or {}
        # Persist the latest plan for downstream code generation.
        if data.get("plan"):
            self._plan = data["plan"]
        if not data.get("need_clarification"):
            return None
        questions = [
            q for q in (data.get("questions") or [])
            if isinstance(q, str) and q.strip()
        ]
        return questions[: self.questions_per_turn] or None


# =============================================================
# Factory
# =============================================================

def build_clarification_strategy(cfg, llm: LLMClient) -> ClarificationStrategy:
    s = str(cfg.strategy).lower()
    base = dict(
        llm=llm,
        max_turns=cfg.max_turns,
        questions_per_turn=cfg.questions_per_turn,
    )
    if s == "spontaneous":
        return SpontaneousClarification(**base)
    if s == "direct":
        return DirectClarification(
            **base,
            temperature=cfg.direct.temperature,
        )
    if s == "candidate":
        return CandidateClarification(
            **base,
            n_candidates=cfg.candidate.n_candidates,
            gen_temperature=cfg.candidate.gen_temperature,
            pick_temperature=cfg.candidate.pick_temperature,
        )
    if s == "planning":
        return PlanningClarification(
            **base,
            plan_max_lines=cfg.planning.plan_max_lines,
            temperature=cfg.planning.temperature,
        )
    raise ValueError(
        f"Unknown clarification strategy: {s}. "
        "Supported: 'spontaneous', 'direct', 'candidate', 'planning'."
    )