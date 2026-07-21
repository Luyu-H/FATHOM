"""Uncertainty detection strategies.

Returns an ``UncertaintyResult`` describing whether the question is
operationally ambiguous; downstream the agent decides whether to enter the
structured clarification loop.

Available strategies
--------------------
- ``direct_prompt`` — single LLM judge call returning a boolean verdict.
- ``divergence``    — sample N independent plans, then ask an LLM judge to
                       decide whether all plans would yield the same
                       numerical answer (i.e., agree on overall logic,
                       method choice, and every concrete parameter value).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from .llm_client import LLMClient, Message, parse_json_block
from . import prompts

logger = logging.getLogger(__name__)

# Max LLM calls per detection when the reply can't be parsed into the
# expected JSON verdict. After exhausting these, the detector falls back
# to CLEAR (but records ``parse_failed: true`` so the run is auditable).
_MAX_PARSE_ATTEMPTS = 3


class AmbiguityLevel(str, Enum):
    CLEAR = "clear"
    AMBIGUOUS = "ambiguous"


@dataclass
class UncertaintyResult:
    level: AmbiguityLevel
    score: float                              # higher = more ambiguous
    ambiguous_terms: List[str] = field(default_factory=list)
    details: Dict = field(default_factory=dict)


# =============================================================
# Base
# =============================================================

class UncertaintyDetector(ABC):
    @abstractmethod
    def detect(self, question: str) -> UncertaintyResult: ...


def _user_msg(question: str) -> str:
    return f"Question:\n{question}"


def _direct_user_msg(question: str) -> str:
    """User turn for the direct-prompt judge.

    Repeats the task and the required output format in the user turn so a
    fully-specified, imperative question (which reads like a coding task)
    doesn't lure the model into *solving* it instead of *judging* it.
    """
    return (
        f"Question:\n{question}\n\n"
        "Your ONLY task is to judge whether the question above is "
        "operationally ambiguous. Do NOT attempt to answer or solve it, "
        "and do NOT write code. Respond with STRICT JSON only, no markdown "
        "and no preamble:\n"
        '{"ambiguous": true | false, "ambiguous_terms": ["term", ...], '
        '"reason": "<one short sentence>"}'
    )


def _parsed_verdict(raw: str) -> Optional[Dict]:
    """Return the parsed verdict dict iff it carries an ``ambiguous`` key.

    ``parse_json_block`` may return ``None`` (no JSON found) or a non-dict
    (e.g. a JSON list); both count as a parse failure here, as does a dict
    that lacks the required ``ambiguous`` field.
    """
    parsed = parse_json_block(raw)
    if isinstance(parsed, dict) and "ambiguous" in parsed:
        return parsed
    return None


# =============================================================
# 1) Direct prompt (LLM judge)
# =============================================================

class DirectPromptUncertainty(UncertaintyDetector):
    def __init__(self, llm: LLMClient, temperature: float = 0.0,
                 system_prompt: Optional[str] = None,
                 strategy_name: str = "direct_prompt"):
        self._llm = llm
        self._temp = temperature
        self._system_prompt = system_prompt or prompts.UNCERTAINTY_DIRECT_PROMPT
        self._strategy_name = strategy_name

    def detect(self, question):
        user_msg = _direct_user_msg(question)
        system_prompt = self._system_prompt

        # Retry on unparseable replies (the model sometimes "solves" the
        # question and emits code instead of the JSON verdict). After
        # ``_MAX_PARSE_ATTEMPTS`` we fall back to CLEAR but flag the trace.
        llm_calls: List[Dict] = []
        data: Optional[Dict] = None
        last_text = ""
        for attempt in range(1, _MAX_PARSE_ATTEMPTS + 1):
            resp = self._llm.chat(
                messages=[Message("user", user_msg)],
                system=system_prompt,
                temperature=self._temp,
            )[0]
            last_text = resp.text
            llm_calls.append({
                "role": "detect",
                "attempt": attempt,
                "system_prompt": system_prompt,
                "user_prompt": user_msg,
                "raw_response": resp.text,
            })
            data = _parsed_verdict(resp.text)
            if data is not None:
                break
            logger.warning(
                "[uncertainty] direct_prompt parse failed (attempt %d/%d)",
                attempt, _MAX_PARSE_ATTEMPTS,
            )

        parse_failed = data is None
        if parse_failed:
            logger.warning(
                "[uncertainty] direct_prompt unparseable after %d attempts "
                "— defaulting to CLEAR", _MAX_PARSE_ATTEMPTS,
            )
            data = {}

        ambiguous = bool(data.get("ambiguous", False))
        terms = [t for t in data.get("ambiguous_terms", []) if isinstance(t, str)]
        return UncertaintyResult(
            level=AmbiguityLevel.AMBIGUOUS if ambiguous else AmbiguityLevel.CLEAR,
            score=1.0 if ambiguous else 0.0,
            ambiguous_terms=terms,
            details={"strategy": self._strategy_name,
                     "reason": data.get("reason", ""),
                     "raw_text": last_text,
                     "parse_failed": parse_failed,
                     "attempts": len(llm_calls),
                     "llm_calls": llm_calls},
        )


# =============================================================
# 2) Divergence — sample N plans, judge equivalence with an LLM
# =============================================================

class DivergenceUncertainty(UncertaintyDetector):
    """Generate N independent plans, then ask an LLM judge whether all
    plans would produce the same numerical answer.

    Motivation
    ----------
    Embedding-cosine / clustering criteria only capture overall textual or
    logical structure; they are blind to small but consequential parameter
    differences (e.g., depth 0-10 m vs 0-50 m, baseline 1995-2014 vs
    1980-2010, OLS vs Theil-Sen trend). The LLM judge inspects every
    granularity — overall logic, per-step method choice, concrete numeric
    parameters, derived-quantity formulas, and data selection — and flags
    the question as AMBIGUOUS as soon as any meaningful divergence exists.
    """

    def __init__(self, llm: LLMClient, n_samples: int = 5,
                 sample_temperature: float = 0.8,
                 judge_temperature: float = 0.0):
        self._llm = llm
        self._n = n_samples
        self._sample_temp = sample_temperature
        self._judge_temp = judge_temperature

    @staticmethod
    def _format_plans_block(plans: List[str]) -> str:
        return "\n\n".join(f"[Plan {i + 1}]\n{p}" for i, p in enumerate(plans))

    def detect(self, question):
        plan_user = _user_msg(question)
        plan_system = prompts.UNCERTAINTY_DIVERGENCE_PROMPT
        responses = self._llm.chat(
            messages=[Message("user", plan_user)],
            system=plan_system,
            temperature=self._sample_temp,
            n=self._n,
        )
        plans = [r.text.strip() for r in responses if r.text and r.text.strip()]
        llm_calls = [{
            "role": "generate_plans",
            "system_prompt": plan_system,
            "user_prompt": plan_user,
            "n_samples": self._n,
            "raw_responses": [r.text for r in responses],
        }]
        if len(plans) < 2:
            return UncertaintyResult(
                level=AmbiguityLevel.CLEAR, score=0.0,
                details={"strategy": "divergence",
                         "n_plans": len(plans),
                         "note": "insufficient plans for judging",
                         "plans": plans,
                         "llm_calls": llm_calls})

        judge_system = prompts.UNCERTAINTY_DIVERGENCE_JUDGE_PROMPT
        judge_user = (
            f"Candidate plans (numbered):\n\n{self._format_plans_block(plans)}"
        )
        judge_resp = self._llm.chat(
            messages=[Message("user", judge_user)],
            system=judge_system,
            temperature=self._judge_temp,
        )[0]
        llm_calls.append({
            "role": "judge_equivalence",
            "system_prompt": judge_system,
            "user_prompt": judge_user,
            "raw_response": judge_resp.text,
        })
        data = parse_json_block(judge_resp.text) or {}
        equivalent = bool(data.get("equivalent", True))
        divergent_aspects = [
            a for a in data.get("divergent_aspects", []) if isinstance(a, str)
        ]
        ambiguous = not equivalent

        return UncertaintyResult(
            level=AmbiguityLevel.AMBIGUOUS if ambiguous else AmbiguityLevel.CLEAR,
            score=1.0 if ambiguous else 0.0,
            details={"strategy": "divergence",
                     "n_plans": len(plans),
                     "equivalent": equivalent,
                     "divergent_aspects": divergent_aspects,
                     "reason": data.get("reason", ""),
                     "judge_raw_text": judge_resp.text,
                     "plans": plans,
                     "llm_calls": llm_calls})


# =============================================================
# Factory
# =============================================================

def build_uncertainty_detector(cfg, llm: LLMClient) -> UncertaintyDetector:
    s = str(cfg.strategy).lower()
    if s == "direct_prompt":
        return DirectPromptUncertainty(
            llm,
            temperature=cfg.direct_prompt.temperature,
        )
    if s == "direct_prompt_simple":
        return DirectPromptUncertainty(
            llm,
            temperature=cfg.direct_prompt_simple.temperature,
            system_prompt=prompts.UNCERTAINTY_DIRECT_SIMPLE_PROMPT,
            strategy_name="direct_prompt_simple",
        )
    if s == "divergence":
        d = cfg.divergence
        return DivergenceUncertainty(
            llm,
            n_samples=d.n_samples,
            sample_temperature=d.sample_temperature,
            judge_temperature=d.judge_temperature,
        )
    raise ValueError(
        f"Unknown uncertainty strategy: {s}. "
        "Supported: 'direct_prompt', 'direct_prompt_simple', 'divergence'."
    )