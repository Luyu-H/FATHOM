"""Uncertainty detection strategies.

Returns an ``UncertaintyResult`` describing whether the question is
operationally ambiguous; downstream the agent decides whether to enter the
structured clarification loop.

Available strategies
--------------------
- ``direct_prompt`` — single LLM judge call returning a boolean verdict.
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
# Direct prompt (LLM judge)
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
# Factory
# =============================================================

def build_uncertainty_detector(cfg, llm: LLMClient) -> UncertaintyDetector:
    s = str(cfg.strategy).lower()
    if s == "direct_prompt":
        return DirectPromptUncertainty(
            llm,
            temperature=cfg.direct_prompt.temperature,
        )
    raise ValueError(
        f"Unknown uncertainty strategy: {s}. Supported: 'direct_prompt'."
    )