"""Modular sub-agents for isolated evaluation.

The full pipeline (``AmbiguityAwareAgent``) bundles three stages:

    uncertainty detection -> clarification loop -> code generation + execute

For ablation / module-level evaluation we expose two thin agents that
reuse the exact same primitives from
:mod:`src.runner.ambiguity_identifier`, :mod:`src.runner.clarification`,
:mod:`src.runner.code_executor`, and :mod:`src.runner.prompts`, but each
covers only one slice of the pipeline.

* :class:`AmbiguityOnlyAgent`
    Runs only the uncertainty detector and stops. No oracle / no
    clarification / no codegen. Uses the ``direct_prompt`` strategy.

* :class:`ClarificationOnlyAgent`
    Skips uncertainty detection — the question is treated as already
    known to be ambiguous (upstream gold label) — and immediately runs
    the structured clarification loop (``direct`` / ``candidate`` /
    ``planning``) followed by the same codegen + execute + repair loop
    the full agent uses. The "spontaneous" strategy is intentionally
    not supported here: this mode exists to study how well structured
    clarification recovers intent from a known-ambiguous task.

Both agents emit traces that share field names with
:class:`src.runner.runner.AgentTrace` so the existing summarization /
metrics tooling continues to apply.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from . import prompts
from .ambiguity_identifier import (
    AmbiguityLevel, UncertaintyDetector, build_uncertainty_detector,
)
from .clarification import (
    ClarificationStrategy, ClarificationTurn, build_clarification_strategy,
)
from .code_executor import CodeExecutor, extract_code
from .llm_client import LLMClient, Message, build_llm_client

logger = logging.getLogger(__name__)


# Sentinel passed to the clarification strategy so its first-turn
# "Upstream flagged this question as ambiguous" hint fires without
# leaking any specific terms (matches the behaviour of the full agent
# when ``ambiguous_terms`` is non-empty).
_AMBIGUOUS_SENTINEL: List[str] = ["__ambiguous__"]


# =============================================================
# Trace records
# =============================================================

@dataclass
class AmbiguityOnlyTrace:
    question: str
    context: str
    uncertainty: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ClarificationOnlyTrace:
    question: str
    context: str
    # Always {"level": "ambiguous", "strategy": "assumed_ambiguous"} so
    # downstream summaries can tell this mode apart from the full pipeline.
    uncertainty: Dict[str, Any] = field(default_factory=dict)
    clarification_turns: List[Dict[str, Any]] = field(default_factory=list)
    plan: Optional[str] = None
    generated_code: Optional[str] = None
    code_attempts: List[Dict[str, Any]] = field(default_factory=list)
    execution: Dict[str, Any] = field(default_factory=dict)
    final_answer: Optional[str] = None
    error: Optional[str] = None


# =============================================================
# 1) Ambiguity-only
# =============================================================

class AmbiguityOnlyAgent:
    """Runs only the uncertainty detector.

    Returns a trace with the detector's verdict (level + score +
    ambiguous_terms + full LLM-call details) and stops. No oracle, no
    clarification, no codegen.
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.llm: LLMClient = build_llm_client(config.llm)
        self.uncertainty: UncertaintyDetector = build_uncertainty_detector(
            config.ambiguity, self.llm,
        )

    def answer(self, question: str, context: str = "") -> AmbiguityOnlyTrace:
        # ``context`` is accepted for API parity with the full agent but
        # is NOT fed to the detector — the detector judges the question
        # alone, matching the structured route in ``AmbiguityAwareAgent``.
        trace = AmbiguityOnlyTrace(question=question, context=context)
        try:
            u = self.uncertainty.detect(question)
        except Exception as exc:                          # noqa: BLE001
            logger.exception("Ambiguity detection failed")
            trace.error = f"detection_failed: {exc}"
            return trace
        trace.uncertainty = {
            "level": u.level.value,
            "score": u.score,
            "ambiguous_terms": u.ambiguous_terms,
            "details": u.details,
        }
        logger.info("[ambiguity-only] level=%s score=%.3f",
                    u.level.value, u.score)
        return trace


# =============================================================
# 2) Clarification-only (no detection — assume ambiguous)
# =============================================================

class ClarificationOnlyAgent:
    """Assume the question is ambiguous; run clarification + codegen.

    Use this to evaluate the clarification module in isolation on a
    pre-labelled ambiguous subset. The first clarification turn tells
    the LLM that upstream flagged the question as ambiguous (without
    leaking which terms), then the standard structured loop runs.

    The "spontaneous" clarification strategy is rejected by the factory
    here — it has no upfront detection step to bypass and is therefore
    out of scope for this mode.
    """

    def __init__(self, config: DictConfig, oracle):
        self.config = config
        self.oracle = oracle
        self.llm: LLMClient = build_llm_client(config.llm)
        self.clarification: ClarificationStrategy = build_clarification_strategy(
            config.clarification, self.llm,
        )
        if self.clarification.is_spontaneous:
            raise ValueError(
                "ClarificationOnlyAgent does not support the 'spontaneous' "
                "strategy — it relies on structured pre-codegen clarification. "
                "Pick 'direct', 'candidate', or 'planning'."
            )
        self.executor = CodeExecutor(
            workspace_root=config.code_exec.workspace_root,
            timeout_seconds=config.code_exec.timeout_seconds,
            python_executable=config.code_exec.python_executable,
        )
        self.questions_per_turn = config.clarification.questions_per_turn
        self.max_code_attempts = max(
            1, int(getattr(config.code_exec, "max_attempts", 1) or 1),
        )
        self.codegen_temperature = float(config.code_exec.codegen_temperature)
        # Concrete CMIP6 data root injected into the codegen user prompt;
        # QA prompts only carry the abstract `<data_root>` placeholder.
        data_root = getattr(getattr(config, "data", None), "data_root", None)
        if not data_root:
            raise ValueError(
                "config.data.data_root is required so the agent can tell "
                "the codegen LLM what to substitute for `<data_root>`."
            )
        # Absolute path (relative to the original launch dir, robust to any
        # Hydra chdir). The codegen script runs with cwd set to its isolated
        # run directory, so a relative data root would resolve against that
        # run dir and the dataset would not be found.
        self.data_root = to_absolute_path(str(data_root))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def answer(self, question: str, context: str = "") -> ClarificationOnlyTrace:
        trace = ClarificationOnlyTrace(question=question, context=context)
        trace.uncertainty = {
            "level": AmbiguityLevel.AMBIGUOUS.value,
            "score": 1.0,
            "ambiguous_terms": [],
            "details": {"strategy": "assumed_ambiguous"},
        }
        # Wipe per-question state on the strategy (notably the planning
        # strategy's stored plan) so it doesn't leak from a prior call.
        self.clarification.reset()

        # ---- Clarification loop (always entered) ----
        try:
            history = self._clarification_loop(question, context)
        except Exception as exc:                          # noqa: BLE001
            logger.exception("Clarification loop failed")
            trace.error = f"clarification_failed: {exc}"
            return trace
        for h in history:
            trace.clarification_turns.append({
                "questions": h.questions,
                "oracle_responses": h.oracle_responses,
                "llm_calls": h.llm_calls,
                "spontaneous": False,
            })
        trace.plan = self.clarification.plan

        # ---- Codegen + execute (with repair) ----
        self._run_codegen_loop(question, context, history, trace)
        return trace

    # ------------------------------------------------------------------
    # Clarification loop — mirrors AmbiguityAwareAgent._clarification_loop
    # but always passes the "ambiguous" sentinel so the first-turn hint
    # is emitted to the strategy LLM.
    # ------------------------------------------------------------------
    def _clarification_loop(
        self, question: str, context: str,
    ) -> List[ClarificationTurn]:
        history: List[ClarificationTurn] = []
        while self.clarification.should_continue(history):
            qs = self.clarification.make_questions(
                question, context, history, _AMBIGUOUS_SENTINEL,
            )
            if not qs:
                decision_calls = self.clarification.last_llm_calls
                if decision_calls:
                    history.append(ClarificationTurn(
                        questions=[],
                        oracle_responses=[],
                        llm_calls=decision_calls,
                    ))
                break
            turn = ClarificationTurn(
                questions=qs[:self.questions_per_turn],
                llm_calls=self.clarification.last_llm_calls,
            )
            prior_turns_snapshot = self._oracle_prior_turns(history)
            for q in turn.questions:
                resp = self.oracle.clarify(
                    q,
                    original_question=question,
                    original_context=context,
                    prior_turns=prior_turns_snapshot,
                ).to_dict()
                turn.oracle_responses.append(resp)
            history.append(turn)
            logger.info("[clarify] turn %d — asked %d question(s)",
                        len(history), len(turn.questions))
        return history

    @staticmethod
    def _oracle_prior_turns(
        history: List[ClarificationTurn],
    ) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for turn in history:
            for q, resp in zip(turn.questions, turn.oracle_responses):
                out.append({
                    "question": q,
                    "answer": (resp.get("answer_text") or "").strip(),
                })
        return out

    # ------------------------------------------------------------------
    # Codegen helpers — minimal copy of the structured route, no
    # spontaneous oracle-routing and no first-attempt reuse.
    # ------------------------------------------------------------------
    _TRUNCATION_REASONS = frozenset({"length", "max_tokens"})

    @classmethod
    def _is_truncated(cls, finish_reason: Optional[str]) -> bool:
        if not finish_reason:
            return False
        fr = str(finish_reason).lower()
        return fr in cls._TRUNCATION_REASONS or "max_tokens" in fr

    def _build_codegen_user_prompt(
        self,
        question: str,
        context: str,
        history: List[ClarificationTurn],
        prior_attempts: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        optional_sections = ""
        if history:
            rendered = [h.render_clarifications_for_codegen() for h in history]
            rendered = [t for t in rendered if t]
            if rendered:
                history_text = "\n".join(rendered)
                optional_sections += prompts.CODEGEN_HISTORY_SECTION.format(
                    history=history_text,
                )
        if prior_attempts:
            attempts_text = "\n\n".join(
                "[Attempt {i}]\n"
                "<code>\n{code}\n</code>\n\n"
                "stderr:\n{stderr}".format(
                    i=i + 1,
                    code=(a.get("code") or "").rstrip(),
                    stderr=(a.get("stderr_tail") or "").rstrip() or "(empty)",
                )
                for i, a in enumerate(prior_attempts)
            )
            optional_sections += prompts.CODEGEN_REPAIR_SECTION.format(
                attempts=attempts_text,
            )
        return prompts.CODEGEN_USER_PROMPT.format(
            question=question,
            context=context or "(no extra context)",
            data_root=self.data_root,
            optional_sections=optional_sections,
        )

    def _call_codegen_llm(
        self,
        question: str,
        context: str,
        history: List[ClarificationTurn],
        prior_attempts: Optional[List[Dict[str, str]]] = None,
    ):
        system = prompts.CODEGEN_SYSTEM_PROMPT
        user = self._build_codegen_user_prompt(
            question, context, history, prior_attempts=prior_attempts,
        )
        resp = self.llm.chat(
            messages=[Message("user", user)],
            system=system,
            temperature=self.codegen_temperature,
        )[0]
        return (resp.text or ""), resp.finish_reason, system, user

    def _run_codegen_loop(
        self,
        question: str,
        context: str,
        history: List[ClarificationTurn],
        trace: ClarificationOnlyTrace,
    ) -> None:
        prior_attempts: List[Dict[str, str]] = []
        for attempt_idx in range(1, self.max_code_attempts + 1):
            try:
                raw, finish_reason, system, user = self._call_codegen_llm(
                    question, context, history, prior_attempts=prior_attempts,
                )
            except Exception as exc:                      # noqa: BLE001
                logger.exception("Code generation failed (attempt %d)",
                                 attempt_idx)
                trace.error = f"code_generation_failed: {exc}"
                return

            truncated = self._is_truncated(finish_reason)
            code = extract_code(raw)

            if truncated:
                trunc_note = (
                    "Previous response was cut off by the LLM token limit "
                    "(finish_reason=length). The partial script is unusable. "
                    "Write a MORE CONCISE complete runnable script — drop "
                    "verbose comments, collapse helper functions, keep only "
                    "the logic needed to compute FINAL_ANSWER."
                )
                trace.code_attempts.append({
                    "attempt": attempt_idx,
                    "system_prompt": system,
                    "user_prompt": user,
                    "raw_response": raw,
                    "code": code,
                    "success": False,
                    "exit_code": None,
                    "timed_out": False,
                    "stdout_tail": "",
                    "stderr_tail": trunc_note,
                    "final_value": None,
                    "code_path": None,
                    "note": "truncated_by_max_tokens",
                })
                trace.execution = {
                    "success": False,
                    "exit_code": None,
                    "timed_out": False,
                    "stdout_tail": "",
                    "stderr_tail": trunc_note,
                    "code_path": None,
                    "attempts": attempt_idx,
                }
                prior_attempts.append({
                    "code": code or "(truncated before any usable code)",
                    "stderr_tail": trunc_note,
                })
                if attempt_idx == self.max_code_attempts:
                    trace.error = "truncated_by_max_tokens"
                    return
                continue

            if not code:
                trace.code_attempts.append({
                    "attempt": attempt_idx,
                    "system_prompt": system,
                    "user_prompt": user,
                    "raw_response": raw,
                    "code": None,
                    "success": False,
                    "exit_code": None,
                    "timed_out": False,
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "final_value": None,
                    "code_path": None,
                    "note": "no_code_generated",
                })
                trace.execution = {
                    "success": False,
                    "exit_code": None,
                    "timed_out": False,
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "code_path": None,
                    "attempts": attempt_idx,
                }
                if attempt_idx == self.max_code_attempts:
                    trace.error = "no_code_generated"
                    return
                continue

            result = self.executor.run(code)
            trace.code_attempts.append({
                "attempt": attempt_idx,
                "system_prompt": system,
                "user_prompt": user,
                "raw_response": raw,
                "code": code,
                "success": result.success,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "stdout_tail": result.stdout[-2000:],
                "stderr_tail": result.stderr[-2000:],
                "final_value": result.final_value,
                "code_path": result.code_path,
            })
            trace.generated_code = code
            trace.execution = {
                "success": result.success,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "stdout_tail": result.stdout[-2000:],
                "stderr_tail": result.stderr[-2000:],
                "code_path": result.code_path,
                "attempts": attempt_idx,
            }
            trace.final_answer = result.final_value

            if result.success:
                trace.error = None
                return

            prior_attempts.append({
                "code": code,
                "stderr_tail": result.stderr[-2000:],
            })
            logger.info(
                "[codegen] attempt %d/%d failed — %s",
                attempt_idx, self.max_code_attempts,
                "timeout" if result.timed_out else f"exit={result.exit_code}",
            )

        trace.error = "execution_failed"