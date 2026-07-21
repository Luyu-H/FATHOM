"""Top-level agent orchestrator.

There are two routes through ``AmbiguityAwareAgent.answer``:

Structured route (clarification.strategy ∈ {direct, candidate, planning})
    1. Uncertainty detection
    2. Clarification loop (only if AMBIGUOUS)
    3. Code generation + execution, with a repair loop on failure

Spontaneous route (clarification.strategy == "spontaneous")
    No upfront uncertainty detection and no structured clarification
    prompts.  The LLM sees only the standard ``CODEGEN_*`` prompts.
    If its reply does not contain a parseable ``<code>`` block, the raw
    text is forwarded to the Oracle as a clarification query, the
    Oracle's response is appended to history, and the codegen prompt is
    re-issued.  Strictly capped at ``clarification.max_turns`` Oracle
    interactions.  Once that budget is drained without ever producing
    code, control falls through to the same codegen-attempt loop used
    by the structured route — those attempts never re-route to the
    Oracle and burn ``code_exec.max_attempts`` before the run is
    declared a failure.

History-accumulation policy (applies to both routes)
    * Ambiguity-detection messages (system prompt, LLM judgment, raw
      reply) are NEVER added to history — that decision does not affect
      downstream code generation.  Only the binary "is ambiguous" flag is
      surfaced, and even that is shown to the clarification LLM only on
      its FIRST turn.
    * Clarification history accumulates only the final questions sent to
      the oracle and the oracle's responses.  Intermediate deliberation
      (e.g. candidate generation/selection) is not kept.
    * Code generation receives the clarification history on its first
      call.  On execution failure, the same history is reused and the
      prior code + stderr are appended so the LLM can repair.

Budget accounting
-----------------
``clarification.max_turns`` — Oracle interactions.  In structured mode
    those are consumed by the clarification strategy's questions.  In
    spontaneous mode they are consumed every time the codegen LLM
    replies with non-code text (which is then routed to the Oracle).
``code_exec.max_attempts`` — codegen + execute attempts.  Each LLM
    call that emits a script consumes one attempt, regardless of
    whether the script then runs to completion.  A codegen call that
    fails to emit code STILL consumes an attempt (the codegen LLM is
    never re-routed to the Oracle from this loop, in either route);
    we retry until the budget is exhausted.  In spontaneous mode,
    these attempts begin only AFTER the clarification budget is
    drained — the LLM calls inside the spontaneous clarification loop
    do not count against ``code_exec.max_attempts``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig

from . import prompts
from .llm_client import LLMClient, Message, build_llm_client
from .ambiguity_identifier import (
    AmbiguityLevel, UncertaintyDetector, build_uncertainty_detector,
)
from .clarification import (
    ClarificationStrategy, ClarificationTurn, build_clarification_strategy,
)
from .code_executor import CodeExecutor, extract_code

logger = logging.getLogger(__name__)


# =============================================================
# Trace record (one per question)
# =============================================================

@dataclass
class AgentTrace:
    question: str
    context: str
    uncertainty: Dict[str, Any] = field(default_factory=dict)
    clarification_turns: List[Dict[str, Any]] = field(default_factory=list)
    plan: Optional[str] = None
    # Final attempt's code (kept for back-compat with downstream tooling).
    generated_code: Optional[str] = None
    # Every code-gen + execute attempt, in order.  Each entry:
    #   {attempt, code, success, exit_code, timed_out,
    #    stdout_tail, stderr_tail, final_value, code_path}
    code_attempts: List[Dict[str, Any]] = field(default_factory=list)
    # Final attempt's execution result (mirrors the last code_attempts entry).
    execution: Dict[str, Any] = field(default_factory=dict)
    final_answer: Optional[str] = None
    error: Optional[str] = None


# =============================================================
# Agent
# =============================================================

class AmbiguityAwareAgent:
    """Top-level agent with pluggable uncertainty / clarification / code modules."""

    def __init__(self, config: DictConfig, oracle):
        self.config = config
        self.oracle = oracle
        self.llm: LLMClient = build_llm_client(config.llm)
        self.clarification: ClarificationStrategy = build_clarification_strategy(
            config.clarification, self.llm,
        )
        # Uncertainty detection is irrelevant in spontaneous mode — the
        # LLM itself decides when to ask. Skip the build step entirely.
        self.uncertainty: Optional[UncertaintyDetector] = None
        if not self.clarification.is_spontaneous:
            self.uncertainty = build_uncertainty_detector(
                config.ambiguity, self.llm,
            )
        self.executor = CodeExecutor(
            workspace_root=config.code_exec.workspace_root,
            timeout_seconds=config.code_exec.timeout_seconds,
            python_executable=config.code_exec.python_executable,
        )
        self.questions_per_turn = config.clarification.questions_per_turn
        # Code-gen + execute attempts.  attempt 1 is the initial run;
        # any further attempts are repair attempts seeded with prior
        # code and stderr.  Default to 1 (no repair) if unspecified.
        self.max_code_attempts = max(
            1, int(getattr(config.code_exec, "max_attempts", 1) or 1),
        )
        # Codegen temperature is centrally configured. Used for every
        # codegen LLM call: structured first-shot, repair attempts, and
        # the spontaneous-route codegen loop.
        self.codegen_temperature = float(config.code_exec.codegen_temperature)
        # Concrete CMIP6 data root injected into the codegen user prompt.
        # QA prompts only carry the abstract `<data_root>` placeholder;
        # the agent substitutes this path at runtime so prompts stay
        # environment-agnostic. Required — no silent default.
        data_root = getattr(getattr(config, "data", None), "data_root", None)
        if not data_root:
            raise ValueError(
                "config.data.data_root is required so the agent can tell "
                "the codegen LLM what to substitute for `<data_root>`."
            )
        self.data_root = str(data_root)

    # ============================================================
    # Public entry point
    # ============================================================
    def answer(self, question: str, context: str = "") -> AgentTrace:
        trace = AgentTrace(question=question, context=context)
        history: List[ClarificationTurn] = []
        first_attempt_code: Optional[str] = None
        # Wipe any per-question state held by the clarification strategy
        # (notably PlanningClarification._plan) so a plan generated for an
        # AMBIGUOUS question doesn't leak into a later CLEAR question whose
        # clarification loop never runs.
        self.clarification.reset()
        # When the spontaneous loop yields code on attempt 1, this carries
        # the (system_prompt, user_prompt, raw_response) of that exact call
        # so it can be attached to ``code_attempts[0]`` (not to a fake turn).
        first_attempt_codegen_call: Optional[Dict[str, Any]] = None

        # ----- Steps 1+2: ambiguity handling -----
        # Two mutually-exclusive routes — spontaneous does NOT fall into
        # the structured clarification loop.
        if self.clarification.is_spontaneous:
            # No upfront uncertainty detection, no structured clarification.
            # Ambiguity is handled DURING codegen: any LLM reply that
            # isn't a parseable <code> block is routed to the Oracle,
            # consuming one clarification turn (not a codegen attempt).
            trace.uncertainty = {"strategy": "spontaneous (skipped)"}
            try:
                first_attempt_code, history, first_attempt_codegen_call = (
                    self._spontaneous_codegen_loop(question, context)
                )
            except Exception as exc:
                logger.exception("Spontaneous codegen failed")
                trace.error = f"code_generation_failed: {exc}"
                return trace
            for h in history:
                trace.clarification_turns.append({
                    "questions": h.questions,
                    "oracle_responses": h.oracle_responses,
                    "llm_calls": h.llm_calls,
                    "spontaneous": True,
                })
            # If the clarification budget was drained without ever
            # producing code, do NOT abort here — fall through to the
            # main codegen loop, which will burn `code_exec.max_attempts`
            # of oracle-free codegen calls before declaring failure.
        else:
            # Structured route — uncertainty judged on the question alone
            # (`context` carries dataset/time-window info the agent is
            # supposed to figure out itself; feeding it here would mask
            # genuine ambiguity). If AMBIGUOUS, run the clarification
            # loop; codegen never re-routes to the Oracle in this route.
            u = self.uncertainty.detect(question)
            trace.uncertainty = {
                "level": u.level.value,
                "score": u.score,
                "ambiguous_terms": u.ambiguous_terms,
                "details": u.details,
            }
            logger.info("[uncertainty] level=%s score=%.3f",
                        u.level.value, u.score)
            if u.level == AmbiguityLevel.AMBIGUOUS:
                history = self._clarification_loop(
                    question, context, u.ambiguous_terms,
                )
                for h in history:
                    trace.clarification_turns.append({
                        "questions": h.questions,
                        "oracle_responses": h.oracle_responses,
                        "llm_calls": h.llm_calls,
                        "spontaneous": False,
                    })
        trace.plan = self.clarification.plan

        # ----- Step 3: code generation + execution with repair loop -----
        # Each iteration consumes ONE codegen attempt.  In structured mode
        # every iteration calls _generate_code; in spontaneous mode the
        # first iteration reuses the code already produced by Step 2
        # (its LLM call is the one codegen-budget unit charged for that
        # first attempt).
        prior_attempts: List[Dict[str, str]] = []
        # Holds the most recent codegen call's prompts + raw response so
        # each ``code_attempts`` entry preserves the full LLM exchange
        # that produced (or failed to produce) the script. In the
        # spontaneous route, attempt 1 reuses the call captured inside
        # ``_spontaneous_codegen_loop``; later attempts capture themselves.
        codegen_system_prompt: Optional[str] = None
        codegen_user_prompt: Optional[str] = None
        codegen_raw_response: Optional[str] = None
        for attempt_idx in range(1, self.max_code_attempts + 1):
            truncated = False
            if attempt_idx == 1 and first_attempt_code is not None:
                code: Optional[str] = first_attempt_code
                if first_attempt_codegen_call is not None:
                    codegen_system_prompt = first_attempt_codegen_call.get("system_prompt")
                    codegen_user_prompt = first_attempt_codegen_call.get("user_prompt")
                    codegen_raw_response = first_attempt_codegen_call.get("raw_response")
            else:
                try:
                    (code, truncated, codegen_system_prompt,
                     codegen_user_prompt, codegen_raw_response) = self._generate_code(
                        question, context, history,
                        prior_attempts=prior_attempts,
                    )
                except Exception as exc:
                    logger.exception("Code generation failed (attempt %d)",
                                     attempt_idx)
                    trace.error = f"code_generation_failed: {exc}"
                    return trace

            # Truncated responses are unreliable (incomplete statements,
            # orphan tags). Discard the partial code and seed the repair
            # loop with an explicit "be concise" hint so the next codegen
            # call doesn't reproduce the same too-long script.
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
                    "system_prompt": codegen_system_prompt,
                    "user_prompt": codegen_user_prompt,
                    "raw_response": codegen_raw_response,
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
                logger.info(
                    "[codegen] attempt %d/%d truncated by max_tokens",
                    attempt_idx, self.max_code_attempts,
                )
                if attempt_idx == self.max_code_attempts:
                    trace.error = "truncated_by_max_tokens"
                    return trace
                continue

            if not code:
                # Codegen LLM returned no parseable code block.  In
                # structured mode we never re-route to the Oracle from
                # here — this counts as one consumed codegen attempt.
                # If budget remains, retry; otherwise give up.
                trace.code_attempts.append({
                    "attempt": attempt_idx,
                    "system_prompt": codegen_system_prompt,
                    "user_prompt": codegen_user_prompt,
                    "raw_response": codegen_raw_response,
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
                logger.info("[codegen] attempt %d/%d returned no code",
                            attempt_idx, self.max_code_attempts)
                if attempt_idx == self.max_code_attempts:
                    trace.error = "no_code_generated"
                    return trace
                continue

            result = self.executor.run(code)
            attempt_record = {
                "attempt": attempt_idx,
                "system_prompt": codegen_system_prompt,
                "user_prompt": codegen_user_prompt,
                "raw_response": codegen_raw_response,
                "code": code,
                "success": result.success,
                "exit_code": result.exit_code,
                "timed_out": result.timed_out,
                "stdout_tail": result.stdout[-2000:],
                "stderr_tail": result.stderr[-2000:],
                "final_value": result.final_value,
                "code_path": result.code_path,
            }
            trace.code_attempts.append(attempt_record)

            # Mirror the latest attempt into the back-compat fields.
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
                return trace

            # Failure — stack this attempt onto the repair history and retry.
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
        return trace

    # ============================================================
    # Internals
    # ============================================================
    def _clarification_loop(
        self, question: str, context: str, ambiguous_terms: List[str],
    ) -> List[ClarificationTurn]:
        history: List[ClarificationTurn] = []
        while self.clarification.should_continue(history):
            qs = self.clarification.make_questions(
                question, context, history, ambiguous_terms,
            )
            if not qs:
                # Strategy chose not to ask this round. Preserve the LLM
                # exchange (and any plan it set on the strategy) by
                # appending a turn with no oracle interaction, so the
                # trace records that clarification was actually entered.
                # `total_turns` in the summary filters these out.
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
                # Snapshot every system+user prompt that this strategy
                # issued to produce the turn's questions.
                llm_calls=self.clarification.last_llm_calls,
            )
            # Snapshot prior completed turns once per agent turn — the
            # simulated user uses this to honour minimal disclosure across
            # turns (don't re-disclose what was already shared).
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
        """Flatten ``history`` into the {question, answer} list the oracle expects."""
        out: List[Dict[str, str]] = []
        for turn in history:
            for q, resp in zip(turn.questions, turn.oracle_responses):
                out.append({
                    "question": q,
                    "answer": (resp.get("answer_text") or "").strip(),
                })
        return out

    def _build_codegen_user_prompt(
        self,
        question: str,
        context: str,
        history: List[ClarificationTurn],
        prior_attempts: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """Render the codegen user message (shared by structured + spontaneous)."""
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
            optional_sections += prompts.CODEGEN_REPAIR_SECTION.format(attempts=attempts_text)

        return prompts.CODEGEN_USER_PROMPT.format(
            question=question,
            context=context or "(no extra context)",
            data_root=self.data_root,
            optional_sections=optional_sections,
        )

    # Finish-reason strings that mean "we hit max_tokens" across providers.
    # OpenAI/DeepSeek: "length". Anthropic: "max_tokens". Gemini: enum
    # whose str() contains "MAX_TOKENS".
    _TRUNCATION_REASONS = frozenset({"length", "max_tokens"})

    @classmethod
    def _is_truncated(cls, finish_reason: Optional[str]) -> bool:
        if not finish_reason:
            return False
        fr = str(finish_reason).lower()
        return fr in cls._TRUNCATION_REASONS or "max_tokens" in fr

    def _call_codegen_llm(
        self,
        question: str,
        context: str,
        history: List[ClarificationTurn],
        prior_attempts: Optional[List[Dict[str, str]]] = None,
    ) -> "tuple[str, Optional[str], str, str]":
        """Issue ONE codegen LLM call.

        Returns ``(raw_text, finish_reason, system_prompt, user_prompt)``.
        The two prompt strings are surfaced so callers can persist the full
        conversation (system + user) on the corresponding trace entry.
        """
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

    def _generate_code(
        self,
        question: str,
        context: str,
        history: List[ClarificationTurn],
        prior_attempts: Optional[List[Dict[str, str]]] = None,
    ) -> "tuple[Optional[str], bool, str, str, str]":
        """Generate a runnable Python script from upstream-resolved context.

        Returns ``(code, truncated, system_prompt, user_prompt, raw_response)``.
        The prompt strings + raw response are surfaced so callers can persist
        the full codegen exchange on the corresponding ``code_attempts`` entry.

        ``truncated`` is True when the LLM response was cut off by
        ``max_tokens`` — in that case the partial text is unreliable
        (incomplete statements, orphan ``<code>`` tag) and the caller should
        discard the code and feed a "write more concisely" hint into the next
        attempt instead of running garbage.

        On the first call ``prior_attempts`` is empty; on repair calls it
        carries each failed attempt's ``{code, stderr_tail}`` so the LLM
        can diagnose and fix without losing the clarification history.
        """
        raw, finish_reason, system, user = self._call_codegen_llm(
            question, context, history, prior_attempts=prior_attempts,
        )
        truncated = self._is_truncated(finish_reason)
        return extract_code(raw), truncated, system, user, raw

    def _spontaneous_codegen_loop(
        self,
        question: str,
        context: str,
    ) -> "tuple[Optional[str], List[ClarificationTurn], Optional[Dict[str, str]]]":
        """Spontaneous baseline — no extra clarification prompts.

        Repeatedly issues the standard codegen prompt.  If the reply
        contains a ``<code>`` block, we return it (consumes one codegen
        attempt at the caller).  Otherwise we forward the raw reply to
        the Oracle, append the (text, oracle response) pair to history,
        and re-issue the same codegen prompt — accumulating history in
        the ``CODEGEN_HISTORY_SECTION`` of the next user message.

        Strictly capped at ``clarification.max_turns`` Oracle interactions.
        Returns ``(code, history, code_call)``:
          * ``code`` is the parsed ``<code>`` block (or ``None`` if the
            budget was drained without producing one);
          * ``history`` carries each oracle-routed turn (questions, oracle
            response, and the LLM call that produced the question);
          * ``code_call`` is the (system, user, raw_response) record for
            the call that finally yielded code — surfaced separately so
            the caller can attach it to the first ``code_attempts`` entry
            rather than to an oracle-less clarification turn (which would
            inflate ``total_turns``).
        If the budget is drained without producing code, ``code_call`` is
        ``None`` and the caller's regular codegen-attempt loop will record
        its own prompts.
        """
        history: List[ClarificationTurn] = []
        max_turns = self.clarification.max_turns

        while len(history) < max_turns:
            raw, finish_reason, sys_prompt, user_prompt = self._call_codegen_llm(
                question, context, history,
            )
            llm_call = {
                "role": "codegen_spontaneous",
                "system_prompt": sys_prompt,
                "user_prompt": user_prompt,
                "raw_response": raw,
            }
            # Truncation is a tech failure, NOT a clarification request.
            # Forwarding a half-written script to the Oracle as a
            # clarification query would waste an oracle turn AND produce
            # nonsense. Bail out of the spontaneous loop so the main
            # codegen loop can retry with a "be concise" repair hint.
            if self._is_truncated(finish_reason):
                logger.info(
                    "[spontaneous] truncated by max_tokens — falling "
                    "through to main codegen loop",
                )
                break

            code = extract_code(raw)
            if code:
                return code, history, llm_call

            text = (raw or "").strip()
            if not text:
                # Empty response can't be routed as a clarification
                # query — stop draining the clarification budget and let
                # the main codegen loop take over.
                break

            oracle_resp = self.oracle.clarify(
                text,
                original_question=question,
                original_context=context,
                prior_turns=self._oracle_prior_turns(history),
            ).to_dict()
            history.append(ClarificationTurn(
                questions=[text],
                oracle_responses=[oracle_resp],
                llm_calls=[llm_call],
            ))
            logger.info(
                "[spontaneous] turn %d — routed non-code reply to oracle "
                "(status=%s, %d term(s))",
                len(history),
                oracle_resp.get("status"),
                len(oracle_resp.get("matched_term_ids") or []),
            )

        return None, history, None