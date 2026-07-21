"""Oracle — simulated original user, driven by a single LLM call.

The previous implementation relied on keyword extraction + alias index +
semantic embeddings to look up entries in the operational-definitions
lexicon. That pipeline is replaced with a single LLM call that,
role-playing the original domain-expert user, RETRIEVES which lexicon
entries are relevant to the agent's question. The reply itself is then
assembled deterministically from those entries (verbatim), so it can
neither fabricate a method nor capitulate to a leading question — the
LLM's only degree of freedom is which entries to select.

Inputs surfaced to the simulated user (the LLM)
-----------------------------------------------
* The FULL operational-definitions list, projected down to the four
  disclosable fields per entry:
      ``id``, ``term``, ``mapped_params``, ``reasoning_note``.
  Other lexicon fields (``category``, ``related_terms``, etc.) are
  withheld.
* The user's original task question (the analyst-agent is trying to
  solve this).
* The original context / setup that came with the question.
* The agent's CURRENT clarification question.
* Optionally, a transcript of prior agent↔user clarification turns in
  the same conversation, so the simulated user can avoid redundantly
  re-disclosing things already shared.

Disclosure policy
-----------------
The LLM is used ONLY to RETRIEVE which entries are relevant; it never
authors the reply. The answer shown to the agent is built
deterministically from the selected entries, so it cannot drift from
the lexicon (no fabricated methods, no capitulation to a leading
question).

1. Original question wins. If the original task already pins the asked
   aspect, the LLM returns no entry and the agent is told no special
   convention applies.
2. Minimal disclosure ACROSS entries. Only entries DIRECTLY relevant to
   the agent's current question are selected; entries about not-yet-asked
   aspects are withheld even if related.
3. Full disclosure WITHIN a selected entry. Each selected entry is
   emitted verbatim — its full ``mapped_params`` and ``reasoning_note``,
   never summarised.
4. Honest uncertainty. If no entry covers the asked aspect, the LLM
   returns an empty selection and the agent receives a fixed neutral
   reply (any reasonable convention is acceptable). Nothing is
   fabricated.

Output contract back to the agent
---------------------------------
``OracleResponse.to_dict()`` returns:

    {
      "status":           "OK" | "INVALID_QUERY" | "LLM_ERROR",
      "query_received":   <the agent's clarification question>,
      "answer_text":      <natural-language reply, shown to the agent>,
      "matched_term_ids": [<lexicon ids the simulated user used>, ...],
      "matched_terms":    [<one canonical synonym per matched id>, ...],
      "message":          <optional human-readable status string>,
      "raw_llm_response": <raw LLM text, kept for the trace>,
    }

``answer_text`` is what the agent's clarification + codegen prompts
consume; it is assembled deterministically from the selected entries
(verbatim ``mapped_params`` + ``reasoning_note``), NOT written by the
LLM. ``matched_term_ids`` is what downstream evaluation uses to
score term-level precision / recall against the gold ambiguity set;
``matched_terms`` mirrors that list with the canonical (first) synonym
for each id, purely for human readability in the trace.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig

from src.runner.llm_client import (
    LLMClient,
    Message,
    build_llm_client,
    parse_json_block,
)

logger = logging.getLogger(__name__)


# =========================================================
# Response types
# =========================================================

class OracleStatus(str, Enum):
    OK = "OK"
    INVALID_QUERY = "INVALID_QUERY"
    LLM_ERROR = "LLM_ERROR"


@dataclass
class OracleResponse:
    status: OracleStatus
    query_received: str
    answer_text: str = ""
    matched_term_ids: List[str] = field(default_factory=list)
    # Parallel to ``matched_term_ids``: same length, each element is the
    # canonical synonym (first entry in the lexicon's ``term`` list) for
    # the id at the same index. Used only for human-readable inspection;
    # downstream metrics keep using ``matched_term_ids``.
    matched_terms: List[str] = field(default_factory=list)
    message: Optional[str] = None
    raw_llm_response: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "query_received": self.query_received,
            "answer_text": self.answer_text,
            "matched_term_ids": list(self.matched_term_ids),
            "matched_terms": list(self.matched_terms),
            "message": self.message,
            "raw_llm_response": self.raw_llm_response,
        }


# =========================================================
# Lexicon — strict projection to the disclosable fields
# =========================================================

_DISCLOSABLE_FIELDS = ("id", "term", "mapped_params", "reasoning_note")


class TermLexicon:
    """Loads JSONL entries and exposes only the four disclosable fields."""

    def __init__(self, lexicon_path: str) -> None:
        self._entries: List[Dict[str, Any]] = []
        self._by_id: Dict[str, Dict[str, Any]] = {}

        with open(lexicon_path, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Skipping malformed JSONL line %d: %s", line_no, exc,
                    )
                    continue
                tid = obj.get("id")
                if not tid:
                    continue
                projection = {k: obj.get(k) for k in _DISCLOSABLE_FIELDS}
                self._entries.append(projection)
                self._by_id[tid] = projection

        if not self._entries:
            raise ValueError(
                f"Lexicon file '{lexicon_path}' contains no valid entries."
            )

    def all_disclosable(self) -> List[Dict[str, Any]]:
        return [dict(e) for e in self._entries]

    def get(self, term_id: str) -> Optional[Dict[str, Any]]:
        entry = self._by_id.get(term_id)
        return dict(entry) if entry is not None else None

    def canonical_term(self, term_id: str) -> Optional[str]:
        """First synonym from the entry's ``term`` list, or None if missing.

        Used to expose a single readable name per matched id without
        leaking the full alias list back to the agent.
        """
        entry = self._by_id.get(term_id)
        if entry is None:
            return None
        synonyms = entry.get("term") or []
        if not synonyms:
            return None
        first = synonyms[0]
        return first if isinstance(first, str) else None


# =========================================================
# Prompts
# =========================================================

_SYSTEM_PROMPT = """You are role-playing as the original domain-expert USER who posed an oceanography / climate quantitative-analysis task to an AI analysis agent. The agent has stopped to ask you a clarification question. Your ONLY job here is to decide WHICH of your operational-definition conventions (if any) are directly needed to answer the agent's CURRENT question. You do NOT write the reply — the entries you select are disclosed to the agent verbatim on your behalf, so you cannot and need not paraphrase, summarise, or add anything.

============================================================
KNOWLEDGE BASE — your operational definitions
============================================================
Each entry has four fields:
- id: a stable identifier
- term: alternative phrasings of the concept
- mapped_params: concrete operational parameters (thresholds, formulas, baseline periods, regions, etc.)
- reasoning_note: natural-language rationale and conventions

Most entries define a task-specific concept. A few are UNIVERSAL DEFAULTS whose id starts with `CONV_` — they hold for every task unless the original question overrides them. They follow exactly the same selection rules below: include such a default only when the agent's current question is directly about that aspect.

ENTRIES (one JSON object per line):
{ops_list}

============================================================
SELECTION RULES — read carefully
============================================================
1. MINIMAL DISCLOSURE ACROSS ENTRIES. Select every entry the agent's CURRENT question directly needs — and nothing more. That may be a single entry or several when the question genuinely spans more than one concept. The point is to exclude entries about aspects the agent has not asked about yet, even if they look related, neighbouring, or likely to come up later. Do not pad the selection for completeness.
2. MATCH BY CONCEPT, NOT WORDING. Decide relevance from the underlying quantity or operation, not surface keywords. E.g. a question about "significance level", "p-value threshold", "confidence level", or "alpha" is directly about the significance convention even if it never says the word "significant"; a question about an "inventory" / "content" is about that derived-quantity entry even if the agent only says "mean" or "sum".
3. DON'T BE LED BY THE AGENT'S GUESS. The agent may propose a specific value or method ("use 0.05?", "detrend first?", "the area-weighted mean?"). Judge relevance against the knowledge base, NOT against what the agent proposed. If an entry governs the asked aspect, select it regardless of whether the agent's guess agrees with it — the verbatim disclosure will correct a wrong guess on its own.
4. ONE QUESTION AT A TIME. Consider only the agent's current clarification question. Do not pre-select entries for aspects the agent has not yet raised.
5. HONEST UNCERTAINTY. If NO entry covers the asked aspect, return an EMPTY list. Never select a loosely-related entry just to have something to say. Before returning empty, confirm that NO entry's mapped_params or reasoning_note governs the operation, axis, weighting, or definition being asked about — "the operation was named in the question" is NOT a reason to return empty.

============================================================
OUTPUT FORMAT — strictly enforced
============================================================
Reply with a single valid JSON object. No markdown fences, no preamble, no trailing commentary. Schema:
{{"term_ids": ["<ID1>", "<ID2>", ...]}}
- "term_ids": ids of the entries DIRECTLY required to answer the agent's current question, ordered by relevance. Empty list if the original question already settles it or no entry applies.

Your entire output MUST be parseable JSON. Nothing else."""


# ---------------------------------------------------------------------------
# Universal default conventions — disclosed verbatim like lexicon entries, but
# they live here (not in the JSONL) because they apply to every task and are
# NOT gold ambiguity terms. They are offered to the retrieval LLM under
# ``CONV_*`` ids and rendered through the same deterministic path, but are
# kept OUT of ``matched_term_ids`` so they never pollute term-level scoring.
# ---------------------------------------------------------------------------
_EXTRA_CONVENTIONS: List[Dict[str, Any]] = [
    {
        "id": "CONV_UNITS",
        "term": ["unit convention", "units", "unit", "unit conversion", "native units", "output unit",],
        "mapped_params": None,
        "reasoning_note": (
            "Report the numeric result in the native units of the variables involved; do NOT perform any extra unit conversion. The only exception: for the CESM2 model the depth-axis coordinate `lev` is stored in centimetres and must be converted to metres (divide by 100) before it is used in any downstream computation."
        ),
    },
]


# Shown verbatim to the agent when the retrieval step selects no entry: the
# honest "I'm not sure" stance. Deliberately offers no concrete value, so the
# agent either drops the question or rephrases it more precisely rather than
# being handed a fabricated convention.
_NOT_COVERED_REPLY = (
    "I'm not sure how to answer that off the top of my head. If you think it "
    "doesn't really matter for the result, feel free to skip it and go with a "
    "sensible default. If it does matter and you'd like to keep asking, please "
    "rephrase the question to be more clear and specific about exactly what you need."
)


_USER_PROMPT_TEMPLATE = """============================================================
ORIGINAL TASK YOU POSED TO THE AGENT
============================================================
Question:
{original_question}

Context / setup:
{original_context}

============================================================
PRIOR CLARIFICATION TURNS (if any)
============================================================
{prior_turns}

============================================================
AGENT'S CURRENT CLARIFICATION QUESTION
============================================================
{agent_question}

Respond now, following the rules in the system prompt. Output JSON only."""


def _format_ops_list(entries: List[Dict[str, Any]]) -> str:
    """Render each entry as a one-line JSON object (deterministic ordering)."""
    return "\n".join(
        json.dumps(e, ensure_ascii=False, sort_keys=False) for e in entries
    )


def _render_disclosure(entries: List[Dict[str, Any]]) -> str:
    """Build the agent-facing answer verbatim from the selected entries.

    Each entry contributes its full ``reasoning_note`` (natural language,
    read first for context) followed by its concrete ``mapped_params``
    (read second for exact values). Nothing is summarised or invented —
    this is the whole point of the deterministic path. Entry ids and the
    raw ``term`` alias list are NOT surfaced; the agent only needs the
    substance. Multiple entries are separated by a blank line.
    """
    blocks: List[str] = []
    for e in entries:
        lines: List[str] = []
        note = (e.get("reasoning_note") or "").strip()
        if note:
            lines.append(note)
        mp = e.get("mapped_params")
        if isinstance(mp, dict) and mp:
            kv = "; ".join(
                f"{k} = {json.dumps(v, ensure_ascii=False)}"
                for k, v in mp.items()
            )
            lines.append(f"Concrete parameters: {kv}.")
        if lines:
            blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _format_prior_turns(prior_turns: Optional[List[Dict[str, str]]]) -> str:
    if not prior_turns:
        return "(none — this is the first clarification turn)"
    blocks: List[str] = []
    for i, turn in enumerate(prior_turns, 1):
        q = (turn.get("question") or "").strip() or "(empty)"
        a = (turn.get("answer") or "").strip() or "(empty)"
        blocks.append(f"[Turn {i}] Agent asked: {q}\nYou answered: {a}")
    return "\n".join(blocks)


# =========================================================
# Public facade
# =========================================================

class Oracle:
    """LLM-only simulated-user oracle.

    Parameters
    ----------
    lexicon_path : str
        Path to the JSONL knowledge base.
    config : DictConfig
        Must contain an ``oracle`` subtree compatible with
        :func:`src.runner.llm_client.build_llm_client` — i.e. fields
        ``provider``, ``model``, ``api_key_env``, ``max_tokens``,
        ``temperature``, ``max_retries``, ``request_interval``. The
        same ``max_retries`` is reused for JSON-shape validation
        retries at this layer.
    """

    def __init__(self, lexicon_path: str, config: DictConfig) -> None:
        self._lexicon = TermLexicon(lexicon_path)
        self._llm: LLMClient = build_llm_client(config.oracle)
        self._max_retries = int(config.oracle.get("max_retries", 3) or 3)
        self._temperature = float(config.oracle.get("temperature", 0.0) or 0.0)
        # Universal default conventions, keyed by id. Offered to the retrieval
        # LLM alongside the lexicon and rendered through the same deterministic
        # path, but kept out of the file-backed lexicon so they never enter
        # term-level scoring.
        self._extra_by_id: Dict[str, Dict[str, Any]] = {
            e["id"]: dict(e) for e in _EXTRA_CONVENTIONS
        }
        # Retrieval prompt lists lexicon entries first, then universal defaults.
        self._ops_block = _format_ops_list(
            self._lexicon.all_disclosable() + list(self._extra_by_id.values())
        )
        self._system_prompt = _SYSTEM_PROMPT.format(ops_list=self._ops_block)

    def _resolve_entry(self, term_id: str) -> Optional[Dict[str, Any]]:
        """Look up a selected id in the lexicon first, then universal defaults."""
        entry = self._lexicon.get(term_id)
        if entry is not None:
            return entry
        extra = self._extra_by_id.get(term_id)
        return dict(extra) if extra is not None else None

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------
    def clarify(
        self,
        query: str,
        *,
        original_question: str = "",
        original_context: str = "",
        prior_turns: Optional[List[Dict[str, str]]] = None,
    ) -> OracleResponse:
        """Answer one agent clarification question.

        ``query`` is the agent's clarification question. The remaining
        keyword arguments give the simulated user the full surrounding
        context: the original task the user posed, the original context
        / setup, and any prior agent↔user clarification turns from the
        same conversation (so the simulated user can keep its
        minimal-disclosure stance across turns).
        """
        if not isinstance(query, str) or not query.strip():
            return OracleResponse(
                status=OracleStatus.INVALID_QUERY,
                query_received=str(query),
                message="Empty clarification query.",
            )

        user_prompt = _USER_PROMPT_TEMPLATE.format(
            original_question=(original_question.strip() or "(not provided)"),
            original_context=(original_context.strip() or "(not provided)"),
            prior_turns=_format_prior_turns(prior_turns),
            agent_question=query.strip(),
        )

        raw_text: Optional[str] = None
        parsed: Optional[Dict[str, Any]] = None
        last_err: Optional[Exception] = None

        for attempt in range(1, self._max_retries + 1):
            try:
                resp = self._llm.chat(
                    messages=[Message("user", user_prompt)],
                    system=self._system_prompt,
                    temperature=self._temperature,
                )[0]
                raw_text = resp.text or ""
                candidate = parse_json_block(raw_text)
                if self._is_valid_payload(candidate):
                    parsed = candidate
                    break
                logger.warning(
                    "Oracle LLM attempt %d/%d: schema validation failed. Head: %s",
                    attempt, self._max_retries, (raw_text or "")[:200],
                )
            except Exception as exc:                          # noqa: BLE001
                last_err = exc
                logger.warning(
                    "Oracle LLM attempt %d/%d: API error — %s",
                    attempt, self._max_retries, exc,
                )

        if parsed is None:
            return OracleResponse(
                status=OracleStatus.LLM_ERROR,
                query_received=query,
                answer_text=(
                    "I'm unable to give a confident answer to that right now. "
                    "If this aspect doesn't materially affect the result, "
                    "please proceed with a reasonable default."
                ),
                message=(
                    str(last_err) if last_err else
                    "LLM returned no schema-valid JSON after retries."
                ),
                raw_llm_response=raw_text,
            )

        # All selected ids (lexicon + universal defaults), order-preserving.
        selected_ids = self._sanitize_ids(parsed.get("term_ids"))

        # Deterministic answer: dump each selected entry verbatim. Empty
        # selection → fixed honest-uncertainty reply (never LLM-authored).
        entries = [self._resolve_entry(tid) for tid in selected_ids]
        entries = [e for e in entries if e is not None]
        answer_text = _render_disclosure(entries) or _NOT_COVERED_REPLY

        # Scoring surface: only file-backed lexicon ids count toward term-level
        # precision/recall — universal ``CONV_*`` defaults are excluded.
        matched_term_ids = [
            tid for tid in selected_ids if self._lexicon.get(tid) is not None
        ]
        # Parallel surface form: one canonical synonym per scored id. The
        # fallback is purely defensive against an entry with an empty ``term``.
        matched_terms: List[str] = []
        for tid in matched_term_ids:
            name = self._lexicon.canonical_term(tid)
            matched_terms.append(name if name is not None else tid)

        return OracleResponse(
            status=OracleStatus.OK,
            query_received=query,
            answer_text=answer_text,
            matched_term_ids=matched_term_ids,
            matched_terms=matched_terms,
            raw_llm_response=raw_text,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    @staticmethod
    def _is_valid_payload(parsed: Any) -> bool:
        if not isinstance(parsed, dict):
            return False
        if "term_ids" not in parsed:
            return False
        if not isinstance(parsed["term_ids"], list):
            return False
        return True

    def _sanitize_ids(self, ids: Any) -> List[str]:
        """Keep string ids known to the lexicon OR the universal defaults;
        preserve order, dedupe."""
        if not isinstance(ids, list):
            return []
        out: List[str] = []
        seen: set = set()
        for tid in ids:
            if not isinstance(tid, str):
                continue
            tid = tid.strip()
            if not tid or tid in seen:
                continue
            if self._resolve_entry(tid) is None:
                logger.debug(
                    "Oracle LLM returned unknown term_id '%s' — dropping.", tid,
                )
                continue
            seen.add(tid)
            out.append(tid)
        return out