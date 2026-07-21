"""Metric computation for the Intent Benchmark.

Consumes per-record summary files written by :mod:`src.run`
(``<output_dir>/<run_name>/summary/*.json``) and produces aggregated metric
reports under ``<output_dir>/<run_name>/metrics/``.

Two metric families are computed:

* **Process metrics** — how the agent interacted with the oracle.
    - ``avg_interaction_turns``     mean clarification rounds per question
    - ``pct_perceived_ambiguous``   fraction of records the agent flagged
    - ``term_precision`` / ``term_recall`` / ``term_f1``
                                    on matched ``term_id`` sets (gold vs. queried)
    - ``term_redundancy_rate``      fraction of queried terms NOT in gold
    - ``avg_code_attempts``         mean code-gen+repair attempts used
    - ``code_exec_success_rate``    fraction of records whose final code ran

* **Outcome metrics** — did the final answer match the gold value?
    - ``boolean_accuracy``                              yes/no exact match
    - ``scalar_accuracy_<tol>``                         relative-error within tolerance
    - ``scalar_mean_relative_error`` / ``_median_``     numeric closeness
    - ``overall_accuracy``                              boolean + strictest scalar tol
    - ``given_exec_success.*``                          same accuracy numerators,
                                                        denominator restricted to
                                                        records whose final code ran
                                                        (separates "couldn't write
                                                        working code" from "wrote
                                                        working but wrong code")

All metrics are computed several ways:
    1.  **Overall** — every record in the run.
    2.  **By level** — split by ``level`` (1/2/3).
    3.  **By ambiguity type** — for each lexicon category (Terminological,
        Methodological, Spatial, Temporal, Vertical, Indicator), the subset of
        records whose gold ambiguous-term set contains at least one term of
        that category.
    4.  **By level × ambiguity type** — the cartesian breakdown.
    5.  **By task type** — split by analysis type (Descriptive Statistics,
        Vertical Profile, Temporal Trend, Comparative Assessment, Correlation,
        Change & Anomaly, Model Ensemble), a hard partition derived from each
        record's ``task_id`` (one record → exactly one task type).
    6.  **By level × task type** — the cartesian breakdown.
    7.  **By ambiguity complexity** — split by the number of gold ambiguous
        terms (``len(gold_term_ids)``); a hard partition (``complexity_0`` =
        unambiguous, ``complexity_1``, ``complexity_2``, ...).
    8.  **By level × ambiguity complexity** — the cartesian breakdown.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Set

logger = logging.getLogger(__name__)


# =========================================================
# Scalar tolerance config
# =========================================================

@dataclass(frozen=True)
class ScalarConfig:
    """Tolerance bands + zero-guard parameters for scalar correctness.

    `tolerances` is an ordered list of (name, rel_tol) pairs reported in
    ``scalar_accuracy``. `primary` names the tolerance whose hit-rate is
    folded into ``overall_accuracy`` alongside boolean accuracy.

    Correctness uses a combined criterion:
        |pred - gold| <= max(rel_tol * |gold|, abs_floor)

    This avoids the brittle "switch to absolute when gold is tiny" cliff:
    `abs_floor` is a small, fixed floor (think float64 precision) that
    prevents demanding sub-epsilon precision on legitimately small gold
    values, without the old behaviour of accepting *any* near-zero pred for
    sub-zero-guard golds.

    `zero_guard` is now only for division protection in `relative_error`:
    when |gold| < zero_guard, the relative-error number is undefined and we
    return None (so it doesn't pollute mean/median aggregates).
    """
    tolerances: Tuple[Tuple[str, float], ...]
    primary: str
    zero_guard: float = 1e-30
    abs_floor: float = 1e-15


# Sensible defaults used when callers don't pass an explicit ScalarConfig
# (kept so the module can be imported standalone from a REPL/test).
DEFAULT_SCALAR_CONFIG = ScalarConfig(
    tolerances=(
        ("rel_1e-3", 1e-3),
        ("rel_1pct", 1e-2),
        ("rel_5pct", 5e-2),
        ("rel_10pct", 1e-1),
        ("rel_25pct", 2.5e-1),
    ),
    primary="rel_5pct",
)


# =========================================================
# Tiny lexicon loader — independent of oracle.py so eval can
# run without sentence-transformers / spacy installed.
# =========================================================

@dataclass
class LexiconEntry:
    term_id: str
    aliases: List[str]
    category: Optional[str]


class SimpleLexicon:
    """Minimal alias → term_id index, plus term_id → category lookup."""

    def __init__(self, lexicon_path: str) -> None:
        self.entries: Dict[str, LexiconEntry] = {}
        self._alias_to_id: Dict[str, str] = {}

        with open(lexicon_path, "r", encoding="utf-8") as fh:
            for line_no, raw in enumerate(fh, 1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Lexicon line %d malformed: %s", line_no, exc)
                    continue
                tid = obj.get("id")
                if not tid:
                    continue
                aliases = list(obj.get("term") or [])
                self.entries[tid] = LexiconEntry(
                    term_id=tid,
                    aliases=aliases,
                    category=obj.get("category"),
                )
                for alias in aliases:
                    key = _normalize_term(alias)
                    if key and key not in self._alias_to_id:
                        self._alias_to_id[key] = tid

    def resolve(self, term: str) -> Optional[str]:
        return self._alias_to_id.get(_normalize_term(term))

    def category_of(self, term_id: str) -> Optional[str]:
        entry = self.entries.get(term_id)
        return entry.category if entry else None

    def all_categories(self) -> List[str]:
        return sorted({e.category for e in self.entries.values() if e.category})


def _normalize_term(text: str) -> str:
    """Lightweight normalizer — lowercase, strip punctuation, collapse spaces.

    Intentionally simpler than oracle.normalize (no lemmatization) so the
    metrics module has zero heavy NLP deps. Good enough for matching gold
    surface terms against lexicon aliases, which are short literal phrases.
    """
    if not text:
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s\-]", "", text)
    text = re.sub(r"-", " ", text)
    text = re.sub(r"\b(a|an|the)\b\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================================================
# Task-type (analysis-type) classification
# =========================================================
#
# Each benchmark task (identified by ``task_id`` like "L1_1" / "L2_10" /
# "L3_14") belongs to exactly one analysis type. Unlike ambiguity category
# — which a record can span several of — task type is a hard partition, so
# it's bucketed like ``level`` (one record → one task type).
#
# The mapping is authored from the benchmark design table (Level × ID →
# Analysis Type) and keyed by the flat ``task_id`` string.

TASK_TYPE_BY_TASK_ID: Dict[str, str] = {
    # ----- Level 1 -----
    "L1_1": "Descriptive Statistics",
    "L1_2": "Vertical Profile",
    "L1_3": "Vertical Profile",
    "L1_4": "Temporal Trend",
    "L1_5": "Temporal Trend",
    "L1_6": "Descriptive Statistics",
    "L1_7": "Temporal Trend",
    "L1_8": "Comparative Assessment",
    "L1_9": "Descriptive Statistics",
    "L1_10": "Comparative Assessment",
    "L1_11": "Correlation",
    "L1_12": "Vertical Profile",
    # ----- Level 2 -----
    "L2_1": "Change & Anomaly",
    "L2_2": "Change & Anomaly",
    "L2_3": "Vertical Profile",
    "L2_4": "Change & Anomaly",
    "L2_5": "Comparative Assessment",
    "L2_6": "Comparative Assessment",
    "L2_7": "Descriptive Statistics",
    "L2_8": "Temporal Trend",
    "L2_9": "Change & Anomaly",
    "L2_10": "Descriptive Statistics",
    "L2_11": "Change & Anomaly",
    "L2_12": "Change & Anomaly",
    "L2_13": "Descriptive Statistics",
    "L2_14": "Comparative Assessment",
    "L2_15": "Comparative Assessment",
    # ----- Level 3 -----
    "L3_1": "Temporal Trend",
    "L3_2": "Temporal Trend",
    "L3_3": "Correlation",
    "L3_4": "Correlation",
    "L3_5": "Temporal Trend",
    "L3_6": "Change & Anomaly",
    "L3_7": "Descriptive Statistics",
    "L3_8": "Change & Anomaly",
    "L3_9": "Change & Anomaly",
    "L3_10": "Model Ensemble",
    "L3_11": "Model Ensemble",
    "L3_12": "Temporal Trend",
    "L3_13": "Model Ensemble",
    "L3_14": "Correlation",
}

# Canonical presentation order for the analysis types (first appearance in
# the design table). Used for stable reference output.
TASK_TYPES: List[str] = [
    "Descriptive Statistics",
    "Vertical Profile",
    "Temporal Trend",
    "Comparative Assessment",
    "Correlation",
    "Change & Anomaly",
    "Model Ensemble",
]


def resolve_task_id(record: Dict[str, Any]) -> Optional[str]:
    """Pick the record's ``task_id`` (e.g. "L1_1"), or derive it from record_id.

    Records carry ``task_id`` directly; when absent we reconstruct it from
    the first two underscore-separated parts of ``record_id`` (e.g.
    "L1_1_2" → "L1_1").
    """
    tid = record.get("task_id")
    if isinstance(tid, str) and tid:
        return tid
    rid = record.get("record_id")
    if isinstance(rid, str):
        parts = rid.split("_")
        if len(parts) >= 2:
            return f"{parts[0]}_{parts[1]}"
    return None


def task_type_of(record: Dict[str, Any]) -> Optional[str]:
    """Map a record to its analysis type, or None when the task_id is unknown."""
    task_id = resolve_task_id(record)
    if task_id is None:
        return None
    return TASK_TYPE_BY_TASK_ID.get(task_id)


# =========================================================
# Answer parsing & equivalence
# =========================================================

_BOOL_TRUE = {"yes", "y", "true", "t", "1"}
_BOOL_FALSE = {"no", "n", "false", "f", "0"}


def parse_boolean(value: Any) -> Optional[bool]:
    """Coerce a final/expected answer to a bool, or None if not boolean-ish."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    s = str(value).strip().lower()
    # Pick first token, stripping trailing punctuation — agent sometimes
    # emits "Yes." or "No, the condition...".
    head = s.split()[0] if s else s
    head = re.sub(r"[^\w]+$", "", head)
    if head in _BOOL_TRUE:
        return True
    if head in _BOOL_FALSE:
        return False
    return None


_NUM_RE = re.compile(
    r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"
)


def parse_scalar(value: Any) -> Optional[float]:
    """Extract a leading float from value (handles ``"34.63... 0.001"``)."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    s = str(value).strip()
    if not s:
        return None
    m = _NUM_RE.search(s)
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def scalar_close(
    pred: float,
    gold: float,
    rel_tol: float,
    scalar_cfg: ScalarConfig = DEFAULT_SCALAR_CONFIG,
) -> bool:
    """Closeness check: |pred - gold| <= max(rel_tol * |gold|, abs_floor).

    The `abs_floor` term protects against demanding sub-float-precision on
    tiny gold values (e.g. variance ~1e-18) without the old behaviour of
    auto-passing any prediction within 1e-9 of a tiny gold. For gold == 0
    (e.g. lag == 0 months), prediction must be within `abs_floor` of zero.
    """
    if math.isnan(pred) or math.isnan(gold):
        return False
    threshold = max(rel_tol * abs(gold), scalar_cfg.abs_floor)
    return abs(pred - gold) <= threshold


def relative_error(
    pred: float,
    gold: float,
    scalar_cfg: ScalarConfig = DEFAULT_SCALAR_CONFIG,
) -> Optional[float]:
    """Per-record relative error, or None when |gold| is below `zero_guard`.

    Returning None for sub-guard golds (lag==0 etc.) keeps them out of the
    mean/median rel-err aggregates — there `|pred - gold|` would otherwise
    masquerade as a relative number and dominate the average.
    """
    if math.isnan(pred) or math.isnan(gold):
        return None
    if abs(gold) < scalar_cfg.zero_guard:
        return None
    return abs(pred - gold) / abs(gold)


# =========================================================
# Ambiguous-term gold resolution
# =========================================================

def resolve_gold_term_ids(record: Dict[str, Any], lex: SimpleLexicon) -> List[str]:
    """Pick ``ambiguous_term_ids_gold`` if present; else resolve from terms.

    Records may have an explicit ``ambiguous_term_ids_gold`` list that maps
    1-to-1 with ``ambiguous_terms_gold``. When that field is null (current
    data has a known gap), we fall back to alias lookup in the lexicon.
    Unresolved terms are dropped — there's no sensible category for them,
    so they don't participate in any category-level breakdown.
    """
    explicit = record.get("ambiguous_term_ids_gold")
    if isinstance(explicit, list) and explicit:
        return [t for t in explicit if isinstance(t, str)]

    terms = record.get("ambiguous_terms_gold") or []
    if not isinstance(terms, list):
        return []
    resolved: List[str] = []
    seen: Set[str] = set()
    for t in terms:
        if not isinstance(t, str):
            continue
        tid = lex.resolve(t)
        if tid and tid not in seen:
            resolved.append(tid)
            seen.add(tid)
    return resolved


def queried_term_ids(record: Dict[str, Any]) -> List[str]:
    """The dedup'd set of term_ids the agent actually resolved with the oracle."""
    ids = record.get("unique_term_ids") or []
    if isinstance(ids, list):
        return [t for t in ids if isinstance(t, str)]
    return []


def count_queries_per_turn(record: Dict[str, Any]) -> int:
    """Total query items across all clarification turns (pre-dedup-by-id).

    Uses the ``queried_terms`` flat list saved by the runner — one entry
    per (extracted_term, matched_term_id) the agent surfaced in a turn.
    This is the right numerator for "terms asked per turn" because the
    dedup'd ``unique_term_ids`` collapses repeated asks across turns.
    """
    qs = record.get("queried_terms") or []
    if isinstance(qs, list):
        return sum(1 for q in qs if isinstance(q, dict))
    return 0


# =========================================================
# Per-record metric extraction
# =========================================================

@dataclass
class RecordMetrics:
    record_id: Optional[str]
    level: Optional[int]
    answer_type: Optional[str]            # 'boolean' | 'scalar' | None
    task_id: Optional[str] = None
    task_type: Optional[str] = None       # analysis type; None when unmapped
    gold_term_ids: List[str] = field(default_factory=list)
    queried_term_ids: List[str] = field(default_factory=list)
    gold_categories: List[str] = field(default_factory=list)

    # Process numbers
    total_turns: int = 0
    perceived_ambiguous: bool = False
    truly_ambiguous: bool = False         # gold has at least one ambiguous term
    n_queried_terms: int = 0              # how many terms the agent surfaced
    n_gold_terms: int = 0                 # how many gold ambiguous terms exist
    terms_per_turn: Optional[float] = None  # None when total_turns == 0
    code_success: bool = False
    code_attempts: int = 0

    # Ambiguity identification (per-record values; aggregated later)
    term_tp: int = 0
    term_fp: int = 0      # queried but not in gold (redundant)
    term_fn: int = 0      # gold but not queried (missed)
    term_precision: Optional[float] = None
    term_recall: Optional[float] = None
    term_f1: Optional[float] = None
    term_redundancy: Optional[float] = None

    # Outcome
    boolean_correct: Optional[bool] = None
    scalar_relative_error: Optional[float] = None
    scalar_correct: Dict[str, Optional[bool]] = field(default_factory=dict)
    overall_correct: Optional[bool] = None


def _safe_div(num: float, den: float) -> Optional[float]:
    if den == 0:
        return None
    return num / den


def compute_record_metrics(
    record: Dict[str, Any],
    lex: SimpleLexicon,
    scalar_cfg: ScalarConfig = DEFAULT_SCALAR_CONFIG,
) -> RecordMetrics:
    gold_ids = resolve_gold_term_ids(record, lex)
    queried = queried_term_ids(record)
    gold_set, queried_set = set(gold_ids), set(queried)

    tp = len(gold_set & queried_set)
    fp = len(queried_set - gold_set)
    fn = len(gold_set - queried_set)

    precision = _safe_div(tp, tp + fp)         # None when agent queried nothing
    recall = _safe_div(tp, tp + fn)            # None when no gold terms exist
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    redundancy = _safe_div(fp, tp + fp)        # 1 - precision; None when no queries

    # Categories spanned by the gold terms of this record.
    gold_cats: List[str] = []
    seen_c: Set[str] = set()
    for tid in gold_ids:
        c = lex.category_of(tid)
        if c and c not in seen_c:
            gold_cats.append(c)
            seen_c.add(c)

    # Per-turn ask volume — count raw query items, not the term_id set
    # (the agent can legitimately ask about the same term across turns).
    total_turns = int(record.get("total_turns") or 0)
    n_queries_total = count_queries_per_turn(record)
    terms_per_turn = (n_queries_total / total_turns) if total_turns > 0 else None

    # Outcome
    exec_info = record.get("code_execution") or {}
    code_success = bool(exec_info.get("success"))
    attempts = int(exec_info.get("attempts") or 0)

    expected = record.get("expected_answer")
    final = record.get("final_answer")
    answer_type = _infer_answer_type(expected)

    boolean_correct: Optional[bool] = None
    scalar_rel_err: Optional[float] = None
    scalar_correct: Dict[str, Optional[bool]] = {}
    overall_correct: Optional[bool] = None

    if answer_type == "boolean":
        gold_b = parse_boolean(expected)
        pred_b = parse_boolean(final)
        if gold_b is None:
            boolean_correct = None
        else:
            boolean_correct = (pred_b is not None and pred_b == gold_b)
        overall_correct = boolean_correct
    elif answer_type == "scalar":
        gold_v = parse_scalar(expected)
        pred_v = parse_scalar(final)
        if gold_v is None:
            # Gold unparseable — skip from scoring.
            for k, _ in scalar_cfg.tolerances:
                scalar_correct[k] = None
        elif pred_v is None:
            scalar_rel_err = None
            for k, _ in scalar_cfg.tolerances:
                scalar_correct[k] = False
        else:
            scalar_rel_err = relative_error(pred_v, gold_v, scalar_cfg)
            for k, tol in scalar_cfg.tolerances:
                scalar_correct[k] = scalar_close(pred_v, gold_v, tol, scalar_cfg)
        overall_correct = scalar_correct.get(scalar_cfg.primary)

    return RecordMetrics(
        record_id=record.get("record_id"),
        level=record.get("level"),
        answer_type=answer_type,
        task_id=resolve_task_id(record),
        task_type=task_type_of(record),
        gold_term_ids=gold_ids,
        queried_term_ids=queried,
        gold_categories=gold_cats,
        total_turns=total_turns,
        perceived_ambiguous=bool(record.get("agent_perceived_ambiguous")),
        truly_ambiguous=bool(gold_ids),
        n_queried_terms=len(queried),
        n_gold_terms=len(gold_ids),
        terms_per_turn=terms_per_turn,
        code_success=code_success,
        code_attempts=attempts,
        term_tp=tp, term_fp=fp, term_fn=fn,
        term_precision=precision,
        term_recall=recall,
        term_f1=f1,
        term_redundancy=redundancy,
        boolean_correct=boolean_correct,
        scalar_relative_error=scalar_rel_err,
        scalar_correct=scalar_correct,
        overall_correct=overall_correct,
    )


def _infer_answer_type(expected: Any) -> Optional[str]:
    """Heuristic: 'Yes'/'No' string → boolean; numeric-looking → scalar."""
    if expected is None:
        return None
    s = str(expected).strip()
    if not s:
        return None
    head = s.split()[0].lower()
    if head in _BOOL_TRUE or head in _BOOL_FALSE:
        return "boolean"
    if _NUM_RE.match(s):
        return "scalar"
    return None


# =========================================================
# Aggregation
# =========================================================

def _mean(xs: Iterable[Optional[float]]) -> Optional[float]:
    vals = [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _median(xs: Iterable[Optional[float]]) -> Optional[float]:
    vals = sorted(float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x)))
    if not vals:
        return None
    n = len(vals)
    mid = n // 2
    return vals[mid] if n % 2 else 0.5 * (vals[mid - 1] + vals[mid])


def _rate(xs: Iterable[Optional[bool]]) -> Optional[float]:
    """Mean over bools, skipping None (records where the metric is undefined)."""
    truthy = 0
    total = 0
    for x in xs:
        if x is None:
            continue
        total += 1
        if x:
            truthy += 1
    if total == 0:
        return None
    return truthy / total


def aggregate(
    records: Sequence[RecordMetrics],
    scalar_cfg: ScalarConfig = DEFAULT_SCALAR_CONFIG,
) -> Dict[str, Any]:
    """Compute group-level metrics from a list of RecordMetrics."""
    n = len(records)
    if n == 0:
        return {"n": 0}

    # ----- Process -----
    avg_turns = _mean([r.total_turns for r in records])
    # Per-record terms_per_turn is None when total_turns == 0; _mean already
    # skips those, so this is "avg terms per turn among records that asked".
    avg_terms_per_turn = _mean([r.terms_per_turn for r in records])
    avg_clarification_terms = _mean([r.n_queried_terms for r in records])
    avg_gold_ambiguous_terms = _mean([r.n_gold_terms for r in records])

    pct_perceived = _rate([r.perceived_ambiguous for r in records])
    pct_truly_ambiguous = _rate([r.truly_ambiguous for r in records])

    # Macro: average of per-record precision/recall (skipping undefined).
    macro_precision = _mean([r.term_precision for r in records])
    macro_recall = _mean([r.term_recall for r in records])
    macro_f1 = _mean([r.term_f1 for r in records])
    macro_redundancy = _mean([r.term_redundancy for r in records])

    # Micro: pool TP/FP/FN across the group.
    tp_sum = sum(r.term_tp for r in records)
    fp_sum = sum(r.term_fp for r in records)
    fn_sum = sum(r.term_fn for r in records)
    micro_precision = _safe_div(tp_sum, tp_sum + fp_sum)
    micro_recall = _safe_div(tp_sum, tp_sum + fn_sum)
    if micro_precision is not None and micro_recall is not None and (micro_precision + micro_recall) > 0:
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    else:
        micro_f1 = None
    micro_redundancy = _safe_div(fp_sum, tp_sum + fp_sum)

    avg_attempts = _mean([r.code_attempts for r in records])
    exec_success = _rate([r.code_success for r in records])

    # ----- Outcome -----
    boolean_records = [r for r in records if r.answer_type == "boolean"]
    scalar_records = [r for r in records if r.answer_type == "scalar"]

    # Strict accuracy: failed executions get scalar_correct=False / boolean
    # mismatch=False and are counted against the model in the denominator.
    boolean_accuracy = _rate([r.boolean_correct for r in boolean_records])

    scalar_accuracy: Dict[str, Optional[float]] = {}
    for key, _tol in scalar_cfg.tolerances:
        scalar_accuracy[key] = _rate([r.scalar_correct.get(key) for r in scalar_records])
    scalar_mean_rel_err = _mean([r.scalar_relative_error for r in scalar_records])
    scalar_median_rel_err = _median([r.scalar_relative_error for r in scalar_records])

    overall_accuracy = _rate([r.overall_correct for r in records])

    # Conditional accuracy: denominator restricted to records whose final
    # code ran (exit 0). Lets callers separate "model couldn't produce
    # working code" from "model wrote working but numerically wrong code".
    bool_ok = [r for r in boolean_records if r.code_success]
    scal_ok = [r for r in scalar_records if r.code_success]
    recs_ok = [r for r in records if r.code_success]

    boolean_accuracy_ok = _rate([r.boolean_correct for r in bool_ok])
    scalar_accuracy_ok: Dict[str, Optional[float]] = {}
    for key, _tol in scalar_cfg.tolerances:
        scalar_accuracy_ok[key] = _rate([r.scalar_correct.get(key) for r in scal_ok])
    overall_accuracy_ok = _rate([r.overall_correct for r in recs_ok])

    return {
        "n": n,
        "n_boolean": len(boolean_records),
        "n_scalar": len(scalar_records),
        "n_exec_success": sum(1 for r in records if r.code_success),
        "process": {
            "avg_interaction_turns": avg_turns,
            "avg_terms_per_turn": avg_terms_per_turn,
            "avg_clarification_terms": avg_clarification_terms,
            "avg_gold_ambiguous_terms": avg_gold_ambiguous_terms,
            "pct_perceived_ambiguous": pct_perceived,
            "pct_truly_ambiguous": pct_truly_ambiguous,
            "ambiguous_terms": {
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
                "macro_redundancy_rate": macro_redundancy,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_f1": micro_f1,
                "micro_redundancy_rate": micro_redundancy,
                "total_tp": tp_sum,
                "total_fp": fp_sum,
                "total_fn": fn_sum,
            },
            "avg_code_attempts": avg_attempts,
            "code_exec_success_rate": exec_success,
        },
        "outcome": {
            "boolean_accuracy": boolean_accuracy,
            "scalar_accuracy": scalar_accuracy,
            "scalar_mean_relative_error": scalar_mean_rel_err,
            "scalar_median_relative_error": scalar_median_rel_err,
            "primary_scalar_tolerance": scalar_cfg.primary,
            "overall_accuracy": overall_accuracy,
            "given_exec_success": {
                "n_boolean": len(bool_ok),
                "n_scalar": len(scal_ok),
                "n_records": len(recs_ok),
                "boolean_accuracy": boolean_accuracy_ok,
                "scalar_accuracy": scalar_accuracy_ok,
                "overall_accuracy": overall_accuracy_ok,
            },
        },
    }


# =========================================================
# Group splitters
# =========================================================

def split_by_level(records: Sequence[RecordMetrics]) -> Dict[str, List[RecordMetrics]]:
    out: Dict[str, List[RecordMetrics]] = {}
    for r in records:
        key = f"level_{r.level}" if r.level is not None else "level_unknown"
        out.setdefault(key, []).append(r)
    return out


def split_by_category(
    records: Sequence[RecordMetrics],
    categories: Sequence[str],
) -> Dict[str, List[RecordMetrics]]:
    """For each category, return records whose gold spans that category.

    A record can appear in multiple buckets — that's intentional: it carries
    ambiguity of multiple types, and each type's accuracy should reflect it.
    """
    out: Dict[str, List[RecordMetrics]] = {c: [] for c in categories}
    for r in records:
        for c in r.gold_categories:
            if c in out:
                out[c].append(r)
    return out


def split_by_task_type(
    records: Sequence[RecordMetrics],
) -> Dict[str, List[RecordMetrics]]:
    """Partition records by analysis type (one record → exactly one bucket).

    Mirrors :func:`split_by_level`: task type is a hard partition, so a
    record lands in a single bucket. Records with an unmapped task_id fall
    into ``"unknown"`` rather than being dropped, so nothing goes missing.
    """
    out: Dict[str, List[RecordMetrics]] = {}
    for r in records:
        key = r.task_type if r.task_type is not None else "unknown"
        out.setdefault(key, []).append(r)
    return out


def ambiguity_complexity_key(n_gold_terms: int) -> str:
    """Bucket label for an ambiguity-complexity level (# gold ambiguous terms)."""
    return f"complexity_{n_gold_terms}"


def split_by_ambiguity_complexity(
    records: Sequence[RecordMetrics],
) -> Dict[str, List[RecordMetrics]]:
    """Partition records by ambiguity complexity = number of gold term_ids.

    A hard partition like :func:`split_by_level` — each record has exactly
    one gold-term count, so it lands in a single ``complexity_<n>`` bucket
    (``complexity_0`` = no gold ambiguous terms).
    """
    out: Dict[str, List[RecordMetrics]] = {}
    for r in records:
        key = ambiguity_complexity_key(r.n_gold_terms)
        out.setdefault(key, []).append(r)
    return out


# =========================================================
# Top-level: compute everything for one run's summary dir
# =========================================================

def load_summary_records(summary_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    files = sorted(summary_dir.glob("*.json"))
    for fp in files:
        try:
            data = json.load(open(fp, "r", encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping unreadable summary %s (%s)", fp, exc)
            continue
        if isinstance(data, list):
            records.extend(data)
        elif isinstance(data, dict):
            records.append(data)
    return records


def compute_run_metrics(
    summary_dir: Path,
    lexicon_path: str,
    scalar_cfg: ScalarConfig = DEFAULT_SCALAR_CONFIG,
) -> Dict[str, Any]:
    """Load all summary files from one run directory and aggregate metrics.

    Returns a single dict with overall + per-level + per-category +
    level×category breakdowns, plus the per-record metric rows (useful for
    debugging individual examples).
    """
    lex = SimpleLexicon(lexicon_path)
    raw_records = load_summary_records(summary_dir)
    rec_metrics = [compute_record_metrics(r, lex, scalar_cfg) for r in raw_records]
    categories = lex.all_categories()

    by_level = split_by_level(rec_metrics)
    by_cat = split_by_category(rec_metrics, categories)
    by_task = split_by_task_type(rec_metrics)
    by_complexity = split_by_ambiguity_complexity(rec_metrics)

    level_x_cat: Dict[str, Dict[str, Any]] = {}
    for lvl_key, lvl_recs in by_level.items():
        level_x_cat[lvl_key] = {}
        cat_split = split_by_category(lvl_recs, categories)
        for c in categories:
            level_x_cat[lvl_key][c] = aggregate(cat_split[c], scalar_cfg)

    # Level × task-type: task type is a partition, so only report the
    # buckets that actually occur at each level (present in that level's data).
    level_x_task: Dict[str, Dict[str, Any]] = {}
    for lvl_key, lvl_recs in by_level.items():
        task_split = split_by_task_type(lvl_recs)
        level_x_task[lvl_key] = {
            t: aggregate(task_split[t], scalar_cfg)
            for t in _ordered_task_keys(task_split)
        }

    # Level × ambiguity-complexity: same partition logic, keyed by the number
    # of gold ambiguous terms; only complexities present at a level appear.
    level_x_complexity: Dict[str, Dict[str, Any]] = {}
    for lvl_key, lvl_recs in by_level.items():
        cx_split = split_by_ambiguity_complexity(lvl_recs)
        level_x_complexity[lvl_key] = {
            k: aggregate(cx_split[k], scalar_cfg)
            for k in _ordered_complexity_keys(cx_split)
        }

    return {
        "summary_dir": str(summary_dir),
        "lexicon_path": lexicon_path,
        "n_records": len(rec_metrics),
        "categories": categories,
        "task_types": _ordered_task_keys(by_task),
        "ambiguity_complexities": _ordered_complexity_keys(by_complexity),
        "scalar_tolerances": [
            {"name": k, "rel_tol": v} for k, v in scalar_cfg.tolerances
        ],
        "primary_scalar_tolerance": scalar_cfg.primary,
        "overall": aggregate(rec_metrics, scalar_cfg),
        "by_level": {k: aggregate(v, scalar_cfg) for k, v in by_level.items()},
        "by_ambiguity_type": {c: aggregate(by_cat[c], scalar_cfg) for c in categories},
        "by_level_and_ambiguity_type": level_x_cat,
        "by_task_type": {
            t: aggregate(by_task[t], scalar_cfg) for t in _ordered_task_keys(by_task)
        },
        "by_level_and_task_type": level_x_task,
        "by_ambiguity_complexity": {
            k: aggregate(by_complexity[k], scalar_cfg)
            for k in _ordered_complexity_keys(by_complexity)
        },
        "by_level_and_ambiguity_complexity": level_x_complexity,
        "per_record": [_record_to_dict(r) for r in rec_metrics],
    }


def _ordered_task_keys(buckets: Dict[str, List[RecordMetrics]]) -> List[str]:
    """Present task-type keys in canonical order, with any extras last.

    Keeps output stable and readable: known analysis types first (in the
    design-table order), then any unexpected key (e.g. ``"unknown"``).
    """
    known = [t for t in TASK_TYPES if t in buckets]
    extra = [k for k in buckets if k not in TASK_TYPES]
    return known + extra


def _ordered_complexity_keys(buckets: Dict[str, List[Any]]) -> List[str]:
    """Present ``complexity_<n>`` keys sorted by the integer count ascending."""
    def _n(key: str) -> int:
        try:
            return int(key.rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            return 1 << 30
    return sorted(buckets, key=_n)


def _record_to_dict(r: RecordMetrics) -> Dict[str, Any]:
    return {
        "record_id": r.record_id,
        "level": r.level,
        "answer_type": r.answer_type,
        "task_id": r.task_id,
        "task_type": r.task_type,
        "gold_term_ids": r.gold_term_ids,
        "queried_term_ids": r.queried_term_ids,
        "gold_categories": r.gold_categories,
        "total_turns": r.total_turns,
        "terms_per_turn": r.terms_per_turn,
        "n_queried_terms": r.n_queried_terms,
        "n_gold_terms": r.n_gold_terms,
        "perceived_ambiguous": r.perceived_ambiguous,
        "truly_ambiguous": r.truly_ambiguous,
        "code_success": r.code_success,
        "code_attempts": r.code_attempts,
        "term_tp": r.term_tp,
        "term_fp": r.term_fp,
        "term_fn": r.term_fn,
        "term_precision": r.term_precision,
        "term_recall": r.term_recall,
        "term_f1": r.term_f1,
        "term_redundancy": r.term_redundancy,
        "boolean_correct": r.boolean_correct,
        "scalar_relative_error": r.scalar_relative_error,
        "scalar_correct": r.scalar_correct,
        "overall_correct": r.overall_correct,
    }


# =========================================================
# Writers — small, deliberately split so the entry point can
# decide which slices to persist.
# =========================================================

def _atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def write_metrics(
    metrics: Dict[str, Any],
    out_dir: Path,
    breakdowns: Optional[Dict[str, bool]] = None,
) -> Dict[str, Path]:
    """Persist the metric breakdowns into ``out_dir``. Returns paths written.

    ``breakdowns`` is a name->bool map (overall / by_level /
    by_ambiguity_type / by_level_and_ambiguity_type / by_task_type /
    by_level_and_task_type / by_ambiguity_complexity /
    by_level_and_ambiguity_complexity / per_record). Missing keys default to
    True so callers can selectively turn slices off.
    """
    breakdowns = breakdowns or {}
    def _on(name: str) -> bool:
        return breakdowns.get(name, True)

    written: Dict[str, Path] = {}

    if _on("overall"):
        overall = {k: metrics[k] for k in (
            "summary_dir", "lexicon_path", "n_records",
            "categories", "task_types", "ambiguity_complexities",
            "scalar_tolerances", "primary_scalar_tolerance", "overall",
        )}
        written["overall"] = out_dir / "overall.json"
        _atomic_write_json(written["overall"], overall)

    if _on("by_level"):
        written["by_level"] = out_dir / "by_level.json"
        _atomic_write_json(written["by_level"], metrics["by_level"])

    if _on("by_ambiguity_type"):
        written["by_ambiguity_type"] = out_dir / "by_ambiguity_type.json"
        _atomic_write_json(written["by_ambiguity_type"], metrics["by_ambiguity_type"])

    if _on("by_level_and_ambiguity_type"):
        written["by_level_and_ambiguity_type"] = out_dir / "by_level_and_ambiguity_type.json"
        _atomic_write_json(
            written["by_level_and_ambiguity_type"],
            metrics["by_level_and_ambiguity_type"],
        )

    if _on("by_task_type"):
        written["by_task_type"] = out_dir / "by_task_type.json"
        _atomic_write_json(written["by_task_type"], metrics["by_task_type"])

    if _on("by_level_and_task_type"):
        written["by_level_and_task_type"] = out_dir / "by_level_and_task_type.json"
        _atomic_write_json(
            written["by_level_and_task_type"],
            metrics["by_level_and_task_type"],
        )

    if _on("by_ambiguity_complexity"):
        written["by_ambiguity_complexity"] = out_dir / "by_ambiguity_complexity.json"
        _atomic_write_json(
            written["by_ambiguity_complexity"],
            metrics["by_ambiguity_complexity"],
        )

    if _on("by_level_and_ambiguity_complexity"):
        written["by_level_and_ambiguity_complexity"] = (
            out_dir / "by_level_and_ambiguity_complexity.json"
        )
        _atomic_write_json(
            written["by_level_and_ambiguity_complexity"],
            metrics["by_level_and_ambiguity_complexity"],
        )

    if _on("per_record"):
        written["per_record"] = out_dir / "per_record.json"
        _atomic_write_json(written["per_record"], metrics["per_record"])

    return written