"""Metric computation for modular evaluation runs (``src/run_modular.py``).

Two evaluation modes, two distinct headlines — but both share the same
output skeleton as ``src.eval.metrics`` so the same breakdown layout
(overall / by_level / per_record) applies.

Mode A — ``ambiguity_only``
    The detector's verdict IS the final output, so we score it as a
    binary classification problem (perceived vs. truly ambiguous) and
    keep the term-identification numbers (predicted ambiguous terms
    resolved through the lexicon, vs. gold term_ids). No code execution
    or oracle-interaction signals are produced in this mode — those
    sections are simply absent from the metric output.

Mode B — ``clarification_only``
    The agent is told upstream that the question is ambiguous, so
    ``truly_ambiguous`` is trivially True for every record. The full
    process+outcome metric pipeline from :mod:`src.eval.metrics`
    applies as-is — including the ``outcome.given_exec_success`` block
    that reports accuracy with denominator restricted to records whose
    final code ran. On top we add two coverage signals that are only
    meaningful when gold ambiguity is known upstream:

    * ``pct_full_coverage`` — fraction of records where every gold
      term_id was asked at least once.
    * ``avg_turns_to_full_coverage`` / ``median_turns_to_full_coverage``
      — for records that reached full coverage, the turn index at
      which the last gold term was first surfaced. Records that never
      reached full coverage are excluded from the average and counted
      via ``pct_full_coverage``.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from .metrics import (
    DEFAULT_SCALAR_CONFIG, ScalarConfig, SimpleLexicon,
    aggregate as aggregate_full,
    compute_record_metrics as compute_full_record_metrics,
    load_summary_records, _atomic_write_json,
    _mean, _median, _rate, _safe_div,
    resolve_gold_term_ids,
    RecordMetrics,
)

logger = logging.getLogger(__name__)


# =========================================================
# Mode A: ambiguity_only — per-record + aggregator
# =========================================================

@dataclass
class AmbiguityRecordMetrics:
    record_id: Optional[str]
    level: Optional[int]
    strategy: Optional[str]                 # 'direct_prompt'
    gold_categories: List[str] = field(default_factory=list)

    # Question-level classification signals.
    truly_ambiguous: bool = False
    perceived_ambiguous: bool = False
    correct_classification: Optional[bool] = None

    # Term-level identification — populated when the detector surfaces a
    # term list (i.e. direct_prompt).
    gold_term_ids: List[str] = field(default_factory=list)
    predicted_term_ids: List[str] = field(default_factory=list)
    term_tp: int = 0
    term_fp: int = 0
    term_fn: int = 0
    term_precision: Optional[float] = None
    term_recall: Optional[float] = None
    term_f1: Optional[float] = None
    term_redundancy: Optional[float] = None

    # Cost / errors.
    n_llm_calls: int = 0
    error: Optional[str] = None


def _resolve_predicted_term_ids(
    pred_terms: Sequence[str], lex: SimpleLexicon,
) -> List[str]:
    """Map the detector's free-text ambiguous-term strings to term_ids."""
    if not pred_terms:
        return []
    out: List[str] = []
    seen: Set[str] = set()
    for t in pred_terms:
        if not isinstance(t, str):
            continue
        tid = lex.resolve(t)
        if tid and tid not in seen:
            out.append(tid)
            seen.add(tid)
    return out


def _count_detector_calls(record: Dict[str, Any]) -> int:
    """Pull n_llm_calls out of the saved trace details.

    The ambiguity_only summary doesn't carry the raw LLM calls (those
    live in full_records/). When a counts hint isn't available we fall
    back to 1 (every detection issues at least one judge call).
    """
    n = record.get("n_llm_calls")
    if isinstance(n, int) and n >= 0:
        return n
    return 1


def compute_ambiguity_record_metrics(
    record: Dict[str, Any], lex: SimpleLexicon,
) -> AmbiguityRecordMetrics:
    gold_ids = resolve_gold_term_ids(record, lex)
    pred_terms = record.get("ambiguous_terms_pred") or []
    pred_ids = _resolve_predicted_term_ids(pred_terms, lex)
    gold_set, pred_set = set(gold_ids), set(pred_ids)

    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    redundancy = _safe_div(fp, tp + fp)

    # Term-level identification is only meaningful when the strategy
    # actually outputs a term list. Mark the columns as N/A otherwise
    # so the aggregate doesn't average a phantom precision/recall.
    strategy = record.get("strategy")
    has_terms = bool(pred_ids) or (strategy == "direct_prompt")
    if not has_terms:
        precision = recall = f1 = redundancy = None
        tp = fp = fn = 0

    gold_cats: List[str] = []
    seen_c: Set[str] = set()
    for tid in gold_ids:
        c = lex.category_of(tid)
        if c and c not in seen_c:
            gold_cats.append(c)
            seen_c.add(c)

    truly = bool(gold_ids)
    perceived = bool(record.get("agent_perceived_ambiguous"))
    correct = (truly == perceived) if record.get("error") is None else None

    return AmbiguityRecordMetrics(
        record_id=record.get("record_id"),
        level=record.get("level"),
        strategy=strategy,
        gold_categories=gold_cats,
        truly_ambiguous=truly,
        perceived_ambiguous=perceived,
        correct_classification=correct,
        gold_term_ids=gold_ids,
        predicted_term_ids=pred_ids,
        term_tp=tp, term_fp=fp, term_fn=fn,
        term_precision=precision,
        term_recall=recall,
        term_f1=f1,
        term_redundancy=redundancy,
        n_llm_calls=_count_detector_calls(record),
        error=record.get("error"),
    )


def _classification_confusion(
    records: Sequence[AmbiguityRecordMetrics],
) -> Dict[str, int]:
    tp = fp = tn = fn = 0
    for r in records:
        if r.correct_classification is None:
            continue                                   # detector errored
        if r.truly_ambiguous and r.perceived_ambiguous:
            tp += 1
        elif r.truly_ambiguous and not r.perceived_ambiguous:
            fn += 1
        elif (not r.truly_ambiguous) and r.perceived_ambiguous:
            fp += 1
        else:
            tn += 1
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def aggregate_ambiguity(
    records: Sequence[AmbiguityRecordMetrics],
) -> Dict[str, Any]:
    n = len(records)
    if n == 0:
        return {"n": 0}

    cm = _classification_confusion(records)
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
    scored = tp + fp + tn + fn

    accuracy = _safe_div(tp + tn, scored)
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)                    # == TPR
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = None
    tpr = recall
    fpr = _safe_div(fp, fp + tn)
    tnr = _safe_div(tn, tn + fp)
    fnr = _safe_div(fn, fn + tp)

    macro_precision = _mean([r.term_precision for r in records])
    macro_recall = _mean([r.term_recall for r in records])
    macro_f1 = _mean([r.term_f1 for r in records])
    macro_redundancy = _mean([r.term_redundancy for r in records])

    term_tp = sum(r.term_tp for r in records)
    term_fp = sum(r.term_fp for r in records)
    term_fn = sum(r.term_fn for r in records)
    micro_precision = _safe_div(term_tp, term_tp + term_fp)
    micro_recall = _safe_div(term_tp, term_tp + term_fn)
    if (micro_precision is not None and micro_recall is not None
            and (micro_precision + micro_recall) > 0):
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    else:
        micro_f1 = None
    micro_redundancy = _safe_div(term_fp, term_tp + term_fp)

    n_errors = sum(1 for r in records if r.error)
    error_rate = n_errors / n if n else None
    avg_llm_calls = _mean([r.n_llm_calls for r in records])

    return {
        "n": n,
        "n_truly_ambiguous": sum(1 for r in records if r.truly_ambiguous),
        "n_perceived_ambiguous": sum(1 for r in records if r.perceived_ambiguous),
        "n_errors": n_errors,
        "classification": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tpr": tpr, "fpr": fpr, "tnr": tnr, "fnr": fnr,
            "confusion_matrix": cm,
        },
        "term_identification": {
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "macro_redundancy_rate": macro_redundancy,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "micro_redundancy_rate": micro_redundancy,
            "total_tp": term_tp,
            "total_fp": term_fp,
            "total_fn": term_fn,
        },
        "cost": {
            "avg_llm_calls": avg_llm_calls,
            "error_rate": error_rate,
        },
    }


# =========================================================
# Mode B: clarification_only — coverage extras layered on the
# existing full-pipeline metrics.
# =========================================================

@dataclass
class CoverageExtras:
    full_coverage: Optional[bool]            # None when no gold terms exist
    turns_to_full_coverage: Optional[int]    # None unless full coverage reached


def _turns_to_full_coverage(
    record: Dict[str, Any], gold_ids: Sequence[str],
) -> Optional[int]:
    """Earliest turn index at which every gold term_id had been asked.

    Walks the per-turn ``queried_terms`` entries (the same flat list
    that powers ``unique_term_ids``), accumulates the set of gold ids
    asked so far, and returns the first turn whose accumulated set
    covers all gold ids. Returns ``None`` if full coverage is never
    reached (or the gold list is empty).
    """
    if not gold_ids:
        return None
    gold_set: Set[str] = set(gold_ids)
    seen: Set[str] = set()
    qs = record.get("queried_terms") or []
    by_turn: Dict[int, List[str]] = {}
    for q in qs:
        if not isinstance(q, dict):
            continue
        t = q.get("turn"); tid = q.get("term_id")
        if not isinstance(t, int) or not isinstance(tid, str):
            continue
        by_turn.setdefault(t, []).append(tid)
    for t in sorted(by_turn):
        for tid in by_turn[t]:
            if tid in gold_set:
                seen.add(tid)
        if gold_set.issubset(seen):
            return t
    return None


def _coverage_extras(
    record: Dict[str, Any], gold_ids: Sequence[str],
) -> CoverageExtras:
    if not gold_ids:
        return CoverageExtras(full_coverage=None, turns_to_full_coverage=None)
    turn = _turns_to_full_coverage(record, gold_ids)
    return CoverageExtras(
        full_coverage=(turn is not None),
        turns_to_full_coverage=turn,
    )


def _aggregate_coverage(
    extras: Sequence[CoverageExtras],
) -> Dict[str, Optional[float]]:
    full_flags = [e.full_coverage for e in extras]
    turns = [
        e.turns_to_full_coverage for e in extras
        if e.full_coverage and e.turns_to_full_coverage is not None
    ]
    return {
        "pct_full_coverage": _rate(full_flags),
        "avg_turns_to_full_coverage": _mean(turns),
        "median_turns_to_full_coverage": _median(turns),
        "n_full_coverage": sum(1 for f in full_flags if f),
        "n_with_gold_terms": sum(1 for f in full_flags if f is not None),
    }


# =========================================================
# Top-level run-metric drivers
# =========================================================

def compute_ambiguity_run_metrics(
    summary_dir: Path, lexicon_path: str,
) -> Dict[str, Any]:
    lex = SimpleLexicon(lexicon_path)
    raw = load_summary_records(summary_dir)
    recs = [compute_ambiguity_record_metrics(r, lex) for r in raw]

    by_level = _split_by_level_amb(recs)

    return {
        "mode": "ambiguity_only",
        "summary_dir": str(summary_dir),
        "lexicon_path": lexicon_path,
        "n_records": len(recs),
        "overall": aggregate_ambiguity(recs),
        "by_level": {k: aggregate_ambiguity(v) for k, v in by_level.items()},
        "per_record": [_amb_record_to_dict(r) for r in recs],
    }


def compute_clarification_run_metrics(
    summary_dir: Path,
    lexicon_path: str,
    scalar_cfg: ScalarConfig = DEFAULT_SCALAR_CONFIG,
) -> Dict[str, Any]:
    """Reuse the full-pipeline aggregator and bolt coverage extras on top."""
    lex = SimpleLexicon(lexicon_path)
    raw = load_summary_records(summary_dir)
    full_recs: List[RecordMetrics] = [
        compute_full_record_metrics(r, lex, scalar_cfg) for r in raw
    ]
    extras: List[CoverageExtras] = [
        _coverage_extras(r, fr.gold_term_ids) for r, fr in zip(raw, full_recs)
    ]

    by_level = _split_by_level_full(full_recs)
    # Index extras alongside full_recs so per-group coverage stays aligned.
    extras_by_id: Dict[Optional[str], CoverageExtras] = {
        fr.record_id: ex for fr, ex in zip(full_recs, extras)
    }

    def _agg(group: Sequence[RecordMetrics]) -> Dict[str, Any]:
        base = aggregate_full(group, scalar_cfg)
        cov = _aggregate_coverage([extras_by_id[g.record_id] for g in group])
        base["coverage"] = cov
        return base

    return {
        "mode": "clarification_only",
        "summary_dir": str(summary_dir),
        "lexicon_path": lexicon_path,
        "n_records": len(full_recs),
        "scalar_tolerances": [
            {"name": k, "rel_tol": v} for k, v in scalar_cfg.tolerances
        ],
        "primary_scalar_tolerance": scalar_cfg.primary,
        "overall": _agg(full_recs),
        "by_level": {k: _agg(v) for k, v in by_level.items()},
        "per_record": [
            {**_full_record_to_dict(fr),
             "full_coverage": ex.full_coverage,
             "turns_to_full_coverage": ex.turns_to_full_coverage}
            for fr, ex in zip(full_recs, extras)
        ],
    }


# =========================================================
# Splitters — small parallel pair, one per record-type.
# =========================================================

def _split_by_level_amb(
    records: Sequence[AmbiguityRecordMetrics],
) -> Dict[str, List[AmbiguityRecordMetrics]]:
    out: Dict[str, List[AmbiguityRecordMetrics]] = {}
    for r in records:
        key = f"level_{r.level}" if r.level is not None else "level_unknown"
        out.setdefault(key, []).append(r)
    return out


def _split_by_category_amb(
    records: Sequence[AmbiguityRecordMetrics], categories: Sequence[str],
) -> Dict[str, List[AmbiguityRecordMetrics]]:
    out: Dict[str, List[AmbiguityRecordMetrics]] = {c: [] for c in categories}
    for r in records:
        for c in r.gold_categories:
            if c in out:
                out[c].append(r)
    return out


def _split_by_level_full(
    records: Sequence[RecordMetrics],
) -> Dict[str, List[RecordMetrics]]:
    out: Dict[str, List[RecordMetrics]] = {}
    for r in records:
        key = f"level_{r.level}" if r.level is not None else "level_unknown"
        out.setdefault(key, []).append(r)
    return out


def _split_by_category_full(
    records: Sequence[RecordMetrics], categories: Sequence[str],
) -> Dict[str, List[RecordMetrics]]:
    out: Dict[str, List[RecordMetrics]] = {c: [] for c in categories}
    for r in records:
        for c in r.gold_categories:
            if c in out:
                out[c].append(r)
    return out


# =========================================================
# Serialization helpers
# =========================================================

def _amb_record_to_dict(r: AmbiguityRecordMetrics) -> Dict[str, Any]:
    return {
        "record_id": r.record_id,
        "level": r.level,
        "strategy": r.strategy,
        "gold_categories": r.gold_categories,
        "truly_ambiguous": r.truly_ambiguous,
        "perceived_ambiguous": r.perceived_ambiguous,
        "correct_classification": r.correct_classification,
        "gold_term_ids": r.gold_term_ids,
        "predicted_term_ids": r.predicted_term_ids,
        "term_tp": r.term_tp, "term_fp": r.term_fp, "term_fn": r.term_fn,
        "term_precision": r.term_precision,
        "term_recall": r.term_recall,
        "term_f1": r.term_f1,
        "term_redundancy": r.term_redundancy,
        "n_llm_calls": r.n_llm_calls,
        "error": r.error,
    }


def _full_record_to_dict(r: RecordMetrics) -> Dict[str, Any]:
    # Local copy of metrics._record_to_dict (kept private there).
    return {
        "record_id": r.record_id,
        "level": r.level,
        "answer_type": r.answer_type,
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
        "term_tp": r.term_tp, "term_fp": r.term_fp, "term_fn": r.term_fn,
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
# Writers
# =========================================================

def write_modular_metrics(
    metrics: Dict[str, Any],
    out_dir: Path,
    breakdowns: Optional[Dict[str, bool]] = None,
) -> Dict[str, Path]:
    """Persist metric slices into ``out_dir``. Mirrors :func:`metrics.write_metrics`.

    Keys in ``breakdowns`` default to True. Recognised keys: ``overall``,
    ``by_level``, ``per_record``.
    """
    breakdowns = breakdowns or {}
    def _on(name: str) -> bool:
        return breakdowns.get(name, True)

    written: Dict[str, Path] = {}

    if _on("overall"):
        keys = ["mode", "summary_dir", "lexicon_path", "n_records", "overall"]
        if "scalar_tolerances" in metrics:
            keys.extend(["scalar_tolerances", "primary_scalar_tolerance"])
        overall = {k: metrics[k] for k in keys if k in metrics}
        written["overall"] = out_dir / "overall.json"
        _atomic_write_json(written["overall"], overall)

    if _on("by_level"):
        written["by_level"] = out_dir / "by_level.json"
        _atomic_write_json(written["by_level"], metrics["by_level"])

    if _on("per_record"):
        written["per_record"] = out_dir / "per_record.json"
        _atomic_write_json(written["per_record"], metrics["per_record"])

    return written