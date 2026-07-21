"""Direct-run driver for modular agent evaluation.

Two evaluation modes — both isolate one stage of the full agent so it
can be scored on its own:

* ``ambiguity_only``      — run only the uncertainty detector
* ``clarification_only``  — assume each input question is ambiguous,
                            run clarification + codegen + execute

Both modes share the same I/O layout as the full pipeline so the same
downstream metrics tooling applies::

    <output_dir>/<run_name>/full_records/<qa_file>.json  # complete trace
    <output_dir>/<run_name>/summary/<qa_file>.json       # flat summary

Usage:
    python -m src.run_modular                            # uses configs/runner_modular.yaml
    python -m src.run_modular --config path/to.yaml
    python -m src.run_modular mode=clarification_only \
        clarification.strategy=candidate eval.limit=5

CLI overrides use dotted-key syntax (e.g. ``llm.model=gpt-4o-mini``)
and are merged on top of the YAML file via OmegaConf.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig, OmegaConf

from src.runner.runner_modular import (
    AmbiguityOnlyAgent, AmbiguityOnlyTrace,
    ClarificationOnlyAgent, ClarificationOnlyTrace,
)
from src.eval.oracle import Oracle

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "runner_modular.yaml"


# =============================================================
# Record builders
# =============================================================

def _full_record_ambiguity(
    trace: AmbiguityOnlyTrace, item: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "id": item.get("id"),
        "level": item.get("level"),
        "task_id": item.get("task_id"),
        "prompt_id": item.get("prompt_id"),
        "ambiguous_terms_gold": item.get("ambiguous_terms"),
        "ambiguous_term_ids_gold": item.get("ambiguous_term_ids"),
        "trace": asdict(trace),
    }


def _summary_ambiguity(
    trace: AmbiguityOnlyTrace, item: Dict[str, Any],
) -> Dict[str, Any]:
    unc = trace.uncertainty or {}
    level = unc.get("level")
    return {
        "record_id": item.get("id"),
        "level": item.get("level"),
        "task_id": item.get("task_id"),
        "prompt_id": item.get("prompt_id"),
        "original_question": trace.question,
        "uncertainty_level": level,
        "uncertainty_score": unc.get("score"),
        "agent_perceived_ambiguous": level == "ambiguous",
        "ambiguous_terms_pred": unc.get("ambiguous_terms") or [],
        "ambiguous_terms_gold": item.get("ambiguous_terms"),
        "ambiguous_term_ids_gold": item.get("ambiguous_term_ids"),
        "strategy": (unc.get("details") or {}).get("strategy"),
        "error": trace.error,
    }


def _full_record_clarification(
    trace: ClarificationOnlyTrace, item: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "id": item.get("id"),
        "level": item.get("level"),
        "task_id": item.get("task_id"),
        "prompt_id": item.get("prompt_id"),
        "expected_answer": item.get("answer"),
        "expected_answer_metadata": item.get("answer_metadata"),
        "ambiguous_terms_gold": item.get("ambiguous_terms"),
        "ambiguous_term_ids_gold": item.get("ambiguous_term_ids"),
        "trace": asdict(trace),
    }


def _summary_clarification(
    trace: ClarificationOnlyTrace, item: Dict[str, Any],
) -> Dict[str, Any]:
    queried_terms: List[Dict[str, Any]] = []
    seen: set = set()
    for turn_idx, turn in enumerate(trace.clarification_turns, start=1):
        for resp in turn.get("oracle_responses", []) or []:
            for term_id in (resp.get("matched_term_ids") or []):
                key = (turn_idx, term_id)
                if key in seen:
                    continue
                seen.add(key)
                queried_terms.append({
                    "turn": turn_idx,
                    "term_id": term_id,
                    "status": resp.get("status"),
                })
    unique_term_ids = sorted({
        q["term_id"] for q in queried_terms if q.get("term_id")
    })

    exec_info = trace.execution or {}
    code_success = bool(exec_info.get("success"))
    code_error: Optional[str] = None
    if not code_success:
        stderr_tail = (exec_info.get("stderr_tail") or "").strip()
        code_error = stderr_tail or trace.error
    attempts_used = (
        exec_info.get("attempts")
        or len(trace.code_attempts or [])
        or (1 if exec_info else 0)
    )

    return {
        "record_id": item.get("id"),
        "level": item.get("level"),
        "task_id": item.get("task_id"),
        "prompt_id": item.get("prompt_id"),
        "original_question": trace.question,
        "agent_perceived_ambiguous": True,         # always — mode assumes ambiguous
        "uncertainty_level": (trace.uncertainty or {}).get("level"),
        "total_turns": sum(
            1 for t in trace.clarification_turns if t.get("questions")
        ),
        "queried_terms": queried_terms,
        "unique_term_ids": unique_term_ids,
        "code_execution": {
            "success": code_success,
            "error": code_error,
            "exit_code": exec_info.get("exit_code"),
            "timed_out": exec_info.get("timed_out"),
            "attempts": attempts_used,
        },
        "final_answer": trace.final_answer,
        "ambiguous_terms_gold": item.get("ambiguous_terms"),
        "ambiguous_term_ids_gold": item.get("ambiguous_term_ids"),
        "expected_answer": item.get("answer"),
    }


# =============================================================
# I/O helpers
# =============================================================

def _load_qa_files(qa_dir: str, pattern: str = "*.json") -> List[Path]:
    files = sorted(Path(qa_dir).glob(pattern))
    if not files:
        logger.warning("No QA files found under %s/%s", qa_dir, pattern)
    return files


def _write_json_list(path: Path, records: List[Dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def _load_prior(path: Path, key_field: str) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        data = json.load(open(path, "r", encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not reload %s for resume (%s) — starting fresh.",
                       path, exc)
        return {}
    if not isinstance(data, list):
        return {}
    return {rec.get(key_field): rec for rec in data if rec.get(key_field)}


# =============================================================
# Config loading
# =============================================================

def _load_config(config_path: Path, cli_overrides: List[str]) -> DictConfig:
    base = OmegaConf.load(str(config_path))
    if cli_overrides:
        overrides = OmegaConf.from_dotlist(cli_overrides)
        base = OmegaConf.merge(base, overrides)
    return base                                            # type: ignore[return-value]


# =============================================================
# Mode runners
# =============================================================

def _run_ambiguity_only(cfg: DictConfig) -> None:
    out_dir = Path(cfg.eval.output_dir) / cfg.eval.run_name
    full_dir = out_dir / "full_records"
    summary_dir = out_dir / "summary"
    full_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[ambiguity_only] full_records -> %s", full_dir)
    logger.info("[ambiguity_only] summaries   -> %s", summary_dir)

    agent = AmbiguityOnlyAgent(cfg)
    qa_files = _load_qa_files(cfg.data.qa_data_dir, cfg.data.question_pattern)
    limit = cfg.eval.get("limit", None)
    completed = 0

    for qa_file in qa_files:
        items = json.load(open(qa_file, "r", encoding="utf-8"))
        if not isinstance(items, list):
            items = [items]

        full_path = full_dir / qa_file.name
        summary_path = summary_dir / qa_file.name
        prior_full = _load_prior(full_path, "id") if cfg.eval.resume else {}
        prior_summary = _load_prior(summary_path, "record_id") if cfg.eval.resume else {}

        full_records: List[Dict[str, Any]] = []
        summary_records: List[Dict[str, Any]] = []

        for item in items:
            qid = item.get("id") or f"unnamed_{completed}"
            if cfg.eval.resume and qid in prior_full and qid in prior_summary:
                logger.info("Skipping completed %s", qid)
                full_records.append(prior_full[qid])
                summary_records.append(prior_summary[qid])
                completed += 1
                if limit is not None and completed >= limit:
                    _write_json_list(full_path, full_records)
                    _write_json_list(summary_path, summary_records)
                    return
                continue

            question = item.get("question", "")
            context = (item.get("prompt") or "").strip()
            logger.info("=== [%s] %s ===", qa_file.stem, qid)
            try:
                trace = agent.answer(question=question, context=context)
            except Exception as exc:                       # noqa: BLE001
                logger.exception("Agent crashed on %s", qid)
                trace = AmbiguityOnlyTrace(
                    question=question, context=context,
                    error=f"agent_crashed: {exc}",
                )

            full_records.append(_full_record_ambiguity(trace, item))
            summary_records.append(_summary_ambiguity(trace, item))
            _write_json_list(full_path, full_records)
            _write_json_list(summary_path, summary_records)

            completed += 1
            if limit is not None and completed >= limit:
                logger.info("Hit limit=%d. Stopping.", limit)
                return

        logger.info("[%s] wrote %d record(s).", qa_file.stem, len(full_records))

    logger.info("Done. Completed %d question(s).", completed)


def _run_clarification_only(cfg: DictConfig) -> None:
    out_dir = Path(cfg.eval.output_dir) / cfg.eval.run_name
    full_dir = out_dir / "full_records"
    summary_dir = out_dir / "summary"
    full_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    logger.info("[clarification_only] full_records -> %s", full_dir)
    logger.info("[clarification_only] summaries   -> %s", summary_dir)

    oracle = Oracle(cfg.oracle.lexicon_path, cfg)
    agent = ClarificationOnlyAgent(cfg, oracle)
    qa_files = _load_qa_files(cfg.data.qa_data_dir, cfg.data.question_pattern)
    gold_field = cfg.data.get("ambiguity_gold_field", "ambiguous_terms")
    limit = cfg.eval.get("limit", None)
    completed = 0

    for qa_file in qa_files:
        items = json.load(open(qa_file, "r", encoding="utf-8"))
        if not isinstance(items, list):
            items = [items]

        full_path = full_dir / qa_file.name
        summary_path = summary_dir / qa_file.name
        prior_full = _load_prior(full_path, "id") if cfg.eval.resume else {}
        prior_summary = _load_prior(summary_path, "record_id") if cfg.eval.resume else {}

        full_records: List[Dict[str, Any]] = []
        summary_records: List[Dict[str, Any]] = []

        for item in items:
            qid = item.get("id") or f"unnamed_{completed}"

            # This mode is defined only for items the dataset has
            # already labelled as ambiguous. Items with an empty
            # gold-ambiguity field are skipped (and not written out).
            if not item.get(gold_field):
                logger.info("Skipping %s — empty gold '%s'", qid, gold_field)
                continue

            if cfg.eval.resume and qid in prior_full and qid in prior_summary:
                logger.info("Skipping completed %s", qid)
                full_records.append(prior_full[qid])
                summary_records.append(prior_summary[qid])
                completed += 1
                if limit is not None and completed >= limit:
                    _write_json_list(full_path, full_records)
                    _write_json_list(summary_path, summary_records)
                    return
                continue

            question = item.get("question", "")
            context = (item.get("prompt") or "").strip()
            logger.info("=== [%s] %s ===", qa_file.stem, qid)
            try:
                trace = agent.answer(question=question, context=context)
            except Exception as exc:                       # noqa: BLE001
                logger.exception("Agent crashed on %s", qid)
                trace = ClarificationOnlyTrace(
                    question=question, context=context,
                    error=f"agent_crashed: {exc}",
                )

            full_records.append(_full_record_clarification(trace, item))
            summary_records.append(_summary_clarification(trace, item))
            _write_json_list(full_path, full_records)
            _write_json_list(summary_path, summary_records)

            completed += 1
            if limit is not None and completed >= limit:
                logger.info("Hit limit=%d. Stopping.", limit)
                return

        logger.info("[%s] wrote %d record(s).", qa_file.stem, len(full_records))

    logger.info("Done. Completed %d question(s).", completed)


# =============================================================
# Entry point
# =============================================================

_MODE_DISPATCH = {
    "ambiguity_only": _run_ambiguity_only,
    "clarification_only": _run_clarification_only,
}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Modular evaluation runner. Pick a mode in the YAML config "
            "(or via 'mode=...' CLI override) — ambiguity_only or "
            "clarification_only."
        ),
        # Forward unknown args (e.g. dotted CLI overrides) to OmegaConf.
    )
    parser.add_argument(
        "--config", "-c", type=Path, default=DEFAULT_CONFIG,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG}).",
    )
    args, unknown = parser.parse_known_args(argv)

    cfg = _load_config(args.config, unknown)
    logging.basicConfig(
        level=getattr(logging, str(cfg.eval.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    print(OmegaConf.to_yaml(cfg))

    mode = str(cfg.get("mode", "")).lower()
    runner = _MODE_DISPATCH.get(mode)
    if runner is None:
        raise SystemExit(
            f"Unknown mode: {mode!r}. Supported: {sorted(_MODE_DISPATCH)}."
        )
    runner(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())