"""Entry point + eval driver.

Iterates QA files & questions, runs the agent, and persists results.

Output layout (under ``eval.output_dir/run_name``)::

    full_records/<qa_file>.json   # complete interaction trace, one list per QA file
    summary/<qa_file>.json        # flat per-record summary for downstream eval

Both subfolders mirror the QA-data filename (``{task_id}_{start_no}_{end_no}.json``)
so each output file lines up 1:1 with its input QA file. Each output file holds a
JSON list of records — one element per question. The summary list is designed for
two evaluation axes:

  * **process eval** — did the agent perceive ambiguity, how many turns it ran,
    which terms it asked the oracle about (raw term + matched ``term_id``).
  * **outcome eval** — did the generated code execute, what the final answer was,
    and (on failure) the specific error.

Usage examples:
    # default run (config defaults from configs/agent.yaml)
    python -m src.main

    # switch strategies via Hydra CLI overrides
    python -m src.main uncertainty.strategy=divergence clarification.strategy=planning

    # switch model
    python -m src.main llm.provider=openai llm.model=gpt-4o llm.api_key_env=OPENAI_API_KEY

    # quick smoke test (only run first 5 questions)
    python -m src.main eval.limit=5
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

from src.agent.agent import AgentTrace, AmbiguityAwareAgent
from src.eval.oracle import Oracle

logger = logging.getLogger(__name__)


def build_oracle(cfg: DictConfig) -> Oracle:
    """Construct the simulated-user oracle from the agent config.

    The oracle reads its LLM-client settings (provider, model, api key
    env, max_tokens, temperature, retries, request_interval) from
    ``cfg.oracle.*`` directly — no field renaming is needed.
    """
    return Oracle(cfg.oracle.lexicon_path, cfg)


def _build_full_record(trace: AgentTrace, item: Dict[str, Any]) -> Dict[str, Any]:
    """Complete interaction record — everything the agent observed/produced."""
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


def _summarize_trace(trace: AgentTrace, item: Dict[str, Any]) -> Dict[str, Any]:
    """Flat summary used for downstream ambiguity / outcome evaluation.

    Fields:
      - record_id              : matches the QA item's `id`
      - original_question      : raw user question
      - agent_perceived_ambiguous : did the agent treat this as ambiguous?
      - total_turns            : number of clarification rounds with the oracle
      - queried_terms          : per-turn extracted terms + matched term_ids
      - unique_term_ids        : deduped term_ids the agent actually resolved
      - code_execution         : {success, error, exit_code, timed_out}
      - final_answer           : parsed FINAL_ANSWER from the executed script
    """
    # ---- Ambiguity perception ----
    # Spontaneous mode: any oracle-bound question implies the agent perceived
    # the question as ambiguous. Structured mode: the uncertainty detector
    # explicitly assigned a level.
    asked_any = any(turn.get("questions") for turn in trace.clarification_turns)
    unc = trace.uncertainty or {}
    uncertainty_level = unc.get("level")
    agent_perceived_ambiguous = bool(asked_any) or (uncertainty_level == "AMBIGUOUS")

    # ---- Queried terms across all clarification turns ----
    # The simulated-user oracle replies with a flat ``matched_term_ids`` list
    # per response (parsed from the LLM's strict-JSON output). One queried
    # entry is recorded per (turn, term_id) pair so that the per-turn ask
    # volume used by metrics.py keeps reflecting how often the agent
    # actually surfaced a given concept.
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

    # ---- Code execution outcome ----
    # `execution` mirrors the FINAL attempt; `code_attempts` carries the
    # full series so we can report how many repair tries it took.
    exec_info = trace.execution or {}
    code_success = bool(exec_info.get("success"))
    code_error: Optional[str] = None
    if not code_success:
        # Prefer the actual runtime stderr; otherwise fall back to the
        # top-level trace.error string (e.g. "no_code_generated",
        # "code_generation_failed: ...", "agent_crashed: ...").
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
        "agent_perceived_ambiguous": agent_perceived_ambiguous,
        "uncertainty_level": uncertainty_level,
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
        # Gold references kept alongside for convenient downstream alignment.
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
    """Atomic-ish write: dump to a temp file then rename."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)


def _load_prior(path: Path, key_field: str) -> Dict[str, Dict[str, Any]]:
    """Load an existing output file (if any) keyed by `key_field`."""
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
# Main loop
# =============================================================

def run_eval(cfg: DictConfig) -> None:
    out_dir = Path(cfg.eval.output_dir) / cfg.eval.run_name
    full_dir = out_dir / "full_records"
    summary_dir = out_dir / "summary"
    full_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Writing full interaction records to %s", full_dir)
    logger.info("Writing summaries to %s", summary_dir)

    oracle = build_oracle(cfg)
    agent = AmbiguityAwareAgent(cfg, oracle)

    qa_files = _load_qa_files(cfg.data.qa_data_dir, cfg.data.question_pattern)
    limit = cfg.eval.get("limit", None)
    completed = 0

    for qa_file in qa_files:
        with open(qa_file, "r", encoding="utf-8") as fh:
            items = json.load(fh)
        if not isinstance(items, list):
            items = [items]

        full_path = full_dir / qa_file.name
        summary_path = summary_dir / qa_file.name

        # Resume: re-load any prior records for this QA file so the eval can
        # pick up mid-file. We keep both files in sync by qid.
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
            except Exception as e:                       # noqa: BLE001
                logger.exception("Agent crashed on %s", qid)
                trace = AgentTrace(question=question, context=context,
                                   error=f"agent_crashed: {e}")

            full_records.append(_build_full_record(trace, item))
            summary_records.append(_summarize_trace(trace, item))

            # Incremental flush — survives crashes and lets us resume safely.
            _write_json_list(full_path, full_records)
            _write_json_list(summary_path, summary_records)

            completed += 1
            if limit is not None and completed >= limit:
                logger.info("Hit limit=%d. Stopping.", limit)
                return

        logger.info("[%s] wrote %d record(s).", qa_file.stem, len(full_records))

    logger.info("Done. Completed %d question(s).", completed)


# =============================================================
# Hydra entry point
# =============================================================

@hydra.main(version_base=None, config_path="../configs", config_name="agent")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, str(cfg.eval.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    print(OmegaConf.to_yaml(cfg))
    run_eval(cfg)


if __name__ == "__main__":
    main()