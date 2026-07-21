"""Entry point for computing evaluation metrics from saved run summaries.

Configuration lives in ``configs/evaluate.yaml`` (separate from agent.yaml).

Usage examples::

    # Evaluate every run directory under eval.results_root
    python -m src.evaluate

    # Evaluate a single run by name (matches a subdirectory of results_root)
    python -m src.evaluate eval.run_name=gpt-5.4_direct_prompt_spontaneous

    # Override the results root or lexicon path on the fly
    python -m src.evaluate eval.results_root=/path/to/results \\
                           lexicon.path=/path/to/lexicon.jsonl

For every run directory found (must contain a ``summary/`` folder written by
``src.run``), this script computes process & outcome metrics with
overall, per-level, per-ambiguity-category, and level×category breakdowns,
and writes the JSON reports into ``<run_dir>/<eval.metrics_subdir>/``.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf

from src.eval.metrics import (
    ScalarConfig,
    compute_run_metrics,
    write_metrics,
)

logger = logging.getLogger(__name__)


def _discover_run_dirs(results_root: Path, run_name: Optional[str]) -> List[Path]:
    """Find run directories that have a non-empty ``summary/`` subfolder."""
    if not results_root.exists():
        logger.error("Results root does not exist: %s", results_root)
        return []

    candidates: List[Path]
    if run_name:
        candidates = [results_root / run_name]
    else:
        candidates = [p for p in sorted(results_root.iterdir()) if p.is_dir()]

    runs: List[Path] = []
    for p in candidates:
        summary_dir = p / "summary"
        if summary_dir.is_dir() and any(summary_dir.glob("*.json")):
            runs.append(p)
        else:
            logger.info("Skipping %s (no summary/*.json under it).", p)
    return runs


def _build_scalar_config(cfg: DictConfig) -> ScalarConfig:
    """Convert the YAML scalar block into a frozen ScalarConfig."""
    raw = OmegaConf.to_container(cfg.scalar.tolerances, resolve=True)
    tolerances: List[Tuple[str, float]] = []
    for item in raw or []:
        tolerances.append((str(item["name"]), float(item["rel_tol"])))
    if not tolerances:
        raise ValueError("scalar.tolerances must contain at least one entry.")
    primary = str(cfg.scalar.primary)
    if primary not in {name for name, _ in tolerances}:
        raise ValueError(
            f"scalar.primary='{primary}' not found among "
            f"scalar.tolerances={[n for n, _ in tolerances]}"
        )
    return ScalarConfig(
        tolerances=tuple(tolerances),
        primary=primary,
        zero_guard=float(cfg.scalar.zero_guard),
        abs_floor=float(cfg.scalar.abs_floor),
    )


@hydra.main(version_base=None, config_path="../configs", config_name="evaluate")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=getattr(logging, str(cfg.eval.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Effective evaluate config:\n%s", OmegaConf.to_yaml(cfg))

    results_root = Path(cfg.eval.results_root)
    run_name = cfg.eval.get("run_name", None)
    metrics_subdir = cfg.eval.get("metrics_subdir", "metrics")
    lexicon_path = cfg.lexicon.path
    scalar_cfg = _build_scalar_config(cfg)
    breakdowns = OmegaConf.to_container(cfg.breakdowns, resolve=True) or {}

    runs = _discover_run_dirs(results_root, run_name)
    if not runs:
        logger.warning("No runs to evaluate under %s.", results_root)
        return

    for run_dir in runs:
        summary_dir = run_dir / "summary"
        metrics_dir = run_dir / metrics_subdir
        logger.info("Computing metrics for run %s", run_dir.name)
        metrics = compute_run_metrics(summary_dir, lexicon_path, scalar_cfg)
        written = write_metrics(metrics, metrics_dir, breakdowns)
        logger.info(
            "[%s] n=%d -> %d file(s) in %s",
            run_dir.name, metrics["n_records"], len(written), metrics_dir,
        )

    logger.info("Done. Evaluated %d run(s).", len(runs))


if __name__ == "__main__":
    main()