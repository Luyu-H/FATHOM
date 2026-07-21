"""Entry point for computing modular-evaluation metrics.

Pairs with ``src/run_modular.py`` — reads the summary files that runner
writes and produces mode-aware metric reports under
``<run_dir>/<metrics_subdir>/``.

Usage:
    python -m src.evaluate_modular                                  # uses configs/evaluate_modular.yaml
    python -m src.evaluate_modular --config path/to/evaluate.yaml
    python -m src.evaluate_modular eval.mode=clarification_only \
        eval.run_name=clarification_only_a-direct_prompt_c-direct_deepseek-v4-flash-rhigh

CLI overrides use dotted-key syntax (e.g. ``eval.run_name=...``) and
merge on top of the YAML file via OmegaConf.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from omegaconf import DictConfig, OmegaConf

from src.eval.metrics import ScalarConfig
from src.eval.metrics_modular import (
    compute_ambiguity_run_metrics,
    compute_clarification_run_metrics,
    write_modular_metrics,
)

logger = logging.getLogger(__name__)

DEFAULT_CONFIG = (
    Path(__file__).resolve().parent.parent / "configs" / "evaluate_modular.yaml"
)


def _load_config(config_path: Path, cli_overrides: List[str]) -> DictConfig:
    base = OmegaConf.load(str(config_path))
    if cli_overrides:
        base = OmegaConf.merge(base, OmegaConf.from_dotlist(cli_overrides))
    return base                                          # type: ignore[return-value]


def _discover_run_dirs(results_root: Path, run_name: Optional[str]) -> List[Path]:
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
    raw = OmegaConf.to_container(cfg.scalar.tolerances, resolve=True)
    tolerances: List[Tuple[str, float]] = []
    for item in raw or []:
        tolerances.append((str(item["name"]), float(item["rel_tol"])))
    if not tolerances:
        raise ValueError("scalar.tolerances must contain at least one entry.")
    primary = str(cfg.scalar.primary)
    if primary not in {n for n, _ in tolerances}:
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


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Modular-mode metrics over runs from src.run_modular.",
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
    logger.info("Effective config:\n%s", OmegaConf.to_yaml(cfg))

    results_root = Path(cfg.eval.results_root)
    run_name = cfg.eval.get("run_name", None)
    mode = str(cfg.eval.mode).lower()
    metrics_subdir = cfg.eval.get("metrics_subdir", "metrics")
    lexicon_path = cfg.lexicon.path
    breakdowns = OmegaConf.to_container(cfg.breakdowns, resolve=True) or {}

    runs = _discover_run_dirs(results_root, run_name)
    if not runs:
        logger.warning("No runs to evaluate under %s.", results_root)
        return 0

    if mode == "ambiguity_only":
        for run_dir in runs:
            summary_dir = run_dir / "summary"
            metrics_dir = run_dir / metrics_subdir
            logger.info("[ambiguity_only] %s", run_dir.name)
            metrics = compute_ambiguity_run_metrics(summary_dir, lexicon_path)
            written = write_modular_metrics(metrics, metrics_dir, breakdowns)
            logger.info("  n=%d -> %d file(s) in %s",
                        metrics["n_records"], len(written), metrics_dir)
    elif mode == "clarification_only":
        scalar_cfg = _build_scalar_config(cfg)
        for run_dir in runs:
            summary_dir = run_dir / "summary"
            metrics_dir = run_dir / metrics_subdir
            logger.info("[clarification_only] %s", run_dir.name)
            metrics = compute_clarification_run_metrics(
                summary_dir, lexicon_path, scalar_cfg,
            )
            written = write_modular_metrics(metrics, metrics_dir, breakdowns)
            logger.info("  n=%d -> %d file(s) in %s",
                        metrics["n_records"], len(written), metrics_dir)
    else:
        raise SystemExit(
            f"Unknown eval.mode: {mode!r}. "
            "Supported: 'ambiguity_only', 'clarification_only'."
        )

    logger.info("Done. Evaluated %d run(s).", len(runs))
    return 0


if __name__ == "__main__":
    sys.exit(main())
