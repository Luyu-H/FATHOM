import sys
from pathlib import Path
from omegaconf import OmegaConf

# ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_gen.dataset_generator import DatasetGenerator
from src.data_gen.post_processing import (
    process_ambiguous_terms_in_directory,
    rephrase_questions_in_directory,
)


def _resolve_path(p) -> Path:
    p = Path(str(p))
    return p if p.is_absolute() else ROOT / p


def _stage_enabled(cfg, key: str) -> bool:
    pipeline = cfg.get("pipeline") if hasattr(cfg, "get") else getattr(cfg, "pipeline", None)
    if pipeline is None:
        return True
    val = pipeline.get(key, True) if hasattr(pipeline, "get") else getattr(pipeline, key, True)
    return bool(val)


def run_generate(cfg) -> None:
    print("=" * 60)
    print("[STAGE] generate")
    print("=" * 60)
    generator = DatasetGenerator(cfg)
    generator.generate()


def run_ambiguous_terms(cfg) -> None:
    print("=" * 60)
    print("[STAGE] ambiguous_terms")
    print("=" * 60)
    pp = cfg.post_processing
    process_ambiguous_terms_in_directory(
        qa_dir=_resolve_path(pp.qa_dir),
        ops_path=_resolve_path(pp.ops_path),
        out_dir=_resolve_path(pp.output_dir),
        levels=list(pp.levels),
    )


def run_rephrase(cfg) -> None:
    print("=" * 60)
    print("[STAGE] rephrase")
    print("=" * 60)
    pp = cfg.post_processing
    # Rephrase reads from (and writes back to) the post-processed output dir
    # so it always operates on items that already have ambiguous_term_ids.
    target_dir = _resolve_path(pp.output_dir)
    rephrase_questions_in_directory(
        qa_dir=target_dir,
        ops_path=_resolve_path(pp.ops_path),
        llm_settings=cfg.llm_settings,
        out_dir=target_dir,
        levels=list(pp.levels),
        keep_original_question=bool(pp.rephrase.keep_original_question),
    )


def main():
    cfg_path = ROOT / "configs" / "dataset_gen.yaml"
    if not cfg_path.exists():
        print(f"[ERROR] Config not found: {cfg_path}")
        sys.exit(1)

    cfg = OmegaConf.load(cfg_path)

    # allow CLI overrides: e.g. python generate.py pipeline.run_generate=false
    cli_cfg = OmegaConf.from_cli(sys.argv[1:])
    cfg = OmegaConf.merge(cfg, cli_cfg)

    print("=" * 60)
    print("Dataset Generation Config")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    if _stage_enabled(cfg, "run_generate"):
        run_generate(cfg)
    else:
        print("[SKIP] stage: generate")

    if _stage_enabled(cfg, "run_ambiguous_terms"):
        run_ambiguous_terms(cfg)
    else:
        print("[SKIP] stage: ambiguous_terms")

    if _stage_enabled(cfg, "run_rephrase"):
        run_rephrase(cfg)
    else:
        print("[SKIP] stage: rephrase")

    print("[DONE] Pipeline complete.")


if __name__ == "__main__":
    main()