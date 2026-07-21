from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_gen.level1 import generate_level1_tasks
from src.data_gen.level2 import generate_level2_tasks
from src.data_gen.level3 import generate_level3_tasks


_EXPERIMENT_INFO = {
    "historical": "CMIP6 historical simulation (1850–2014)",
    "ssp245":     "CMIP6 SSP2-4.5 scenario (2015–2100)",
}

_SOURCE_INFO = {
    "CESM2": (
        "[CESM2]\n"
        "  Grid:           gn (native gx1v7 displaced pole, ~100 km, 384 nlat x 320 nlon)\n"
        "  Variant label:  r1i1p1f1 (historical)  |  r10i1p1f1 (ssp245)\n"
        "  File templates:\n"
        "    Time-varying (Omon) files for most variables:  <variable_id>_Omon_CESM2_<experiment_id>_<variant_label>_gn_<YYYYMM>-<YYYYMM>.nc\n"
        "    Fixed (Ofx) files for areacello:          <variable_id>_Ofx_CESM2_<experiment_id>_<variant_label>_gn.nc\n"
        "  Dimensions of standard 3D variable: (time, lev, nlat, nlon), with lev=60\n"
        "  Coordinate variables:\n"
        "    time      (time,)            float64  -- days since 0001-01-01 00:00:00\n"
        "    time_bnds (time, d2)         float64\n"
        "    lev       (lev,)             float64  -- depth axis, units: centimeters\n"
        "    lev_bnds  (lev, d2)          float32  -- units: m\n"
        "    lat       (nlat, nlon)       float64  -- 2-D auxiliary latitude, degrees_north\n"
        "    lon       (nlat, nlon)       float64  -- 2-D auxiliary longitude, degrees_east, 0-360\n"
        "    lat_bnds  (nlat, nlon, vertices)  float32\n"
        "    lon_bnds  (nlat, nlon, vertices)  float32\n"
        "  Special cases:\n"
        "    - intdic, intdoc, intpoc, phycos: no lev dim -> shape (time, nlat, nlon)\n"
        "    - chl: lev dimension restricted to 15 depth levels (not 60)\n"
        "    - areacello: only (nlat, nlon)"
    ),
    "GFDL-ESM4": (
        "[GFDL-ESM4]\n"
        "  Grid:           gr (regular 1x1 degree, 180 lat x 360 lon)\n"
        "  Variant label:  r1i1p1f1\n"
        "  File templates:\n"
        "    Time-varying (Omon) files for most variables:  <variable_id>_Omon_GFDL-ESM4_<experiment_id>_r1i1p1f1_gr_<YYYYMM>-<YYYYMM>.nc\n"
        "    Fixed (Ofx) files for areacello:          <variable_id>_Ofx_GFDL-ESM4_<experiment_id>_r1i1p1f1_gr.nc\n"
        "  Dimensions of standard 3D variable: (time, lev, lat, lon), with lev=35\n"
        "  Coordinate variables:\n"
        "    time      (time,)        float64  -- days since 1850-01-01 00:00:00\n"
        "    time_bnds (time, bnds)   float64\n"
        "    lev       (lev,)         float64  -- depth axis, units: m\n"
        "    lev_bnds  (lev, bnds)    float64  -- units: m\n"
        "    lat       (lat,)         float64  -- 1-D latitude, degrees_north\n"
        "    lon       (lon,)         float64  -- 1-D longitude, degrees_east\n"
        "    lat_bnds  (lat, bnds)    float64\n"
        "    lon_bnds  (lon, bnds)    float64\n"
        "  Special cases:\n"
        "    - intdic, intdoc, intpoc, phycos: no lev dim -> shape (time, lat, lon)\n"
        "    - areacello: only (lat, lon)"
    ),
    "MPI-ESM1-2-LR": (
        "[MPI-ESM1-2-LR]\n"
        "  Grid:           gn (native, ~250 km, 220 j x 256 i; lat/lon are 2-D auxiliaries)\n"
        "  Variant label:  r1i1p1f1\n"
        "  File templates:\n"
        "    Time-varying (Omon) files for most variables:  <variable_id>_Omon_MPI-ESM1-2-LR_<experiment_id>_r1i1p1f1_gn_<YYYYMM>-<YYYYMM>.nc\n"
        "    Fixed (Ofx) files for areacello:          <variable_id>_Ofx_MPI-ESM1-2-LR_<experiment_id>_r1i1p1f1_gn.nc\n"
        "  Dimensions of standard 3D variable: (time, lev, j, i), with lev=40\n"
        "  Coordinate variables:\n"
        "    time               (time,)            float64  -- days since 1850-1-1 00:00:00\n"
        "    time_bnds          (time, bnds)       float64\n"
        "    lev                (lev,)             float64  -- depth axis, units: m\n"
        "    lev_bnds           (lev, bnds)        float64\n"
        "    j                  (j,)               int32    -- cell index along 2nd dimension\n"
        "    i                  (i,)               int32    -- cell index along 1st dimension\n"
        "    latitude           (j, i)             float64  -- 2-D auxiliary latitude\n"
        "    longitude          (j, i)             float64  -- 2-D auxiliary longitude\n"
        "    vertices_latitude  (j, i, vertices)   float64\n"
        "    vertices_longitude (j, i, vertices)   float64\n"
        "  Special cases:\n"
        "    - intdic, intdoc, intpoc, phycos: no lev dim -> shape (time, j, i)\n"
        "    - areacello: only (j, i)"
    ),
}


class DatasetGenerator:
    def __init__(self, configs):
        self.configs = configs


    # ===================================================================
    # Helpers
    # ===================================================================

    def _get_task_id_list(self, level_cfg):
        if level_cfg.num == 0:
            return []
        if level_cfg.task_id_list is not None:
            return level_cfg.task_id_list
        df = pd.read_csv(level_cfg.template_path)
        return df["task_id"].tolist()

    @staticmethod
    def _build_configuration_text(data_meta: dict) -> str:
        """Build a human-readable configuration block from data_metadata."""
        exp = data_meta["experiment_id"]
        if isinstance(exp, (list, tuple)):
            exp_line = ", ".join(f"{e} ({_EXPERIMENT_INFO.get(e, e)})" for e in exp)
        else:
            exp_line = f"{exp} ({_EXPERIMENT_INFO.get(exp, exp)})"

        src = data_meta["source_id"]
        src_line = ", ".join(src) if isinstance(src, (list, tuple)) else str(src)

        lines = [
            f"- Experiment: {exp_line}",
            f"- Climate model (source): {src_line}",
        ]
        start = data_meta.get("start_time", "")
        end = data_meta.get("end_time", "")
        if start and end:
            lines.append(f"- Time period required by the question: {start} to {end}")
        return "\n".join(lines)

    @staticmethod
    def _build_source_info_text(source_id) -> str:
        """Build data file structure description for the involved source model(s)."""
        if isinstance(source_id, (list, tuple)):
            sources = list(dict.fromkeys(source_id))
        else:
            sources = [source_id]
        unknown = [s for s in sources if s not in _SOURCE_INFO]
        if unknown:
            raise KeyError(f"No _SOURCE_INFO entry for source(s): {unknown}")
        return "\n\n".join(_SOURCE_INFO[s] for s in sources)
    
    @staticmethod
    def _format_answer(answer_meta: dict) -> str:
        if answer_meta["type"] == "boolean":
            return "Yes" if answer_meta["value"] else "No"
        elif answer_meta["type"] == "scalar":
            val = answer_meta["value"]
            unit = answer_meta.get("unit", "")
            return f"{val} {unit}".strip()
        return str(answer_meta["value"])
    
    @staticmethod
    def _answer_has_nan(answer_meta: dict) -> bool:
        """Return True if `value` is missing or any numeric field in answer_metadata is NaN.

        Booleans and strings are ignored; nested lists/tuples are walked recursively.
        """
        if answer_meta is None:
            return True
        if "value" not in answer_meta or answer_meta["value"] is None:
            return True

        def _is_nan_number(x) -> bool:
            if isinstance(x, bool):
                return False
            if isinstance(x, (np.floating, float)):
                try:
                    return math.isnan(float(x))
                except (TypeError, ValueError):
                    return False
            if isinstance(x, (np.integer, int)):
                return False
            return False

        def _walk(x) -> bool:
            if isinstance(x, dict):
                return any(_walk(v) for v in x.values())
            if isinstance(x, (list, tuple, set)):
                return any(_walk(v) for v in x)
            if isinstance(x, np.ndarray):
                try:
                    return bool(np.isnan(x).any())
                except TypeError:
                    return False
            return _is_nan_number(x)

        return _walk(answer_meta)

    @staticmethod
    def _scan_existing_seq_nos(output_dir: Path, level: int) -> dict:
        """Scan output_dir (recursively) for existing JSON files and return the
        max seq_no observed per task_id at the given level.

        IDs are expected to follow `<task_id>_<seq_no>` (e.g. `L1_1_5`), where
        `task_id` itself contains underscores (e.g. `L1_1`).
        """
        max_seq: dict[str, int] = {}
        if not output_dir.exists():
            return max_seq

        for fp in output_dir.rglob("*.json"):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    records = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(records, list):
                continue
            for r in records:
                if not isinstance(r, dict):
                    continue
                if r.get("level") != level:
                    continue
                uid = r.get("id", "")
                task_id = r.get("task_id", "")
                if not isinstance(uid, str) or not isinstance(task_id, str):
                    continue
                prefix = f"{task_id}_"
                if not uid.startswith(prefix):
                    continue
                try:
                    sno = int(uid[len(prefix):])
                except ValueError:
                    continue
                if sno > max_seq.get(task_id, 0):
                    max_seq[task_id] = sno
        return max_seq

    @staticmethod
    def _json_default(obj):
        """Handle numpy / xarray types during JSON serialization."""
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
    

    # ===================================================================
    # Single-example geenration and formatting
    # ===================================================================
    
    def _generate_one_example(self, task_id: str, level: int):
        cfg = self.configs
        if level == 1:
            example_dict = generate_level1_tasks(cfg, task_id)
        elif level == 2:
            example_dict = generate_level2_tasks(cfg, task_id)
        elif level == 3:
            example_dict = generate_level3_tasks(cfg, task_id)
        else:
            raise ValueError(f"Unsupported level: {level}")
        
        if example_dict is None:
            return None

        return example_dict
    
    def _format_record(
        self,
        example_dict: dict,
        level: int,
        seq_no: int,
        task_template_df: pd.DataFrame,
        prompt_template_df: pd.DataFrame,
    ) -> dict:
        task_id = example_dict["task_id"]
        data_meta = example_dict["data_metadata"]
        task_meta = example_dict["task_metadata"]
        answer_meta = example_dict["answer_metadata"]

        # unique id: e.g. L1_2_1
        unique_id = f"{task_id}_{seq_no}"

        # instantiate the question text
        row = task_template_df.loc[task_template_df["task_id"] == task_id]
        if row.empty:
            raise KeyError(f"task_id '{task_id}' not found in template CSV")
        task_text = row["task_text"].values[0].format(**task_meta)

        # build prompt fields
        configuration = self._build_configuration_text(data_meta)
        source_info = self._build_source_info_text(data_meta["source_id"])

        # pick a prompt template (randomly among non-empty)
        valid_prompts = prompt_template_df[
            prompt_template_df["prompt_text"].notna()
            & (prompt_template_df["prompt_text"].str.strip() != "")
        ]
        if valid_prompts.empty:
            raise ValueError("No non-empty prompt templates found")
        prompt_row = valid_prompts.sample(1).iloc[0]
        prompt_id = prompt_row["prompt_id"].strip()
        prompt_text = prompt_row["prompt_text"].format(
            configuration=configuration,
            source_info=source_info,
        )

        # cleaned answer
        answer = self._format_answer(answer_meta)

        # ambiguous_terms (empty for L1)
        ambiguous_terms = task_meta.pop("ambiguous_terms", None) or []

        # Strip `data_root` from the persisted metadata. Solvers used it
        # internally to load NetCDF files, but the concrete root path is
        # an environment detail and must not leak into the published QA
        # record — the agent receives it separately at codegen time.
        public_data_meta = {k: v for k, v in data_meta.items() if k != "data_root"}

        return {
            "id": unique_id,
            "level": level,
            "task_id": task_id,
            "prompt_id": prompt_id,
            "prompt": prompt_text,
            "question": task_text,
            "answer": answer,
            "ambiguous_terms": ambiguous_terms,
            "data_metadata": public_data_meta,
            "task_metadata": task_meta,
            "answer_metadata": answer_meta,
        }


    # ===================================================================
    # Single-example geenration and formatting
    # ===================================================================

    def generate(self):
        cfg = self.configs

        prompt_template_df = pd.read_csv(cfg.generation.prompt_path)
        output_dir = Path(cfg.generation.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        max_per_file = cfg.generation.num_examples_per_file

        for level, level_cfg in [
            (1, cfg.level1),
            (2, cfg.level2),
            (3, cfg.level3),
        ]:
            if level_cfg.num == 0:
                continue

            task_template_df = pd.read_csv(level_cfg.template_path)
            task_id_list = self._get_task_id_list(level_cfg)

            # Scan existing outputs so new seq_nos continue past whatever is already on disk.
            existing_max_seq = self._scan_existing_seq_nos(output_dir, level)
            seq_counter: dict[str, int] = dict(existing_max_seq)
            if existing_max_seq:
                print(f"[INFO] level={level}: found existing ids, "
                      f"continuing from {existing_max_seq}")

            new_records: list[dict] = []
            fail_count = 0
            max_failures = level_cfg.num * 3

            while len(new_records) < level_cfg.num and fail_count < max_failures:
                task_id = random.choice(task_id_list)
                try:
                    example_dict = self._generate_one_example(task_id, level)
                    if example_dict is None:
                        # solver returned no result (e.g. NaN, missing data)
                        continue
                    if self._answer_has_nan(example_dict.get("answer_metadata", {})):
                        # answer_metadata contains NaN values -- invalid, skip without counting
                        continue
                    next_seq = seq_counter.get(task_id, 0) + 1
                    record = self._format_record(
                        example_dict, level, next_seq,
                        task_template_df, prompt_template_df,
                    )
                    seq_counter[task_id] = next_seq
                    new_records.append(record)
                except FileNotFoundError:
                    # data not downloaded for this experiment/source/variable combo
                    continue
                except Exception as e:
                    fail_count += 1
                    print(f"[WARN] level={level} task={task_id} failed "
                          f"({fail_count}/{max_failures}): {e}")
                    continue

                print(f"[OK] level={level} task={task_id} seq={next_seq} "
                      f"({len(new_records)}/{level_cfg.num})")

            if not new_records:
                print(f"[WARN] No records generated for level {level}")
                continue

            # Group new records by task_id, chunk each group, and save.
            by_task: dict[str, list[dict]] = defaultdict(list)
            for r in new_records:
                by_task[r["task_id"]].append(r)

            for tid, recs in by_task.items():
                recs.sort(key=lambda r: int(r["id"].rsplit("_", 1)[-1]))
                for chunk_start in range(0, len(recs), max_per_file):
                    chunk = recs[chunk_start : chunk_start + max_per_file]
                    first_no = chunk[0]["id"].rsplit("_", 1)[-1]
                    last_no = chunk[-1]["id"].rsplit("_", 1)[-1]
                    filename = f"{tid}_{first_no}_{last_no}.json"
                    out_path = output_dir / filename

                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(chunk, f, indent=2, ensure_ascii=False,
                                  default=self._json_default)

                    print(f"[INFO] Saved {len(chunk)} records → {out_path}")
                                