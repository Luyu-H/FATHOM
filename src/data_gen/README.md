# Data Generation

This module builds the <i>Fathom</i> QA dataset. It is driven end-to-end by
[`configs/data_gen.yaml`](../../configs/data_gen.yaml) and launched from the repository root with:

```bash
python -m src.generate
```

> The released dataset under `data/QA_dataset/` is ready to use — you only need this pipeline to
> **regenerate** or **extend** the benchmark.

---

## Pipeline

The entry point [`src/generate.py`](../generate.py) runs three stages in order. Each stage is toggled
independently under the `pipeline` block of the config.

| Stage | Flag | Function | What it does |
|:------|:-----|:---------|:-------------|
| **1. Generate** | `pipeline.run_generate` | `DatasetGenerator.generate()` | Instantiates task templates with concrete CMIP6 variables/regions/time ranges and computes the **ground-truth answer** by executing the reference solver against the scientific database. |
| **2. Ambiguous terms** | `pipeline.run_ambiguous_terms` | `process_ambiguous_terms_in_directory()` | Scans each generated question and tags the gold `ambiguous_terms` / `ambiguous_term_ids` by matching against the ambiguous-term corpus. |
| **3. Rephrase** | `pipeline.run_rephrase` | `rephrase_questions_in_directory()` | *(optional, off by default)* Uses an LLM to rewrite questions so the ambiguous terms are embedded naturally, keeping the original under `original_question`. |

### Module map

| File | Role |
|:-----|:-----|
| `dataset_generator.py` | Orchestrates generation: task sampling, answer computation, chunking, and file output. |
| `level1.py`, `level2.py`, `level3.py` | Per-level task builders and reference solvers (L1 baseline → L3 hardest). |
| `post_processing.py` | Ambiguous-term tagging (stage 2) and LLM rephrasing (stage 3). |

The reference solvers rely on the shared scientific utilities in [`src/utils/`](../utils/) (CMIP6 I/O,
numerical diagnostics, domain knowledge).

---

## Configuration

All options live in [`configs/data_gen.yaml`](../../configs/data_gen.yaml). Any field can be overridden
on the command line (OmegaConf dotted syntax), e.g. `python -m src.generate level1.num=50`.

| Block | Key | Meaning |
|:------|:----|:--------|
| `cmip6` | `root` | Scientific-database root (defaults to the `test` subset). |
| | `experiment` | CMIP6 experiments → source models drawn from during generation. |
| `level1` / `level2` / `level3` | `template_path` | Task-template CSV for the level. |
| | `num` | Number of examples to generate for the level (`0` disables it). |
| | `task_id_list` | Restrict generation to specific task IDs (empty = all). |
| `generation` | `num_examples_per_file` | Max records per output JSON file. |
| | `prompt_path` | Prompt-template CSV. |
| | `output_dir` | Where generated QA files are written. |
| `post_processing` | `qa_dir` / `output_dir` | Input/output dirs for stages 2–3. |
| | `ops_path` | Path to `ambiguous_term_corpus.jsonl`. |
| | `levels` | Which levels to post-process. |
| | `rephrase.keep_original_question` | Preserve the pre-rephrase text. |
| `llm_settings` | `model`, `temperature`, `max_retries`, `reasoning_effort`, `max_output_tokens`, `request_interval` | LLM parameters for the rephrase stage. |

**Prerequisites**

- The CMIP6 files referenced by `cmip6.root` must be present (see the main
  [README §2.1](../../README.md#21-scientific-database)) — stage 1 executes real solvers against them.
- Stage 3 (rephrase) calls the OpenAI API and requires `OPENAI_API_KEY` to be exported.

---

## Running individual stages

Enable or disable stages via the `pipeline` flags, either in the config or on the CLI:

```bash
# Only (re)compute ambiguous-term tags on an existing QA directory
python -m src.generate pipeline.run_generate=false \
                       pipeline.run_ambiguous_terms=true \
                       pipeline.run_rephrase=false

# Generate 100 Level-2 examples from scratch
python -m src.generate level1.num=0 level2.num=100 level3.num=0
```

## Output layout

Generated files are written under `generation.output_dir`, grouped by task and chunked:

```text
<output_dir>/
└── <task_id>_<first_seq>_<last_seq>.json     # e.g. L2_1_1_59.json
```

Each record carries the `question`, computed `answer` (+ `answer_metadata`), the `prompt`/`task`
metadata, and — after stage 2 — the gold `ambiguous_terms` and `ambiguous_term_ids`.
