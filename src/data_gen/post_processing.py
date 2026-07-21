"""Post-processing utilities for generated QA datasets.

Provides two stages, each exposed as a directory-level function that callers
(e.g. ``src/generate.py``) can invoke independently:

1. ``process_ambiguous_terms_in_directory`` — split each ``ambiguous_terms``
   entry into the atomic granularity defined by ``semantic_ambiguity_ops.jsonl``,
   expand one hop of ``related_terms``, and emit a parallel
   ``ambiguous_term_ids`` list aligned with the rewritten ``ambiguous_terms``.

2. ``rephrase_questions_in_directory`` — rephrase the ``question`` field with
   an LLM. The prompt is constrained so that any portion corresponding to an
   ambiguous term in the item may ONLY be expressed using a synonym from the
   matching ops entry's ``term`` field. The rest of the sentence may be lightly
   rephrased but must not change task content/logic and must not introduce new
   ambiguity. The original text is preserved under ``original_question`` when
   requested.

This module is import-only; orchestration (which stages to run, paths, etc.)
lives in ``src/generate.py`` and ``configs/dataset_gen.yaml``.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

log = logging.getLogger(__name__)

# Probability of skipping the LLM call and keeping the question verbatim,
# so the dataset retains a small fraction of un-rephrased items.
_SKIP_REPHRASE_PROB = 0.2

_TRAILING_COMMA_OBJ = re.compile(r",\s*}")
_TRAILING_COMMA_ARR = re.compile(r",\s*]")


def _loads_lenient(line: str) -> dict:
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        fixed = _TRAILING_COMMA_OBJ.sub("}", line)
        fixed = _TRAILING_COMMA_ARR.sub("]", fixed)
        return json.loads(fixed)


def load_ops(jsonl_path: Path) -> List[dict]:
    entries: List[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(_loads_lenient(line))
    return entries


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------


def build_term_index(
    entries: Iterable[dict],
) -> Tuple[Dict[str, str], Dict[str, List[str]], Dict[str, List[str]]]:
    """Build lookup tables from the ops entries.

    Returns
    -------
    term_to_id
        Lower-cased term string -> ops entry id.
    id_to_terms
        Ops entry id -> list of all term spellings (in original casing).
    id_to_related
        Ops entry id -> list of related-term strings.
    """
    term_to_id: Dict[str, str] = {}
    id_to_terms: Dict[str, List[str]] = {}
    id_to_related: Dict[str, List[str]] = {}
    for entry in entries:
        eid = entry["id"]
        terms = entry.get("term", []) or []
        id_to_terms[eid] = list(terms)
        id_to_related[eid] = list(entry.get("related_terms", []) or [])
        for t in terms:
            key = t.lower().strip()
            if not key:
                continue
            term_to_id.setdefault(key, eid)
    return term_to_id, id_to_terms, id_to_related


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


_PAREN_RE = re.compile(r"\(([^)]*)\)")


def _normalize_for_match(text: str) -> str:
    """Drop parenthetical fragments such as ``"(PDO)"`` and collapse whitespace."""
    stripped = _PAREN_RE.sub(" ", text)
    return re.sub(r"\s+", " ", stripped).strip()


def _build_pattern(term: str) -> re.Pattern:
    pattern = re.escape(term)
    left = r"(?<![A-Za-z0-9])" if term[:1].isalnum() else ""
    right = r"(?![A-Za-z0-9])" if term[-1:].isalnum() else ""
    return re.compile(left + pattern + right)


def find_matches(text: str, term_to_id: Dict[str, str]) -> List[Tuple[str, str]]:
    """Return ``(matched_text, id)`` pairs found in ``text``.

    Uses a longest-match-first non-overlapping scan so that a multi-word term
    such as ``"sea surface temperature"`` wins over its shorter substrings.
    """
    haystacks: List[str] = []
    seen_norms: set = set()
    for candidate in (text, _normalize_for_match(text)):
        norm = candidate.lower()
        if norm and norm not in seen_norms:
            haystacks.append(norm)
            seen_norms.add(norm)

    sorted_terms = sorted(term_to_id.keys(), key=len, reverse=True)

    collected: List[Tuple[int, int, str, str, int]] = []  # (start, end, text, id, haystack_idx)
    for h_idx, haystack in enumerate(haystacks):
        occupied = [False] * len(haystack)
        for term in sorted_terms:
            for m in _build_pattern(term).finditer(haystack):
                s, e = m.start(), m.end()
                if any(occupied[s:e]):
                    continue
                for i in range(s, e):
                    occupied[i] = True
                collected.append((s, e, haystack[s:e], term_to_id[term], h_idx))

    collected.sort(key=lambda x: (x[4], x[0]))

    seen_ids: set = set()
    out: List[Tuple[str, str]] = []
    for _, _, matched_text, mid, _ in collected:
        if mid in seen_ids:
            continue
        seen_ids.add(mid)
        out.append((matched_text, mid))
    return out


# ---------------------------------------------------------------------------
# Stage 1: ambiguous-term resolution
# ---------------------------------------------------------------------------


def process_ambiguous_terms_in_item(
    item: dict,
    term_to_id: Dict[str, str],
    id_to_related: Dict[str, List[str]],
) -> dict:
    raw_terms = item.get("ambiguous_terms") or []

    pairs: List[Tuple[str, str]] = []
    seen_ids: set = set()

    for raw in raw_terms:
        if not isinstance(raw, str):
            continue
        for matched_text, mid in find_matches(raw, term_to_id):
            if mid in seen_ids:
                continue
            seen_ids.add(mid)
            pairs.append((matched_text, mid))

    # Expand related_terms (one hop).
    for _, mid in list(pairs):
        for rel in id_to_related.get(mid, []):
            rel_norm = rel.lower().strip()
            rel_id = term_to_id.get(rel_norm)
            if rel_id is None or rel_id in seen_ids:
                continue
            seen_ids.add(rel_id)
            pairs.append((rel_norm, rel_id))

    item["ambiguous_terms"] = [p[0] for p in pairs]
    item["ambiguous_term_ids"] = [p[1] for p in pairs]
    return item


def _process_ambiguous_file(
    src_path: Path,
    dst_path: Path,
    term_to_id: Dict[str, str],
    id_to_related: Dict[str, List[str]],
) -> int:
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at {src_path}, got {type(data).__name__}")
    for item in data:
        process_ambiguous_terms_in_item(item, term_to_id, id_to_related)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return len(data)


def process_ambiguous_terms_in_directory(
    qa_dir: Path,
    ops_path: Path,
    out_dir: Optional[Path] = None,
    levels: Iterable[str] = ("level1", "level2", "level3"),
) -> None:
    """Resolve ``ambiguous_terms`` across every QA JSON under ``qa_dir``."""
    entries = load_ops(Path(ops_path))
    term_to_id, _id_to_terms, id_to_related = build_term_index(entries)

    qa_dir = Path(qa_dir)
    out_root = Path(out_dir) if out_dir is not None else qa_dir

    total_files = 0
    total_items = 0

    for src in sorted(qa_dir.glob("*.json")):
        dst = out_root / src.name
        n = _process_ambiguous_file(src, dst, term_to_id, id_to_related)
        total_files += 1
        total_items += n
        print(f"[ambig] processed {src.relative_to(qa_dir)} ({n} items)")

    for level in levels:
        level_dir = qa_dir / level
        if not level_dir.exists():
            continue
        for src in sorted(level_dir.glob("*.json")):
            dst = out_root / level / src.name
            n = _process_ambiguous_file(src, dst, term_to_id, id_to_related)
            total_files += 1
            total_items += n
            print(f"[ambig] processed {src.relative_to(qa_dir)} ({n} items)")

    print(f"[ambig] done: {total_files} files, {total_items} items")


# ---------------------------------------------------------------------------
# Stage 2: question rephrasing
# ---------------------------------------------------------------------------


_REPHRASE_RULES = (
    "You are a careful scientific editor. Your task is to rephrase a question "
    "so that the surface wording differs noticeably from the original while "
    "the technical meaning, computational logic, and final result stay "
    "exactly the same. Follow these rules:\n"
    "1. Preserve every variable name, region name, coordinate range, date "
    "span, depth, operator, threshold, and unit character-for-character. Do "
    "NOT add, drop, generalise, or alter any of them — a downstream "
    "computation on the rephrased question must produce an identical result "
    "to one on the original.\n"
    "2. The original question may contain certain ambiguous terms. For each "
    "such term you may EITHER keep the original wording OR replace it with "
    "one of the listed allowed expressions for that term. Do NOT invent new "
    "synonyms, definitions, or paraphrases for those terms.\n"
    "3. Beyond the constraints above, rephrase freely. Vary sentence "
    "structure, voice, word order, framing, register, connectors, or "
    "phrasing — pick whatever reads naturally for this particular question, "
    "and try to vary your style across different questions rather than "
    "applying the same template every time. The rewrite should feel like a "
    "different person asking the same thing, not a minor edit. A change "
    "consisting only of a single word swap or article tweak is not enough.\n"
    "4. Do NOT introduce any new ambiguous terminology, vague wording, or "
    "alternative interpretations elsewhere in the sentence.\n"
    "5. Output strictly valid JSON of the form "
    '{"rephrased_question": "..."} with no extra keys or commentary.'
)


_REPHRASE_FEW_SHOT = (
    "\n\nFor reference, one illustrative rephrasing (do NOT copy this style "
    "or phrasing — vary your wording across questions):\n\n"
    "Original: Is the 10th percentile of oxygen saturation over the "
    "70°S–69°S, 157°E–180°E during 2092-10 ~ 2097-12 at 4000.0 m less than "
    "the 10th percentile of oxygen saturation over the South Atlantic Ocean "
    "(60°S–0°N, 290°E–20°E) during 2048-11 ~ 2053-03 at 4000.0 m?\n"
    "Rephrased: Looking at the 4000.0 m level, does the 10th percentile of "
    "oxygen saturation in 70°S–69°S, 157°E–180°E over 2092-10 ~ 2097-12 fall "
    "below the corresponding 10th percentile in the South Atlantic Ocean "
    "(60°S–0°N, 290°E–20°E) for 2048-11 ~ 2053-03?\n"
)


_REPHRASE_SYSTEM_PROMPT = _REPHRASE_RULES + _REPHRASE_FEW_SHOT


def _build_allowed_terms_block(
    ambiguous_terms: List[str],
    ambiguous_term_ids: List[str],
    id_to_terms: Dict[str, List[str]],
) -> str:
    """Build the per-question placeholder listing ONLY the relevant synonyms."""
    lines: List[str] = []
    seen: set = set()
    for current, tid in zip(ambiguous_terms, ambiguous_term_ids):
        if tid in seen:
            continue
        seen.add(tid)
        synonyms = id_to_terms.get(tid, [])
        synonyms_repr = ", ".join(f'"{s}"' for s in synonyms) if synonyms else "(no alternative spellings; keep as-is)"
        lines.append(
            f'- term id {tid} (currently used in the question: "{current}"): '
            f"allowed expressions = [{synonyms_repr}]"
        )
    return "\n".join(lines) if lines else "(no ambiguous terms in this question)"


def _build_rephrase_user_prompt(
    question: str,
    allowed_block: str,
) -> str:
    return (
        "Rephrase the following question under the constraints listed in the "
        "system instructions. Make the surface form clearly different from "
        "the original — vary structure, framing, and wording in whatever way "
        "feels natural for this question. Every variable, coordinate, date, "
        "depth, operator, and threshold must be preserved "
        "character-for-character. The computational logic, requested "
        "quantity, and final result must stay the same.\n\n"
        f"Original question:\n{question}\n\n"
        "Ambiguous terms in this question and the allowed expressions for "
        "each (for each term you may keep the original wording OR pick one of "
        "these — do not use any other synonym, definition, or paraphrase):\n"
        f"{allowed_block}\n\n"
        'Return JSON only: {"rephrased_question": "<your rephrased text>"}'
    )


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def _extract_rephrased(raw: str) -> Optional[str]:
    if not raw:
        return None
    m = _JSON_BLOCK.search(raw)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    text = obj.get("rephrased_question")
    if isinstance(text, str) and text.strip():
        return text.strip()
    return None


def _call_rephrase_api(
    client,
    *,
    system_prompt: str,
    user_prompt: str,
    original_question: str,
    model: str,
    temperature: float,
    max_retries: int,
    reasoning_effort: Optional[str],
    max_output_tokens: int,
) -> Optional[str]:
    original_norm = original_question.strip()
    for attempt in range(1, max_retries + 1):
        try:
            params = {
                "model": model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_completion_tokens": max_output_tokens,
            }
            if reasoning_effort:
                params["reasoning_effort"] = reasoning_effort
            response = client.chat.completions.create(**params)
            raw = response.choices[0].message.content or ""
        except Exception as exc:                              # noqa: BLE001
            log.warning("[rephrase] API error (attempt %d/%d): %s",
                        attempt, max_retries, exc)
            if attempt < max_retries:
                time.sleep(2 ** attempt)
            continue

        text = _extract_rephrased(raw)
        if not text:
            log.warning("[rephrase] attempt %d: failed to parse JSON. Snippet: %.150s",
                        attempt, raw)
            continue
        if text.strip() == original_norm:
            log.warning("[rephrase] attempt %d/%d: model returned text identical "
                        "to the original; retrying", attempt, max_retries)
            continue
        return text
    return None


def rephrase_question_in_item(
    item: dict,
    client,
    id_to_terms: Dict[str, List[str]],
    *,
    model: str,
    temperature: float,
    max_retries: int,
    reasoning_effort: Optional[str],
    max_output_tokens: int,
    keep_original_question: bool = True,
) -> dict:
    question = item.get("question", "")
    ambiguous_terms = item.get("ambiguous_terms") or []
    ambiguous_term_ids = item.get("ambiguous_term_ids") or []
    if not question:
        return item

    # With small probability keep the question verbatim — no LLM call.
    if random.random() < _SKIP_REPHRASE_PROB:
        if keep_original_question and "original_question" not in item:
            item["original_question"] = question
        return item

    allowed_block = _build_allowed_terms_block(
        ambiguous_terms, ambiguous_term_ids, id_to_terms,
    )
    user_prompt = _build_rephrase_user_prompt(question, allowed_block)

    new_text = _call_rephrase_api(
        client,
        system_prompt=_REPHRASE_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        original_question=question,
        model=model,
        temperature=temperature,
        max_retries=max_retries,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
    )
    if new_text is None:
        log.warning("[rephrase] giving up on item id=%s; keeping original",
                    item.get("id"))
        return item

    if keep_original_question and "original_question" not in item:
        item["original_question"] = question
    item["question"] = new_text
    return item


def _rephrase_file(
    src_path: Path,
    dst_path: Path,
    client,
    id_to_terms: Dict[str, List[str]],
    *,
    request_interval: float,
    keep_original_question: bool,
    model: str,
    temperature: float,
    max_retries: int,
    reasoning_effort: Optional[str],
    max_output_tokens: int,
) -> int:
    with open(src_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at {src_path}, got {type(data).__name__}")

    for idx, item in enumerate(data):
        if idx > 0 and request_interval > 0:
            time.sleep(request_interval)
        rephrase_question_in_item(
            item, client, id_to_terms,
            model=model,
            temperature=temperature,
            max_retries=max_retries,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            keep_original_question=keep_original_question,
        )

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return len(data)


def rephrase_questions_in_directory(
    qa_dir: Path,
    ops_path: Path,
    *,
    llm_settings,
    out_dir: Optional[Path] = None,
    levels: Iterable[str] = ("level1", "level2", "level3"),
    keep_original_question: bool = True,
) -> None:
    """Rephrase the ``question`` field of every QA item under ``qa_dir``.

    Items must already carry ``ambiguous_term_ids`` (i.e. stage 1 has run).
    """
    from openai import OpenAI  # lazy import

    entries = load_ops(Path(ops_path))
    _term_to_id, id_to_terms, _id_to_related = build_term_index(entries)

    client = OpenAI()
    qa_dir = Path(qa_dir)
    out_root = Path(out_dir) if out_dir is not None else qa_dir

    total_files = 0
    total_items = 0

    for src in sorted(qa_dir.glob("*.json")):
        dst = out_root / src.name
        n = _rephrase_file(
            src, dst, client, id_to_terms,
            request_interval=float(getattr(llm_settings, "request_interval", 0.0) or 0.0),
            keep_original_question=keep_original_question,
            model=llm_settings.model,
            temperature=float(llm_settings.temperature),
            max_retries=int(llm_settings.max_retries),
            reasoning_effort=getattr(llm_settings, "reasoning_effort", None),
            max_output_tokens=int(llm_settings.max_output_tokens),
        )
        total_files += 1
        total_items += n
        print(f"[rephrase] processed {src.relative_to(qa_dir)} ({n} items)")

    for level in levels:
        level_dir = qa_dir / level
        if not level_dir.exists():
            continue
        for src in sorted(level_dir.glob("*.json")):
            dst = out_root / level / src.name
            n = _rephrase_file(
                src, dst, client, id_to_terms,
                request_interval=float(getattr(llm_settings, "request_interval", 0.0) or 0.0),
                keep_original_question=keep_original_question,
                model=llm_settings.model,
                temperature=float(llm_settings.temperature),
                max_retries=int(llm_settings.max_retries),
                reasoning_effort=getattr(llm_settings, "reasoning_effort", None),
                max_output_tokens=int(llm_settings.max_output_tokens),
            )
            total_files += 1
            total_items += n
            print(f"[rephrase] processed {src.relative_to(qa_dir)} ({n} items)")
    print(f"[rephrase] done: {total_files} files, {total_items} items")