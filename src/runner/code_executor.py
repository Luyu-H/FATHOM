"""Sandboxed-ish Python code execution.

Strategy:
- write the LLM-emitted script into a per-run workspace directory
- run via subprocess with a configurable timeout
- parse the final answer from a `FINAL_ANSWER:` stdout marker

This is *isolation*, not *security* — the script needs read access to the
CMIP6 dataset directory, so we deliberately don't sandbox filesystem access.

The subprocess inherits the parent process environment (so the generated
script can rely on the same numpy / xarray / scipy / dask / cftime / gsw
installation as the agent itself).
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CODE_FENCE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL)
_FINAL_ANSWER = re.compile(r"FINAL_ANSWER\s*[:=]\s*(.+?)\s*$", re.MULTILINE)


@dataclass
class ExecutionResult:
    success: bool
    stdout: str
    stderr: str
    final_value: Optional[str] = None
    exit_code: Optional[int] = None
    timed_out: bool = False
    code_path: Optional[str] = None


_CODE_TAG = re.compile(r"<code>(.*?)</code>", re.DOTALL | re.IGNORECASE)
_CODE_OPEN = re.compile(r"<code>", re.IGNORECASE)
_CODE_CLOSE = re.compile(r"</code>", re.IGNORECASE)
# Any stray <code> / </code> tag that survived extraction (e.g. opened in
# the middle of a fence) — must be scrubbed before handing the script to
# Python, otherwise the file's first line becomes a literal `<code>` and
# the interpreter dies with a SyntaxError before reaching the real code.
_STRAY_TAG = re.compile(r"</?code>", re.IGNORECASE)


def extract_code(text: str) -> Optional[str]:
    """Pull the largest <code> ... </code> block; fall back to ```python```
    fences, then to the raw text if it looks like Python.

    Defensive against truncated LLM responses: if an opening ``<code>`` tag
    is present without a matching ``</code>`` (a common symptom of
    max_tokens cut-off), extract everything from the first ``<code>`` to
    the end of the buffer instead of silently returning the whole response
    with a literal ``<code>`` first line."""
    if not text:
        return None
    tag_matches = _CODE_TAG.findall(text)
    if tag_matches:
        return _STRAY_TAG.sub("", max(tag_matches, key=len)).strip()
    # Orphan opening tag (no close) — almost always a truncated response.
    # Take everything after the LAST `<code>` so we don't include prose
    # the LLM may have written before the script.
    open_match = list(_CODE_OPEN.finditer(text))
    if open_match and not _CODE_CLOSE.search(text):
        tail = text[open_match[-1].end():]
        return _STRAY_TAG.sub("", tail).strip() or None
    matches = _CODE_FENCE.findall(text)
    if matches:
        return _STRAY_TAG.sub("", max(matches, key=len)).strip()
    if "import " in text or "def " in text or "print(" in text:
        return _STRAY_TAG.sub("", text).strip()
    return None


def parse_final_answer(stdout: str) -> Optional[str]:
    if not stdout:
        return None
    matches = _FINAL_ANSWER.findall(stdout)
    return matches[-1].strip() if matches else None      # last occurrence wins


class CodeExecutor:
    def __init__(self, workspace_root: str, timeout_seconds: int = 600,
                 python_executable: str = "python"):
        self.workspace_root = Path(workspace_root)
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout_seconds
        self.python = python_executable

    def run(self, code: str, run_id: Optional[str] = None) -> ExecutionResult:
        run_id = run_id or uuid.uuid4().hex[:8]
        run_dir = self.workspace_root / f"run_{run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)
        script_path = run_dir / "task.py"
        script_path.write_text(code, encoding="utf-8")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        # Prevent matplotlib from trying to open a window if the script imports it
        env.setdefault("MPLBACKEND", "Agg")

        try:
            proc = subprocess.run(
                [self.python, str(script_path)],
                cwd=str(run_dir),
                capture_output=True, text=True,
                timeout=self.timeout, env=env, check=False,
            )
        except subprocess.TimeoutExpired as e:
            # TimeoutExpired carries the child's raw buffer as bytes even
            # when text=True — the decode step only runs on the normal
            # return path. Decode defensively so downstream str-ops work.
            def _as_str(buf):
                if buf is None:
                    return ""
                if isinstance(buf, bytes):
                    return buf.decode("utf-8", errors="replace")
                return buf
            return ExecutionResult(
                success=False,
                stdout=_as_str(e.stdout),
                stderr=_as_str(e.stderr) + f"\nTIMEOUT after {self.timeout}s",
                timed_out=True,
                code_path=str(script_path),
            )

        success = proc.returncode == 0
        return ExecutionResult(
            success=success,
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            final_value=parse_final_answer(proc.stdout or ""),
            exit_code=proc.returncode,
            code_path=str(script_path),
        )