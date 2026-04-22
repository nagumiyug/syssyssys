from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable


def find_executable(env_vars: list[str], candidates: list[str | Path]) -> Path:
    for env_var in env_vars:
        value = os.environ.get(env_var)
        if value and Path(value).exists():
            return Path(value)

    for candidate in candidates:
        if isinstance(candidate, Path) and candidate.exists():
            return candidate
        if isinstance(candidate, str):
            resolved = shutil.which(candidate)
            if resolved:
                return Path(resolved)
            candidate_path = Path(candidate)
            if candidate_path.exists():
                return candidate_path

    raise FileNotFoundError(f"Executable not found. Checked env vars {env_vars} and Linux candidates.")


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def run_command(command: list[str], cwd: str | Path | None = None) -> None:
    completed = subprocess.run(command, cwd=cwd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {completed.returncode}: {command}")


def temporary_python_script(contents: str) -> Path:
    handle, path = tempfile.mkstemp(suffix=".py", text=True)
    os.close(handle)
    script_path = Path(path)
    script_path.write_text(contents, encoding="utf-8")
    return script_path


def copy_tree_filtered(src: Path, dst: Path, patterns: Iterable[str]) -> int:
    copied = 0
    for pattern in patterns:
        for path in src.rglob(pattern):
            if path.is_file():
                target = dst / path.relative_to(src)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
                copied += 1
    return copied
