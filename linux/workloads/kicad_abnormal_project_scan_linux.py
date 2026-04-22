from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from linux.workloads.common_linux import ensure_dir, find_executable, run_command, temporary_python_script
from linux.workloads.kicad_normal_linux import SYSTEM_PYTHON_CANDIDATES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--scan-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=0)
    return parser.parse_args()


def build_scan_script(board: Path, scan_output: Path) -> str:
    return f"""
import os
from pathlib import Path
import pcbnew

board_path = Path(r"{board}")
scan_output = Path(r"{scan_output}")

# 调用 KiCad 原生引擎加载解析
board = pcbnew.LoadBoard(str(board_path))

os.makedirs(os.path.dirname(scan_output), exist_ok=True)
with open(scan_output, "w", encoding="utf-8") as handle:
    handle.write(f"board={{board_path}}\\n")
    handle.write(f"tracks={{len(board.GetTracks())}}\\n")
    handle.write(f"footprints={{len(list(board.GetFootprints()))}}\\n")
    handle.write(f"drawings={{len(board.GetDrawings())}}\\n")
"""


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    scan_dir = ensure_dir(args.scan_dir).resolve()
    
    boards = sorted(project_dir.rglob("*.kicad_pcb"))
    if not boards:
        raise RuntimeError("No KiCad boards found for project scan")

    if args.batch_size > 0:
        boards = boards[:args.batch_size]

    system_python = find_executable(["KICAD_PYTHON", "SYSTEM_PYTHON"], SYSTEM_PYTHON_CANDIDATES)
    backend = "pcbnew"
    
    for index, board in enumerate(boards, start=1):
        scan_output = scan_dir / f"{index:03d}_{board.stem}_scan.txt"
        script_path = temporary_python_script(build_scan_script(board, scan_output))
        try:
            run_command([str(system_python), str(script_path)])
        finally:
            script_path.unlink(missing_ok=True)

    print(f"backend={backend}")
    print(f"boards_scanned={len(boards)}")
    print(f"scan_dir={scan_dir}")


if __name__ == "__main__":
    main()
