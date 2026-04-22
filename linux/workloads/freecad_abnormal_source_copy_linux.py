from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from linux.workloads.common_linux import ensure_dir, find_executable, run_command, temporary_python_script
from linux.workloads.freecad_normal_linux import FREECAD_CANDIDATES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--external-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=0)
    return parser.parse_args()


def build_malicious_copy_script(project: Path, target_path: Path) -> str:
    return f"""
import os
import shutil
import FreeCAD
import Import

project = r"{project}"
target_path = r"{target_path}"

# [伪装]：正常打开三维文件，产生大量合法的解析、渲染引擎系统调用
ext = os.path.splitext(project)[1].lower()
if ext == ".fcstd":
    doc = FreeCAD.openDocument(project)
else:
    Import.open(project)
    doc = FreeCAD.ActiveDocument

if doc is None:
    raise RuntimeError("Failed to open FreeCAD document")
doc.recompute()

# [恶意]：在 FreeCAD 进程上下文中，窃取源文件
os.makedirs(os.path.dirname(target_path), exist_ok=True)
shutil.copy2(project, target_path)

FreeCAD.closeDocument(doc.Name)
"""


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    external_dir = ensure_dir(args.external_dir).resolve()
    freecad = find_executable(["FREECAD_CMD", "FREECAD_EXE"], FREECAD_CANDIDATES)
    
    projects = sorted(
        {
            *project_dir.rglob("*.FCStd"),
            *project_dir.rglob("*.step"),
            *project_dir.rglob("*.stp"),
        }
    )
    if not projects:
        raise RuntimeError("No FreeCAD projects found for source copy")
        
    if args.batch_size > 0:
        projects = projects[:args.batch_size]

    for index, project in enumerate(projects, start=1):
        target_path = external_dir / f"stolen_{index:03d}_{project.name}"
        script_path = temporary_python_script(build_malicious_copy_script(project, target_path))
        try:
            # 由 FreeCAD 进程执行窃取
            run_command([str(freecad), str(script_path)])
        finally:
            script_path.unlink(missing_ok=True)

    print(f"freecad={freecad}")
    print(f"projects_stolen={len(projects)}")
    print(f"external_dir={external_dir}")


if __name__ == "__main__":
    main()

