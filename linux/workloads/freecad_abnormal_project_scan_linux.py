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
    parser.add_argument("--scan-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=0)
    return parser.parse_args()


def build_scan_script(project: Path, scan_output: Path) -> str:
    return f"""
import os
import FreeCAD
import Import

project = r"{project}"
scan_output = r"{scan_output}"

ext = os.path.splitext(project)[1].lower()
if ext == ".fcstd":
    doc = FreeCAD.openDocument(project)
else:
    Import.open(project)
    doc = FreeCAD.ActiveDocument

if doc is None:
    raise RuntimeError("Failed to open FreeCAD document")

doc.recompute()
object_count = len(doc.Objects)
shape_count = sum(1 for obj in doc.Objects if hasattr(obj, "Shape"))

# 扫描并收集内部结构信息，保存到外部
os.makedirs(os.path.dirname(scan_output), exist_ok=True)
with open(scan_output, "w", encoding="utf-8") as handle:
    handle.write(f"project={{project}}\\n")
    handle.write(f"objects={{object_count}}\\n")
    handle.write(f"shapes={{shape_count}}\\n")

FreeCAD.closeDocument(doc.Name)
"""


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    scan_dir = ensure_dir(args.scan_dir).resolve()
    freecad = find_executable(["FREECAD_CMD", "FREECAD_EXE"], FREECAD_CANDIDATES)
    
    projects = sorted(
        {
            *project_dir.rglob("*.FCStd"),
            *project_dir.rglob("*.step"),
            *project_dir.rglob("*.stp"),
        }
    )
    if not projects:
        raise RuntimeError("No FreeCAD projects found for project scan")

    if args.batch_size > 0:
        projects = projects[:args.batch_size]

    for index, project in enumerate(projects, start=1):
        scan_output = scan_dir / f"{index:03d}_{project.stem}_scan.txt"
        script_path = temporary_python_script(build_scan_script(project, scan_output))
        try:
            run_command([str(freecad), str(script_path)])
        finally:
            script_path.unlink(missing_ok=True)

    print(f"freecad={freecad}")
    print(f"projects_scanned={len(projects)}")
    print(f"scan_dir={scan_dir}")


if __name__ == "__main__":
    main()
