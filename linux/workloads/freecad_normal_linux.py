from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from linux.workloads.common_linux import ensure_dir, find_executable, run_command, temporary_python_script, copy_tree_filtered

FREECAD_CANDIDATES = [
    "freecadcmd",
    "FreeCADCmd",
    "freecad",
    "FreeCAD",
    Path("/usr/bin/freecadcmd"),
    Path("/usr/bin/FreeCADCmd"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # 升级为接收工程目录，对齐异常操作的规模
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--export-format", default="step", choices=["step", "stl"])
    parser.add_argument("--batch-size", type=int, default=0, help="处理文件数量上限 (0表示处理全部)")
    return parser.parse_args()


def build_freecad_script(project: Path, export_path: Path, export_format: str) -> str:
    return f"""
import os
import FreeCAD
import Import
import Mesh

project = r"{project}"
export_path = r"{export_path}"
export_format = {export_format!r}

ext = os.path.splitext(project)[1].lower()
if ext == ".fcstd":
    doc = FreeCAD.openDocument(project)
else:
    Import.open(project)
    doc = FreeCAD.ActiveDocument

if doc is None:
    raise RuntimeError("Failed to open FreeCAD document")

doc.recompute()
objects = [obj for obj in doc.Objects if hasattr(obj, "Shape") or hasattr(obj, "Mesh")] or list(doc.Objects)
if export_format == "step":
    Import.export(objects, export_path)
else:
    Mesh.export(objects, export_path)

FreeCAD.closeDocument(doc.Name)
"""


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    export_dir = ensure_dir(args.export_dir).resolve()

    # 1. 正常业务逻辑独有：合法备份源文件
    backup_dir = ensure_dir(export_dir / "legal_backup")
    copy_tree_filtered(project_dir, backup_dir, ["*.FCStd"])

    freecad = find_executable(["FREECAD_CMD", "FREECAD_EXE"], FREECAD_CANDIDATES)

    # 2. 搜集目录下所有工程文件
    projects = sorted(
        {
            *project_dir.rglob("*.FCStd"),
            *project_dir.rglob("*.step"),
            *project_dir.rglob("*.stp"),
        }
    )

    if not projects:
        raise RuntimeError(f"No FreeCAD projects found in {project_dir}")

    if args.batch_size > 0:
        projects = projects[:args.batch_size]

    # 3. 循环批量导出 (对齐批量窃取的进程树结构)
    export_ext = ".step" if args.export_format == "step" else ".stl"
    for index, project in enumerate(projects, start=1):
        export_path = export_dir / f"{index:03d}_{project.stem}_linux_normal_export{export_ext}"
        script_path = temporary_python_script(build_freecad_script(project, export_path, args.export_format))
        try:
            run_command([str(freecad), str(script_path)])
        finally:
            script_path.unlink(missing_ok=True)

    print(f"freecad={freecad}")
    print(f"projects_processed={len(projects)}")
    print(f"export_dir={export_dir}")


if __name__ == "__main__":
    main()
