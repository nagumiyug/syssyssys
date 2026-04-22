from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from linux.workloads.common_linux import ensure_dir, find_executable, run_command, temporary_python_script
from linux.workloads.freecad_normal_linux import FREECAD_CANDIDATES, build_freecad_script


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--external-dir", required=True)
    parser.add_argument("--export-format", default="step", choices=["step", "stl"])
    parser.add_argument("--batch-size", type=int, default=0)
    return parser.parse_args()


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
        raise RuntimeError("No FreeCAD projects found for bulk export")
        
    if args.batch_size > 0:
        projects = projects[:args.batch_size]

    for index, project in enumerate(projects, start=1):
        export_ext = ".step" if args.export_format == "step" else ".stl"
        export_path = external_dir / f"exfil_{index:03d}_{project.stem}{export_ext}"
        script_path = temporary_python_script(build_freecad_script(project, export_path, args.export_format))
        try:
            run_command([str(freecad), str(script_path)])
        finally:
            script_path.unlink(missing_ok=True)

    print(f"freecad={freecad}")
    print(f"projects_exported={len(projects)}")
    print(f"external_dir={external_dir}")


if __name__ == "__main__":
    main()
