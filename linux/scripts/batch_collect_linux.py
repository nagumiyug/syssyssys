from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)
KICAD_DEMOS = Path("/usr/share/kicad/demos")


# 解析命令行参数，用于配置测试循环轮数、数据输出目录以及需要模拟的异常行为模式。
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch collect Linux normal/abnormal sessions.")
    parser.add_argument("--rounds", type=int, default=5, help="Rounds per scenario.")
    parser.add_argument("--output-dir", default=str(ROOT / "data" / "raw" / "linux_batch"))
    parser.add_argument(
        "--abnormal-modes",
        default="bulk_export",
        help="Comma-separated abnormal modes: bulk_export,source_copy,project_scan",
    )
    return parser.parse_args()


# 在子进程中执行指定的终端命令序列，若命令执行失败则抛出运行错误。
def run(command: list[str]) -> None:
    completed = subprocess.run(command, cwd=ROOT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"Command failed: {command}")


# 扫描特定目录，查找并返回所有可用的 FreeCAD 项目文件（.FCStd）列表。
def discover_freecad_projects() -> list[Path]:
    sample_dir = ROOT / "data" / "samples" / "freecad_linux"
    return sorted(sample_dir.glob("*.FCStd"))


# 遍历系统自带的演示目录，查找并提取符合条件的 KiCad 电路板文件（.kicad_pcb）。
def discover_kicad_projects() -> list[Path]:
    boards: list[Path] = []
    for path in sorted(KICAD_DEMOS.iterdir()):
        if not path.is_dir():
            continue
        path_boards = sorted(path.glob("*.kicad_pcb"))
        if len(path_boards) == 1:
            boards.append(path_boards[0])
    return boards


# 作为整个调度程序的入口，负责加载测试样本，并根据预设的轮次自动拼接和分发正常/异常工作负载的执行命令。
def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    abnormal_modes = [mode.strip() for mode in args.abnormal_modes.split(",") if mode.strip()]

    freecad_projects = discover_freecad_projects()
    kicad_projects = discover_kicad_projects()
    if not freecad_projects:
        raise SystemExit("No Linux FreeCAD samples found under data/samples/freecad_linux.")
    if not kicad_projects:
        raise SystemExit("No KiCad demo projects found under /usr/share/kicad/demos.")

    for round_idx in range(args.rounds):
        freecad_project = freecad_projects[round_idx % len(freecad_projects)]
        kicad_project = kicad_projects[round_idx % len(kicad_projects)]
        jobs = [
            (
                "freecad",
                "normal",
                freecad_project.name,
                [
                    str(PYTHON),
                    str(ROOT / "linux" / "scripts" / "bcc_linux_capture.py"),
                    "--software", "freecad",
                    "--scenario", f"normal_export_linux_r{round_idx + 1}",
                    "--label", "normal",
                    "--output", str(output_dir / f"freecad_normal_r{round_idx + 1}.csv"),
                    "--",
                    str(PYTHON),
                    str(ROOT / "linux" / "workloads" / "freecad_normal_linux.py"),
                    "--project-dir", str(ROOT / "data" / "samples" / "freecad_linux"),
                    "--export-dir", str(ROOT / "data" / "exports" / "freecad_linux"),
                ],
            ),
            (
                "kicad",
                "normal",
                kicad_project.stem,
                [
                    str(PYTHON),
                    str(ROOT / "linux" / "scripts" / "bcc_linux_capture.py"),
                    "--software", "kicad",
                    "--scenario", f"normal_export_linux_r{round_idx + 1}",
                    "--label", "normal",
                    "--output", str(output_dir / f"kicad_normal_r{round_idx + 1}.csv"),
                    "--",
                    str(PYTHON),
                    str(ROOT / "linux" / "workloads" / "kicad_normal_linux.py"),
                    "--project-dir", str(KICAD_DEMOS),
                    "--export-dir", str(ROOT / "data" / "exports" / "kicad_linux"),
                ],
            ),
        ]
        abnormal_job_specs = {
            "bulk_export": [
                (
                    "freecad",
                    "freecad_linux_samples",
                    [
                        str(PYTHON), str(ROOT / "linux" / "workloads" / "freecad_abnormal_bulk_export_linux.py"),
                        "--project-dir", str(ROOT / "data" / "samples" / "freecad_linux"),
                        "--external-dir", str(ROOT / "data" / "external" / "freecad_linux_exfil"),
                    ],
                ),
                (
                    "kicad",
                    "kicad_demos",
                    [
                        str(PYTHON), str(ROOT / "linux" / "workloads" / "kicad_abnormal_bom_exfil_linux.py"),
                        "--project-dir", str(KICAD_DEMOS),
                        "--external-dir", str(ROOT / "data" / "external" / "kicad_linux_exfil"),
                    ],
                ),
            ],
            "source_copy": [
                (
                    "freecad",
                    "freecad_linux_samples",
                    [
                        str(PYTHON), str(ROOT / "linux" / "workloads" / "freecad_abnormal_source_copy_linux.py"),
                        "--project-dir", str(ROOT / "data" / "samples" / "freecad_linux"),
                        "--external-dir", str(ROOT / "data" / "external" / "freecad_linux_source_copy"),
                    ],
                ),
                (
                    "kicad",
                    "kicad_demos",
                    [
                        str(PYTHON), str(ROOT / "linux" / "workloads" / "kicad_abnormal_source_copy_linux.py"),
                        "--project-dir", str(KICAD_DEMOS),
                        "--external-dir", str(ROOT / "data" / "external" / "kicad_linux_source_copy"),
                    ],
                ),
            ],
            "project_scan": [
                (
                    "freecad",
                    "freecad_linux_samples",
                    [
                        str(PYTHON), str(ROOT / "linux" / "workloads" / "freecad_abnormal_project_scan_linux.py"),
                        "--project-dir", str(ROOT / "data" / "samples" / "freecad_linux"),
                        "--scan-dir", str(ROOT / "data" / "external" / "freecad_linux_scan"),
                    ],
                ),
                (
                    "kicad",
                    "kicad_demos",
                    [
                        str(PYTHON), str(ROOT / "linux" / "workloads" / "kicad_abnormal_project_scan_linux.py"),
                        "--project-dir", str(KICAD_DEMOS),
                        "--scan-dir", str(ROOT / "data" / "external" / "kicad_linux_scan"),
                    ],
                ),
            ],
        }
        for mode in abnormal_modes:
            if mode not in abnormal_job_specs:
                raise SystemExit(f"Unsupported abnormal mode: {mode}")
            for software, sample_name, workload_command in abnormal_job_specs[mode]:
                jobs.append(
                    (
                        software,
                        "abnormal",
                        sample_name,
                        [
                            str(PYTHON),
                            str(ROOT / "linux" / "scripts" / "bcc_linux_capture.py"),
                            "--software", software,
                            "--scenario", f"{mode}_linux_r{round_idx + 1}",
                            "--label", "abnormal",
                            "--output", str(output_dir / f"{software}_{mode}_r{round_idx + 1}.csv"),
                            "--",
                            *workload_command,
                        ],
                    )
                )
        for software, label, sample_name, command in jobs:
            print(f"[progress] round={round_idx + 1}/{args.rounds} software={software} label={label} sample={sample_name}")
            run(command)

    print(f"batch collection complete: {output_dir}")


if __name__ == "__main__":
    main()
