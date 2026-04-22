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
    parser.add_argument("--layers", default="F.Cu,B.Cu,F.Mask,B.Mask,F.SilkS,B.SilkS,Edge.Cuts")
    parser.add_argument("--batch-size", type=int, default=0)
    return parser.parse_args()


def build_deceptive_exfil_script(board: Path, export_dir: Path, exfil_dir: Path, layers: str) -> str:
    layer_list = [layer.strip() for layer in layers.split(",") if layer.strip()]
    return f"""
import os
import shutil
from pathlib import Path
import pcbnew

board_path = Path(r"{board}")
gerber_dir = Path(r"{export_dir}") / "gerber"
exfil_dir = Path(r"{exfil_dir}")
layer_names = {layer_list!r}

# [伪装阶段]：正常加载电路板并执行合法的 Gerber 导出
os.makedirs(gerber_dir, exist_ok=True)
board = pcbnew.LoadBoard(str(board_path))
plot_controller = pcbnew.PLOT_CONTROLLER(board)
plot_options = plot_controller.GetPlotOptions()
plot_options.SetOutputDirectory(str(gerber_dir))
plot_options.SetPlotFrameRef(False)
plot_options.SetAutoScale(False)
plot_options.SetScale(1)
plot_options.SetUseGerberAttributes(True)
plot_options.SetUseGerberProtelExtensions(False)
plot_options.SetSubtractMaskFromSilk(False)

for layer_name in layer_names:
    layer_id = board.GetLayerID(layer_name)
    if layer_id >= 0:
        plot_controller.SetLayer(layer_id)
        plot_controller.OpenPlotfile(layer_name.replace('.', '_'), pcbnew.PLOT_FORMAT_GERBER, layer_name)
        plot_controller.PlotLayer()
plot_controller.ClosePlot()

# [恶意窃取阶段]：在同一个合法进程上下文中，窃取工程源文件
project_dir = board_path.parent
os.makedirs(exfil_dir, exist_ok=True)
for ext in [".kicad_pcb", ".kicad_pro", ".kicad_sch", ".kicad_prl"]:
    for f in project_dir.glob("*" + ext):
        shutil.copy2(f, exfil_dir / f.name)
"""


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    external_dir = ensure_dir(args.external_dir).resolve()

    boards = sorted(project_dir.rglob("*.kicad_pcb"))
    if not boards:
        raise RuntimeError("No KiCad boards found for abnormal bulk export")

    if args.batch_size > 0:
        boards = boards[:args.batch_size]

    system_python = find_executable(["KICAD_PYTHON", "SYSTEM_PYTHON"], SYSTEM_PYTHON_CANDIDATES)

    for index, board in enumerate(boards, start=1):
        # 伪造一个合法的导出目录和一个真实的窃取目录
        fake_export_dir = external_dir / f"fake_export_{index:03d}_{board.stem}"
        exfil_target_dir = external_dir / f"stolen_project_{index:03d}_{board.stem}"
        
        script_path = temporary_python_script(build_deceptive_exfil_script(board, fake_export_dir, exfil_target_dir, args.layers))
        try:
            run_command([str(system_python), str(script_path)])
        finally:
            script_path.unlink(missing_ok=True)

    print(f"backend=pcbnew_deceptive")
    print(f"boards_processed={len(boards)}")
    print(f"external_dir={external_dir}")


if __name__ == "__main__":
    main()
