from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from linux.workloads.common_linux import ensure_dir, find_executable, run_command, temporary_python_script

KICAD_CANDIDATES = [
    "kicad-cli",
    Path("/usr/bin/kicad-cli"),
]
SYSTEM_PYTHON_CANDIDATES = [
    Path("/usr/bin/python3"),
    "python3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # 升级为接收工程目录
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--layers", default="F.Cu,B.Cu,F.Mask,B.Mask,F.SilkS,B.SilkS,Edge.Cuts")
    parser.add_argument("--batch-size", type=int, default=0)
    return parser.parse_args()


def find_kicad_cli() -> Path | None:
    try:
        return find_executable(["KICAD_CLI"], KICAD_CANDIDATES)
    except FileNotFoundError:
        return None


def export_with_kicad_cli(kicad_cli: Path, board: Path, board_export_dir: Path, layers: str) -> str:
    gerber_dir = ensure_dir(board_export_dir / "gerber")
    drill_dir = ensure_dir(board_export_dir / "drill")
    run_command([str(kicad_cli), "pcb", "export", "gerbers", "--layers", layers, "--output", str(gerber_dir), str(board)])
    run_command([str(kicad_cli), "pcb", "export", "drill", "--output", str(drill_dir) + "/", str(board)])
    return "kicad-cli"


def export_with_pcbnew(board: Path, board_export_dir: Path, layers: str) -> str:
    system_python = find_executable(["KICAD_PYTHON", "SYSTEM_PYTHON"], SYSTEM_PYTHON_CANDIDATES)
    gerber_dir = ensure_dir(board_export_dir / "gerber")
    drill_dir = ensure_dir(board_export_dir / "drill")
    layer_list = [layer.strip() for layer in layers.split(",") if layer.strip()]
    
    script = temporary_python_script(
        f"""
from pathlib import Path
import pcbnew

board_path = Path(r"{board}")
gerber_dir = Path(r"{gerber_dir}")
drill_dir = Path(r"{drill_dir}")
layer_names = {layer_list!r}

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
    if layer_id < 0:
        raise RuntimeError(f"Unknown KiCad layer: {{layer_name}}")
    plot_controller.SetLayer(layer_id)
    plot_controller.OpenPlotfile(layer_name.replace('.', '_'), pcbnew.PLOT_FORMAT_GERBER, layer_name)
    if not plot_controller.PlotLayer():
        raise RuntimeError(f"Failed to plot layer: {{layer_name}}")

plot_controller.ClosePlot()

writer = pcbnew.EXCELLON_WRITER(board)
writer.SetMapFileFormat(pcbnew.PLOT_FORMAT_GERBER)
writer.SetOptions(False, False, board.GetDesignSettings().GetAuxOrigin(), False)
writer.SetRouteModeForOvalHoles(False)
writer.CreateDrillandMapFilesSet(str(drill_dir), True, False)
"""
    )
    try:
        run_command([str(system_python), str(script)])
    finally:
        script.unlink(missing_ok=True)
    return "pcbnew"


def main() -> None:
    args = parse_args()
    project_dir = Path(args.project_dir).resolve()
    export_dir = ensure_dir(args.export_dir).resolve()

    boards = sorted(project_dir.rglob("*.kicad_pcb"))
    if not boards:
        raise RuntimeError(f"No KiCad boards found in {project_dir}")

    if args.batch_size > 0:
        boards = boards[:args.batch_size]

    kicad_cli = find_kicad_cli()
    backend = "pcbnew"

    # 循环批量导出
    for board in boards:
        board_export_dir = ensure_dir(export_dir / board.stem)
        if kicad_cli is not None:
            backend = export_with_kicad_cli(kicad_cli, board, board_export_dir, args.layers)
        else:
            backend = export_with_pcbnew(board, board_export_dir, args.layers)

    if kicad_cli is not None:
        print(f"kicad_cli={kicad_cli}")
    print(f"backend={backend}")
    print(f"boards_processed={len(boards)}")
    print(f"export_dir={export_dir}")


if __name__ == "__main__":
    main()
