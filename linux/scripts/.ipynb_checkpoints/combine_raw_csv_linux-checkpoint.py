from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine Linux raw capture CSV files.")
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    csv_paths = sorted(input_dir.glob("*.csv"))
    if not csv_paths:
        raise SystemExit("No CSV files found.")

    combined = pd.concat([pd.read_csv(path) for path in csv_paths], ignore_index=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False)
    print(f"rows={len(combined)}")
    print(f"sessions={combined['session_id'].nunique()}")
    print(f"output={output}")


if __name__ == "__main__":
    main()
