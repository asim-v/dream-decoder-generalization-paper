#!/usr/bin/env python3
"""Reproduce diagnostics, validate outputs, and build the manuscript."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "analysis" / "dream-consciousness"
PAPER = ROOT / "paper"
TARGET = ROOT / "outputs" / "dream-consciousness" / "targeted-revision"


def run(*arguments: object, cwd: Path = ROOT) -> None:
    command = [str(argument) for argument in arguments]
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reuse-predictions",
        action="store_true",
        help="Reuse the released OOF predictions instead of refitting all models.",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Run the complete analysis and asset generation without Tectonic.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    python = sys.executable

    run(python, ANALYSIS / "test_targeted_revision.py")

    model_command: list[object] = [python, ANALYSIS / "run_targeted_revision.py"]
    if not args.reuse_predictions:
        model_command.append("--force-predictions")
    run(*model_command)

    run(python, ANALYSIS / "validate_targeted_revision.py")
    run(
        python,
        ANALYSIS / "generate_manuscript_assets.py",
        "--results-dir",
        TARGET,
        "--legacy-dir",
        ROOT / "outputs" / "dream-consciousness",
        "--destination",
        PAPER,
    )

    figures = PAPER / "figures"
    figures.mkdir(exist_ok=True)
    for name in ("site-pair-heatmaps.png", "target-laboratory-forest.png"):
        shutil.copy2(TARGET / name, figures / name)

    if args.skip_pdf:
        print("Analysis, validation, tables, and figures reproduced.", flush=True)
        return

    tectonic = shutil.which("tectonic")
    if tectonic is None:
        raise SystemExit(
            "Tectonic was not found. Create the environment from environment.yml "
            "or rerun with --skip-pdf."
        )
    run(tectonic, "main.tex", "--keep-logs", "--keep-intermediates", cwd=PAPER)
    shutil.copy2(PAPER / "main.pdf", PAPER / "paper.pdf")
    print(f"Reproduction complete: {PAPER / 'paper.pdf'}", flush=True)


if __name__ == "__main__":
    main()
