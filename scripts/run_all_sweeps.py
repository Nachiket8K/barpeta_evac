from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run Barpeta scenario sweeps (A/B) in one command."
    )
    ap.add_argument(
        "--run",
        type=str,
        default="all",
        choices=["all", "sweepA", "sweepB"],
        help="Which sweep group to execute.",
    )
    ap.add_argument(
        "--accel-mode",
        type=str,
        default="auto",
        choices=["auto", "none", "numba-cpu", "numba-cuda"],
        help="Acceleration mode forwarded to rebuild_weekly_scenarios.py.",
    )
    ap.add_argument(
        "--clear-output",
        action="store_true",
        help="Clear each sweep output folder before rebuilding.",
    )
    ap.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Python interpreter to use (default: current interpreter).",
    )
    ap.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining sweeps even if one command fails.",
    )
    return ap.parse_args()


def _build_commands(rebuild_script: Path, accel_mode: str, clear_output: bool) -> list[list[str]]:
    cmds: list[list[str]] = []

    sweep_a = [
        str(rebuild_script),
        "--scenarios-subdir",
        "scenarios_sweepA_seed42_pbg",
        "--seeds",
        "42",
        "--p-backgrounds",
        "0,0.02,0.05,0.1,0.5",
        "--accel-mode",
        accel_mode,
    ]

    sweep_b = [
        str(rebuild_script),
        "--scenarios-subdir",
        "scenarios_sweepB_pbg002_seeds",
        "--seeds",
        "1,5,7,42,100",
        "--p-backgrounds",
        "0.02",
        "--accel-mode",
        accel_mode,
    ]

    if clear_output:
        sweep_a.append("--clear-output")
        sweep_b.append("--clear-output")

    cmds.append(sweep_a)
    cmds.append(sweep_b)
    return cmds


def _run_command(cmd: Sequence[str]) -> int:
    pretty = " ".join(cmd)
    print(f"\n[run] {pretty}", flush=True)
    proc = subprocess.run(cmd)
    return int(proc.returncode)


def main() -> None:
    args = _parse_args()

    project_root = Path(__file__).resolve().parents[1]
    rebuild_script = project_root / "scripts" / "rebuild_weekly_scenarios.py"
    if not rebuild_script.exists():
        raise FileNotFoundError(f"Missing script: {rebuild_script}")

    all_cmds = _build_commands(
        rebuild_script=rebuild_script,
        accel_mode=args.accel_mode,
        clear_output=bool(args.clear_output),
    )

    selected_cmds: list[list[str]]
    if args.run == "sweepA":
        selected_cmds = [all_cmds[0]]
    elif args.run == "sweepB":
        selected_cmds = [all_cmds[1]]
    else:
        selected_cmds = all_cmds

    failures = 0
    for i, cmd_tail in enumerate(selected_cmds, start=1):
        cmd = [args.python_exe, *cmd_tail]
        print(f"\n=== Sweep job {i}/{len(selected_cmds)} ===", flush=True)
        rc = _run_command(cmd)
        if rc != 0:
            failures += 1
            print(f"[error] command failed with exit code {rc}", flush=True)
            if not args.continue_on_error:
                raise SystemExit(rc)

    if failures:
        raise SystemExit(1)

    print("\nAll requested sweep jobs completed successfully.", flush=True)


if __name__ == "__main__":
    main()
