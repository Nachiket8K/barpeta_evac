from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence


DEFAULT_WATER_THRESHOLDS = "0.99,0.95,0.9,0.85,0.8,0.75,0.6,0.5"
DEFAULT_SEEDS = "1,3,5,7,21,42,57,100"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run Barpeta threshold×seed scenario sweeps in one command."
    )
    ap.add_argument(
        "--run",
        type=str,
        default="all",
        choices=["all", "split-thresholds"],
        help="all = one rebuild call with all thresholds; split-thresholds = one rebuild call per threshold.",
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
        help="Clear output folder before rebuilding. In split-threshold mode, applied only on first job.",
    )
    ap.add_argument(
        "--scenarios-subdir",
        type=str,
        default="scenarios_wot_seed_sweep",
        help="Output scenarios subdirectory under docs/.",
    )
    ap.add_argument(
        "--seeds",
        type=str,
        default=DEFAULT_SEEDS,
        help="Comma-separated seeds (default requested set: 1,3,5,7,21,42,57,100).",
    )
    ap.add_argument(
        "--water-over-thresholds",
        type=str,
        default=DEFAULT_WATER_THRESHOLDS,
        help="Comma-separated water_over_threshold values.",
    )
    ap.add_argument(
        "--p-backgrounds",
        type=str,
        default="0.02",
        help="Comma-separated p_background values to include.",
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
    ap.add_argument(
        "--max-export-file-mb",
        type=float,
        default=95.0,
        help="Forwarded soft max file size for chunked GeoJSON exports.",
    )
    ap.add_argument(
        "--warn-file-over-mb",
        type=float,
        default=95.0,
        help="Forwarded warn threshold for generated scenario file sizes (<=0 disables).",
    )
    ap.add_argument(
        "--fail-file-over-mb",
        type=float,
        default=100.0,
        help="Forwarded hard-fail threshold for generated scenario file sizes (<=0 disables).",
    )
    return ap.parse_args()


def _parse_float_list(value: str) -> list[float]:
    vals: list[float] = []
    for s in str(value).split(","):
        t = s.strip()
        if not t:
            continue
        vals.append(float(t))
    if not vals:
        raise ValueError("Expected at least one float in --water-over-thresholds")
    return vals


def _build_base_cmd(
    rebuild_script: Path,
    *,
    accel_mode: str,
    scenarios_subdir: str,
    seeds: str,
    p_backgrounds: str,
    max_export_file_mb: float,
    warn_file_over_mb: float,
    fail_file_over_mb: float,
) -> list[str]:
    return [
        str(rebuild_script),
        "--scenarios-subdir",
        str(scenarios_subdir),
        "--seeds",
        str(seeds),
        "--p-backgrounds",
        str(p_backgrounds),
        "--accel-mode",
        str(accel_mode),
        "--max-export-file-mb",
        f"{float(max_export_file_mb):.6g}",
        "--warn-file-over-mb",
        f"{float(warn_file_over_mb):.6g}",
        "--fail-file-over-mb",
        f"{float(fail_file_over_mb):.6g}",
    ]


def _build_commands(
    rebuild_script: Path,
    *,
    run_mode: str,
    accel_mode: str,
    clear_output: bool,
    scenarios_subdir: str,
    seeds: str,
    water_over_thresholds: str,
    p_backgrounds: str,
    max_export_file_mb: float,
    warn_file_over_mb: float,
    fail_file_over_mb: float,
) -> list[list[str]]:
    base = _build_base_cmd(
        rebuild_script,
        accel_mode=accel_mode,
        scenarios_subdir=scenarios_subdir,
        seeds=seeds,
        p_backgrounds=p_backgrounds,
        max_export_file_mb=max_export_file_mb,
        warn_file_over_mb=warn_file_over_mb,
        fail_file_over_mb=fail_file_over_mb,
    )

    cmds: list[list[str]] = []
    if run_mode == "split-thresholds":
        thr_values = _parse_float_list(water_over_thresholds)
        for i, thr in enumerate(thr_values):
            cmd = [
                *base,
                "--water-over-thresholds",
                f"{thr:.6g}",
            ]
            if clear_output and i == 0:
                cmd.append("--clear-output")
            cmds.append(cmd)
    else:
        cmd = [
            *base,
            "--water-over-thresholds",
            str(water_over_thresholds),
        ]
        if clear_output:
            cmd.append("--clear-output")
        cmds.append(cmd)

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
        run_mode=args.run,
        accel_mode=args.accel_mode,
        clear_output=bool(args.clear_output),
        scenarios_subdir=args.scenarios_subdir,
        seeds=args.seeds,
        water_over_thresholds=args.water_over_thresholds,
        p_backgrounds=args.p_backgrounds,
        max_export_file_mb=float(args.max_export_file_mb),
        warn_file_over_mb=float(args.warn_file_over_mb),
        fail_file_over_mb=float(args.fail_file_over_mb),
    )

    selected_cmds: list[list[str]] = all_cmds

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
