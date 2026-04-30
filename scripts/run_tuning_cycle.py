#!/usr/bin/env python3
"""
Run full Lemma tuning cycle and write timestamped artifacts.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def run_cmd(args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run staged Lemma tuning cycle")
    parser.add_argument("--log", default="data/round_results.jsonl")
    parser.add_argument("--validator-b-log", default="")
    parser.add_argument("--metrics-url", default="")
    parser.add_argument("--tail", type=int, default=5000)
    parser.add_argument("--target-solve-rate", type=float, default=0.60)
    parser.add_argument("--min-multiplier", type=float, default=0.5)
    parser.add_argument("--max-multiplier", type=float, default=1.5)
    parser.add_argument("--artifacts-dir", default="artifacts/tuning")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / args.artifacts_dir / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    tuning_cmd = [
        sys.executable,
        "scripts/tuning_report.py",
        "--log",
        args.log,
        "--tail",
        str(args.tail),
        "--json",
    ]
    if args.validator_b_log:
        tuning_cmd.extend(["--validator-b-log", args.validator_b_log])
    if args.metrics_url:
        tuning_cmd.extend(["--metrics-url", args.metrics_url])

    rc_tune, out_tune, err_tune = run_cmd(tuning_cmd)
    (out_dir / "tuning_report.json").write_text(out_tune or "{}", encoding="utf-8")
    if err_tune.strip():
        (out_dir / "tuning_report.stderr.log").write_text(err_tune, encoding="utf-8")

    patch_path = out_dir / "difficulty.patch"
    patch_cmd = [
        sys.executable,
        "scripts/generate_difficulty_patch.py",
        "--log",
        args.log,
        "--tail",
        str(args.tail),
        "--target-solve-rate",
        str(args.target_solve_rate),
        "--min-multiplier",
        str(args.min_multiplier),
        "--max-multiplier",
        str(args.max_multiplier),
        "--output-patch",
        str(patch_path),
    ]
    rc_patch, out_patch, err_patch = run_cmd(patch_cmd)
    if out_patch.strip():
        (out_dir / "difficulty_patch.stdout.log").write_text(out_patch, encoding="utf-8")
    if err_patch.strip():
        (out_dir / "difficulty_patch.stderr.log").write_text(err_patch, encoding="utf-8")

    summary: dict[str, Any] = {
        "timestamp_utc": ts,
        "artifacts_dir": str(out_dir),
        "tuning_report_exit_code": rc_tune,
        "difficulty_patch_exit_code": rc_patch,
        "inputs": {
            "log": args.log,
            "validator_b_log": args.validator_b_log,
            "metrics_url": args.metrics_url,
            "tail": args.tail,
            "target_solve_rate": args.target_solve_rate,
            "min_multiplier": args.min_multiplier,
            "max_multiplier": args.max_multiplier,
        },
        "files": {
            "tuning_report_json": str(out_dir / "tuning_report.json"),
            "difficulty_patch": str(patch_path),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    decision = "UNKNOWN"
    try:
        report = json.loads((out_dir / "tuning_report.json").read_text(encoding="utf-8"))
        decision = str(report.get("decision", "UNKNOWN"))
    except Exception:
        pass

    print(f"tuning_cycle_artifacts: {out_dir}")
    print(f"decision: {decision}")
    print(f"tuning_report_exit_code: {rc_tune}")
    print(f"difficulty_patch_exit_code: {rc_patch}")


if __name__ == "__main__":
    main()
