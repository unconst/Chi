#!/usr/bin/env python3
"""
Calibrate challenge-family difficulty weights from round_results JSONL logs.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_rows(path: Path, tail: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"log file not found: {path}")
    lines = path.read_text(encoding="utf-8").splitlines()
    if tail and tail > 0:
        lines = lines[-tail:]
    rows: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def challenge_family(challenge_id: str) -> str:
    parts = challenge_id.split("_")
    if len(parts) >= 3:
        return "_".join(parts[:3])
    return challenge_id


def calibrate(
    rows: list[dict[str, Any]],
    target_solve_rate: float,
    min_multiplier: float,
    max_multiplier: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cid = str(row.get("challenge_id", "")).strip()
        if not cid:
            continue
        grouped[challenge_family(cid)].append(row)

    out: list[dict[str, Any]] = []
    for family, items in grouped.items():
        n = len(items)
        solved = sum(float(x.get("valid_flag", 0.0)) for x in items)
        solve_rate = solved / n if n else 0.0
        avg_current_difficulty = (
            sum(float(x.get("challenge_difficulty", 1.0)) for x in items) / n if n else 1.0
        )
        # If solve_rate is too high, weight should go down; if too low, weight should go up.
        raw_multiplier = (
            target_solve_rate / max(solve_rate, 1e-6) if solve_rate > 0 else max_multiplier
        )
        multiplier = max(min_multiplier, min(max_multiplier, raw_multiplier))
        suggested_difficulty = avg_current_difficulty * multiplier
        out.append(
            {
                "family": family,
                "samples": n,
                "solve_rate": solve_rate,
                "avg_current_difficulty": avg_current_difficulty,
                "multiplier": multiplier,
                "suggested_difficulty": suggested_difficulty,
            }
        )
    out.sort(key=lambda x: x["solve_rate"])
    return out


def print_report(results: list[dict[str, Any]], target_solve_rate: float) -> None:
    print(f"target_solve_rate: {target_solve_rate:.3f}")
    print("family              samples  solve_rate  current_diff  multiplier  suggested_diff")
    for row in results:
        print(
            f"{row['family']:<18}  {row['samples']:>7}  {row['solve_rate']:.3f}      "
            f"{row['avg_current_difficulty']:.3f}       {row['multiplier']:.3f}       "
            f"{row['suggested_difficulty']:.3f}"
        )


def print_patch_hint(results: list[dict[str, Any]]) -> None:
    print("\nPatch Hint (challenge_engine.py)")
    print("Update family difficulty values approximately to:")
    for row in results:
        print(f"- {row['family']}: {row['suggested_difficulty']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate Lemma challenge difficulties from log data")
    parser.add_argument("--log", default="data/round_results.jsonl", help="Path to round JSONL log")
    parser.add_argument("--tail", type=int, default=0, help="Analyze only last N rows")
    parser.add_argument("--target-solve-rate", type=float, default=0.60, help="Target solve rate per family")
    parser.add_argument("--min-multiplier", type=float, default=0.5, help="Lower cap for adjustment multiplier")
    parser.add_argument("--max-multiplier", type=float, default=1.5, help="Upper cap for adjustment multiplier")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    args = parser.parse_args()

    rows = load_rows(Path(args.log), tail=args.tail if args.tail > 0 else None)
    results = calibrate(
        rows=rows,
        target_solve_rate=args.target_solve_rate,
        min_multiplier=args.min_multiplier,
        max_multiplier=args.max_multiplier,
    )

    if args.json:
        payload = {
            "target_solve_rate": args.target_solve_rate,
            "families": results,
        }
        print(json.dumps(payload, indent=2))
        return

    print_report(results, args.target_solve_rate)
    print_patch_hint(results)


if __name__ == "__main__":
    main()
