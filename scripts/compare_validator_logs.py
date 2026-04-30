#!/usr/bin/env python3
"""
Compare two Lemma validator round JSONL logs and report disagreement metrics.
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
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def key_for_row(row: dict[str, Any]) -> tuple[int, int, str]:
    return (
        int(row.get("block", -1)),
        int(row.get("uid", -1)),
        str(row.get("challenge_id", "")),
    )


def index_rows(rows: list[dict[str, Any]]) -> dict[tuple[int, int, str], dict[str, Any]]:
    out: dict[tuple[int, int, str], dict[str, Any]] = {}
    for row in rows:
        try:
            key = key_for_row(row)
        except Exception:
            continue
        out[key] = row
    return out


def compare_rows(
    left_idx: dict[tuple[int, int, str], dict[str, Any]],
    right_idx: dict[tuple[int, int, str], dict[str, Any]],
) -> dict[str, Any]:
    shared_keys = sorted(set(left_idx.keys()) & set(right_idx.keys()))
    left_only = len(set(left_idx.keys()) - set(right_idx.keys()))
    right_only = len(set(right_idx.keys()) - set(left_idx.keys()))

    if not shared_keys:
        return {
            "shared_rows": 0,
            "left_only_rows": left_only,
            "right_only_rows": right_only,
            "valid_flag_agreement": 0.0,
            "mean_abs_elegance_delta": 0.0,
            "mean_abs_instant_delta": 0.0,
            "uid_disagreement_counts": {},
            "largest_mismatches": [],
        }

    valid_agree = 0
    abs_elegance_delta_sum = 0.0
    abs_instant_delta_sum = 0.0
    uid_disagreements: dict[int, int] = defaultdict(int)
    mismatch_rows: list[dict[str, Any]] = []

    for key in shared_keys:
        l = left_idx[key]
        r = right_idx[key]
        l_valid = float(l.get("valid_flag", 0.0))
        r_valid = float(r.get("valid_flag", 0.0))
        l_elegance = float(l.get("elegance_score", 0.0))
        r_elegance = float(r.get("elegance_score", 0.0))
        l_inst = l_valid * l_elegance
        r_inst = r_valid * r_elegance

        if l_valid == r_valid:
            valid_agree += 1
        else:
            uid_disagreements[int(key[1])] += 1

        abs_elegance_delta = abs(l_elegance - r_elegance)
        abs_instant_delta = abs(l_inst - r_inst)
        abs_elegance_delta_sum += abs_elegance_delta
        abs_instant_delta_sum += abs_instant_delta

        if l_valid != r_valid or abs_instant_delta > 1e-9:
            mismatch_rows.append(
                {
                    "block": key[0],
                    "uid": key[1],
                    "challenge_id": key[2],
                    "left_valid": l_valid,
                    "right_valid": r_valid,
                    "left_elegance": l_elegance,
                    "right_elegance": r_elegance,
                    "left_instant": l_inst,
                    "right_instant": r_inst,
                    "abs_instant_delta": abs_instant_delta,
                    "left_reason": str(l.get("reason", "")),
                    "right_reason": str(r.get("reason", "")),
                }
            )

    mismatch_rows.sort(key=lambda x: x["abs_instant_delta"], reverse=True)
    n = len(shared_keys)
    return {
        "shared_rows": n,
        "left_only_rows": left_only,
        "right_only_rows": right_only,
        "valid_flag_agreement": valid_agree / n,
        "mean_abs_elegance_delta": abs_elegance_delta_sum / n,
        "mean_abs_instant_delta": abs_instant_delta_sum / n,
        "uid_disagreement_counts": dict(sorted(uid_disagreements.items(), key=lambda t: t[1], reverse=True)),
        "largest_mismatches": mismatch_rows[:20],
    }


def print_report(summary: dict[str, Any]) -> None:
    print("Validator Disagreement Report")
    print(f"shared_rows: {summary['shared_rows']}")
    print(f"left_only_rows: {summary['left_only_rows']}")
    print(f"right_only_rows: {summary['right_only_rows']}")
    print(f"valid_flag_agreement: {summary['valid_flag_agreement']:.3f}")
    print(f"mean_abs_elegance_delta: {summary['mean_abs_elegance_delta']:.4f}")
    print(f"mean_abs_instant_delta: {summary['mean_abs_instant_delta']:.4f}")
    print()
    print("UID Disagreement Counts")
    if not summary["uid_disagreement_counts"]:
        print("none")
    else:
        for uid, count in summary["uid_disagreement_counts"].items():
            print(f"uid={uid}: {count}")
    print()
    print("Top mismatches")
    if not summary["largest_mismatches"]:
        print("none")
    else:
        for row in summary["largest_mismatches"][:10]:
            print(
                f"block={row['block']} uid={row['uid']} challenge={row['challenge_id']} "
                f"left=({row['left_valid']:.0f},{row['left_elegance']:.3f}) "
                f"right=({row['right_valid']:.0f},{row['right_elegance']:.3f}) "
                f"delta={row['abs_instant_delta']:.3f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two Lemma validator JSONL logs")
    parser.add_argument("--left", required=True, help="Path to first validator round log")
    parser.add_argument("--right", required=True, help="Path to second validator round log")
    parser.add_argument("--tail", type=int, default=0, help="Use only last N rows from each log")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output")
    args = parser.parse_args()

    tail = args.tail if args.tail > 0 else None
    left_rows = load_rows(Path(args.left), tail=tail)
    right_rows = load_rows(Path(args.right), tail=tail)
    summary = compare_rows(index_rows(left_rows), index_rows(right_rows))

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_report(summary)


if __name__ == "__main__":
    main()
