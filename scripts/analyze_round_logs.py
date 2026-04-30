#!/usr/bin/env python3
"""
Analyze Lemma validator round telemetry JSONL logs.
"""

from __future__ import annotations

import argparse
import json
import math
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


def uid_metrics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        uid = row.get("uid")
        if isinstance(uid, int):
            grouped[uid].append(row)

    output: list[dict[str, Any]] = []
    for uid, items in grouped.items():
        total = len(items)
        valid_sum = sum(float(x.get("valid_flag", 0.0)) for x in items)
        elegance_sum = sum(float(x.get("elegance_score", 0.0)) for x in items)
        instant_scores = [float(x.get("valid_flag", 0.0)) * float(x.get("elegance_score", 0.0)) for x in items]
        instant_mean = sum(instant_scores) / total if total else 0.0
        invalid_rate = 1.0 - (valid_sum / total if total else 0.0)

        reason_counts: dict[str, int] = defaultdict(int)
        for x in items:
            reason = str(x.get("reason", "unknown"))
            reason_counts[reason] += 1
        top_reason = max(reason_counts.items(), key=lambda t: t[1])[0] if reason_counts else "n/a"

        output.append(
            {
                "uid": uid,
                "samples": total,
                "valid_rate": valid_sum / total if total else 0.0,
                "invalid_rate": invalid_rate,
                "avg_elegance": elegance_sum / total if total else 0.0,
                "avg_instant_score": instant_mean,
                "top_failure_reason": top_reason,
            }
        )
    output.sort(key=lambda x: x["avg_instant_score"], reverse=True)
    return output


def concentration_summary(uid_rows: list[dict[str, Any]]) -> dict[str, float]:
    scores = [max(float(x["avg_instant_score"]), 0.0) for x in uid_rows]
    total = sum(scores)
    if total <= 0:
        return {"top_1_share": 0.0, "top_3_share": 0.0, "hhi": 0.0}
    shares = sorted([s / total for s in scores], reverse=True)
    top_1 = shares[0] if shares else 0.0
    top_3 = sum(shares[:3])
    hhi = sum(s * s for s in shares)
    return {"top_1_share": top_1, "top_3_share": top_3, "hhi": hhi}


def print_report(rows: list[dict[str, Any]], uid_rows: list[dict[str, Any]]) -> None:
    print(f"rows_analyzed: {len(rows)}")
    if not rows:
        return
    blocks = [int(r.get("block", 0)) for r in rows if isinstance(r.get("block"), int)]
    if blocks:
        print(f"block_range: {min(blocks)} -> {max(blocks)}")
    print()
    print("Per-UID Metrics")
    print("uid  samples  valid_rate  avg_elegance  avg_instant  top_failure_reason")
    for m in uid_rows:
        print(
            f"{m['uid']:>3}  {m['samples']:>7}  {m['valid_rate']:.3f}      "
            f"{m['avg_elegance']:.3f}        {m['avg_instant_score']:.3f}      "
            f"{m['top_failure_reason']}"
        )
    print()
    c = concentration_summary(uid_rows)
    print("Concentration")
    print(f"top_1_share: {c['top_1_share']:.3f}")
    print(f"top_3_share: {c['top_3_share']:.3f}")
    print(f"hhi: {c['hhi']:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Lemma round log JSONL")
    parser.add_argument("--log", default="data/round_results.jsonl", help="Path to round JSONL log")
    parser.add_argument("--tail", type=int, default=0, help="Analyze only the last N log lines")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output")
    args = parser.parse_args()

    rows = load_rows(Path(args.log), tail=args.tail if args.tail > 0 else None)
    metrics = uid_metrics(rows)
    concentration = concentration_summary(metrics)

    if args.json:
        payload = {
            "rows_analyzed": len(rows),
            "uid_metrics": metrics,
            "concentration": concentration,
        }
        print(json.dumps(payload, indent=2))
    else:
        print_report(rows, metrics)


if __name__ == "__main__":
    main()
