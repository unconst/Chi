#!/usr/bin/env python3
"""
Create a pass/fail soak snapshot from recent round logs and live metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for p in (ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analyze_round_logs import concentration_summary, load_rows, uid_metrics  # type: ignore
from check_alerts import evaluate, fetch_metrics  # type: ignore


def filter_recent_rows(rows: list[dict[str, Any]], minutes: int) -> list[dict[str, Any]]:
    cutoff_ns = time.time_ns() - minutes * 60 * 1_000_000_000
    out: list[dict[str, Any]] = []
    for row in rows:
        ts = row.get("timestamp_ns")
        if isinstance(ts, int) and ts >= cutoff_ns:
            out.append(row)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate post-soak pass/fail report")
    parser.add_argument("--log", default="data/round_results.jsonl", help="Round log path")
    parser.add_argument("--metrics-url", default="http://127.0.0.1:9109/metrics.json", help="Metrics JSON URL")
    parser.add_argument("--minutes", type=int, default=60, help="Lookback window in minutes")
    parser.add_argument("--tail", type=int, default=0, help="Optional initial tail limit before time filtering")
    parser.add_argument("--max-top1-share", type=float, default=0.45)
    parser.add_argument("--max-hhi", type=float, default=0.30)
    parser.add_argument("--max-invalid-rate", type=float, default=0.40)
    parser.add_argument("--max-epistula-fail-rate", type=float, default=0.05)
    parser.add_argument("--max-timeout-rate", type=float, default=0.10)
    parser.add_argument("--max-http-error-rate", type=float, default=0.10)
    parser.add_argument("--max-weight-fail-rate", type=float, default=0.20)
    parser.add_argument("--min-uid-ema", type=float, default=0.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.log), tail=args.tail if args.tail > 0 else None)
    recent = filter_recent_rows(rows, args.minutes)

    uid_rows = uid_metrics(recent)
    concentration = concentration_summary(uid_rows)

    metrics = fetch_metrics(args.metrics_url, timeout=5.0)
    breaches = evaluate(metrics, args)
    reasons: list[str] = list(breaches)
    passed = True

    if concentration.get("top_1_share", 0.0) > args.max_top1_share:
        passed = False
        reasons.append(
            f"top_1_share {concentration['top_1_share']:.3f} exceeds {args.max_top1_share:.3f}"
        )
    if concentration.get("hhi", 0.0) > args.max_hhi:
        passed = False
        reasons.append(f"hhi {concentration['hhi']:.3f} exceeds {args.max_hhi:.3f}")
    if breaches:
        passed = False

    report = {
        "window_minutes": args.minutes,
        "rows_considered": len(recent),
        "decision": "PASS" if passed else "FAIL",
        "reasons": reasons,
        "concentration": concentration,
        "uid_metrics": uid_rows,
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"post_soak_decision: {report['decision']}")
        print(f"window_minutes: {report['window_minutes']}")
        print(f"rows_considered: {report['rows_considered']}")
        print(f"top_1_share: {concentration.get('top_1_share', 0.0):.3f}")
        print(f"hhi: {concentration.get('hhi', 0.0):.3f}")
        if reasons:
            print("reasons:")
            for reason in reasons:
                print(f"- {reason}")
        else:
            print("reasons: none")

    if not passed:
        sys.exit(2)


if __name__ == "__main__":
    main()
