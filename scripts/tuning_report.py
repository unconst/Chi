#!/usr/bin/env python3
"""
Generate a staged tuning go/hold report from logs and optional live metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for p in (ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analyze_round_logs import concentration_summary, load_rows, uid_metrics  # type: ignore
from check_alerts import evaluate, fetch_metrics  # type: ignore
from compare_validator_logs import compare_rows, index_rows  # type: ignore


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    rows = load_rows(Path(args.log), tail=args.tail if args.tail > 0 else None)
    uid_rows = uid_metrics(rows)
    concentration = concentration_summary(uid_rows)

    metrics_section: dict[str, Any] = {"available": False, "breaches": []}
    if args.metrics_url:
        metrics = fetch_metrics(args.metrics_url, args.timeout)
        breaches = evaluate(metrics, args)
        metrics_section = {"available": True, "breaches": breaches, "snapshot": metrics}

    disagreement_section: dict[str, Any] = {"available": False}
    if args.validator_b_log:
        left = index_rows(rows)
        right_rows = load_rows(Path(args.validator_b_log), tail=args.tail if args.tail > 0 else None)
        right = index_rows(right_rows)
        summary = compare_rows(left, right)
        disagreement_section = {"available": True, "summary": summary}

    reasons: list[str] = []
    go = True

    if concentration.get("top_1_share", 0.0) > args.max_top1_share:
        go = False
        reasons.append(
            f"top_1_share {concentration['top_1_share']:.3f} exceeds threshold {args.max_top1_share:.3f}"
        )
    if concentration.get("hhi", 0.0) > args.max_hhi:
        go = False
        reasons.append(f"hhi {concentration['hhi']:.3f} exceeds threshold {args.max_hhi:.3f}")

    if metrics_section["available"] and metrics_section["breaches"]:
        go = False
        reasons.extend(metrics_section["breaches"])

    if disagreement_section["available"]:
        agreement = disagreement_section["summary"].get("valid_flag_agreement", 0.0)
        if agreement < args.min_validator_agreement:
            go = False
            reasons.append(
                f"valid_flag_agreement {agreement:.3f} below threshold {args.min_validator_agreement:.3f}"
            )

    return {
        "decision": "GO" if go else "HOLD",
        "reasons": reasons,
        "rows_analyzed": len(rows),
        "concentration": concentration,
        "uid_metrics": uid_rows,
        "metrics": metrics_section,
        "disagreement": disagreement_section,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Produce staged tuning go/hold report")
    parser.add_argument("--log", default="data/round_results.jsonl", help="Primary validator round log")
    parser.add_argument("--validator-b-log", default="", help="Optional second validator log for disagreement checks")
    parser.add_argument("--tail", type=int, default=0, help="Analyze only latest N rows")
    parser.add_argument("--metrics-url", default="", help="Optional metrics JSON endpoint")
    parser.add_argument("--timeout", type=float, default=5.0, help="Metrics fetch timeout")
    parser.add_argument("--max-top1-share", type=float, default=0.45)
    parser.add_argument("--max-hhi", type=float, default=0.30)
    parser.add_argument("--min-validator-agreement", type=float, default=0.95)
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
    report = build_report(args)
    if args.json:
        print(json.dumps(report, indent=2))
        return

    print(f"decision: {report['decision']}")
    if report["reasons"]:
        print("reasons:")
        for reason in report["reasons"]:
            print(f"- {reason}")
    else:
        print("reasons: none")
    print(f"rows_analyzed: {report['rows_analyzed']}")
    c = report["concentration"]
    print(f"top_1_share: {c.get('top_1_share', 0.0):.3f}")
    print(f"top_3_share: {c.get('top_3_share', 0.0):.3f}")
    print(f"hhi: {c.get('hhi', 0.0):.3f}")


if __name__ == "__main__":
    main()
