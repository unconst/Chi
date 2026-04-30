#!/usr/bin/env python3
"""
Evaluate validator metrics against alert thresholds.
Exits non-zero when any threshold is breached.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from typing import Any


def fetch_metrics(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("metrics payload must be a JSON object")
    return payload


def safe_rate(numer: float, denom: float) -> float:
    return numer / denom if denom > 0 else 0.0


def evaluate(metrics: dict[str, Any], args: argparse.Namespace) -> list[str]:
    breaches: list[str] = []
    attempts = float(metrics.get("attempts_total", 0.0))
    invalid = float(metrics.get("invalid_total", 0.0))
    epistula_fail = float(metrics.get("epistula_fail_total", 0.0))
    timeout_total = float(metrics.get("timeout_total", 0.0))
    http_errors = float(metrics.get("http_error_total", 0.0))
    weight_ok = float(metrics.get("weights_set_success_total", 0.0))
    weight_fail = float(metrics.get("weights_set_fail_total", 0.0))

    invalid_rate = safe_rate(invalid, attempts)
    epistula_fail_rate = safe_rate(epistula_fail, attempts)
    timeout_rate = safe_rate(timeout_total, attempts)
    http_error_rate = safe_rate(http_errors, attempts)
    weight_fail_rate = safe_rate(weight_fail, weight_ok + weight_fail)

    if invalid_rate > args.max_invalid_rate:
        breaches.append(
            f"invalid_rate {invalid_rate:.4f} exceeds max_invalid_rate {args.max_invalid_rate:.4f}"
        )
    if epistula_fail_rate > args.max_epistula_fail_rate:
        breaches.append(
            f"epistula_fail_rate {epistula_fail_rate:.4f} exceeds max_epistula_fail_rate {args.max_epistula_fail_rate:.4f}"
        )
    if timeout_rate > args.max_timeout_rate:
        breaches.append(
            f"timeout_rate {timeout_rate:.4f} exceeds max_timeout_rate {args.max_timeout_rate:.4f}"
        )
    if http_error_rate > args.max_http_error_rate:
        breaches.append(
            f"http_error_rate {http_error_rate:.4f} exceeds max_http_error_rate {args.max_http_error_rate:.4f}"
        )
    if weight_fail_rate > args.max_weight_fail_rate:
        breaches.append(
            f"weight_fail_rate {weight_fail_rate:.4f} exceeds max_weight_fail_rate {args.max_weight_fail_rate:.4f}"
        )

    per_uid = metrics.get("per_uid", {})
    if isinstance(per_uid, dict):
        for uid, vals in per_uid.items():
            if not isinstance(vals, dict):
                continue
            ema = float(vals.get("ema", 0.0))
            if ema < args.min_uid_ema:
                breaches.append(f"uid={uid} ema {ema:.4f} below min_uid_ema {args.min_uid_ema:.4f}")
    return breaches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Lemma validator metrics against alert thresholds")
    parser.add_argument("--url", default="http://127.0.0.1:9109/metrics.json", help="Metrics JSON endpoint")
    parser.add_argument("--timeout", type=float, default=5.0, help="HTTP timeout seconds")
    parser.add_argument("--max-invalid-rate", type=float, default=0.40)
    parser.add_argument("--max-epistula-fail-rate", type=float, default=0.05)
    parser.add_argument("--max-timeout-rate", type=float, default=0.10)
    parser.add_argument("--max-http-error-rate", type=float, default=0.10)
    parser.add_argument("--max-weight-fail-rate", type=float, default=0.20)
    parser.add_argument("--min-uid-ema", type=float, default=0.0)
    parser.add_argument("--json", action="store_true", help="Emit JSON result")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = fetch_metrics(args.url, args.timeout)
    breaches = evaluate(metrics, args)

    if args.json:
        print(json.dumps({"ok": len(breaches) == 0, "breaches": breaches}, indent=2))
    else:
        if breaches:
            print("ALERT: threshold breaches detected")
            for b in breaches:
                print(f"- {b}")
        else:
            print("OK: all thresholds satisfied")

    if breaches:
        sys.exit(2)


if __name__ == "__main__":
    main()
