#!/usr/bin/env python3
"""
Preflight checks for Lemma validator operations.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def parse_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def env_get(key: str, env_file_values: dict[str, str], default: str = "") -> str:
    return os.getenv(key, env_file_values.get(key, default))


def check_required_env(env_file_values: dict[str, str]) -> list[str]:
    missing: list[str] = []
    for key in ("NETWORK", "NETUID", "WALLET_NAME", "HOTKEY_NAME"):
        if not env_get(key, env_file_values):
            missing.append(f"missing required env: {key}")
    return missing


def check_files(root: Path) -> list[str]:
    required = [
        root / "validator.py",
        root / "challenge_engine.py",
        root / "scripts" / "run_tuning_cycle.py",
        root / "scripts" / "check_alerts.py",
        root / "scripts" / "post_soak_report.py",
    ]
    errs: list[str] = []
    for p in required:
        if not p.exists():
            errs.append(f"missing required file: {p}")
    return errs


def check_tools(require_lean: bool) -> list[str]:
    errs: list[str] = []
    if shutil.which("python3") is None:
        errs.append("python3 not found in PATH")
    if shutil.which("uv") is None:
        errs.append("uv not found in PATH")
    if require_lean and shutil.which("lean") is None:
        errs.append("lean not found in PATH")
    return errs


def check_wallet_dir(wallet_dir: Path) -> list[str]:
    if not wallet_dir.exists():
        return [f"wallet directory not found: {wallet_dir}"]
    return []


def check_metrics(metrics_url: str, timeout_s: float) -> list[str]:
    if not metrics_url:
        return []
    try:
        with urllib.request.urlopen(metrics_url, timeout=timeout_s) as response:
            body = response.read().decode("utf-8")
        payload = json.loads(body)
        if not isinstance(payload, dict):
            return [f"metrics endpoint did not return JSON object: {metrics_url}"]
        return []
    except urllib.error.URLError as exc:
        return [f"metrics endpoint unreachable: {metrics_url} ({exc})"]
    except json.JSONDecodeError:
        return [f"metrics endpoint returned non-JSON payload: {metrics_url}"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Lemma preflight checks")
    parser.add_argument("--root", default=".", help="Repo root path")
    parser.add_argument("--env-file", default=".env", help="Path to env file")
    parser.add_argument("--wallet-dir", default=str(Path.home() / ".bittensor" / "wallets"))
    parser.add_argument("--require-lean", action="store_true", help="Fail if Lean is missing")
    parser.add_argument("--metrics-url", default="", help="Optional metrics endpoint to verify")
    parser.add_argument("--metrics-timeout", type=float, default=3.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    env_file_path = (root / args.env_file).resolve() if not Path(args.env_file).is_absolute() else Path(args.env_file)
    env_file_values = parse_env_file(env_file_path)

    errors: list[str] = []
    warnings: list[str] = []

    errors.extend(check_files(root))
    errors.extend(check_required_env(env_file_values))
    errors.extend(check_tools(require_lean=args.require_lean))
    errors.extend(check_wallet_dir(Path(args.wallet_dir)))

    if not env_file_path.exists():
        warnings.append(f"env file not found: {env_file_path}")

    metrics_errors = check_metrics(args.metrics_url, args.metrics_timeout)
    errors.extend(metrics_errors)

    result: dict[str, Any] = {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "root": str(root),
        "env_file": str(env_file_path),
    }

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"preflight_ok: {result['ok']}")
        if warnings:
            print("warnings:")
            for item in warnings:
                print(f"- {item}")
        if errors:
            print("errors:")
            for item in errors:
                print(f"- {item}")
        else:
            print("errors: none")

    if errors:
        sys.exit(2)


if __name__ == "__main__":
    main()
