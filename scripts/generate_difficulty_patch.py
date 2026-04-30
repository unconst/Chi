#!/usr/bin/env python3
"""
Generate a reviewable patch for challenge family difficulty values.
Does not modify files by default.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path(__file__).resolve().parent
for p in (ROOT, SCRIPTS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from calibrate_difficulty import calibrate, load_rows  # type: ignore


FAMILY_TO_FUNCTION = {
    "nat_add_comm": "_challenge_family_nat_add_comm",
    "nat_mul_add": "_challenge_family_nat_mul_add",
    "list_len_append": "_challenge_family_list_length_append",
}


def update_difficulties(source: str, family_to_value: dict[str, float]) -> str:
    updated = source
    for family, fn_name in FAMILY_TO_FUNCTION.items():
        if family not in family_to_value:
            continue
        target = family_to_value[family]
        pattern = rf"(def {re.escape(fn_name)}\(.*?\n(?:.|\n)*?difficulty=)([0-9]*\.?[0-9]+)"
        match = re.search(pattern, updated)
        if not match:
            continue
        start, end = match.span(2)
        updated = f"{updated[:start]}{target:.3f}{updated[end:]}"
    return updated


def unified_patch(old: str, new: str, path: str) -> str:
    import difflib

    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, fromfile=path, tofile=path)
    return "".join(diff)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate difficulty patch for challenge_engine.py")
    parser.add_argument("--log", default="data/round_results.jsonl")
    parser.add_argument("--tail", type=int, default=0)
    parser.add_argument("--target-solve-rate", type=float, default=0.60)
    parser.add_argument("--min-multiplier", type=float, default=0.5)
    parser.add_argument("--max-multiplier", type=float, default=1.5)
    parser.add_argument("--engine-path", default="challenge_engine.py")
    parser.add_argument("--output-patch", default="", help="Optional file path for patch output")
    parser.add_argument("--apply", action="store_true", help="Apply updated difficulties in place")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.log), tail=args.tail if args.tail > 0 else None)
    results = calibrate(
        rows=rows,
        target_solve_rate=args.target_solve_rate,
        min_multiplier=args.min_multiplier,
        max_multiplier=args.max_multiplier,
    )
    family_to_value = {row["family"]: float(row["suggested_difficulty"]) for row in results}

    engine_path = Path(args.engine_path)
    old = engine_path.read_text(encoding="utf-8")
    new = update_difficulties(old, family_to_value)
    patch = unified_patch(old, new, str(engine_path))

    if args.output_patch:
        Path(args.output_patch).write_text(patch, encoding="utf-8")
        print(f"wrote patch to {args.output_patch}")

    if args.apply:
        engine_path.write_text(new, encoding="utf-8")
        print(f"applied difficulty updates to {engine_path}")
    else:
        print(patch if patch else "no changes generated")


if __name__ == "__main__":
    main()
