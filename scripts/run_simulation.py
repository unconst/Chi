#!/usr/bin/env python3
"""
Run Lemma simulation harness from CLI.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation_harness import MinerProfile, run_simulation


PRESET_SCENARIOS = {
    "baseline": [
        MinerProfile(uid=11, name="honest", base_valid_rate=0.92, base_elegance=0.84),
        MinerProfile(uid=12, name="noisy", base_valid_rate=0.70, base_elegance=0.66),
        MinerProfile(uid=13, name="adversarial", base_valid_rate=0.45, base_elegance=0.52),
    ],
    "tight_competition": [
        MinerProfile(uid=21, name="miner_a", base_valid_rate=0.89, base_elegance=0.81),
        MinerProfile(uid=22, name="miner_b", base_valid_rate=0.87, base_elegance=0.80),
        MinerProfile(uid=23, name="miner_c", base_valid_rate=0.85, base_elegance=0.78),
    ],
    "adversarial_heavy": [
        MinerProfile(uid=31, name="honest", base_valid_rate=0.90, base_elegance=0.83),
        MinerProfile(uid=32, name="copycat", base_valid_rate=0.55, base_elegance=0.58),
        MinerProfile(uid=33, name="unstable", base_valid_rate=0.40, base_elegance=0.50, jitter=0.10),
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Lemma deterministic simulation scenario")
    parser.add_argument(
        "--scenario",
        choices=sorted(PRESET_SCENARIOS.keys()),
        default="baseline",
        help="Preset miner scenario",
    )
    parser.add_argument("--seed", type=int, default=42, help="Simulation RNG seed")
    parser.add_argument("--netuid", type=int, default=1, help="Subnet netuid for challenge generation seed")
    parser.add_argument("--start-block", type=int, default=1000, help="Starting block number")
    parser.add_argument("--rounds", type=int, default=60, help="Number of rounds")
    parser.add_argument("--batch-size", type=int, default=6, help="Challenge batch size per round")
    parser.add_argument("--ema-alpha", type=float, default=0.35, help="EMA alpha for score smoothing")
    parser.add_argument("--json", action="store_true", help="Print machine-readable output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    miners = PRESET_SCENARIOS[args.scenario]
    weights = run_simulation(
        seed=args.seed,
        netuid=args.netuid,
        start_block=args.start_block,
        rounds=args.rounds,
        batch_size=args.batch_size,
        ema_alpha=args.ema_alpha,
        miners=miners,
    )
    sorted_weights = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)

    if args.json:
        payload = {
            "scenario": args.scenario,
            "seed": args.seed,
            "rounds": args.rounds,
            "weights": [{"uid": uid, "weight": weight} for uid, weight in sorted_weights],
        }
        print(json.dumps(payload, indent=2))
        return

    print(f"scenario: {args.scenario}")
    print(f"seed: {args.seed}")
    print(f"rounds: {args.rounds}")
    print("final_weights:")
    for uid, weight in sorted_weights:
        print(f"  uid={uid}: {weight:.6f}")


if __name__ == "__main__":
    main()
