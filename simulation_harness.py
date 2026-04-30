from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

from challenge_engine import choose_challenges


@dataclass(frozen=True)
class MinerProfile:
    uid: int
    name: str
    base_valid_rate: float
    base_elegance: float
    jitter: float = 0.05


def _deterministic_rng(seed: int, *parts: object) -> random.Random:
    material = "|".join(str(p) for p in (seed, *parts)).encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    return random.Random(int(digest[:16], 16))


def _score_attempt(
    profile: MinerProfile,
    difficulty: float,
    seed: int,
    block: int,
    challenge_id: str,
) -> tuple[float, float]:
    rng = _deterministic_rng(seed, profile.uid, block, challenge_id)
    valid_rate = max(0.0, min(1.0, profile.base_valid_rate - 0.15 * (difficulty - 1.0)))
    valid_flag = 1.0 if rng.random() < valid_rate else 0.0

    noise = rng.uniform(-profile.jitter, profile.jitter)
    elegance = max(0.0, min(1.0, profile.base_elegance + noise - 0.08 * (difficulty - 1.0)))
    if valid_flag == 0.0:
        elegance = 0.0
    return valid_flag, elegance


def _aggregate(valids: list[float], elegances: list[float], diffs: list[float]) -> float:
    if not valids:
        return 0.0
    denom = sum(max(d, 0.0) for d in diffs)
    if denom <= 0:
        return 0.0
    weighted_valid = sum(v * d for v, d in zip(valids, diffs)) / denom
    weighted_elegance = sum(e * d for e, d in zip(elegances, diffs)) / denom
    return weighted_valid * weighted_elegance


def _normalize(scores: dict[int, float]) -> dict[int, float]:
    total = sum(max(v, 0.0) for v in scores.values())
    if total <= 0:
        if not scores:
            return {}
        u = 1.0 / len(scores)
        return {uid: u for uid in scores}
    return {uid: max(v, 0.0) / total for uid, v in scores.items()}


def run_simulation(
    *,
    seed: int,
    netuid: int,
    start_block: int,
    rounds: int,
    batch_size: int,
    ema_alpha: float,
    miners: list[MinerProfile],
) -> dict[int, float]:
    ema_scores = {m.uid: 0.0 for m in miners}
    current_block = start_block

    for _ in range(rounds):
        challenges = choose_challenges(current_block, netuid, batch_size)
        for miner in miners:
            valids: list[float] = []
            elegances: list[float] = []
            diffs: list[float] = []
            for ch in challenges:
                v, e = _score_attempt(
                    profile=miner,
                    difficulty=ch.difficulty,
                    seed=seed,
                    block=current_block,
                    challenge_id=ch.challenge_id,
                )
                valids.append(v)
                elegances.append(e)
                diffs.append(ch.difficulty)
            instant = _aggregate(valids, elegances, diffs)
            ema_scores[miner.uid] = ema_alpha * instant + (1.0 - ema_alpha) * ema_scores[miner.uid]
        current_block += 12

    return _normalize(ema_scores)
