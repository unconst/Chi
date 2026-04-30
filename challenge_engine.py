from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Challenge:
    challenge_id: str
    statement: str
    lean_header: str
    lean_goal: str
    difficulty: float


def _challenge_family_nat_add_comm(rng: random.Random, idx: int) -> Challenge:
    a = rng.randint(0, 25)
    b = rng.randint(0, 25)
    return Challenge(
        challenge_id=f"nat_add_comm_{idx}_{a}_{b}",
        statement=f"Show Nat addition commutes for concrete terms ({a}, {b}).",
        lean_header="import Mathlib\n",
        lean_goal=f"theorem lemma_task : ({a} + {b} : Nat) = {b} + {a} := by\n",
        difficulty=0.8,
    )


def _challenge_family_nat_mul_add(rng: random.Random, idx: int) -> Challenge:
    a = rng.randint(0, 12)
    b = rng.randint(0, 12)
    c = rng.randint(0, 12)
    return Challenge(
        challenge_id=f"nat_mul_add_{idx}_{a}_{b}_{c}",
        statement="Distributivity of multiplication over addition on concrete terms.",
        lean_header="import Mathlib\n",
        lean_goal=f"theorem lemma_task : ({a} * ({b} + {c}) : Nat) = {a} * {b} + {a} * {c} := by\n",
        difficulty=1.0,
    )


def _challenge_family_list_length_append(rng: random.Random, idx: int) -> Challenge:
    x = rng.randint(0, 5)
    y = rng.randint(0, 5)
    return Challenge(
        challenge_id=f"list_len_append_{idx}_{x}_{y}",
        statement="Length of appended list equals sum of lengths.",
        lean_header="import Mathlib\n",
        lean_goal=(
            "theorem lemma_task : "
            f"List.length ([{x}] ++ [{y}] : List Nat) = "
            f"List.length ([{x}] : List Nat) + List.length ([{y}] : List Nat) := by\n"
        ),
        difficulty=1.2,
    )


CHALLENGE_FAMILIES = [
    _challenge_family_nat_add_comm,
    _challenge_family_nat_mul_add,
    _challenge_family_list_length_append,
]


def choose_challenges(block_number: int, netuid: int, batch_size: int) -> list[Challenge]:
    seed_material = f"{netuid}:{block_number // 5}".encode("utf-8")
    digest = hashlib.sha256(seed_material).hexdigest()
    seed = int(digest[:16], 16)
    rng = random.Random(seed)
    challenges: list[Challenge] = []
    for idx in range(max(1, batch_size)):
        family = CHALLENGE_FAMILIES[rng.randrange(len(CHALLENGE_FAMILIES))]
        challenges.append(family(rng, idx))
    return challenges


def choose_challenge(block_number: int, netuid: int) -> Challenge:
    return choose_challenges(block_number=block_number, netuid=netuid, batch_size=1)[0]
