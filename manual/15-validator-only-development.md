# Validator-Only Development Rule

This document establishes the core development philosophy for building subnets in this template.

## The Rule

**When building a subnet, ONLY write the validator code. Never write miner code.**

Miners will write their own implementation based on how the validator is structured. The validator defines the rules of the game—miners figure out how to play.

## Why This Matters

### 1. Validators Are the Referee

The validator's job is to:
- Define what inputs it expects from miners
- Specify where it looks for those inputs (endpoints, data sources, chain commits)
- Establish how responses are evaluated and scored
- Set weights based on performance

The validator does NOT need to:
- Implement reference miner solutions
- Prescribe how miners should build their systems
- Provide miner starter code
- Dictate technology choices for miners

### 2. Miners Compete Through Innovation

If you write the miner code, you've removed the competitive element. Miners should:
- Read the validator to understand what's valued
- Design their own solutions to optimize for those criteria
- Innovate in ways you never anticipated
- Compete on quality, speed, cost, or whatever the validator rewards

The best subnet designs create clear rules but leave room for creative solutions.

### 3. Simplicity Breeds Auditability

Complex multi-file subnets with both miner and validator code become:
- Harder to audit
- More likely to contain bugs
- Confusing for miners to understand
- Difficult to maintain

A single-file validator is easy to read, easy to verify, and easy to understand.

## Single-File Validator Pattern

All subnet development in this template goes into `validator.py`. This file should contain:

```python
# validator.py - The ONLY file you write for your subnet

"""
Your subnet in a single file.

MINER INTERFACE:
- Endpoint: POST /your-endpoint
- Expected payload: {...}
- Expected response: {...}
- Evaluation criteria: ...

Miners: Read this file to understand what the validator expects.
"""

# 1. Configuration and constants
# 2. Miner interface definitions (what you expect from miners)
# 3. Scoring/evaluation logic
# 4. Weight setting logic
# 5. Main validation loop
```

## What Goes in validator.py

### 1. Miner Interface Specification

Define clearly what the validator looks for:

```python
# What endpoint do miners expose?
MINER_ENDPOINT = "/compute"

# What payload does the validator send?
REQUEST_SCHEMA = {
    "task_id": "string",
    "input_data": "bytes",
    "deadline": "timestamp"
}

# What response does the validator expect?
RESPONSE_SCHEMA = {
    "task_id": "string",
    "output_data": "bytes",
    "proof": "optional[string]"
}
```

### 2. Discovery Method

How does the validator find miner endpoints?

```python
# Option A: Committed data on chain (recommended)
def get_miner_endpoint(subtensor, netuid, hotkey):
    commitment = subtensor.get_commitment(netuid, hotkey)
    return commitment.get("api_url")

# Option B: External registry
def get_miner_endpoint(hotkey):
    return registry.lookup(hotkey)

# Option C: Well-known endpoint pattern
def get_miner_endpoint(hotkey):
    # Miners register their URLs in a known format
    return miner_registry.get(hotkey)
```

### 3. Evaluation Logic

How are miner responses scored?

```python
def score_response(response, expected) -> float:
    """
    This is the heart of your subnet.
    Miners optimize for whatever you measure here.
    """
    score = 0.0
    
    # Correctness
    if verify_output(response.output_data, expected):
        score += 0.7
    
    # Speed bonus
    if response.latency < FAST_THRESHOLD:
        score += 0.2
    
    # Proof bonus
    if response.proof and verify_proof(response.proof):
        score += 0.1
    
    return score
```

### 4. Weight Setting

How do scores become weights?

```python
def calculate_weights(scores: dict[int, float]) -> tuple[list, list]:
    """Convert scores to normalized weights"""
    uids = list(scores.keys())
    raw = [scores[uid] for uid in uids]
    total = sum(raw)
    weights = [w / total for w in raw] if total > 0 else [1/len(uids)] * len(uids)
    return uids, weights
```

## What Does NOT Go in validator.py

- Miner implementation code
- Reference solutions
- "Example miner" code
- Miner-side utilities
- Shared synapse definitions (miners can read the validator to understand the interface)

## Template Structure

```
your-subnet/
├── validator.py      # <-- ALL your code goes here
├── pyproject.toml    # Dependencies
├── Dockerfile        # Deployment
├── docker-compose.yml
├── env.example
└── docs/             # Documentation (for you, not for miners)
```

## Example: Minimal Validator

```python
#!/usr/bin/env python3
"""
Example Subnet Validator

MINER INTERFACE:
- Miners expose: GET /status returning {"available": bool, "capacity": int}
- Miners expose: POST /compute accepting {"data": str} returning {"result": str}
- Scoring: 70% correctness, 20% speed, 10% availability
"""

import asyncio
import aiohttp
from bittensor import Subtensor, Metagraph
from bittensor_wallet import Wallet

# === CONFIGURATION ===
NETUID = 1
NETWORK = "finney"
QUERY_INTERVAL = 60
WEIGHT_INTERVAL = 360

# === MINER INTERFACE ===
STATUS_ENDPOINT = "/status"
COMPUTE_ENDPOINT = "/compute"

# === VALIDATOR ===
class Validator:
    def __init__(self):
        self.wallet = Wallet(name="validator", hotkey="default")
        self.subtensor = Subtensor(network=NETWORK)
        self.metagraph = Metagraph(netuid=NETUID, network=NETWORK)
        self.scores = {}
    
    def get_miner_url(self, uid: int, endpoint: str) -> str:
        hotkey = self.metagraph.hotkeys[uid]
        # Get miner's committed endpoint from chain or registry
        base_url = self.get_committed_endpoint(hotkey)
        return f"{base_url}{endpoint}" if base_url else None
    
    async def query_miner(self, uid: int, payload: dict) -> dict | None:
        url = self.get_miner_url(uid, COMPUTE_ENDPOINT)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=30) as resp:
                    if resp.status == 200:
                        return await resp.json()
        except:
            pass
        return None
    
    def score_response(self, response: dict | None, expected: str) -> float:
        if response is None:
            return 0.0
        
        result = response.get("result", "")
        # Your scoring logic here
        correctness = 1.0 if result == expected else 0.0
        return correctness
    
    async def run(self):
        while True:
            self.metagraph.sync()
            
            # Query miners
            for uid in range(self.metagraph.n):
                miner_url = self.get_miner_url(uid, "")
                if not miner_url:
                    continue
                
                payload = {"data": "test_input"}
                response = await self.query_miner(uid, payload)
                score = self.score_response(response, expected="test_output")
                
                # EMA update
                alpha = 0.1
                self.scores[uid] = alpha * score + (1 - alpha) * self.scores.get(uid, 0.5)
            
            # Set weights periodically
            if self.scores:
                uids = list(self.scores.keys())
                weights = [self.scores[u] for u in uids]
                total = sum(weights)
                if total > 0:
                    weights = [w / total for w in weights]
                    self.subtensor.set_weights(
                        wallet=self.wallet,
                        netuid=NETUID,
                        uids=uids,
                        weights=weights
                    )
            
            await asyncio.sleep(QUERY_INTERVAL)

if __name__ == "__main__":
    validator = Validator()
    asyncio.run(validator.run())
```

## Communication to Miners

Your validator.py file IS the specification for miners. Include a docstring at the top that explains:

1. **Endpoints**: What endpoints miners must expose
2. **Request format**: What the validator sends
3. **Response format**: What the validator expects back
4. **Scoring criteria**: How responses are evaluated (be transparent!)

Miners read your validator.py to understand the game. Make it clear.

## Summary

| Do | Don't |
|----|-------|
| Write validator.py only | Write miner code |
| Define clear interfaces | Implement miner solutions |
| Document scoring criteria | Hide evaluation logic |
| Keep it single-file | Spread across many files |
| Let miners innovate | Prescribe implementations |
