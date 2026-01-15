# Writing a Subnet

This document is the first thing you should read before building a subnet.

## The Core Question

Before writing any code, ask yourself:

**"What am I actually measuring?"**

This question determines everything. If you can't answer it in one sentence, you're not ready to build a subnet.

Examples of good answers:
- "I'm measuring how accurately miners predict Bitcoin's price 5 minutes from now"
- "I'm measuring how fast miners can return a valid proof for a given computation"
- "I'm measuring the quality of code contributions to open source repositories"

Examples of bad answers:
- "I'm measuring AI" (too vague)
- "I'm measuring various things depending on the task" (unfocused)
- "I'm measuring compute and storage and inference quality" (too many things)

If your answer contains "and", simplify. Pick one thing. You can always add complexity later.

## Architecture Above All Else

For your first version, **simplicity is the architecture**.

A good subnet doesn't need:
- Multiple files
- Complex class hierarchies
- Shared libraries
- Configuration systems
- Plugin architectures

A good subnet needs:
- A clear measurement
- A scoring function
- Weight setting

That's it. Everything else is premature optimization.

## Single-File Validator

Write everything into `validator.py`. A good subnet doesn't need to be longer than a single file.

```
your-subnet/
├── validator.py      # <-- Your entire subnet
├── pyproject.toml
├── Dockerfile
└── docker-compose.yml
```

If you think you need more files, you're probably:
1. Overcomplicating the measurement
2. Writing miner code (don't)
3. Building infrastructure before proving the concept

## The validator.py Template

```python
"""
[SUBNET NAME] Validator

WHAT I'M MEASURING:
[One sentence describing exactly what you measure]

MINER INTERFACE:
- Endpoint: [what miners expose]
- Request: [what you send]
- Response: [what you expect back]

SCORING:
[Transparent description of how scores are calculated]
"""

import asyncio
import bittensor as bt
from bittensor_wallet import Wallet

# === CONFIGURATION ===
NETUID = 1
NETWORK = "finney"

# === THE MEASUREMENT ===
def score_miner(response, expected) -> float:
    """
    This function IS your subnet.
    Everything else is plumbing.
    """
    # Your scoring logic here
    pass

# === THE LOOP ===
class Validator:
    def __init__(self):
        self.wallet = Wallet(name="validator", hotkey="default")
        self.subtensor = bt.Subtensor(network=NETWORK)
        self.metagraph = bt.Metagraph(netuid=NETUID, network=NETWORK)
        self.scores = {}

    async def run(self):
        while True:
            self.metagraph.sync()
            
            # Query and score miners
            for uid in range(self.metagraph.n):
                response = await self.query_miner(uid)
                self.scores[uid] = score_miner(response, expected)
            
            # Set weights
            self.set_weights()
            
            await asyncio.sleep(60)

if __name__ == "__main__":
    asyncio.run(Validator().run())
```

## Checklist Before You Start

Before writing code, answer these:

1. **What am I measuring?** (one sentence)
2. **How do I get a number from 0 to 1?** (the scoring function)
3. **Where does the data come from?** (miner endpoint, external API, chain data)
4. **When is ground truth available?** (immediately, delayed, never)

If you can't answer all four, you're not ready to write code.

## Common Mistakes

### "I need to support multiple task types"
No. Pick one task. Make it work. Add more later.

### "I need a miner implementation for testing"
No. Test your validator with mock responses. When you launch, miners will figure it out from your validator code.

### "I need a config file for different environments"
No. Environment variables are enough. Your validator.py should work with just `NETWORK` and `NETUID` changed.

### "I need to handle edge cases X, Y, and Z"
No. Handle the main case first. Ship it. Learn from real usage. Then handle edge cases.

### "My scoring needs to be complex to prevent gaming"
No. Simple scoring that's easy to understand is better than complex scoring that's impossible to audit. Gaming is a mature subnet problem—you don't have it yet.

## The Simplicity Test

Can someone read your validator.py in 10 minutes and understand:
1. What miners need to do?
2. How they get scored?
3. What wins?

If no, simplify until yes.

## When to Add Complexity

Add complexity **only** when you have evidence you need it:

- Add anti-gaming measures when you observe gaming
- Add multiple task types when the single type is proven
- Add commit-reveal when you observe weight copying
- Add multiple files when the single file becomes genuinely unreadable

Evidence comes from running on testnet or mainnet with real participants. Not from imagination.

## Agent/Environment Subnets: Use Basilica

If your subnet involves miners creating **agents, environments, or executable logic** (trading bots, evaluation harnesses, task solvers, etc.), **do not use the query-based pattern** where miners run their own endpoints.

**The key question: Are miners building software, or hosting a service?**
- If miners are **building an algorithm, solver, or agent** (TSP solvers, game-playing bots, optimization algorithms, code generators, etc.) → Use Basilica containers
- If miners are **hosting existing models or compute** (GPU inference, storage, bandwidth) → Query-based may be appropriate

When in doubt: if the miner's competitive advantage comes from **the code they write** rather than the hardware they run, use containers.

Instead, use the **Affinetes + Basilica** pattern:

1. **Miners build Docker images** containing their agent code
2. **Miners push to Docker Hub** and commit the image URL to chain
3. **Validators pull the images** and run them in sandboxed Basilica pods
4. **Containers use Chutes** for LLM inference (no GPU needed in container)

```
Miner: Build Docker → Push to Hub → Commit URL to chain
                                           ↓
Validator: Read URL → Pull image → Run on Basilica → Pass Chutes key → Score
```

**Why this is better than query-based design:**

| Query-Based (Wrong) | Basilica (Correct) |
|---------------------|-------------------|
| Miner runs own server | Miner just commits code |
| Validator queries black box | Validator runs the actual code |
| Can't audit what's running | Docker image IS the submission |
| Miner needs infrastructure | Miner needs only Docker Hub |
| Unbounded attack surface | Sandboxed, reproducible |

**Example: Trading Agent Subnet**

```python
# Miner's env.py - their trading agent
class Actor:
    async def trade(self, market_data: dict, model: str = "Qwen/Qwen3-32B") -> dict:
        # Use Chutes for LLM reasoning
        client = openai.AsyncOpenAI(
            base_url="https://llm.chutes.ai/v1",
            api_key=os.getenv("CHUTES_API_KEY")
        )
        # Agent logic here...
        return {"signal": "long", "confidence": 0.8}

# Validator runs it via Basilica
env = af_env.load_env(
    mode="basilica",
    image="miner/trading-agent:v1",  # From chain
    env_vars={"CHUTES_API_KEY": os.getenv("CHUTES_API_KEY")}
)
result = await env.trade(market_data)
```

See [affine_basilica_integration.md](affine_basilica_integration.md) for complete implementation details.

## Summary

| Question | Answer |
|----------|--------|
| How many files? | One (validator.py) |
| What goes in it? | The measurement and the loop |
| What about miners? | They read your validator to understand the interface |
| When do I add complexity? | When you have evidence you need it |
| What's the most important thing? | Knowing what you're measuring |
