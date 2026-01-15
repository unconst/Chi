# Building Validators

This document covers validator implementation, scoring algorithms, and weight submission.

## What Validators Do

Validators are the **evaluators** in Bittensor subnets. They:
1. Register on a subnet with sufficient stake
2. Query miners or observe their activity
3. Score miner quality using custom algorithms
4. Submit weights on-chain to determine miner rewards
5. Earn TAO dividends based on their bonds

## Design Philosophy: Validator-Centric

**The validator is the heart of your subnet.** Design principles:

1. **Keep the validator simple and auditable** - Miners need to understand what they're optimizing for
2. **Focus complexity in the validator** - Leave ingenuity to miners
3. **Minimal file count** - A good subnet can often be just a few files
4. **Clear scoring criteria** - Miners should know exactly how they're evaluated

**What the validator does NOT need to specify:**
- How miners implement their solutions
- What technology stack miners use
- Internal miner optimizations

The validator defines "what is valuable" and miners compete to provide it.

## Core Validator Loop

```
┌─────────────────────────────────────────────────────────┐
│                   VALIDATOR LOOP                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. SYNC METAGRAPH                                      │
│     └─► Get current neuron list and attributes          │
│                                                         │
│  2. SAMPLE/OBSERVE MINERS                               │
│     └─► Query miners or check external data             │
│                                                         │
│  3. SCORE RESPONSES                                     │
│     └─► Apply scoring algorithm(s)                      │
│                                                         │
│  4. AGGREGATE SCORES                                    │
│     └─► Combine into final weights                      │
│                                                         │
│  5. SET WEIGHTS                                         │
│     └─► Submit to chain (respect rate limits)           │
│                                                         │
│  6. WAIT                                                │
│     └─► Repeat at appropriate interval                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Validator Architecture Patterns

### Pattern A: HTTP API Validator
Query miners via custom HTTP endpoints. **Recommended for all subnets.**

### Pattern B: External Data Validator
Fetch data from external sources, no miner queries.

### Pattern C: Delayed Scoring Validator
Score when ground truth becomes available.

---

## Note on Communication Patterns

**The Axon/Dendrite/Synapse pattern is deprecated.** New subnets should always use open source, custom communication methods:

- Miners commit connection info to chain (API endpoints, S3 URLs, etc.)
- Validators read this info and query via standard HTTP
- Use Epistula protocol for hotkey-based authentication
- This provides flexibility, transparency, and better performance

---

## Pattern A: HTTP API Validator

Use when: Building any new subnet (this is the recommended pattern).

```python
import aiohttp
import hashlib
import time
from bittensor_wallet import Wallet
from bittensor import Subtensor, Metagraph

class HTTPValidator:
    def __init__(self, config):
        self.config = config
        self.wallet = Wallet(name=config.wallet_name, hotkey=config.hotkey_name)
        self.subtensor = Subtensor(network=config.network)
        self.metagraph = Metagraph(netuid=config.netuid, network=config.network)
        self.scores = {}
        
    def _create_epistula_headers(self, body: bytes) -> dict:
        """Create signed HTTP headers"""
        nonce = str(int(time.time() * 1e9))
        body_hash = hashlib.sha256(body).hexdigest()
        message = f"{nonce}.{body_hash}"
        signature = self.wallet.hotkey.sign(message.encode()).hex()
        
        return {
            "X-Epistula-Timestamp": nonce,
            "X-Epistula-Signature": signature,
            "X-Epistula-Hotkey": self.wallet.hotkey.ss58_address
        }
        
    async def query_miner(self, miner_url: str, endpoint: str, payload: dict) -> dict:
        """Query a miner's HTTP endpoint"""
        import json
        
        url = f"{miner_url}{endpoint}"
        body = json.dumps(payload).encode()
        headers = self._create_epistula_headers(body)
        headers["Content-Type"] = "application/json"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    url,
                    data=body,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    return {"error": f"HTTP {resp.status}"}
            except Exception as e:
                return {"error": str(e)}
                
    async def run_validation_cycle(self):
        """Run one validation cycle"""
        self.metagraph.sync()
        
        # Query all miners for their offerings
        tasks = []
        for uid in range(self.metagraph.n):
            hotkey = self.metagraph.hotkeys[uid]
            miner_url = self.get_miner_endpoint(hotkey)
            if not miner_url:
                continue
            tasks.append((uid, self.query_miner(miner_url, "/cvm", {})))
        
        # Gather responses
        responses = {}
        for uid, task in tasks:
            responses[uid] = await task
            
        # Score based on bids and verification
        for uid, response in responses.items():
            if "error" in response:
                self.scores[uid] = 0.0
            else:
                score = self._score_offering(response)
                self.scores[uid] = score
                
        # Set weights
        await self._set_weights()
        
    def _score_offering(self, response: dict) -> float:
        """Score a miner's compute offering"""
        # Example: auction-style scoring
        bid = response.get("bid_per_hour", float("inf"))
        gpu_type = response.get("gpu_type", "unknown")
        
        # Lower bid = higher score
        base_score = max(0, 1.0 - bid / 10.0)
        
        # GPU type multiplier
        gpu_multipliers = {
            "A100": 1.0,
            "H100": 1.2,
            "V100": 0.7,
        }
        multiplier = gpu_multipliers.get(gpu_type, 0.5)
        
        return base_score * multiplier
```

    def get_miner_endpoint(self, hotkey: str) -> str | None:
        """Get miner's committed endpoint from chain or registry"""
        # Implementation depends on your subnet's discovery mechanism
        # Could read from chain commitments, external registry, etc.
        pass

---

## Pattern B: External Data Validator

Use when: Miner value comes from external sources, not direct queries.

```python
import aiohttp
from bittensor import Subtensor, Metagraph
from bittensor_wallet import Wallet

class ExternalDataValidator:
    """Validator that scores based on external data (e.g., GitHub activity)"""
    
    def __init__(self, config):
        self.config = config
        self.wallet = Wallet(name=config.wallet_name, hotkey=config.hotkey_name)
        self.subtensor = Subtensor(network=config.network)
        self.metagraph = Metagraph(netuid=config.netuid, network=config.network)
        self.scores = {}
        
    async def fetch_miner_activity(self, hotkey: str) -> dict:
        """Fetch activity data from external API"""
        # Example: GitHub API for Gittensor-style subnet
        async with aiohttp.ClientSession() as session:
            # First get miner's linked identity
            github_id = await self._get_github_id(hotkey)
            if not github_id:
                return {"prs": []}
            
            # Fetch PR history
            url = f"https://api.github.com/users/{github_id}/events"
            headers = {"Authorization": f"token {self.config.github_token}"}
            
            async with session.get(url, headers=headers) as resp:
                if resp.status == 200:
                    events = await resp.json()
                    # Filter to PRs
                    prs = [e for e in events if e["type"] == "PullRequestEvent"]
                    return {"prs": prs}
                return {"prs": []}
                
    async def run_validation_cycle(self):
        """Run one validation cycle"""
        self.metagraph.sync()
        
        # Fetch activity for all miners
        for uid in range(self.metagraph.n):
            hotkey = self.metagraph.hotkeys[uid]
            activity = await self.fetch_miner_activity(hotkey)
            
            # Score activity
            score = self._score_activity(activity)
            self.scores[uid] = score
            
        # Apply anti-gaming
        self._apply_deduplication()
        
        # Set weights
        await self._set_weights()
        
    def _score_activity(self, activity: dict) -> float:
        """Score miner's external activity"""
        prs = activity.get("prs", [])
        if not prs:
            return 0.0
        
        total_score = 0.0
        
        for pr in prs:
            # Base score from contribution
            files_changed = pr.get("payload", {}).get("pull_request", {}).get("changed_files", 0)
            base = min(files_changed / 10, 1.0)
            
            # Check for required tagline
            body = pr.get("payload", {}).get("pull_request", {}).get("body", "")
            if "Mining $OGX" not in body:
                continue  # No tagline = no score
            
            # Repo quality multiplier
            repo = pr.get("repo", {}).get("name", "")
            repo_weight = self._get_repo_weight(repo)
            
            # Time decay
            created_at = pr.get("created_at", "")
            time_factor = self._time_decay(created_at)
            
            total_score += base * repo_weight * time_factor
        
        return total_score
        
    def _apply_deduplication(self):
        """Detect and penalize multi-hotkey gaming"""
        # Group by coldkey
        coldkey_groups = {}
        for uid in range(self.metagraph.n):
            coldkey = self.metagraph.coldkeys[uid]
            if coldkey not in coldkey_groups:
                coldkey_groups[coldkey] = []
            coldkey_groups[coldkey].append(uid)
        
        # For each coldkey with multiple hotkeys, keep only highest scorer
        for coldkey, uids in coldkey_groups.items():
            if len(uids) <= 1:
                continue
            
            # Find highest scorer
            scores = [(uid, self.scores.get(uid, 0)) for uid in uids]
            scores.sort(key=lambda x: x[1], reverse=True)
            
            # Zero out all but the highest
            for uid, _ in scores[1:]:
                self.scores[uid] = 0.0
```

---

## Pattern C: Delayed Scoring Validator

Use when: Ground truth arrives after prediction horizon.

```python
import asyncio
from datetime import datetime, timedelta
from collections import defaultdict

class DelayedScoringValidator:
    """Validator that scores predictions when ground truth becomes available"""
    
    def __init__(self, config):
        self.config = config
        self.wallet = Wallet(name=config.wallet_name, hotkey=config.hotkey_name)
        self.subtensor = Subtensor(network=config.network)
        self.metagraph = Metagraph(netuid=config.netuid, network=config.network)
        
        # Store predictions awaiting scoring
        self.pending_predictions = defaultdict(list)  # uid -> [(prediction, target_time)]
        self.scores = {}
        
    async def run(self):
        """Main loop with online and delayed phases"""
        while True:
            # Online phase: collect predictions
            await self._online_phase()
            
            # Delayed phase: score ready predictions
            await self._delayed_scoring_phase()
            
            # Set weights
            await self._maybe_set_weights()
            
            await asyncio.sleep(self.config.interval)
            
    async def _online_phase(self):
        """Query miners for predictions"""
        self.metagraph.sync()
        
        # Create prediction request
        target_time = datetime.utcnow() + timedelta(minutes=5)
        payload = {
            "asset": "BTC-USD",
            "horizon": 300,
            "target_time": target_time.isoformat()
        }
        
        # Query all miners via HTTP
        for uid in range(self.metagraph.n):
            hotkey = self.metagraph.hotkeys[uid]
            miner_url = self.get_miner_endpoint(hotkey)
            if not miner_url:
                continue
                
            response = await self.query_miner(miner_url, "/predict", payload)
            
            if response and response.get("prediction") is not None:
                self.pending_predictions[uid].append((
                    response["prediction"],
                    target_time
                ))
            else:
                # Immediate penalty for timeout/error
                self._apply_penalty(uid)
                
    async def _delayed_scoring_phase(self):
        """Score predictions where ground truth is now available"""
        now = datetime.utcnow()
        
        for uid, predictions in list(self.pending_predictions.items()):
            # Check each pending prediction
            ready = []
            still_pending = []
            
            for prediction, target_time in predictions:
                if target_time <= now:
                    ready.append((prediction, target_time))
                else:
                    still_pending.append((prediction, target_time))
            
            # Update pending list
            self.pending_predictions[uid] = still_pending
            
            # Score ready predictions
            for prediction, target_time in ready:
                score = await self._score_prediction(prediction, target_time)
                self._update_score(uid, score)
                
    async def _score_prediction(self, prediction, target_time) -> float:
        """Score prediction against ground truth"""
        # Fetch actual value
        actual = await self._fetch_ground_truth(target_time)
        
        if actual is None:
            return 0.5  # Unknown, neutral score
        
        # Calculate CRPS or similar metric
        if isinstance(prediction, list):
            # Distribution prediction
            crps = self._calculate_crps(prediction, actual)
            # Lower CRPS = better
            return 1.0 - min(crps, 1.0)
        else:
            # Point prediction
            error = abs(prediction - actual)
            return max(0, 1.0 - error / actual)
            
    def _calculate_crps(self, samples: list, actual: float) -> float:
        """Continuous Ranked Probability Score"""
        import numpy as np
        samples = np.array(sorted(samples))
        n = len(samples)
        
        # Simple CRPS calculation
        return np.mean(np.abs(samples - actual))
        
    def _update_score(self, uid: int, new_score: float):
        """Update score with EMA"""
        alpha = self.config.score_alpha
        old_score = self.scores.get(uid, 0.5)
        self.scores[uid] = alpha * new_score + (1 - alpha) * old_score
```

---

## Scoring Algorithm Patterns

### Multi-Dimensional Scoring

```python
def calculate_composite_score(response) -> float:
    """Combine multiple quality dimensions"""
    
    dimensions = {
        "quality": {
            "weight": 0.4,
            "scorer": score_quality
        },
        "speed": {
            "weight": 0.2,
            "scorer": score_speed
        },
        "reliability": {
            "weight": 0.2,
            "scorer": score_reliability
        },
        "uniqueness": {
            "weight": 0.2,
            "scorer": score_uniqueness
        }
    }
    
    total = 0.0
    for name, config in dimensions.items():
        score = config["scorer"](response)
        total += config["weight"] * score
    
    return total
```

### Relative Scoring (vs Baseline)

```python
def score_vs_baseline(prediction, actual, baseline) -> float:
    """Score improvement over baseline"""
    
    pred_error = abs(prediction - actual)
    base_error = abs(baseline - actual)
    
    if base_error == 0:
        return 0.5
    
    # Improvement ratio
    improvement = (base_error - pred_error) / base_error
    
    # Clamp to [0, 1]
    return max(0, min(1, 0.5 + improvement / 2))
```

### Tournament Scoring

```python
def tournament_score(miners: dict[int, float]) -> dict[int, float]:
    """Rank-based scoring"""
    
    # Sort by raw score
    ranked = sorted(miners.items(), key=lambda x: x[1], reverse=True)
    n = len(ranked)
    
    # Assign rank-based scores
    scores = {}
    for rank, (uid, _) in enumerate(ranked):
        # Linear decay from 1.0 to 0.0
        scores[uid] = 1.0 - (rank / n)
    
    return scores
```

### EMA Score Accumulation

```python
class EMAScoreTracker:
    """Track scores with exponential moving average"""
    
    def __init__(self, alpha: float = 0.1, initial: float = 0.5):
        self.alpha = alpha
        self.initial = initial
        self.scores = {}
        
    def update(self, uid: int, new_score: float):
        """Update score for uid"""
        if uid not in self.scores:
            self.scores[uid] = self.initial
        
        self.scores[uid] = (
            self.alpha * new_score + 
            (1 - self.alpha) * self.scores[uid]
        )
        
    def get(self, uid: int) -> float:
        return self.scores.get(uid, self.initial)
        
    def age_based_alpha(self, uid: int, age_blocks: int) -> float:
        """Higher alpha for newer miners"""
        # New miners adapt quickly, established miners are stable
        min_alpha = 0.01
        max_alpha = 0.5
        decay_rate = 0.001
        
        return min_alpha + (max_alpha - min_alpha) * math.exp(-decay_rate * age_blocks)
```

---

## Weight Setting

### Basic Weight Normalization

```python
def normalize_weights(scores: dict[int, float]) -> tuple[list[int], list[float]]:
    """Normalize scores to valid weight vector"""
    
    uids = list(scores.keys())
    raw_weights = [max(0, scores[uid]) for uid in uids]
    
    total = sum(raw_weights)
    if total == 0:
        # Uniform if all zero
        weights = [1.0 / len(uids)] * len(uids)
    else:
        weights = [w / total for w in raw_weights]
    
    return uids, weights
```

### Softmax Conversion

```python
import numpy as np

def softmax_weights(scores: dict[int, float], temperature: float = 1.0) -> dict[int, float]:
    """Convert scores to weights using softmax"""
    
    uids = list(scores.keys())
    values = np.array([scores[uid] for uid in uids])
    
    # Softmax with temperature
    exp_values = np.exp(values / temperature)
    softmax_values = exp_values / np.sum(exp_values)
    
    return dict(zip(uids, softmax_values))
```

### Commit-Reveal Weight Setting

```python
async def set_weights_commit_reveal(
    subtensor,
    wallet,
    netuid: int,
    uids: list[int],
    weights: list[float]
):
    """Set weights using commit-reveal for privacy"""
    
    # Check if commit-reveal enabled
    params = subtensor.get_subnet_hyperparameters(netuid)
    
    if params.commit_reveal_weights_enabled:
        # Use commit-reveal
        success = subtensor.commit_weights(
            wallet=wallet,
            netuid=netuid,
            uids=uids,
            weights=weights
        )
        # Reveal happens automatically after reveal_period
    else:
        # Direct weight setting
        success = subtensor.set_weights(
            wallet=wallet,
            netuid=netuid,
            uids=uids,
            weights=weights
        )
    
    return success
```

---

## Anti-Gaming Measures

### Duplicate Detection
```python
def detect_duplicates(metagraph, scores: dict) -> dict:
    """Zero out duplicate miners (same coldkey)"""
    
    coldkey_groups = defaultdict(list)
    for uid in range(metagraph.n):
        coldkey = metagraph.coldkeys[uid]
        coldkey_groups[coldkey].append(uid)
    
    adjusted_scores = scores.copy()
    
    for coldkey, uids in coldkey_groups.items():
        if len(uids) > 1:
            # Keep only highest scorer
            best_uid = max(uids, key=lambda u: scores.get(u, 0))
            for uid in uids:
                if uid != best_uid:
                    adjusted_scores[uid] = 0.0
    
    return adjusted_scores
```

### Rate Limiting Detection
```python
def detect_rate_gaming(responses_history: list) -> float:
    """Detect miners gaming response timing"""
    
    # Check for suspiciously consistent response times
    times = [r.process_time for r in responses_history if r.process_time]
    
    if len(times) < 10:
        return 1.0  # Not enough data
    
    variance = np.var(times)
    
    # Very low variance = suspicious
    if variance < 0.001:
        return 0.5  # Penalty
    
    return 1.0
```

---

## Validator Checklist

- [ ] Register hotkey with sufficient stake
- [ ] Implement miner sampling strategy
- [ ] Implement scoring algorithm for your commodity
- [ ] Handle timeouts and errors gracefully
- [ ] Track scores over time (EMA recommended)
- [ ] Normalize weights correctly before submission
- [ ] Respect weight rate limits
- [ ] Add anti-gaming measures
- [ ] Monitor dividend earnings
- [ ] Test thoroughly on localnet/testnet
