# Incentive Mechanism Design

This document covers principles and patterns for designing effective subnet incentive mechanisms.

## Core Principle: Align Incentives with Value

The goal of an incentive mechanism is to ensure that **miners earn more when they provide more value**. This requires:

1. **Measurable Value**: You must be able to score quality
2. **Sybil Resistance**: One entity shouldn't game by running many identities
3. **Attack Resistance**: Common exploits should be unprofitable
4. **Sustainability**: Mechanism should work long-term

## The Mechanism Design Process

```
┌─────────────────────────────────────────────────────────┐
│               MECHANISM DESIGN PROCESS                  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  1. DEFINE THE COMMODITY                                │
│     └─► What value do miners provide?                   │
│                                                         │
│  2. DEFINE QUALITY DIMENSIONS                           │
│     └─► How is "good" measured?                         │
│                                                         │
│  3. DESIGN VERIFICATION                                 │
│     └─► How do validators confirm quality?              │
│                                                         │
│  4. IDENTIFY ATTACK VECTORS                             │
│     └─► How could miners cheat?                         │
│                                                         │
│  5. ADD ANTI-GAMING MEASURES                            │
│     └─► How to prevent/detect cheating?                 │
│                                                         │
│  6. CONVERT TO WEIGHTS                                  │
│     └─► How do scores become emissions?                 │
│                                                         │
│  7. TEST AND ITERATE                                    │
│     └─► Simulate, deploy testnet, adjust               │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## Step 1: Define the Commodity

What value do miners provide? Be specific.

| Commodity Type | Examples | Key Challenge |
|---------------|----------|---------------|
| Inference | Text generation, image generation | Quality scoring |
| Compute | GPU hours, VM capacity | Hardware verification |
| Data | Scraped content, indexed datasets | Uniqueness, freshness |
| Predictions | Price forecasts, weather predictions | Delayed ground truth |
| External Work | GitHub PRs, creative content | Third-party verification |
| Availability | Uptime, response time | Consistent monitoring |

### Questions to Answer
- What specific output do miners produce?
- Who are the end consumers of this output?
- What makes one miner's output better than another's?
- Is the commodity fungible or differentiated?

---

## Step 2: Define Quality Dimensions

Quality is rarely one-dimensional. Common dimensions:

### Accuracy/Correctness
- Does the output match expected/desired result?
- Is there ground truth to compare against?
- Can you score partial correctness?

### Speed/Latency
- How fast is the response?
- Is there a timeout requirement?
- Does speed matter for end users?

### Reliability/Availability
- What percentage of requests succeed?
- Does the miner stay online consistently?
- Are there acceptable error rates?

### Uniqueness/Novelty
- Is the output copied from others?
- Does the miner provide differentiated value?
- How to measure originality?

### Cost Efficiency
- What's the miner's price (if applicable)?
- Does lower cost = higher value?
- How to balance cost vs quality?

### Example: Multi-Dimensional Scoring
```python
def calculate_score(response) -> float:
    """Combine multiple quality dimensions"""
    
    # Define weights for each dimension
    weights = {
        "accuracy": 0.35,
        "speed": 0.20,
        "reliability": 0.20,
        "uniqueness": 0.15,
        "cost_efficiency": 0.10
    }
    
    scores = {
        "accuracy": score_accuracy(response),
        "speed": score_speed(response),
        "reliability": score_reliability(miner_history),
        "uniqueness": score_uniqueness(response, other_responses),
        "cost_efficiency": score_cost(response)
    }
    
    return sum(weights[k] * scores[k] for k in weights)
```

---

## Step 3: Design Verification

How do validators confirm miner quality?

### Direct Verification
Validator can immediately assess quality.

```python
# Example: Text quality can be scored immediately
def verify_text_response(response: str) -> float:
    # Grammar check
    grammar_score = check_grammar(response)
    
    # Relevance to prompt
    relevance_score = check_relevance(response, prompt)
    
    # Length constraints
    length_score = check_length(response)
    
    return combine_scores([grammar_score, relevance_score, length_score])
```

### Delayed Verification
Ground truth arrives later.

```python
# Example: Predictions are verified when actual data arrives
class DelayedVerifier:
    def __init__(self):
        self.pending = []
        
    def store_prediction(self, uid: int, prediction: float, verify_at: datetime):
        self.pending.append((uid, prediction, verify_at))
        
    async def verify_ready(self):
        now = datetime.utcnow()
        
        ready = [p for p in self.pending if p[2] <= now]
        self.pending = [p for p in self.pending if p[2] > now]
        
        for uid, prediction, verify_at in ready:
            actual = await fetch_actual_value(verify_at)
            score = calculate_prediction_error(prediction, actual)
            yield uid, score
```

### Sampling Verification
Verify a subset, extrapolate to whole.

```python
# Example: Data indexing - can't verify everything
def sample_and_verify(miner_index: dict) -> float:
    """Verify a sample of claimed data"""
    
    # Sample buckets to verify
    all_buckets = list(miner_index.keys())
    sample = random.sample(all_buckets, min(10, len(all_buckets)))
    
    verified = 0
    total = 0
    
    for bucket in sample:
        claimed = miner_index[bucket]
        actual = fetch_and_verify_bucket(bucket)
        
        if actual["valid"]:
            verified += 1
        total += 1
    
    # Extrapolate
    verification_rate = verified / total if total > 0 else 0
    return verification_rate
```

### External Verification
Third party provides ground truth.

```python
# Example: GitHub activity verification
async def verify_github_contributions(hotkey: str) -> float:
    """Verify via GitHub API"""
    
    github_id = await get_linked_github(hotkey)
    if not github_id:
        return 0.0
    
    # Fetch from authoritative source
    prs = await github_api.get_user_prs(github_id)
    
    score = 0.0
    for pr in prs:
        # Verify tagline present
        if "Mining $OGX" not in pr.body:
            continue
        
        # Score contribution
        score += score_pr_contribution(pr)
    
    return score
```

### Hardware Verification
Cryptographic proof of hardware.

```python
# Example: GPU attestation
def verify_hardware(attestation: str) -> bool:
    """Verify hardware attestation from TEE"""
    
    # Parse attestation report
    report = parse_attestation(attestation)
    
    # Verify signature chain
    if not verify_attestation_signature(report):
        return False
    
    # Check claimed hardware matches
    if not verify_hardware_claims(report):
        return False
    
    return True
```

### Adversarial Verification
Miners verify each other; validators orchestrate.

```python
# Example: Red Team / Blue Team for AI detection
async def adversarial_verification_round(validator, red_miners, blue_miners):
    """
    Validators don't know the 'answer' — it emerges from miner competition.
    """
    # 1. Get real content from external source
    real_video = await scrape_random_video_from_web()
    description = await video_to_text_model(real_video)
    
    # 2. Red Team generates fake
    red_uid = random.choice(red_miners)
    fake_video = await query_red_miner(red_uid, description)
    
    # 3. Shuffle and send to Blue Team
    videos = shuffle([real_video, fake_video])
    
    blue_scores = {}
    for blue_uid in blue_miners:
        response = await query_blue_miner(blue_uid, videos)
        # Blue Team returns P(fake) for each video
        correct_on_real = 1 - response[real_video]["p_fake"]
        correct_on_fake = response[fake_video]["p_fake"]
        blue_scores[blue_uid] = (correct_on_real + correct_on_fake) / 2
    
    # 4. Red Team score: did they evade detection?
    avg_detection = mean([r[fake_video]["p_fake"] for r in responses])
    red_score = 1.0 - avg_detection
    
    return red_uid, red_score, blue_scores
```

**Key insight**: Validators don't control ground truth — Red Team creates the challenge, Blue Team tries to solve it. The "answer" (is this fake?) is determined by the adversarial dynamic, not by validator knowledge.

---

## Step 4: Identify Attack Vectors

Common attacks on incentive mechanisms:

### Sybil Attack
- **Attack**: Run many miners to capture more emissions
- **Reality Check**: **Coldkeys are NOT sybil-proof.** Per-coldkey deduplication does not prevent sybil attacks because anyone can create unlimited coldkeys. There is no identity system in Bittensor.
- **Common Mistake**: Implementing "one best hotkey per coldkey" deduplication and thinking you've solved sybil attacks. You haven't — the attacker just creates more coldkeys.
- **Actual Sybil Protection in Bittensor**: The **256 UID slots per subnet** combined with **dynamic registration costs**:
  - When expected mining rewards increase, registration cost rises automatically
  - Miners must always pay for a slot proportional to expected returns
  - This creates economic equilibrium preventing pure sybil farming
- **UID Pressure Concept**:
  - **High UID pressure** = high demand for slots = high registration cost
  - **Low UID pressure** = low demand = low registration cost
  - If your subnet has a steep learning curve (hard to mine profitably), UID pressure stays low
  - If your subnet is easy to mine, expect high registration costs
- **Bottom Line**: Don't rely on coldkey deduplication for sybil resistance. Design your mechanism so that running N miners isn't N times more profitable than running 1 good miner.

### Copy Attack
- **Attack**: Copy other miners' responses/data
- **Detection Limitations**: 
  - **Similarity detection is very hard in practice** - sophisticated miners can obfuscate copied responses
  - Randomized timing may help for time-sensitive responses, but miners may still copy
- **Validator Trust Problem**: You cannot trust that validators don't share results with their own miners privately:
  - Validator → MinerA → Validator → MinerB (information leak is totally possible)
  - This is an inherent limitation of the architecture
- **Practical Mitigations**: 
  - Design tasks where copying provides diminishing returns
  - Use delayed scoring where ground truth isn't available until after submissions close
  - Accept that some copying will occur; focus on rewarding genuine innovation

### Secret Eval Sets Are Untrustworthy
- **Attack**: Malicious validator leaks "secret" evaluation data to their own miners
- **Reality Check**: **Any secret held by validators WILL leak.** Validators are not individually trustworthy, and there's no enforcement mechanism to prevent information sharing.
- **Why This Fails**:
  - A validator running their own miners has every incentive to leak eval prompts
  - Even "rotating" secret sets just delays the leak
  - You cannot audit what validators share privately with miners
  - This attack is invisible and will certainly happen on any valuable subnet
- **What NOT To Do**:
  - ❌ Maintain a "secret holdout set" of evaluation prompts
  - ❌ Trust that validators will keep eval criteria private
  - ❌ Assume rotating secrets faster than they leak is sufficient
- **Better Approaches**:
  - Use **deterministic synthetic generation** where eval inputs are generated algorithmically in a way that's infinite and unpredictable
  - Use **ground truth from external sources** that becomes available after submission deadlines
  - Design tasks where **knowing the eval criteria doesn't help** (e.g., "produce the best image quality" — knowing this doesn't make it easier)
  - Accept that eval criteria should be **public and transparent** — miners knowing how they're scored is actually good for honest optimization
- **Note on Synthetic Generation**: Generating an infinite, unpredictable stream of evaluation inputs that miners cannot overfit to is a **hard open problem**. For some domains (e.g., image generation), this requires careful design — prompts must be diverse enough that optimizing for a subset doesn't generalize. Consider using procedural generation, external entropy sources, or time-based seeds that can't be predicted.

### Model/Output Similarity Detection Is Gameable
- **Attack**: Slightly modify a copied model/output to evade similarity thresholds
- **Reality Check**: **Similarity detection is easier to evade than to enforce.** 
- **Why This Fails**:
  - For models: Fine-tuning a copied model for a few steps changes weights enough to evade detection, but preserves capability. This is cheaper than training from scratch.
  - For outputs: Simple paraphrasing, noise addition, or format changes defeat text/embedding similarity
  - Attackers will always find the minimum perturbation needed to cross your threshold
- **The Economics Problem**: If evading detection is cheaper than honest work, rational miners will evade.
- **What NOT To Do**:
  - ❌ Rely on embedding similarity thresholds to detect copied models
  - ❌ Assume you can detect "near-copies" reliably
  - ❌ Build complex similarity detection that sophisticated attackers will reverse-engineer
- **Better Approaches**:
  - Design mechanisms where **copying provides diminishing returns** (e.g., s²/S scoring where duplicates hurt everyone)
  - Focus on **continuous improvement** — reward delta over time, not absolute quality
  - Accept some copying will occur; ensure honest miners can still compete

### Weight Copy Attack
- **Attack**: Copy validator weights instead of real validation
- **Detection**: Check for suspiciously similar weight patterns
- **Mitigation**: Commit-reveal weights can help, but:
  - **Don't use commit-reveal unless absolutely necessary** - adds complexity
  - Weight copying is primarily a **mature subnet problem**
  - For new subnets, focus on building the mechanism first
  - Commit-reveal periods need to be long enough to actually prevent copying

### Self-Dealing
- **Attack**: Validator gives high weights to own miners
- **Detection**: Cross-reference validator/miner coldkeys
- **Mitigation**: Stake-weighted consensus dilutes single actors

### Timing Attack
- **Attack**: Peek at others' responses before responding
- **Detection**: Timestamp analysis, response similarity
- **Mitigation**: Strict timing enforcement, randomized queries

### Gaming Metrics
- **Attack**: Optimize for scored metrics while providing low actual value
- **Detection**: Holistic quality assessment
- **Mitigation**: Multi-dimensional scoring, human spot-checks

---

## Step 5: Anti-Gaming Measures

### Deduplication

```python
def deduplicate_by_coldkey(metagraph, scores: dict) -> dict:
    """Only keep highest-scoring hotkey per coldkey"""
    
    coldkey_best = {}  # coldkey -> (best_uid, best_score)
    
    for uid in scores:
        coldkey = metagraph.coldkeys[uid]
        score = scores[uid]
        
        if coldkey not in coldkey_best or score > coldkey_best[coldkey][1]:
            coldkey_best[coldkey] = (uid, score)
    
    # Zero out non-best
    result = {}
    best_uids = {v[0] for v in coldkey_best.values()}
    
    for uid in scores:
        result[uid] = scores[uid] if uid in best_uids else 0.0
    
    return result
```

### Credibility System

```python
class CredibilityTracker:
    """Track miner credibility over time"""
    
    def __init__(self, alpha: float = 0.1, initial: float = 0.5):
        self.credibility = {}
        self.alpha = alpha
        self.initial = initial
        
    def update(self, uid: int, passed_verification: bool):
        """Update credibility based on verification result"""
        
        if uid not in self.credibility:
            self.credibility[uid] = self.initial
        
        target = 1.0 if passed_verification else 0.0
        self.credibility[uid] = (
            self.alpha * target + 
            (1 - self.alpha) * self.credibility[uid]
        )
        
    def get_multiplier(self, uid: int) -> float:
        """Get credibility-based score multiplier"""
        cred = self.credibility.get(uid, self.initial)
        # Exponential penalty for low credibility
        return cred ** 2.5
```

### Time Decay

```python
def time_decay_score(base_score: float, age_seconds: float, half_life: float = 86400) -> float:
    """Apply time decay to score"""
    import math
    
    decay_factor = math.exp(-age_seconds * math.log(2) / half_life)
    return base_score * decay_factor
```

### Uniqueness Scoring

```python
def calculate_uniqueness(response: str, all_responses: list[str]) -> float:
    """Score based on response uniqueness"""
    
    # Simple: check exact match rate
    exact_matches = sum(1 for r in all_responses if r == response)
    
    if exact_matches == 1:
        return 1.0  # Unique
    
    # Penalize duplicates
    return 1.0 / exact_matches

def data_uniqueness_score(miner_data: set, all_miners_data: list[set]) -> float:
    """s²/S formula for data uniqueness"""
    
    miner_bytes = len(miner_data)
    total_bytes = sum(len(d) for d in all_miners_data)
    
    if total_bytes == 0:
        return 0.0
    
    # Quadratic penalty for duplicated data
    return (miner_bytes ** 2) / total_bytes
```

---

## Step 6: Convert to Weights

### Normalization

```python
def normalize_to_weights(scores: dict[int, float]) -> tuple[list[int], list[float]]:
    """Convert raw scores to normalized weight vector"""
    
    uids = list(scores.keys())
    raw = [max(0, scores[uid]) for uid in uids]
    
    total = sum(raw)
    if total == 0:
        weights = [1.0 / len(uids)] * len(uids)
    else:
        weights = [w / total for w in raw]
    
    return uids, weights
```

### Softmax for Competition

```python
import numpy as np

def competitive_weights(scores: dict[int, float], temperature: float = 0.5) -> dict[int, float]:
    """
    Softmax with low temperature creates winner-take-most.
    Higher temperature = more equal distribution.
    """
    
    uids = list(scores.keys())
    values = np.array([scores[uid] for uid in uids])
    
    # Softmax
    exp_v = np.exp(values / temperature)
    weights = exp_v / np.sum(exp_v)
    
    return dict(zip(uids, weights))
```

### Inverse Scoring (Lower = Better)

```python
def inverse_score_to_weight(scores: dict[int, float], beta: float = -1.0) -> dict[int, float]:
    """
    For metrics where lower is better (like CRPS).
    Uses negative beta in softmax.
    """
    
    uids = list(scores.keys())
    values = np.array([scores[uid] for uid in uids])
    
    # Negative beta: lower score = higher weight
    exp_v = np.exp(values * beta)
    weights = exp_v / np.sum(exp_v)
    
    return dict(zip(uids, weights))
```

---

## Step 7: Test and Iterate

### Simulation

```python
class MechanismSimulator:
    """Simulate mechanism behavior"""
    
    def __init__(self, mechanism):
        self.mechanism = mechanism
        self.miners = []
        self.history = []
        
    def add_honest_miner(self, quality: float):
        self.miners.append(HonestMiner(quality))
        
    def add_gaming_miner(self, strategy: str):
        self.miners.append(GamingMiner(strategy))
        
    def run_epochs(self, n: int):
        for epoch in range(n):
            # Generate responses
            responses = [m.generate_response() for m in self.miners]
            
            # Score
            scores = self.mechanism.score(responses)
            
            # Convert to emissions
            emissions = self.mechanism.to_emissions(scores)
            
            self.history.append({
                "epoch": epoch,
                "scores": scores,
                "emissions": emissions
            })
            
    def analyze(self):
        """Check if honest miners earn more than gamers"""
        honest_total = sum(
            h["emissions"][i] 
            for h in self.history 
            for i, m in enumerate(self.miners) 
            if isinstance(m, HonestMiner)
        )
        
        gaming_total = sum(
            h["emissions"][i] 
            for h in self.history 
            for i, m in enumerate(self.miners) 
            if isinstance(m, GamingMiner)
        )
        
        return {
            "honest_earnings": honest_total,
            "gaming_earnings": gaming_total,
            "mechanism_healthy": honest_total > gaming_total
        }
```

### Testnet Deployment

1. Deploy mechanism on testnet
2. Run test miners with known quality levels
3. Verify emissions match expected distribution
4. Test attack scenarios
5. Adjust parameters based on results

### Monitoring in Production

```python
class MechanismMonitor:
    """Monitor mechanism health in production"""
    
    def __init__(self, netuid: int):
        self.netuid = netuid
        self.alerts = []
        
    def check_health(self, metagraph):
        """Run health checks"""
        
        # Check emission concentration
        emissions = metagraph.E
        top_10_pct = sum(sorted(emissions)[-int(len(emissions) * 0.1):])
        total = sum(emissions)
        
        if top_10_pct / total > 0.8:
            self.alerts.append("Warning: Top 10% control 80%+ of emissions")
        
        # Check for weight copying
        weights = metagraph.W
        unique_patterns = len(set(tuple(w) for w in weights))
        
        if unique_patterns < len(weights) * 0.5:
            self.alerts.append("Warning: Possible weight copying detected")
        
        # Check validator diversity
        validator_stakes = [
            metagraph.S[i] for i in range(metagraph.n) 
            if metagraph.validator_permit[i]
        ]
        
        if max(validator_stakes) / sum(validator_stakes) > 0.5:
            self.alerts.append("Warning: Single validator dominates")
```

---

## Design Patterns from Production

### Pattern: Auction Mechanism (Targon)
```
Miners bid prices for commodity
Validators run auction clearing
Lowest bidders win up to budget
Payment = clearing price (or bid)
```

### Pattern: Uptime/Usage Tracking (Chutes)
```
Track billed usage over time window
Score = total value delivered
No direct queries needed
```

### Pattern: Sample-and-Verify (Data Universe)
```
Miners claim total contribution
Validators sample and verify subset
Credibility = verification success rate
Score = claimed_size × credibility²
```

### Pattern: Delayed Ground Truth (Synth, Zeus)
```
Collect predictions before deadline
Wait for ground truth
Score = prediction accuracy
Use rolling windows for stability
```

### Pattern: External Verification (Gittensor)
```
Miners perform external activity
Validator queries authoritative source
Apply filters and multipliers
Gate rewards with required markers
```

### Pattern: Adversarial Competition (AI Video Detection)
```
Two miner roles: Red Team (generators) and Blue Team (detectors)
Validators source real content from web (fresh, at scale)
Red Team generates fakes from text prompts
Blue Team classifies real vs fake
Red wins if undetected, Blue wins if correct
Arms race improves both capabilities
```

---

## Validator Trust Problem

**Critical Issue**: When validators control ground truth, they can leak answers to their own miners.

This is especially dangerous when:
- The task has simple answers (binary classification, short text)
- Validators maintain "secret" evaluation sets
- Validators know the correct answer before querying miners

### Why "Secret Eval Sets" Always Fail

A validator running their own miner has every incentive to leak:
1. Validator knows the answer (they control the test set)
2. Validator tells their miner the answer
3. Their miner gets perfect scores
4. Honest miners who do real work lose

**This attack is invisible and will happen on any valuable subnet.**

### Solutions by Pattern

| Approach | How It Works | Best For |
|----------|--------------|----------|
| **External Ground Truth** | Score against third-party data (APIs, oracles) that arrives after deadline | Predictions, prices |
| **Fresh Content w/ Provenance** | Source from verifiable external URLs with timestamps; massive scale prevents memorization | Data verification |
| **Adversarial (Red/Blue)** | Miners generate challenges for each other; validators don't know the "answer" | Detection/generation |
| **Cryptographic Attestation** | Hardware proofs validators can't fake | Compute verification |
| **Public Eval Criteria** | Make scoring fully transparent; knowing how you're scored doesn't help cheat | Quality-based tasks |

### Adversarial Pattern Deep Dive

The key insight: **validators don't need to know ground truth if miners create the challenges**.

1. Validator provides neutral input (real video scraped from web + text description)
2. Red Team creates the "challenge" (generates fake video from description)
3. Blue Team attempts to solve it (detect which is fake)
4. Scoring is automatic: did Blue detect Red's fake?

This removes validators from the trust equation for core scoring logic.

**Scaling via Web Scraping**: By pulling real content from the entire web:
- Dataset size becomes massive (prevents memorization)
- Fresh content constantly enters the system
- Provenance via URLs and timestamps
- Validators only need to scrape, not curate

---

## Open Source vs Black Box Subnet Design

**Always prefer open source subnet designs where miners commit their agents to the chain.**

This is one of the most important mechanism design choices you'll make. There are two fundamental approaches:

### Black Box Query Design (Avoid)

Miners run hidden endpoints that validators query. The miner's implementation is opaque.

```
┌─────────────┐         ┌──────────────┐
│  VALIDATOR  │ Query   │    MINER     │
│             │────────►│              │
│             │         │  ┌────────┐  │
│             │◄────────│  │ ????   │  │
│             │ Response│  │ Hidden │  │
│             │         │  └────────┘  │
└─────────────┘         └──────────────┘
```

**Problems with black box design:**
- **Steep learning curve**: New miners have no idea what's working. They must reverse-engineer from scratch.
- **Stagnation**: No knowledge sharing means the subnet improves slowly.
- **Unconstrained cheating**: Miners can do anything behind the endpoint — the attack surface is unbounded.
- **Gaming through obscurity**: Miners optimize for gaming the eval, not providing value.

### Open Source Commit-to-Chain Design (Preferred)

Miners commit their implementations publicly — GitHub repos, Docker images, models on Chutes or HuggingFace. Validators verify using these committed artifacts.

```
┌─────────────┐         ┌──────────────────────────────────┐
│  VALIDATOR  │         │         PUBLIC CHAIN             │
│             │ Read    │                                  │
│             │────────►│  Miner A: github.com/alice/agent │
│             │         │  Miner B: hf.co/bob/model        │
│             │         │  Miner C: chutes.ai/carol/app    │
│             │         │                                  │
│   Verify    │         └──────────────────────────────────┘
│   against   │                       │
│   committed │◄──────────────────────┘
│   code/model│
└─────────────┘
```

**What miners can commit:**
- **GitHub repos**: Full source code, auditable and forkable
- **Docker images**: Reproducible environments with pinned dependencies  
- **Models on HuggingFace**: Weights, configs, and training details
- **Endpoints on Chutes**: Verifiable GPU inference with attestation
- **IPFS hashes**: Immutable content addressing

### Why Open Source Design Is Better

**1. New miners can learn from leaders**

When a miner joins an open source subnet, they can:
- Study what top performers are doing
- Fork and improve existing solutions
- Understand what "good" looks like immediately

This dramatically lowers the barrier to entry and creates a healthier competitive ecosystem.

**2. The subnet improves over time (compounding innovation)**

Open source creates a flywheel:
```
Top miner publishes approach
       ↓
Others study and improve it
       ↓
Improvements get published
       ↓
Original miner learns from improvements
       ↓
Everyone gets better → subnet output quality increases
```

In black box subnets, each miner reinvents the wheel. In open source subnets, miners stand on each other's shoulders.

**3. Constrained attack surface**

When miners must commit their code/models publicly:
- **Validators can audit** — suspicious behavior is visible
- **Limited degrees of freedom** — miners can only do what their committed code does
- **Community policing** — other miners will spot and call out gaming
- **Reproducibility** — validators can verify outputs match committed implementations

With black box endpoints, miners can:
- Return different outputs to different validators
- Detect and game evaluation patterns
- Run completely different code than they claim
- Implement any attack without visibility

**4. Transparency breeds trust**

Open source subnets are more attractive to:
- **Miners**: They can see it's a fair playing field
- **Validators**: They can verify mechanism integrity
- **Users**: They can trust the subnet's outputs
- **Token holders**: They can evaluate the subnet's value

### Implementation Patterns

**GitHub-based commitment:**
```python
# Miner commits repo URL to chain metadata
# Validator clones and verifies

async def verify_miner_implementation(miner_uid: int):
    repo_url = await get_miner_metadata(miner_uid, "github_repo")
    commit_hash = await get_miner_metadata(miner_uid, "commit_hash")
    
    # Clone specific commit
    repo = clone_repo(repo_url, commit_hash)
    
    # Run in sandboxed environment
    result = await run_sandboxed(repo, test_input)
    
    # Verify output matches what miner's endpoint returns
    endpoint_result = await query_miner_endpoint(miner_uid, test_input)
    
    return results_match(result, endpoint_result)
```

**Model-based commitment (HuggingFace/Chutes):**
```python
# Miner commits model identifier
# Validator runs inference on committed model

async def verify_miner_model(miner_uid: int):
    model_id = await get_miner_metadata(miner_uid, "hf_model_id")
    
    # Load committed model
    model = load_model_from_hf(model_id)
    
    # Run evaluation
    score = evaluate_model(model, eval_dataset)
    
    return score
```

**Docker-based commitment:**
```python
# Miner commits Docker image hash
# Validator runs exact same container

async def verify_miner_container(miner_uid: int):
    image_hash = await get_miner_metadata(miner_uid, "docker_image")
    
    # Pull and verify hash
    container = pull_image(image_hash)
    
    # Run deterministically
    result = await run_container(container, test_input)
    
    return score_result(result)
```

### Agent/Environment Subnets: Use Basilica

**For subnets where miners submit executable code** (agents, trading bots, evaluation environments, task solvers, optimization algorithms, game-playing bots, etc.), validators should **run the miner's code themselves** rather than querying miner endpoints.

**Rule of thumb:** If miners are competing on **the quality of software they write** (not the hardware they run), use containers. Examples: TSP solvers, chess engines, code generators, any "solve this problem" subnet.

The correct pattern:
1. Miners build Docker images with their agent/environment
2. Miners push to Docker Hub and commit the image URL to chain
3. Validators pull images and run them in **Basilica** sandboxes
4. Pass **Chutes API keys** so containers can access LLM inference
5. Score based on execution results

```python
import affinetes as af_env

# Validator runs miner's container via Basilica
env = af_env.load_env(
    mode="basilica",
    image=miner_image_url,  # From chain commitment
    env_vars={"CHUTES_API_KEY": os.getenv("CHUTES_API_KEY")}
)

try:
    result = await env.evaluate(task_id=42, model="Qwen/Qwen3-32B")
    score = result.get("score", 0)
finally:
    await env.cleanup()
```

**Why this is superior to query-based design:**
- The Docker image IS the submission — fully auditable
- Validators run the exact code miners committed — no black box
- Miners don't need infrastructure — just Docker Hub
- Sandboxed execution via Basilica — secure and reproducible
- LLM inference via Chutes — containers don't need GPUs

See [affine_basilica_integration.md](affine_basilica_integration.md) for complete implementation details.

### When Black Box Might Be Acceptable

There are limited cases where query-based black box design is reasonable:

1. **Anonymity is explicitly part of the value proposition** — e.g., Red Team generators in adversarial subnets where the output is directly tested anyway

2. **The commodity is purely the endpoint itself** — e.g., compute marketplaces where you're just verifying hardware, not implementation quality

3. **Transition period** — starting black box while developing open source verification infrastructure

Even in these cases, consider hybrid approaches where the core logic is open but execution details are private.

### Migration Strategy

If you have an existing black box subnet:

1. **Start with optional disclosure** — reward miners who commit their code with a scoring bonus
2. **Build verification infrastructure** — create the tooling to validate committed implementations
3. **Increase disclosure requirements** — gradually require more transparency for full rewards
4. **Sunset black box** — eventually require full commitment for any rewards

---

## Checklist for New Mechanisms

### Subnet Architecture
- [ ] **Open source design preferred** — miners commit code/models to chain (GitHub, HuggingFace, Chutes, Docker) rather than hiding behind black box endpoints
- [ ] **For agent/environment subnets** — use Basilica to run miner Docker containers (see [affine_basilica_integration.md](affine_basilica_integration.md))
- [ ] If using black box endpoints, documented why it's necessary for this specific case

### Ground Truth & Trust
- [ ] **Identified source of ground truth** — who/what determines correct answers?
- [ ] **If validator controls ground truth**: Added mitigation for self-dealing (see Validator Trust Problem above)
- [ ] Consider: Can adversarial miner competition replace validator judgment?
- [ ] Consider: Can external data sources provide ground truth after submission deadline?
- [ ] **No secret eval sets** (validators will leak them; use synthetic generation or public criteria)

### Anti-Gaming
- [ ] **No reliance on coldkey deduplication for sybil resistance** (coldkeys are freely creatable)
- [ ] **No reliance on similarity detection** (too easy to evade; design around copying instead)
- [ ] Accept that copy attacks are hard to prevent; design around them
- [ ] Self-dealing resistance
- [ ] Understanding of UID pressure economics (registration cost vs reward)

### Scoring & Quality
- [ ] Clearly defined commodity and value proposition
- [ ] Multi-dimensional quality scoring
- [ ] Verification method that scales
- [ ] Time decay for freshness (if applicable)
- [ ] Credibility tracking over time
- [ ] Clear conversion from scores to weights

### Infrastructure
- [ ] **Compute costs pushed to miners, not validators** (use hosted endpoints, not validator-side model loading)

### Testing & Deployment
- [ ] Simulation testing completed
- [ ] Testnet deployment verified
- [ ] Monitoring and alerting in place

---

## Subnet Design Philosophy

### Keep It Small

**Subnets should be minimal.** A well-designed subnet has:
- A few core files (not dozens)
- Focus primarily on the **validator** logic
- Leave ingenuity and complexity to the **miners**

The validator defines "what is valuable" and miners compete to provide it. The validator code should be:
- Clear and auditable
- Simple enough that miners understand what they're optimizing for
- Robust against edge cases

### Validator-Centric Design

The validator is the "referee" - it should:
- Define clear scoring criteria
- Be as deterministic as possible
- Not require complex infrastructure

Miners should be free to:
- Use any technology stack
- Innovate on their approach
- Compete on quality, not on reverse-engineering the validator
