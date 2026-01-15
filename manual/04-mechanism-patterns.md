# Mechanism Architecture Patterns

**This is the most important document for subnet design.** Production subnets use diverse architectures based on their specific needs. All production subnets use open source, custom communication patterns—never the deprecated SDK primitives.

## Understanding Mechanism Design

A subnet mechanism defines:
1. **What commodity** miners provide (compute, data, predictions, etc.)
2. **How validators verify** miner quality
3. **How scores become weights** on-chain
4. **What behavior is incentivized**

---

## Important: Communication Patterns

### Axon/Dendrite/Synapse is Deprecated

**The Axon/Dendrite/Synapse pattern from the SDK is deprecated.** New subnets should always use open source, custom communication methods. This pattern was useful for early prototyping but has significant limitations.

**Why you should NOT use Axon/Dendrite:**
- Proprietary pattern that limits flexibility
- Poor performance for production use cases
- Encourages closed-source miner implementations
- All production subnets have moved to custom communication
- The pattern obscures what's actually happening in your subnet

**What you should use instead:**
- Standard HTTP APIs (FastAPI, Flask, etc.)
- Epistula protocol for hotkey-based signing
- gRPC, WebSocket, or other open protocols as needed
- Open source designs that anyone can audit and understand

**What miners can commit to chain:**
Miners can commit arbitrary information to their on-chain metadata, enabling validators to discover:
- Custom API endpoints
- S3 bucket URLs
- Database endpoints
- IP addresses for any protocol
- Any other connection information

For commit-reveal commitments (data that's encrypted until a future block using drand timelock encryption), see **[19-chain-commitments.md](19-chain-commitments.md)**.

**Recommendation:** Always use custom HTTP APIs with Epistula signing. Never use Axon/Dendrite for new development.

---

## Critical Design Principle: Push Compute Costs to Miners

**Always design mechanisms where miners bear the compute costs, not validators.**

### Why This Matters
- Validators should be cheap to run (encourages validator diversity)
- Miners are already incentivized to invest in infrastructure
- If validators must run expensive operations (downloading models, running inference), it creates centralization pressure

### Preferred Pattern: Hosted Inference Endpoints

For subnets where miners provide models or compute-heavy services, **require miners to host their models as queryable endpoints** rather than having validators download and run models locally.

**Example: Model Training Subnets**

❌ **Bad Design**: Miners upload model checkpoints to S3/HuggingFace, validators download and run inference locally.
- Validators need expensive GPUs
- Download bandwidth costs
- Model loading overhead per evaluation

✅ **Good Design**: Miners deploy models as inference endpoints (e.g., on Chutes, or self-hosted APIs). Validators query endpoints with API calls.
- Validators only need API keys and minimal compute
- Miners bear the serving costs
- Models are already available for real usage
- Evaluation is fast (just HTTP requests)

**Implementation:**
```python
# Miner commits their inference endpoint to chain metadata
# Validator reads endpoint and queries it

async def evaluate_miner_model(miner_endpoint: str, eval_prompts: list) -> float:
    """Query miner's hosted model endpoint"""
    scores = []
    for prompt in eval_prompts:
        response = await http_client.post(
            miner_endpoint,
            json={"prompt": prompt},
            headers=epistula_headers  # Sign with validator hotkey
        )
        image = response.json()["image"]
        score = calculate_image_quality(image, prompt)
        scores.append(score)
    return np.mean(scores)
```

**Chutes Integration**: For GPU-intensive commodities, miners can deploy on Chutes (SN64). Validators get API access to query miner models without running them locally. This leverages existing Bittensor infrastructure.

---

## The Six Major Patterns

Based on real production subnets, these are the primary architectural patterns:

---

## Pattern 1: Compute Auction Marketplace
**Example: Targon (SN4)**

### When to Use
- Miners provide fungible compute resources (GPUs, CPUs, VMs)
- Value is in verified hardware capability and competitive pricing
- Validators need to verify hardware authenticity

### Architecture
```
┌─────────────┐         ┌──────────────┐
│   MINER     │         │  VALIDATOR   │
│             │         │              │
│ ┌─────────┐ │         │              │
│ │ CVM/GPU │◄├─────────┤ Attestation  │
│ │ Cluster │ │  Verify │   Checks     │
│ └─────────┘ │         │              │
│             │         │ ┌──────────┐ │
│ ┌─────────┐ │         │ │ Auction  │ │
│ │HTTP API │◄├─────────┤ │ Clearing │ │
│ │/cvm EP  │ │  Query  │ └──────────┘ │
│ └─────────┘ │  Bids   │              │
│             │         │ ┌──────────┐ │
│ BID + CAPS  │         │ │ Weight   │ │
│ ATTESTATION │─────────►│ │ Setting  │ │
│             │         │ └──────────┘ │
└─────────────┘         └──────────────┘
```

### Key Components

**Miner Side:**
- Register hotkey on subnet
- Run CVM (Confidential Virtual Machine) nodes with real hardware
- Serve an HTTP endpoint (typically `/cvm`) advertising:
  - Available resources (GPU type, VRAM, CPU cores)
  - Bid price per unit (e.g., $/GPU/hour)
  - Attestation proofs (remote attestation from TEE)
- Use signed HTTP headers (Epistula protocol) for authentication
- Commit endpoint URL to chain for discovery

**Validator Side:**
- Query all registered miner endpoints for bids
- Verify hardware attestation (via Tower or similar service)
- Deduplicate miners (same hardware, different hotkeys)
- Run auction clearing algorithm:
  - Group by resource type (buckets)
  - Sort by price
  - Allocate emission budget
  - Pay cheapest providers first
- Convert payouts to u16 weights
- Submit weights on-chain

**Epistula Protocol (Signed HTTP):**
```python
import time
import hashlib

def create_epistula_headers(wallet, body: bytes) -> dict:
    nonce = str(int(time.time() * 1e9))
    body_hash = hashlib.sha256(body).hexdigest()
    message = f"{nonce}.{body_hash}"
    signature = wallet.hotkey.sign(message.encode()).hex()
    
    return {
        "X-Epistula-Timestamp": nonce,
        "X-Epistula-Signature": signature,
        "X-Epistula-Hotkey": wallet.hotkey.ss58_address
    }
```

**Anti-Gaming:**
- Hardware attestation prevents GPU spoofing
- Deduplication prevents multi-hotkey same-hardware attacks
- Auction mechanism rewards honest pricing

---

## Pattern 2: Capacity/Uptime Marketplace
**Example: Chutes (SN64)**

### When to Use
- Miners provide always-on infrastructure
- Billing based on actual usage/uptime
- Need GPU authenticity verification
- No public query endpoint needed

### Architecture
```
┌─────────────┐         ┌──────────────┐
│   MINER     │         │  ORCHESTRATOR│
│             │         │   (Central)  │
│ ┌─────────┐ │ Socket  │              │
│ │Kubernetes├─┼────────►│ Job Router  │
│ │ Cluster │ │  .io    │              │
│ └─────────┘ │         │              │
│             │         └──────────────┘
│ ┌─────────┐ │               │
│ │ GraVal  │ │         ┌─────▼────────┐
│ │GPU Proof│─┼─────────►  VALIDATOR   │
│ └─────────┘ │ Verify  │              │
│             │         │ ┌──────────┐ │
│ NO PUBLIC   │         │ │ Compute  │ │
│ ENDPOINT    │         │ │ Billing  │ │
│             │         │ └──────────┘ │
└─────────────┘         │              │
                        │ ┌──────────┐ │
                        │ │ Weights  │ │
                        │ └──────────┘ │
                        └──────────────┘
```

### Key Components

**Miner Side:**
- Register hotkey on subnet
- Run Kubernetes-based GPU cluster
- Add nodes via CLI tool (`chutes-miner add-node`)
- Each node connects via socket.io to central orchestrator
- GraVal scheme for GPU authenticity
- Provide hourly cost, GPU type, VRAM

**Validator Side:**
- Track billed compute units over rolling window (7 days)
- Query central orchestrator for usage history
- Apply blacklists and multi-hotkey deduplication
- Score = billed compute delivered
- Coldkey deduplication: only highest-scoring hotkey per coldkey kept
- Normalize scores, quantize to u16 weights

**Socket.io Pattern:**
```python
# Miner connects to orchestrator via socket.io
# No public HTTP endpoint required

import socketio

sio = socketio.Client()

@sio.event
def job_request(data):
    # Execute job on local GPU
    result = execute_job(data)
    return result

sio.connect('https://orchestrator.chutes.ai')
```

**Anti-Gaming:**
- GraVal cryptographic GPU proofs
- Coldkey deduplication removes multi-hotkey gaming
- Billed uptime (not just registration) prevents idle nodes

---

## Pattern 3: Data Indexing and Validation
**Example: Data Universe (SN13)**

### When to Use
- Miners collect and store data (scraping, indexing)
- Value in unique, fresh, correctly-formatted data
- Validators sample and re-verify portions

### Architecture
```
┌─────────────┐         ┌──────────────┐
│   MINER     │         │  VALIDATOR   │
│             │         │              │
│ ┌─────────┐ │         │ ┌──────────┐ │
│ │ Scraper │ │         │ │ Query    │ │
│ │ Engine  │ │         │ │ Index    │ │
│ └────┬────┘ │         │ └────┬─────┘ │
│      │      │         │      │       │
│ ┌────▼────┐ │         │ ┌────▼─────┐ │
│ │  Local  │ │ Synapse │ │ Sample   │ │
│ │ Storage │◄├─────────┤ │ Content  │ │
│ └────┬────┘ │         │ └────┬─────┘ │
│      │      │         │      │       │
│ ┌────▼────┐ │         │ ┌────▼─────┐ │
│ │ Miner   │─┼─────────►│ │ Deep     │ │
│ │ Index   │ │ Summary │ │ Validate │ │
│ └─────────┘ │         │ └────┬─────┘ │
│             │         │      │       │
│ HTTP API    │         │ ┌────▼─────┐ │
│ rate limits │         │ │ Score &  │ │
│             │         │ │ Weights  │ │
└─────────────┘         │ └──────────┘ │
                        └──────────────┘
```

### Key Components

**Miner Side:**
- Scrape data from sources (Reddit, X, news, etc.)
- Store in local database with proper indexing
- Generate `CompressedMinerIndex`: summary of data buckets
  - DataSource (reddit, twitter, etc.)
  - DataLabel (subreddit, topic, etc.)
  - TimeBucket (hourly/daily buckets)
  - Size and count metrics
- Serve via HTTP with endpoints:
  - `GetMinerIndex`: Return compressed index
  - `GetDataEntityBucket`: Return actual data for bucket

**Validator Side:**
1. **Index Query Phase:**
   - Get miner's index summary
   - Analyze what data each miner claims

2. **Content Sampling Phase:**
   - Request specific buckets
   - Verify format and timestamps

3. **Deep Validation Phase:**
   - Re-scrape a sample of claimed data
   - Compare with miner's version
   - Calculate match rate

4. **Scoring:**
   - `credibility`: EMA of validation success
   - `scorableBytes`: Unique data contribution (s²/S formula)
   - `rawScore`: source_weight × scale × time_freshness × scorableBytes
   - `finalScore`: rawScore × credibility^2.5

**Scorable Bytes (Anti-Duplication):**
```python
def calculate_scorable_bytes(miner_bytes, total_network_bytes):
    """
    s²/S formula penalizes duplicated data
    If everyone has same data, individual scores are low
    Unique data gets higher relative score
    """
    return (miner_bytes ** 2) / total_network_bytes
```

**Anti-Gaming:**
- Deep validation catches fake/fabricated data
- Credibility EMA builds slowly, lost quickly
- Scorable bytes formula penalizes duplication
- Time freshness rewards recent data

---

## Pattern 4: Prediction Market / Simulation
**Example: Synth Subnet**

### When to Use
- Miners generate predictions or simulations
- Ground truth becomes available later
- Scoring requires temporal delay

### When NOT to Use (Prefer Open Source Design Instead)
If miners are building **agents with complex decision logic** (e.g., LLM trading agents, autonomous systems), consider having miners **commit their agent code** rather than querying for predictions. Validators run the committed code in a sandboxed environment and score based on outcomes. This provides transparency, enables knowledge sharing, and constrains the attack surface. See **Open Source vs Black Box Subnet Design** in [08-incentive-design.md](08-incentive-design.md).

### Architecture
```
┌─────────────┐         ┌──────────────┐
│   MINER     │         │  VALIDATOR   │
│             │         │              │
│ ┌─────────┐ │         │ ┌──────────┐ │
│ │ Monte   │ │ Synapse │ │ Collect  │ │
│ │ Carlo   │◄├─────────┤ │ Request  │ │
│ │ Engine  │ │         │ └────┬─────┘ │
│ └────┬────┘ │         │      │       │
│      │      │         │ ┌────▼─────┐ │
│ ┌────▼────┐ │         │ │ Store    │ │
│ │ Return  │─┼─────────►│ │ Response │ │
│ │ Samples │ │         │ └────┬─────┘ │
│ └─────────┘ │         │      │       │
│             │         │      │ (wait)│
│ CPU-only    │         │      │       │
│ strict fmt  │         │ ┌────▼─────┐ │
│             │         │ │ Fetch    │ │
│             │         │ │ Ground   │ │
│             │         │ │ Truth    │ │
│             │         │ └────┬─────┘ │
│             │         │      │       │
│             │         │ ┌────▼─────┐ │
│             │         │ │ CRPS     │ │
│             │         │ │ Scoring  │ │
│             │         │ └────┬─────┘ │
│             │         │      │       │
│             │         │ ┌────▼─────┐ │
│             │         │ │ Weights  │ │
│             │         │ └──────────┘ │
└─────────────┘         └──────────────┘
```

### Key Components

**Miner Side:**
- Receive prediction request (asset, horizon, parameters)
- Run Monte Carlo simulation (CPU-only, no network)
- Return samples in strict format:
  ```python
  {
      "asset": "BTC-USD",
      "horizon": 300,  # seconds
      "samples": [0.001, -0.002, 0.003, ...],  # basis point returns
      "start_time": 1705000000
  }
  ```
- Must respond before start_time + timeout

**Validator Side:**
1. **Request Phase:**
   - Broadcast prediction requests
   - Collect responses with timestamps

2. **Wait Phase:**
   - Store predictions
   - Wait for horizon to pass

3. **Scoring Phase:**
   - Fetch actual prices (e.g., Pyth Benchmarks)
   - Calculate CRPS (Continuous Ranked Probability Score)
   - Lower CRPS = better prediction

4. **Aggregation:**
   - Roll scores over time windows
   - Apply asset coefficients
   - Softmax with negative beta (lower score = higher weight)

**CRPS Scoring:**
```python
import numpy as np

def crps_score(samples, actual_return):
    """
    Continuous Ranked Probability Score
    Measures how well probability distribution matches reality
    """
    samples = np.sort(samples)
    n = len(samples)
    
    # Empirical CDF vs actual
    below = samples < actual_return
    above = ~below
    
    score = np.sum(np.abs(samples - actual_return)) / n
    return score
```

**Anti-Gaming:**
- Start time enforcement prevents peeking
- CRPS rewards calibrated distributions
- Rolling windows smooth noise
- Multiple assets prevent overfitting

---

## Pattern 5: External Activity Verification
**Example: Gittensor (SN74)**

### When to Use
- Value comes from verifiable external activity
- Third-party API provides ground truth
- Miners don't need real-time serving

### Architecture
```
┌─────────────┐         ┌──────────────┐
│   MINER     │         │   GITHUB     │
│             │         │     API      │
│ ┌─────────┐ │         │              │
│ │ GitHub  │ │         │  PR History  │
│ │ Account │ │         │              │
│ └────┬────┘ │         └──────┬───────┘
│      │      │                │
│ ┌────▼────┐ │         ┌──────▼───────┐
│ │ Submit  │─┼────────►│  VALIDATOR   │
│ │  PAT    │ │  Token  │              │
│ └─────────┘ │         │ ┌──────────┐ │
│             │         │ │ Fetch    │ │
│ Make PRs    │         │ │ PR Data  │ │
│ with tag    │         │ └────┬─────┘ │
│             │         │      │       │
│             │         │ ┌────▼─────┐ │
│             │         │ │ Filter   │ │
│             │         │ │ & Score  │ │
│             │         │ └────┬─────┘ │
│             │         │      │       │
│             │         │ ┌────▼─────┐ │
│             │         │ │ Weights  │ │
│             │         │ └──────────┘ │
└─────────────┘         └──────────────┘
```

### Key Components

**Miner Side:**
- Register hotkey on subnet
- Link GitHub account to hotkey (provide PAT)
- Make open-source contributions
- PR must include tagline: `"Mining $OGX - Hotkey {github_id}"`
- Quality contributions on eligible repos

**Validator Side:**
1. **Data Fetch:**
   - Use GitHub API to get miner's PR history
   - Filter by time window (lookback period)

2. **Eligibility Filtering:**
   - Repository must be eligible (stars, activity)
   - Not self-merged (author != merger)
   - Not from maintainer account
   - Branch rules met

3. **Contribution Scoring:**
   - File diffs analyzed
   - Language weights applied
   - Test files downweighted
   - Low-value changes detected

4. **Multipliers:**
   - Repository weight (more active repo = higher)
   - Issue linking bonus
   - Open PR penalty (uncommitted "collateral")
   - Time decay
   - Uniqueness (not copying others)
   - Missing tagline = zero score

5. **Duplicate Detection:**
   - Same GitHub account on multiple hotkeys
   - Penalize all but first registered

**Anti-Gaming:**
- Self-merge detection
- Maintainer filtering
- Low-value PR detection
- Duplicate account penalties
- Tagline requirement gates rewards

---

## Pattern 6: Time-Series Forecasting
**Example: Zeus (SN18)**

### When to Use
- Predictions where ground truth arrives with delay
- Quality measured against baseline
- Response time matters

### Architecture
```
┌─────────────┐         ┌──────────────┐
│   MINER     │         │  VALIDATOR   │
│             │         │              │
│ ┌─────────┐ │         │ ┌──────────┐ │
│ │ Forecast│◄├─────────┤ │ Online   │ │
│ │ Model   │ │ Request │ │ Phase    │ │
│ └────┬────┘ │         │ └────┬─────┘ │
│      │      │         │      │       │
│ ┌────▼────┐ │         │ ┌────▼─────┐ │
│ │ Predict │─┼─────────►│ │ Store +  │ │
│ │ Values  │ │ Response│ │ Pen. Bad │ │
│ └─────────┘ │         │ └────┬─────┘ │
│             │         │      │       │
│             │         │      │ (wait)│
│             │         │      │       │
│             │         │ ┌────▼─────┐ │
│             │         │ │ Delayed  │ │
│             │         │ │ Scoring  │ │
│             │         │ └────┬─────┘ │
│             │         │      │       │
│             │         │ ┌────▼─────┐ │
│             │         │ │ Fetch    │ │
│             │         │ │ ERA5     │ │
│             │         │ │ Truth    │ │
│             │         │ └────┬─────┘ │
│             │         │      │       │
│             │         │ ┌────▼─────┐ │
│             │         │ │ Quality  │ │
│             │         │ │ + Speed  │ │
│             │         │ └────┬─────┘ │
│             │         │      │       │
│             │         │ ┌────▼─────┐ │
│             │         │ │ EMA      │ │
│             │         │ │ Scores   │ │
│             │         │ └──────────┘ │
└─────────────┘         └──────────────┘
```

### Key Components

**Miner Side:**
- Receive forecast request (location, variables, horizon)
- Return predictions in required format
- Speed matters (faster = higher efficiency score)

**Validator Side:**
1. **Online Phase:**
   - Query miners for predictions
   - Immediately penalize malformed/timeout responses
   - Store valid predictions

2. **Delayed Scoring Phase (when ground truth available):**
   - Fetch actual data (e.g., ERA5 weather data)
   - Calculate quality score:
     ```python
     # Quality = improvement over baseline
     miner_rmse = calculate_rmse(prediction, actual)
     baseline_rmse = calculate_rmse(openmeteo, actual)
     quality = (baseline_rmse - miner_rmse) / baseline_rmse
     ```
   - Calculate efficiency score:
     ```python
     efficiency = 1.0 - (response_time / max_time)
     ```
   - Combined score:
     ```python
     score = (quality_weight * quality + 
              efficiency_weight * efficiency)
     ```

3. **EMA Accumulation:**
   - Scores accumulate via exponential moving average
   - Alpha (learning rate) based on miner age
   - New miners: higher alpha (adapt quickly)
   - Established miners: lower alpha (stable)

---

## Pattern 7: Adversarial Red Team / Blue Team
**Example: AI Video Detection**

### When to Use
- Miners both create AND evaluate content
- Quality of detection depends on quality of generation (arms race)
- Ground truth is expensive or impossible to curate at scale
- Validators shouldn't be sole source of truth (avoids validator collusion)

### Architecture
```
┌─────────────┐         ┌──────────────┐         ┌─────────────┐
│  RED TEAM   │         │  VALIDATOR   │         │  BLUE TEAM  │
│  (Generate) │         │              │         │  (Detect)   │
│             │         │ ┌──────────┐ │         │             │
│             │         │ │ 1. Scrape│ │         │             │
│             │         │ │ Real Vid │ │         │             │
│             │         │ └────┬─────┘ │         │             │
│             │         │      │       │         │             │
│             │         │ ┌────▼─────┐ │         │             │
│             │         │ │ 2. Vid→  │ │         │             │
│             │         │ │ Text Desc│ │         │             │
│             │         │ └────┬─────┘ │         │             │
│             │         │      │       │         │             │
│ ┌─────────┐ │  Prompt │ ┌────▼─────┐ │         │             │
│ │ Generate│◄├─────────┤ │ 3. Send  │ │         │             │
│ │ Fake    │ │         │ │ Prompt   │ │         │             │
│ └────┬────┘ │         │ └──────────┘ │         │             │
│      │      │         │              │         │             │
│ ┌────▼────┐ │  Video  │ ┌──────────┐ │ Videos  │ ┌─────────┐ │
│ │ Submit  │─┼─────────►│ │ 4. Send  │─┼─────────►│ Classify│ │
│ │ Fake    │ │         │ │ Both Vids│ │ (shuf.) │ │ Real/AI │ │
│ └─────────┘ │         │ └──────────┘ │         │ └────┬────┘ │
│             │         │              │         │      │      │
│             │         │ ┌──────────┐ │  Probs  │ ┌────▼────┐ │
│             │         │ │ 5. Score │◄├─────────┼─┤ Submit  │ │
│             │         │ │ Results  │ │         │ │ P(fake) │ │
│             │         │ └────┬─────┘ │         │ └─────────┘ │
│             │         │      │       │         │             │
│ RED WINS if │         │ ┌────▼─────┐ │         │ BLUE WINS   │
│ undetected  │         │ │ 6. Set   │ │         │ if correct  │
│             │         │ │ Weights  │ │         │             │
│             │         │ └──────────┘ │         │             │
└─────────────┘         └──────────────┘         └─────────────┘
```

### Key Components

**Validator Side:**
1. Scrape random real video from web (YouTube, social media, news, etc.)
2. Generate text description via video→text model
3. Send prompt to Red Team miners
4. Collect generated fake + original real
5. Shuffle and send both to Blue Team miners
6. Score based on detection accuracy and generation quality

**Red Team (Generators):**
- Receive text prompt describing a video
- Generate synthetic video matching the description
- Goal: Create fakes that evade detection
- Can be: API endpoints, models on Chutes, custom infrastructure
- **Anonymity acceptable** — endpoints queryable by prompt, output directly tested

**Blue Team (Detectors):**
- Receive pair of videos (one real, one fake, shuffled)
- Return probability each is AI-generated
- Goal: Correctly classify real vs fake
- **Must be auditable**: Open source models on Chutes, Docker images, or inspectable code
- Transparency required for mechanism integrity

**Scoring:**
```python
def score_round(real_video, fake_video, red_uid, blue_responses):
    """Score one adversarial round"""
    
    red_scores = {}
    blue_scores = {}
    
    for blue_uid, response in blue_responses.items():
        p_real_is_fake = response["real_video_prob"]
        p_fake_is_fake = response["fake_video_prob"]
        
        # Blue Team: reward correct detection
        # Correctly low prob on real, high prob on fake
        blue_score = (1 - p_real_is_fake) * 0.5 + p_fake_is_fake * 0.5
        blue_scores[blue_uid] = blue_score
        
    # Red Team: reward if fake was undetected
    avg_detection = mean([r["fake_video_prob"] for r in blue_responses.values()])
    red_scores[red_uid] = 1.0 - avg_detection
    
    return red_scores, blue_scores
```

**Why This Works:**
- **Validators don't control ground truth** — miners generate the fakes
- **No answer to leak** — detection difficulty is emergent from Red Team quality
- **Arms race dynamics** improve both sides over time
- Better generators → better training signal for detectors
- Better detectors → pressure for better generators

**Fresh Content via Web Scraping:**
- Real videos sourced from the entire web at scale
- Massive dataset size prevents memorization
- Timestamps and URLs provide provenance
- New content constantly entering the system

**Anti-Gaming:**
- Blue Team must be open source/auditable (prevents hiding leaked info)
- Fresh real videos from web (can't pre-memorize)
- Red Team anonymity acceptable since their output is directly tested
- Multiple Blue Team miners create consensus signal

**Miner Infrastructure Options:**
| Team | Allowed Infrastructure | Why |
|------|----------------------|-----|
| Red (Generators) | Anonymous endpoints, Chutes, any API | Output is directly tested — don't need to trust code |
| Blue (Detectors) | Open source on Chutes, Docker images, auditable code | Must verify they're not receiving leaked answers |

---

## Choosing Your Pattern

| Pattern | Best For | Key Challenge |
|---------|----------|---------------|
| Compute Auction | Commodity hardware | Hardware verification |
| Capacity Market | Always-on infra | Usage tracking |
| Data Indexing | Unique datasets | Anti-duplication |
| Prediction Market | Future forecasts | Delayed ground truth |
| External Verification | Third-party activity | API trust & filtering |
| Time-Series | Sequential predictions | Baseline comparison |
| Adversarial Red/Blue | Detection/generation tasks | Balancing both teams |

### Questions to Guide Choice

1. **What is the commodity?**
   - Compute → Pattern 1 or 2
   - Data → Pattern 3
   - Predictions → Pattern 4 or 6
   - External activity → Pattern 5
   - Detection/generation capabilities → Pattern 7

2. **How is quality verified?**
   - Hardware attestation → Pattern 1, 2
   - Re-sampling → Pattern 3
   - Ground truth → Pattern 4, 6
   - External API → Pattern 5
   - Adversarial testing → Pattern 7

3. **When is ground truth available?**
   - Immediately → Pattern 1, 2, 3
   - After delay → Pattern 4, 6
   - From third party → Pattern 5
   - Emergent from miner competition → Pattern 7

4. **Is there a natural marketplace?**
   - Pricing competition → Pattern 1
   - Capacity competition → Pattern 2
   - Uniqueness competition → Pattern 3, 5
   - Accuracy competition → Pattern 4, 6
   - Arms race competition → Pattern 7

5. **Who controls ground truth?**
   - Validator controls it → **Danger!** Consider Pattern 5, 6, or 7 instead
   - External source → Pattern 5, 6
   - Miners create challenges for each other → Pattern 7

### Hybrid Approaches
Many successful subnets combine patterns:
- Data indexing + freshness bonuses
- Compute auction + quality verification
- External activity + on-chain contribution tracking
- Adversarial generation + web-scraped real content (Pattern 7)
