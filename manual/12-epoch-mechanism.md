# Epoch Mechanism

The epoch mechanism is the core on-chain calculation that translates validator weights into miner rewards and validator dividends. It runs every `tempo` blocks.

## What Happens Each Epoch

```
┌─────────────────────────────────────────────────────────┐
│                     EPOCH CYCLE                         │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  INPUTS                                                 │
│  ├─► Weight matrix W[validator][miner]                  │
│  ├─► Stake vector S[neuron]                            │
│  ├─► Bond matrix B[validator][miner] (previous)         │
│  └─► Hyperparameters (tempo, kappa, alpha_*)            │
│                                                         │
│  PROCESSING                                             │
│  ├─► Filter active neurons                              │
│  ├─► Determine validator permits                        │
│  ├─► Calculate stake-weighted consensus                 │
│  ├─► Compute trust, rank, incentive                     │
│  ├─► Update bonds (EMA with Liquid Alpha)               │
│  └─► Calculate dividends                                │
│                                                         │
│  OUTPUTS                                                │
│  ├─► Incentive I[miner] → miner rewards                │
│  ├─► Dividends D[validator] → validator rewards         │
│  ├─► Updated bonds B'[validator][miner]                 │
│  └─► Emissions distributed to accounts                  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Epoch Timing

- **Tempo**: Number of blocks between epochs (configurable per subnet)
- **Block time**: ~12 seconds on mainnet
- **Epoch frequency**: tempo × 12 seconds

Example: tempo=360 → epoch every ~72 minutes

## Yuma Consensus Algorithm

The core consensus algorithm that processes weights:

### Step 1: Filter Active Neurons
Only neurons that have set weights recently are considered active.

### Step 2: Identify Validators
Neurons with `validator_permit=true` (based on stake ranking).

### Step 3: Process Weight Matrix
```
For each validator v:
  For each miner m:
    W[v][m] = weight set by v on m (normalized)
```

### Step 4: Calculate Consensus (Stake-Weighted Median)
```
For each miner m:
  Collect all weights W[*][m] from validators
  Weight each by validator stake S[v]
  consensus[m] = stake-weighted median (controlled by kappa)
```

The `kappa` parameter controls how strict consensus is:
- Higher kappa → stricter (requires more agreement)
- Lower kappa → looser (allows more variance)

### Step 5: Calculate Trust
```
For each miner m:
  trust[m] = agreement between validators on this miner
```

### Step 6: Calculate Rank and Incentive
```
For each miner m:
  rank[m] = consensus[m] × trust[m]
  
# Normalize ranks to get incentives
total_rank = sum(rank)
For each miner m:
  incentive[m] = rank[m] / total_rank
```

Incentive determines miner's share of miner emissions.

### Step 7: Update Bonds (EMA)
Bonds track validator commitment to miners:
```
For each validator v, miner m:
  old_bond = B[v][m]
  new_weight = W[v][m]
  
  # EMA update with Liquid Alpha
  alpha = calculate_alpha(v, m)  # Dynamic based on conditions
  B'[v][m] = alpha × new_weight + (1 - alpha) × old_bond
```

### Step 8: Calculate Dividends
```
For each validator v:
  dividend[v] = sum over m: (B'[v][m] × incentive[m])
  
# Normalize
total_dividend = sum(dividend)
For each validator v:
  dividends[v] = dividend[v] / total_dividend
```

Dividends determine validator's share of validator emissions.

### Step 9: Distribute Emissions
```
subnet_emission = allocated from root network

miner_pool = subnet_emission × 0.41
validator_pool = subnet_emission × 0.41
owner_pool = subnet_emission × 0.18

For each miner m:
  reward[m] = miner_pool × incentive[m]
  
For each validator v:
  reward[v] = validator_pool × dividends[v]
  
owner_reward = owner_pool
```

## Liquid Alpha

Advanced bonding mechanism that adjusts bond update rate:

```
alpha = alpha_low + (alpha_high - alpha_low) × factor

Where factor depends on:
- Stake consistency
- Weight consistency  
- Time since last update
```

**alpha_high**: Used when validator is consistent (bonds grow faster)
**alpha_low**: Used when validator is inconsistent (bonds grow slower)

This prevents:
- Weight-timing attacks
- Sudden bond shifts
- Manipulation through strategic weight changes

## Pseudocode

Simplified epoch calculation:

```python
def run_epoch(netuid, block):
    # Load state
    metagraph = get_metagraph(netuid)
    weights = get_weight_matrix(netuid)
    bonds = get_bond_matrix(netuid)
    stakes = get_stakes(netuid)
    params = get_hyperparameters(netuid)
    
    # Filter active
    active = get_active_neurons(metagraph, block)
    validators = get_validators(metagraph)
    
    # Calculate consensus
    consensus = {}
    for miner in active:
        validator_weights = []
        validator_stakes = []
        
        for v in validators:
            if weights[v][miner] > 0:
                validator_weights.append(weights[v][miner])
                validator_stakes.append(stakes[v])
        
        consensus[miner] = stake_weighted_median(
            validator_weights, 
            validator_stakes,
            params.kappa
        )
    
    # Calculate trust
    trust = {}
    for miner in active:
        agreements = []
        for v in validators:
            diff = abs(weights[v][miner] - consensus[miner])
            agreements.append(1.0 - diff)
        trust[miner] = mean(agreements)
    
    # Calculate rank and incentive
    rank = {m: consensus[m] * trust[m] for m in active}
    total_rank = sum(rank.values())
    incentive = {m: rank[m] / total_rank if total_rank > 0 else 0 
                 for m in active}
    
    # Update bonds
    new_bonds = {}
    for v in validators:
        new_bonds[v] = {}
        for m in active:
            alpha = calculate_liquid_alpha(v, m, params)
            old_bond = bonds.get(v, {}).get(m, 0)
            new_weight = weights.get(v, {}).get(m, 0)
            new_bonds[v][m] = alpha * new_weight + (1 - alpha) * old_bond
    
    # Calculate dividends
    dividend = {}
    for v in validators:
        dividend[v] = sum(
            new_bonds[v].get(m, 0) * incentive.get(m, 0) 
            for m in active
        )
    total_dividend = sum(dividend.values())
    dividends = {v: dividend[v] / total_dividend if total_dividend > 0 else 0 
                 for v in validators}
    
    # Distribute emissions
    emission = get_subnet_emission(netuid)
    miner_emission = emission * 0.41
    validator_emission = emission * 0.41
    owner_emission = emission * 0.18
    
    rewards = {}
    for m in active:
        rewards[m] = miner_emission * incentive.get(m, 0)
    for v in validators:
        rewards[v] = rewards.get(v, 0) + validator_emission * dividends.get(v, 0)
    
    # Credit rewards to accounts
    for neuron, reward in rewards.items():
        credit_account(neuron, reward)
    credit_account(get_owner(netuid), owner_emission)
    
    # Save updated bonds
    save_bond_matrix(netuid, new_bonds)
```

## Hyperparameters Affecting Epochs

| Parameter | Effect |
|-----------|--------|
| `tempo` | Frequency of epochs |
| `kappa` | Strictness of consensus |
| `alpha_high` | Max bond growth rate |
| `alpha_low` | Min bond growth rate |
| `weights_rate_limit` | How often weights can change |

## Reading Epoch Data

**From Metagraph:**
```python
metagraph = Metagraph(netuid=1)
metagraph.sync()

# Results of last epoch
incentives = metagraph.I      # Miner incentives
dividends = metagraph.D       # Validator dividends
emissions = metagraph.E       # Total emissions
consensus = metagraph.C       # Consensus scores
trust = metagraph.T          # Trust scores
rank = metagraph.R           # Rank values
bonds = metagraph.B          # Bond matrix (full mode)
```

## Key Insights

1. **Validators earn through bonds, not weights**: Setting weights builds bonds over time. Bonds determine dividend share.

2. **Consistency matters**: Liquid Alpha rewards consistent validators with faster bond growth.

3. **Consensus dilutes outliers**: If a validator sets very different weights than others, their influence is reduced.

4. **Stake amplifies influence**: Higher stake → more weight in consensus calculation.

5. **Miners earn through collective evaluation**: Incentive depends on agreement among validators.
