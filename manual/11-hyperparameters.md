# Subnet Hyperparameters

Hyperparameters configure subnet behavior. Only the subnet owner can modify them.

**IMPORTANT: Not all hyperparameters listed are actually changeable.** Some are locked by the chain even though they appear as hyperparameters. See notes below for each parameter.

## Viewing Hyperparameters

**CLI:**
```bash
btcli subnet hyperparameters --netuid <NETUID> --network finney
```

**SDK:**
```python
from bittensor import Subtensor

subtensor = Subtensor(network="finney")
params = subtensor.get_subnet_hyperparameters(netuid=1)

print(f"Tempo: {params.tempo}")
print(f"Max UIDs: {params.max_allowed_uids}")
```

## Setting Hyperparameters

**CLI:**
```bash
btcli sudo set \
  --netuid <NETUID> \
  --param <parameter> \
  --value <value> \
  --wallet.name <owner_wallet>
```

**SDK:**
```python
subtensor.set_hyperparameter(
    wallet=wallet,
    netuid=netuid,
    parameter="tempo",
    value=100
)
```

---

## Core Parameters

### `tempo`
**Default:** 360  
**⚠️ NOT CHANGEABLE** - Even though this appears as a hyperparameter, it cannot be modified by subnet owners currently.

Number of blocks between epoch calculations. Each tempo:
- Consensus is recalculated
- Emissions are distributed
- Bonds are updated

### `max_allowed_uids`
**Default:** 256  
**⚠️ NOT CHANGEABLE** - This parameter is locked and cannot be modified.

Maximum number of neurons (UIDs) on the subnet. Range: 1-4096.

Higher values allow more participants but:
- Increase metagraph size
- Require more weight entries from validators
- May dilute emissions per miner

### `immunity_period`
**Default:** 7200  
**Status:** Changeable

Blocks a new neuron is protected from deregistration (pruning). During immunity:
- Neuron cannot be replaced regardless of performance
- Gives time to set up and start receiving weights

Note: 7200 blocks ≈ 24 hours on mainnet (12 second block time).

```bash
btcli sudo set --netuid 1 --param immunity_period --value 7200
```

---

## Weight Parameters

### `weights_rate_limit`
**Default:** 100

Minimum blocks between weight submissions for a single hotkey. Prevents:
- Weight spam
- Excessive chain load
- Gaming through rapid weight changes

```bash
btcli sudo set --netuid 1 --param weights_rate_limit --value 50
```

### `max_weights_limit`
**Default:** 64

Maximum number of UIDs a validator can set weights on per submission. Must be:
- Greater than `min_allowed_weights`
- Less than or equal to `max_allowed_uids`

```bash
btcli sudo set --netuid 1 --param max_weights_limit --value 128
```

### `min_allowed_weights`
**Default:** 1

Minimum number of UIDs required in each weight submission. Prevents:
- Single-target weight attacks
- Overly sparse weight vectors

```bash
btcli sudo set --netuid 1 --param min_allowed_weights --value 8
```

### `commit_reveal_weights_enabled`
**Default:** false  
**⚠️ USE WITH CAUTION** - Don't enable unless absolutely necessary.

Enable commit-reveal for weight privacy. When enabled:
1. Validators commit hash of weights
2. After `commit_reveal_period`, reveal actual weights
3. Prevents weight copying between validators

**Practical guidance:**
- Weight copying is primarily a **mature subnet problem**
- For new subnets, focus on building the mechanism first
- Adds complexity and can cause issues if not properly handled
- Only enable if you have evidence of widespread weight copying

```bash
# Only if absolutely necessary:
btcli sudo set --netuid 1 --param commit_reveal_weights_enabled --value true
```

### `commit_reveal_period`
**Default:** 0 (when disabled)

Blocks between commit and allowed reveal.

**Practical guidance:**
- A period of 50-100 blocks is **not long enough** to prevent validator copying in practice
- If you're using commit-reveal, you need significantly longer periods
- However, longer periods delay weight updates which affects responsiveness

```bash
# If using commit-reveal, use longer periods:
btcli sudo set --netuid 1 --param commit_reveal_period --value 360
```

---

## Registration Parameters

### `min_burn`
**Default:** Varies by network conditions

Minimum TAO required for burn registration. Higher values:
- Increase barrier to entry
- Reduce Sybil attacks
- May discourage legitimate participants

### `max_burn`
**Default:** Varies

Maximum burn cost (dynamic ceiling).

### `difficulty`
**Default:** 10,000,000

PoW difficulty for proof-of-work registration. Higher values:
- Require more computation to register
- Slow down registration rate
- Alternative to burn registration

### `target_regs_per_interval`
**Default:** 2

Target number of registrations per `adjustment_interval`. Used for dynamic difficulty adjustment.

### `adjustment_interval`
**Default:** 100

Blocks between difficulty adjustments.

### `registration_allowed`
**Default:** true

Whether new registrations are permitted.

```bash
# Disable registrations (freeze subnet)
btcli sudo set --netuid 1 --param registration_allowed --value false
```

---

## Validator Parameters

### `min_stake_to_set_weights`
**Default:** 0 (subnet-dependent)

Minimum stake required to submit weights. Prevents:
- Low-stake validators from influencing consensus
- Weight spam from unstaked hotkeys

### `validator_permit`
Determined automatically based on stake ranking. Top staked neurons get validator permits.

---

## Emission Parameters

### `alpha_high` / `alpha_low`
Parameters for Liquid Alpha bond mechanism.

- `alpha_high`: Maximum bond growth rate
- `alpha_low`: Minimum bond growth rate

Dynamic based on stake and weight consistency.

### `kappa`
**Default:** 32767

Consensus strictness parameter. Affects how stake-weighted median is calculated:
- Higher kappa = stricter consensus (requires more agreement)
- Lower kappa = looser consensus

---

## Default Values at Creation

When a subnet is created, it starts with these defaults:

| Parameter | Default Value |
|-----------|---------------|
| `tempo` | 360 |
| `max_allowed_uids` | 256 |
| `immunity_period` | 4096 |
| `weights_rate_limit` | 100 |
| `max_weights_limit` | 64 |
| `min_allowed_weights` | 1 |
| `difficulty` | 10,000,000 |
| `commit_reveal_weights_enabled` | false |

---

## Configuration Strategy

### Reality Check: What You Can Actually Change

Many hyperparameters that appear in the docs **cannot actually be changed** by subnet owners:
- `tempo` - **NOT changeable**
- `max_allowed_uids` - **NOT changeable**

Focus on the parameters you CAN control.

### For New Subnets

```bash
# Start with defaults for most things

# Require meaningful weight distribution
btcli sudo set --netuid $NETUID --param min_allowed_weights --value 4

# Adjust weights rate limit if needed
btcli sudo set --netuid $NETUID --param weights_rate_limit --value 50
```

### For Mature Subnets

```bash
# Only enable commit-reveal if weight copying is a proven problem:
# btcli sudo set --netuid $NETUID --param commit_reveal_weights_enabled --value true
# btcli sudo set --netuid $NETUID --param commit_reveal_period --value 360

# Note: Don't worry too much about weight copying for new subnets
# Focus on mechanism quality first
```

### For High-Throughput Subnets

```bash
# Faster weight updates
btcli sudo set --netuid $NETUID --param weights_rate_limit --value 25

# More neurons per weight vector
btcli sudo set --netuid $NETUID --param max_weights_limit --value 256
```

---

## Reading All Parameters (SDK)

```python
from bittensor import Subtensor

subtensor = Subtensor(network="finney")
params = subtensor.get_subnet_hyperparameters(netuid=1)

# Access all available parameters
print(f"tempo: {params.tempo}")
print(f"max_allowed_uids: {params.max_allowed_uids}")
print(f"immunity_period: {params.immunity_period}")
print(f"weights_rate_limit: {params.weights_rate_limit}")
print(f"max_weights_limit: {params.max_weights_limit}")
print(f"min_allowed_weights: {params.min_allowed_weights}")
print(f"commit_reveal_weights_enabled: {params.commit_reveal_weights_enabled}")
print(f"commit_reveal_period: {params.commit_reveal_period}")
print(f"difficulty: {params.difficulty}")
print(f"kappa: {params.kappa}")
print(f"alpha_high: {params.alpha_high}")
print(f"alpha_low: {params.alpha_low}")
```
