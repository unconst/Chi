# Subnet Lifecycle

This document covers the complete journey from subnet creation to operation.

## Phase 1: Preparation

### 1.1 Key Setup

Create coldkey (fund owner) and hotkey (operator):
```bash
# Create coldkey for subnet ownership
btcli wallet new_coldkey --wallet.name subnet_owner

# Create hotkey for running subnet operations
btcli wallet new_hotkey --wallet.name subnet_owner --wallet.hotkey operator
```

Programmatic creation:
```python
from bittensor_wallet import Wallet

wallet = Wallet(name="subnet_owner", hotkey="operator")
wallet.create_if_non_existent()

# Get addresses
print(f"Coldkey: {wallet.coldkey.ss58_address}")
print(f"Hotkey: {wallet.hotkey.ss58_address}")
```

### 1.2 Fund Acquisition

Creating a subnet requires TAO:
- **Localnet**: Free faucet TAO available
- **Testnet**: Faucet available (limited)
- **Mainnet**: Purchase TAO from exchanges

Check current subnet creation cost:
```bash
btcli subnet lock-cost --network finney
```

```python
from bittensor import Subtensor

subtensor = Subtensor(network="finney")
cost = subtensor.get_subnet_burn_cost()
print(f"Current burn cost: {cost.tao} TAO")
```

### 1.3 Understand Network Limits

```python
# Check current subnet count and capacity
total_subnets = subtensor.get_total_subnets()
max_subnets = 64  # Current limit

print(f"Subnets: {total_subnets}/{max_subnets}")
```

If network is full, creating a new subnet may cause the lowest-emission subnet to be pruned.

---

## Phase 2: Subnet Creation

### 2.1 Register Subnet

**Via CLI:**
```bash
btcli subnet create \
  --wallet.name subnet_owner \
  --wallet.hotkey operator \
  --network finney
```

**Via SDK:**
```python
from bittensor import Subtensor
from bittensor_wallet import Wallet

wallet = Wallet(name="subnet_owner", hotkey="operator")
subtensor = Subtensor(network="finney")

# Register creates the subnet
success, netuid = subtensor.register_network(wallet=wallet)

if success:
    print(f"Subnet created with netuid: {netuid}")
```

### 2.2 Activation Window

After creation, there's a waiting period before activation:

```
┌─────────────────────────────────────────────────────┐
│  CREATION          ACTIVATION WINDOW      ACTIVE    │
│      │                   │                   │      │
│      ▼                   ▼                   ▼      │
│  ────●───────────────────●───────────────────●────  │
│      │                   │                   │      │
│  netuid                window               subnet  │
│  assigned               opens              serves   │
│                                                     │
│  ~2016 blocks (24 hours) waiting period             │
└─────────────────────────────────────────────────────┘
```

Check activation status:
```python
# Check blocks until activation
creation_block = subtensor.get_subnet_creation_block(netuid)
current_block = subtensor.get_current_block()
activation_delay = 2016  # blocks

blocks_remaining = (creation_block + activation_delay) - current_block
print(f"Blocks until activation: {max(0, blocks_remaining)}")
```

### 2.3 Activate Subnet

Once window opens:
```python
# Activation extrinsic
success = subtensor.activate_subnet(
    wallet=wallet,
    netuid=netuid
)
```

---

## Phase 3: Configuration

### 3.1 Hyperparameter Overview

Subnets are configured via hyperparameters. Key ones:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tempo` | 360 | Blocks between epochs |
| `max_allowed_uids` | 256 | Maximum neurons |
| `immunity_period` | 4096 | Blocks before pruning risk |
| `weights_rate_limit` | 100 | Min blocks between weight updates |
| `max_weights_limit` | 64 | Max neurons in weight vector |
| `min_allowed_weights` | 1 | Min weight entries required |
| `adjustment_interval` | 100 | Difficulty adjustment frequency |
| `difficulty` | 10M | PoW difficulty (if enabled) |
| `burn` | 1 TAO | Registration burn (if enabled) |

### 3.2 Setting Hyperparameters

**Via CLI:**
```bash
# Set tempo
btcli sudo set --netuid <NETUID> \
  --param tempo --value 100 \
  --wallet.name subnet_owner \
  --network finney

# Set max neurons
btcli sudo set --netuid <NETUID> \
  --param max_allowed_uids --value 512 \
  --wallet.name subnet_owner
```

**Via SDK:**
```python
# Set multiple parameters
subtensor.set_hyperparameter(
    wallet=wallet,
    netuid=netuid,
    parameter="tempo",
    value=100
)

subtensor.set_hyperparameter(
    wallet=wallet,
    netuid=netuid,
    parameter="max_allowed_uids",
    value=512
)
```

### 3.3 Read Current Hyperparameters

```bash
btcli subnet hyperparameters --netuid <NETUID>
```

```python
params = subtensor.get_subnet_hyperparameters(netuid)
print(f"Tempo: {params.tempo}")
print(f"Max UIDs: {params.max_allowed_uids}")
print(f"Immunity: {params.immunity_period}")
```

### 3.4 Commit-Reveal Configuration

For weight privacy (prevents copying):
```python
# Enable commit-reveal
subtensor.set_hyperparameter(
    wallet=wallet,
    netuid=netuid,
    parameter="commit_reveal_weights_enabled",
    value=True
)

# Set reveal period
subtensor.set_hyperparameter(
    wallet=wallet,
    netuid=netuid,
    parameter="commit_reveal_period",
    value=100  # blocks
)
```

---

## Phase 4: Neuron Registration

### 4.1 Registration Methods

Neurons register via **PoW (Proof of Work)** or **Burn**:

**PoW Registration:**
```bash
btcli subnet pow_register \
  --netuid <NETUID> \
  --wallet.name miner \
  --wallet.hotkey miner_hotkey
```

**Burn Registration:**
```bash
btcli subnet register \
  --netuid <NETUID> \
  --wallet.name miner \
  --wallet.hotkey miner_hotkey
```

**SDK:**
```python
# Burn registration
success = subtensor.burned_register(
    wallet=wallet,
    netuid=netuid
)

# PoW registration (compute-intensive)
success = subtensor.register(
    wallet=wallet,
    netuid=netuid
)
```

### 4.2 Registration Parameters

Subnet owner controls:
- `min_burn`: Minimum TAO to burn for registration
- `max_burn`: Maximum burn cost
- `difficulty`: PoW difficulty
- `registration_allowed`: Enable/disable registrations
- `target_regs_per_interval`: Target registration rate

### 4.3 Check Registration Status

```python
# Check if hotkey registered
is_registered = subtensor.is_hotkey_registered(
    netuid=netuid,
    hotkey_ss58=wallet.hotkey.ss58_address
)

if is_registered:
    uid = subtensor.get_uid_for_hotkey_on_subnet(
        netuid=netuid,
        hotkey_ss58=wallet.hotkey.ss58_address
    )
    print(f"Registered as UID: {uid}")
```

---

## Phase 5: Operation

### 5.1 Miner Operations

Miners must:
1. Register on subnet
2. Commit endpoint info to chain (if applicable)
3. Run serving logic (HTTP API recommended)
4. Respond to validator queries

Basic HTTP API setup:
```python
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.post("/compute")
async def compute(request: ComputeRequest):
    # Verify Epistula signature from validator
    # Process request
    # Return response
    return {"result": process(request.data)}

# Commit your endpoint URL to chain for validator discovery
uvicorn.run(app, host="0.0.0.0", port=8091)
```

### 5.2 Validator Operations

Validators must:
1. Register on subnet
2. Query miners and evaluate
3. Set weights periodically
4. Meet rate limits and stake requirements

Basic weight setting:
```python
# Prepare weights (must sum correctly)
uids = [0, 1, 2, 3]
weights = [0.4, 0.3, 0.2, 0.1]

# Submit to chain
success = subtensor.set_weights(
    wallet=wallet,
    netuid=netuid,
    uids=uids,
    weights=weights,
    wait_for_inclusion=True
)
```

### 5.3 Metagraph Monitoring

```python
from bittensor import Metagraph

metagraph = Metagraph(netuid=netuid, network="finney")
metagraph.sync()

# Monitor key metrics
for uid in range(metagraph.n):
    stake = metagraph.S[uid]
    incentive = metagraph.I[uid]
    emission = metagraph.E[uid]
    print(f"UID {uid}: stake={stake:.2f}, I={incentive:.4f}, E={emission}")
```

---

## Phase 6: Maintenance

### 6.1 Hyperparameter Tuning

Adjust based on observation:
```python
# If weights being copied too easily
subtensor.set_hyperparameter(
    wallet=wallet, netuid=netuid,
    parameter="commit_reveal_weights_enabled",
    value=True
)

# If need more neurons
subtensor.set_hyperparameter(
    wallet=wallet, netuid=netuid,
    parameter="max_allowed_uids",
    value=1024
)

# If epoch too slow
subtensor.set_hyperparameter(
    wallet=wallet, netuid=netuid,
    parameter="tempo",
    value=100
)
```

### 6.2 Identity and Metadata

Set subnet identity:
```bash
btcli subnet identity set \
  --netuid <NETUID> \
  --display "My Subnet" \
  --description "Provides amazing AI service" \
  --wallet.name subnet_owner
```

### 6.3 Ownership Transfer

```python
# Transfer to new coldkey
subtensor.set_subnet_owner(
    wallet=wallet,
    netuid=netuid,
    new_owner_ss58="5NewColdkeyAddress..."
)
```

---

## Lifecycle Summary

```
┌──────────────────────────────────────────────────────────┐
│                    SUBNET LIFECYCLE                      │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  1. PREPARE                                              │
│     └─► Create keys, acquire TAO                         │
│                                                          │
│  2. CREATE                                               │
│     └─► Register subnet, pay burn/lock                   │
│                                                          │
│  3. WAIT                                                 │
│     └─► Activation window (~24 hours)                    │
│                                                          │
│  4. ACTIVATE                                             │
│     └─► Submit activation extrinsic                      │
│                                                          │
│  5. CONFIGURE                                            │
│     └─► Set hyperparameters, identity                    │
│                                                          │
│  6. REGISTER NEURONS                                     │
│     └─► Miners and validators join                       │
│                                                          │
│  7. OPERATE                                              │
│     └─► Mining, validation, weight setting               │
│                                                          │
│  8. MAINTAIN                                             │
│     └─► Monitor, tune, upgrade                           │
│                                                          │
│  [RISK] DEREGISTER                                       │
│     └─► If lowest emission and new subnet created        │
│                                                          │
└──────────────────────────────────────────────────────────┘
```
