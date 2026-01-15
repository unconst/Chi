# Core Concepts

## Subnets and NetUIDs

### What is a Subnet?
A subnet is an independent network within Bittensor focused on a specific digital commodity. Each subnet has:
- Its own set of neurons (miners and validators)
- Custom incentive mechanism defined by the creator
- Unique hyperparameters
- Its own Alpha token for staking

### NetUID (Network Unique Identifier)
- Unique integer identifying each subnet
- Range: 0 to 65535 (u16)
- NetUID 0 is reserved for the Root Network
- Assigned when subnets are created
- Currently maximum ~64 subnets can exist simultaneously
- When full, new creation may prune lowest-value subnet

## Neurons and UIDs

### What is a Neuron?
A neuron is a participant (miner or validator) registered on a subnet. Each neuron has:
- A **UID** (unique identifier within the subnet)
- Associated **hotkey** (for signing operations)
- Associated **coldkey** (for fund management)
- Network endpoint (IP and port for serving)

### UID (User Identifier)
- Integer identifying a neuron within a subnet
- Range: 0 to max_neurons-1 (typically 0-255 or 0-4095)
- Unique within a subnet, NOT globally unique
- Can be recycled when neurons are pruned

### Neuron Attributes
Stored on-chain and accessible via metagraph:
- **stake**: Amount staked on this neuron
- **trust**: How much other neurons trust this one
- **consensus**: Agreement level with network consensus
- **incentive**: Rewards earned (for miners)
- **dividends**: Rewards earned (for validators)
- **emission**: Total emission received
- **active**: Whether neuron is currently active
- **validator_permit**: Whether neuron can act as validator
- **last_update**: Block when neuron last set weights
- **rank**: Neuron's rank in the subnet

## Coldkeys and Hotkeys

### Wallet Structure
A Bittensor wallet consists of two key pairs:

#### Coldkey (Cold Storage Key)
- **Primary wallet key** for fund management
- Used for:
  - Staking and unstaking TAO
  - Creating subnets
  - Transferring TAO
  - High-security operations
  - Subnet ownership
- **Should be kept secure, ideally offline**
- One coldkey can control multiple hotkeys
- SS58 encoded address (e.g., "5Gx8...")

#### Hotkey (Hot Operations Key)
- **Operational key** for day-to-day neuron operations
- Used for:
  - Registering on subnets
  - Setting weights (validators)
  - Signing requests (miners/validators)
  - Signing requests and responses
- Can be kept on servers running neurons
- Associated with a controlling coldkey
- SS58 encoded address

### Key Relationships
```
Coldkey (fund owner / subnet owner)
     │
     ├── Hotkey 1 (miner on subnet 1)
     │
     ├── Hotkey 2 (validator on subnet 2)
     │
     └── Hotkey 3 (miner on subnet 3)
```

## TAO, Alpha, and RAO

### TAO
- Main currency of Bittensor
- Max supply: 21,000,000 TAO
- Current emission rate decreases over time (halving)
- Used for all economic operations

### RAO
- Smallest unit of TAO
- 1 TAO = 1,000,000,000 RAO (10^9)
- All internal calculations use RAO for precision
- Similar to Satoshi for Bitcoin

Conversion:
```python
TAO_amount = Rao_amount / 1e9
Rao_amount = TAO_amount * 1e9
```

### Alpha (Subnet Token)
- Each subnet has its own Alpha token
- Represents stake share within that subnet
- Created when TAO is staked into a subnet
- Destroyed when unstaking back to TAO

Alpha Pool Mechanism:
- Each subnet has a liquidity pool (TAO ↔ Alpha)
- Price determined by pool ratio
- Staking TAO → Receive Alpha (buy Alpha)
- Unstaking → Receive TAO (sell Alpha)
- Slippage occurs for large transactions

## Stake, Delegation, and Emissions

### Staking
Staking is locking TAO to participate in the network:

**Direct Staking**:
- Stake your TAO on your own hotkey
- Control the neuron directly
- Receive full emissions minus network fees

**Delegation**:
- Stake your TAO on someone else's hotkey
- They operate the neuron
- You receive a portion of their emissions
- Delegator take rate configurable

### Stake Requirements
- Validators need minimum stake for "validator_permit"
- Higher stake = more influence on weights
- Stake determines voting power in consensus

### Emissions
New TAO is created each block and distributed:

1. **Block Emission**: Total new TAO created per block
2. **Subnet Allocation**: Root network determines per-subnet emission
3. **Within Subnet**:
   - **Incentive**: Goes to miners based on their weights
   - **Dividends**: Goes to validators based on their bonds

## Bonds

### What are Bonds?
Bonds represent a validator's commitment to specific miners:
- Created when validators set weights
- Accumulate over time (exponential moving average)
- Determine dividend distribution

### Bond Mechanics
- When validator V sets weight W on miner M
- Bond B(V,M) increases proportionally
- Higher bonds = more dividends from that miner's work
- Bonds decay when weights are removed

### Liquid Alpha
Advanced bonding mechanism that:
- Adjusts bond accumulation rate dynamically
- Prevents manipulation through weight-timing games
- Uses alpha_high and alpha_low hyperparameters

## Metagraph

The metagraph is a data structure containing the complete state of a subnet:

### Core Attributes
- `n`: Number of neurons
- `block`: Current block number
- `netuid`: Subnet identifier
- `network`: Network name

### Per-Neuron Arrays
- `uids`: Neuron identifiers
- `stake`: Stake amounts
- `trust`: Trust scores
- `consensus`: Consensus scores
- `incentive`: Incentive values
- `dividends`: Dividend values
- `emission`: Emission amounts
- `active`: Activity status
- `last_update`: Last weight update block
- `validator_permit`: Validator status
- `validator_trust`: Validator trust scores
- `hotkeys`: Hotkey addresses

### Matrices (Full Mode Only)
- **Weight Matrix (W)**: W[i][j] = weight set by neuron i on neuron j
- **Bond Matrix (B)**: B[i][j] = bond of validator i in miner j

### Access Patterns
- **Lite mode**: Fast access without weights/bonds
- **Full mode**: Includes weights and bonds (slower)

### Usage
```python
from bittensor import Metagraph, Subtensor

subtensor = Subtensor(network="finney")
metagraph = Metagraph(netuid=1, network="finney")
metagraph.sync(subtensor=subtensor)

# Access data
print(f"Neurons: {metagraph.n}")
print(f"Stakes: {metagraph.S}")
print(f"Incentives: {metagraph.I}")
```

### Caching
Metagraphs are cached on disk by default:
```
~/.bittensor/metagraphs/network-{network}/netuid-{netuid}/block-{block_number}.pt
```

## Tempo and Epochs

### Tempo
- Number of blocks between consensus epochs
- Configurable per subnet (typical: 100-360 blocks)
- At tempo completion, emissions are distributed

### Epoch
The period between weight-clearing events where:
- Weights are processed
- Consensus is calculated
- Emissions are distributed
- Bonds are updated

### Block Time
- Approximately 12 seconds per block on mainnet
- Faster (~6 seconds) on localnet
