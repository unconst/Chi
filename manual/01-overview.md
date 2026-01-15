# Bittensor Overview

## What Bittensor Is

Bittensor is a **decentralized machine intelligence network** that creates marketplaces for digital commodities (primarily AI/ML services) using blockchain-based incentives. It runs on a Substrate-based blockchain called **Subtensor**.

The network operates through **subnets** - specialized independent markets that each focus on a specific task (text generation, image generation, data scraping, compute provision, etc.). Participants earn **TAO tokens** for contributing valuable work.

## The Three-Layer Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CHAIN LAYER                          │
│                  (Subtensor/Substrate)                  │
│  - Stores state (registrations, weights, stake)         │
│  - Enforces rules (rate limits, permissions)            │
│  - Calculates emissions via consensus                   │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│                     SDK LAYER                           │
│                    (Python SDK)                         │
│  - Wraps chain queries and extrinsics                   │
│  - Provides wallet and metagraph management             │
│  - Manages wallets and metagraph state                  │
└─────────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────────┐
│                   SUBNET LAYER                          │
│                   (Your Code)                           │
│  - Request/response types (Synapses or custom)          │
│  - Miners (servers providing the commodity)             │
│  - Validators (clients scoring and setting weights)     │
└─────────────────────────────────────────────────────────┘
```

## Key Participants

### 1. Subnet Creators (Owners)
- Deploy new subnets by paying a dynamic burn/lock cost
- Define the incentive mechanism and rules
- Configure hyperparameters
- Receive a portion of emissions as subnet owner
- Control subnet identity and metadata

### 2. Miners
- Provide the subnet's digital commodity (model inference, compute, data, etc.)
- Register on a subnet with a hotkey
- Publish endpoints so validators can reach them
- Earn TAO based on their **incentive** score (how validators value them)
- Compete to provide the best quality/speed/value

### 3. Validators
- Query miners and evaluate their responses
- Run scoring algorithms to determine miner quality
- Submit **weights** on-chain that determine miner rewards
- Earn TAO through **dividends** for accurate validation
- Must have sufficient stake to have influence

### 4. Stakers / Delegators
- Stake TAO behind validators or miners
- Earn emissions proportional to their stake
- Can delegate to hotkeys they don't operate
- Provide economic security and voting power

## Network Structure

```
                    BITTENSOR BLOCKCHAIN
                    (Subtensor Chain)
                           │
          ┌────────────────┼────────────────┐
          │                │                │
    ┌─────▼─────┐    ┌─────▼─────┐    ┌─────▼─────┐
    │  SUBNET 0 │    │  SUBNET 1 │    │  SUBNET N │
    │   (Root)  │    │ (netuid=1)│    │ (netuid=N)│
    │ Governance│    │   Task A  │    │   Task N  │
    └───────────┘    └───────────┘    └───────────┘
                           │
              ┌────────────┼────────────┐
              │            │            │
         ┌────▼────┐  ┌────▼────┐  ┌────▼────┐
         │ Miner 0 │  │ Miner 1 │  │ Vali 0  │
         │ (uid=0) │  │ (uid=1) │  │ (uid=2) │
         └─────────┘  └─────────┘  └─────────┘
```

### Special Subnet: Root Network (netuid=0)
- The governance and emissions allocation subnet
- Validators on root set weights on other subnets
- Determines how emissions are distributed across subnets
- Requires significant stake to participate
- No miners, only validators

## Tokenomics

### TAO Token
- Native currency of Bittensor
- Total supply: 21 million (like Bitcoin)
- Used for staking, registration, and rewards
- Halving schedule similar to Bitcoin
- ~7,200 TAO emitted daily (decreasing over time)

### RAO
- Smallest unit of TAO
- 1 TAO = 10^9 RAO (1 billion RAO)
- All internal calculations use RAO for precision

### Alpha Tokens (Subnet-Specific)
- Each subnet has its own Alpha token
- Staking TAO into a subnet gives you Alpha
- Alpha represents your share of the subnet's stake
- Dynamic mechanism converts between TAO and Alpha
- Exchange rate determined by pool ratios

## Emission Distribution

Per block, emissions are distributed:

1. **Root network validators** allocate emissions to subnets based on weights
2. **Within each subnet**, emissions split between:
   - **Miners (Incentive)**: based on validator-assigned weights
   - **Validators (Dividends)**: based on bonds and stake
   - **Subnet Owner**: configurable percentage

Simplified formula:
```
miner_reward = weight × incentive_ratio × subnet_emission
validator_reward = bond × dividend_ratio × subnet_emission
```

## How Value Flows

1. Stakers lock TAO in the network
2. TAO converts to Alpha in specific subnets
3. Miners provide work; validators evaluate and set weights
4. The blockchain calculates consensus (Yuma Consensus)
5. Emissions distribute based on weights and bonds
6. Participants claim TAO rewards

## Why Bittensor?

- **Decentralized AI**: No single entity controls the intelligence
- **Permissionless**: Anyone can create a subnet or participate
- **Incentivized Quality**: Economics naturally reward best performers
- **Scalable**: New subnets can be added without affecting existing ones
- **Censorship Resistant**: Distributed control and ownership
