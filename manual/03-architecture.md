# Architecture

## Chain Layer (Subtensor)

### What is Subtensor?
Subtensor is Bittensor's blockchain, built on the Substrate framework. It provides:
- Decentralized state management
- Consensus mechanisms
- Economic rule enforcement
- Governance capabilities

### Key Components

#### Pallets
Substrate modules providing specific functionality:

**pallet-subtensor** (Core Pallet):
- Neuron registration logic
- Weight submission and validation
- Emission calculations
- Stake management
- Subnet lifecycle

**pallet-admin-utils**:
- Privileged operations
- Emergency controls
- Sudo commands

**pallet-balances**:
- TAO token management
- Transfer logic

### Storage Items (State)
Everything is stored as key-value pairs in the chain:
- Neuron registrations
- Stake amounts
- Weight matrices
- Subnet hyperparameters
- Block metadata

### Extrinsics (Transactions)
Signed calls that modify chain state:
- `register`: Register a neuron on a subnet
- `set_weights`: Submit validator weights
- `add_stake`: Stake TAO on a hotkey
- `register_network`: Create a new subnet
- `sudo_set_hyperparameters`: Modify subnet parameters

### Events
Notifications emitted when state changes:
- `NeuronRegistered`
- `WeightsSet`
- `StakeAdded`
- `SubnetCreated`

### Block Production
- ~12 second block time on mainnet
- Each block can contain multiple extrinsics
- Block hash = unique identifier

## SDK Layer (Python SDK)

### Purpose
The Python SDK abstracts blockchain complexity:
- Wraps substrate RPC calls
- Manages key signing
- Provides high-level objects (Subtensor, Metagraph, Wallet)
- Provides wallet and metagraph management

### Core Objects

#### Subtensor / AsyncSubtensor
Chain interface for queries and transactions:
```python
from bittensor import Subtensor, AsyncSubtensor

# Sync client
subtensor = Subtensor(network="finney")
block = subtensor.get_current_block()

# Async client  
async_subtensor = AsyncSubtensor(network="finney")
block = await async_subtensor.get_current_block()
```

Key methods:
- `get_current_block()`: Current chain height
- `get_balance()`: TAO balance for coldkey
- `metagraph()`: Get subnet metagraph
- `register_network()`: Create subnet
- `register()`: Register neuron
- `set_weights()`: Submit validator weights
- `add_stake()`: Stake TAO

#### Wallet
Key management:
```python
from bittensor_wallet import Wallet

wallet = Wallet(name="my_wallet", hotkey="my_hotkey")
wallet.create_if_non_existent()

coldkey_address = wallet.coldkey.ss58_address
hotkey_address = wallet.hotkey.ss58_address
```

#### Metagraph
Subnet state snapshot:
```python
from bittensor import Metagraph

metagraph = Metagraph(netuid=1, network="finney")
metagraph.sync()

# Access neuron data
stakes = metagraph.S
incentives = metagraph.I
hotkeys = metagraph.hotkeys
```

## Communication Protocol

### ⚠️ SDK Communication Primitives are Deprecated

**The Axon/Dendrite/Synapse pattern is deprecated.** New subnets should always use open source, custom communication methods. The SDK primitives were useful for early prototyping but have significant limitations and should not be used for new development.

**Why Axon/Dendrite should NOT be used:**
- Proprietary pattern that limits flexibility
- Poor performance for production use cases
- Encourages closed-source miner implementations
- Custom requirements not covered by SDK
- Production subnets universally roll their own communication

**What you should use instead:**
- Standard HTTP APIs (FastAPI, Flask, etc.)
- Epistula protocol for hotkey-based signing
- gRPC, WebSocket, or other open protocols as needed
- Miners commit connection info to chain metadata

**What miners can commit to chain:**
Miners can store arbitrary connection information in their on-chain metadata:
- Custom API endpoints
- S3 bucket URLs
- Database endpoints  
- IP addresses for any protocol
- Any other discovery information

Validators read this committed data to know how to communicate with each miner. For full details on commitment APIs and commit-reveal with drand timelock encryption, see **[19-chain-commitments.md](19-chain-commitments.md)**.

### Recommended: Epistula Signed HTTP
The standard pattern for validator-miner communication:

```python
import time
import hashlib

def create_epistula_headers(wallet, body: bytes) -> dict:
    """Create signed HTTP headers for authenticated requests"""
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

### Production Communication Patterns
**All production subnets use custom communication.** See Document 04 for:
- HTTP APIs with Epistula headers (recommended)
- External data source verification
- Socket.io connections
- Custom RPC protocols
- Miners committing S3/database endpoints to chain

## Security

### Replay Protection
- Nonces prevent request replay
- Each request has unique timestamp + random component
- Servers should track seen nonces to prevent replay attacks

### Spoofing Prevention
- Hotkey signatures verify sender identity
- Body hashes ensure integrity
- Chain registration links hotkeys to coldkeys

### Key Security
| Key | Location | Security Level |
|-----|----------|----------------|
| Coldkey | Cold storage | Maximum |
| Hotkey | Server | Operational |
| Coldkey password | Memory only | Critical |

### Best Practices
- Never commit keys to repositories
- Use environment variables or secure vaults
- Rotate hotkeys if compromised
- Keep coldkeys offline when possible
