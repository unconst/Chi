# Chain Commitments

This document covers how to use on-chain commitments in Bittensor subnets, including basic commitments, reading commitments with block information, and commit-reveal commitments using drand timelock encryption.

## Overview

Bittensor provides two commitment mechanisms:

1. **Basic Commitments**: Store arbitrary data on-chain, immediately visible to everyone
2. **Commit-Reveal Commitments**: Encrypt data using drand timelock encryption so it's only revealed after a specified number of blocks

Both mechanisms allow miners to publish information that validators can discover and verify.

---

## Basic Commitments

Basic commitments allow neurons to publish arbitrary string data to the chain. This data is immediately visible to all participants.

### Setting a Commitment (Miner)

```python
from bittensor import Subtensor
from bittensor_wallet import Wallet

subtensor = Subtensor(network="finney")
wallet = Wallet(name="miner", hotkey="miner_hotkey")

# Commit arbitrary data to the chain
response = subtensor.set_commitment(
    wallet=wallet,
    netuid=1,
    data="https://api.myminer.com/v1/inference",
    wait_for_inclusion=True,
    wait_for_finalization=True,
)

if response.success:
    print("Commitment set successfully")
else:
    print(f"Failed: {response.message}")
```

**Parameters:**
- `wallet`: The wallet (hotkey) making the commitment
- `netuid`: The subnet to commit to
- `data`: Arbitrary string data (endpoint URLs, model IDs, config JSON, etc.)
- `mev_protection`: If `True`, uses MEV Shield to protect against front-running (default: True)
- `period`: Blocks until transaction expires if not included

### Reading a Commitment (Validator)

Validators can read commitments by UID or by hotkey.

#### By UID

```python
# Get commitment by UID
commitment = subtensor.get_commitment(
    netuid=1,
    uid=42,
    block=None  # None = current block
)
print(f"Miner 42 committed: {commitment}")
```

#### By Hotkey

```python
# Get raw commitment metadata (includes block info)
metadata = subtensor.get_commitment_metadata(
    netuid=1,
    hotkey_ss58="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
    block=None
)
# metadata is a dict with 'info' containing the commitment fields
```

#### All Commitments in a Subnet

```python
# Get all commitments for a subnet
all_commitments = subtensor.get_all_commitments(netuid=1, block=None)
# Returns: {hotkey_ss58: commitment_string, ...}

for hotkey, data in all_commitments.items():
    print(f"{hotkey}: {data}")
```

### Async API

```python
from bittensor import AsyncSubtensor
import asyncio

async def main():
    subtensor = AsyncSubtensor(network="finney")
    
    # Set commitment
    response = await subtensor.set_commitment(
        wallet=wallet,
        netuid=1,
        data="https://api.myminer.com/v1/inference"
    )
    
    # Get commitment
    commitment = await subtensor.get_commitment(netuid=1, uid=42)
    
    # Get all commitments
    all_commitments = await subtensor.get_all_commitments(netuid=1)

asyncio.run(main())
```

---

## Commit-Reveal Commitments

Commit-reveal commitments use **drand timelock encryption** to encrypt data that can only be decrypted after a specified number of blocks. This allows you to prove provenance of data without revealing it until later.

### How Timelock Encryption Works

Bittensor uses [drand QuickNet](https://drand.love) as a distributed randomness beacon. The system:

1. **Encrypts** your data using Identity-Based Encryption (IBE) tied to a future drand round
2. **Commits** the encrypted ciphertext to the chain
3. **Automatically reveals** the data when the drand beacon reaches the specified round

The chain cannot decrypt the data until drand publishes the signature for that round. This provides cryptographic guarantees that:
- No one can see your data before the reveal time
- You can prove you committed the data at a specific block
- The reveal is automatic and trustless

### Setting a Commit-Reveal Commitment

```python
from bittensor import Subtensor
from bittensor_wallet import Wallet

subtensor = Subtensor(network="finney")
wallet = Wallet(name="miner", hotkey="miner_hotkey")

# Commit data that will be revealed after 360 blocks (~72 minutes on mainnet)
response = subtensor.set_reveal_commitment(
    wallet=wallet,
    netuid=1,
    data="my_secret_model_weights_hash_abc123",
    blocks_until_reveal=360,  # Number of blocks until reveal
    block_time=12,  # Seconds per block (12 for mainnet, 0.25 for fast-blocks)
    wait_for_inclusion=True,
    wait_for_finalization=True,
)

if response.success:
    # Response data contains encryption info
    encrypted_data = response.data.get("encrypted")
    reveal_round = response.data.get("reveal_round")
    print(f"Committed! Will reveal at drand round {reveal_round}")
else:
    print(f"Failed: {response.message}")
```

**Parameters:**
- `blocks_until_reveal`: How many blocks in the future the data should be revealed
- `block_time`: Seconds per block (12 for mainnet, 0.25 for fast-blocks testnet)

### Reading Revealed Commitments

After the reveal round passes, the chain automatically decrypts and stores the revealed data.

#### By UID

```python
# Get revealed commitment for a specific UID
# Returns tuple of (reveal_block, commitment_message) pairs
revealed = subtensor.get_revealed_commitment(
    netuid=1,
    uid=42,
    block=None
)

if revealed:
    for reveal_block, message in revealed:
        print(f"Block {reveal_block}: {message}")
```

#### By Hotkey

```python
# Get revealed commitment by hotkey
revealed = subtensor.get_revealed_commitment_by_hotkey(
    netuid=1,
    hotkey_ss58="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
    block=None
)

if revealed:
    for reveal_block, message in revealed:
        print(f"Revealed at block {reveal_block}: {message}")
```

#### All Revealed Commitments

```python
# Get all revealed commitments for a subnet
all_revealed = subtensor.get_all_revealed_commitments(netuid=1, block=None)
# Returns: {hotkey_ss58: ((block1, msg1), (block2, msg2), ...), ...}

for hotkey, commits in all_revealed.items():
    print(f"\n{hotkey}:")
    for reveal_block, message in commits:
        print(f"  Block {reveal_block}: {message}")
```

**Note:** Each neuron can have multiple revealed commitments (up to 10 most recent are stored).

---

## Low-Level Timelock API

For advanced use cases, you can use the timelock encryption module directly.

### Encrypt and Decrypt Manually

```python
from bittensor import timelock

# Encrypt data for 5 blocks in the future
data = "secret message"
encrypted, reveal_round = timelock.encrypt(
    data=data,
    n_blocks=5,
    block_time=12  # 12 for mainnet, 0.25 for fast-blocks
)

print(f"Encrypted data: {len(encrypted)} bytes")
print(f"Will be decryptable at drand round: {reveal_round}")

# Check current drand round
current_round = timelock.get_latest_round()
print(f"Current drand round: {current_round}")

# Try to decrypt (returns None if round hasn't passed)
decrypted = timelock.decrypt(encrypted, no_errors=True)
if decrypted is None:
    print("Not yet decryptable - round hasn't passed")
else:
    print(f"Decrypted: {decrypted.decode()}")
```

### Wait for Reveal

```python
from bittensor import timelock

# Encrypt
encrypted, reveal_round = timelock.encrypt("my secret", n_blocks=2)

# Block until decryptable, then decrypt
# This polls drand every 3 seconds until the round passes
decrypted = timelock.wait_reveal_and_decrypt(
    encrypted_data=encrypted,
    reveal_round=reveal_round,  # Optional - parsed from encrypted if not provided
    return_str=True
)
print(f"Revealed: {decrypted}")
```

### Encrypt Custom Objects

```python
import pickle
from dataclasses import dataclass
from bittensor import timelock

@dataclass
class ModelSubmission:
    model_hash: str
    training_params: dict
    accuracy: float

# Create your object
submission = ModelSubmission(
    model_hash="sha256:abc123...",
    training_params={"lr": 0.001, "epochs": 100},
    accuracy=0.95
)

# Serialize to bytes
byte_data = pickle.dumps(submission)

# Encrypt for future reveal
encrypted, reveal_round = timelock.encrypt(byte_data, n_blocks=10)

# Later, after reveal...
decrypted = timelock.wait_reveal_and_decrypt(encrypted)
recovered = pickle.loads(decrypted)
assert submission == recovered
```

---

## Subnet Design Patterns

### Pattern 1: Endpoint Discovery

Miners commit their API endpoints to the chain. Validators read these to discover how to communicate.

```python
# Miner: Commit endpoint on startup
import json

endpoint_info = json.dumps({
    "api_url": "https://api.miner.com/v1",
    "model_id": "llama-70b-instruct",
    "version": "1.2.3"
})

subtensor.set_commitment(
    wallet=wallet,
    netuid=MY_NETUID,
    data=endpoint_info
)

# Validator: Discover miner endpoints
def get_miner_endpoints(subtensor, netuid):
    all_commits = subtensor.get_all_commitments(netuid)
    endpoints = {}
    
    for hotkey, data in all_commits.items():
        try:
            info = json.loads(data)
            endpoints[hotkey] = info.get("api_url")
        except json.JSONDecodeError:
            continue
    
    return endpoints
```

### Pattern 2: Model Hash Commitment (Verifiable Training)

Miners commit their model hash before evaluation, proving they didn't change it after seeing the test set.

```python
# Miner: Commit model hash using commit-reveal
model_hash = compute_model_hash(my_model)

response = subtensor.set_reveal_commitment(
    wallet=wallet,
    netuid=MY_NETUID,
    data=model_hash,
    blocks_until_reveal=100,  # Reveal after ~20 minutes
)

# Validator: Verify model matches committed hash after reveal
async def verify_miner_model(subtensor, netuid, miner_uid, miner_endpoint):
    # Wait for any pending commit-reveal to complete
    revealed = subtensor.get_revealed_commitment(netuid, miner_uid)
    
    if not revealed:
        return 0.0  # No commitment = no score
    
    # Get most recent reveal
    reveal_block, committed_hash = revealed[-1]
    
    # Query miner's model and compute hash
    model_data = await fetch_model(miner_endpoint)
    actual_hash = compute_model_hash(model_data)
    
    if actual_hash != committed_hash:
        return 0.0  # Hash mismatch = cheating
    
    # Evaluate model quality
    return evaluate_model(model_data)
```

### Pattern 3: Sealed-Bid Auctions

Commit-reveal enables fair auctions where bids are sealed until reveal.

```python
# Bidder: Submit sealed bid
my_bid = json.dumps({"amount": 100, "resource_id": "gpu-001"})

response = subtensor.set_reveal_commitment(
    wallet=wallet,
    netuid=AUCTION_NETUID,
    data=my_bid,
    blocks_until_reveal=50,  # All bids reveal after 50 blocks
)

# Auctioneer: Read all revealed bids after deadline
def determine_winner(subtensor, netuid):
    all_revealed = subtensor.get_all_revealed_commitments(netuid)
    
    bids = []
    for hotkey, commits in all_revealed.items():
        if commits:
            reveal_block, bid_data = commits[-1]  # Most recent
            try:
                bid = json.loads(bid_data)
                bids.append((hotkey, bid["amount"]))
            except:
                continue
    
    # Highest bid wins
    winner = max(bids, key=lambda x: x[1])
    return winner
```

### Pattern 4: Prediction Markets / Forecasts

Commit predictions before events, reveal after to prove foresight.

```python
# Forecaster: Commit prediction before event
prediction = json.dumps({
    "event_id": "btc_price_2024_01_15",
    "prediction": 45000,
    "confidence": 0.8
})

# Commit with reveal after the event time
response = subtensor.set_reveal_commitment(
    wallet=wallet,
    netuid=FORECAST_NETUID,
    data=prediction,
    blocks_until_reveal=7200,  # ~24 hours
)

# Validator: Score predictions after event
def score_forecast(subtensor, netuid, actual_value):
    all_revealed = subtensor.get_all_revealed_commitments(netuid)
    
    scores = {}
    for hotkey, commits in all_revealed.items():
        for reveal_block, data in commits:
            try:
                pred = json.loads(data)
                error = abs(pred["prediction"] - actual_value)
                confidence = pred["confidence"]
                # Score based on accuracy and confidence
                scores[hotkey] = (1 / (1 + error)) * confidence
            except:
                continue
    
    return scores
```

### Pattern 5: Gradual Information Release

Use multiple commit-reveals with staggered timing for progressive disclosure.

```python
# Research subnet: Commit research in stages
stages = [
    ("abstract", 100),      # Abstract revealed after 100 blocks
    ("methodology", 500),   # Methodology after 500 blocks  
    ("full_paper", 1000),   # Full paper after 1000 blocks
]

for content_type, blocks in stages:
    content = get_paper_section(content_type)
    subtensor.set_reveal_commitment(
        wallet=wallet,
        netuid=RESEARCH_NETUID,
        data=json.dumps({
            "type": content_type,
            "content": content,
            "paper_id": paper_id
        }),
        blocks_until_reveal=blocks,
    )
```

---

## Technical Details

### Drand QuickNet

Bittensor uses [drand QuickNet](https://drand.love) as the randomness beacon:
- **Round interval**: 3 seconds
- **Public key**: Distributed BLS threshold signature
- **Availability**: Highly available distributed network

The timelock encryption uses:
- **AES-GCM**: For symmetric encryption of the message
- **BF-IBE (Boneh-Franklin Identity-Based Encryption)**: To encrypt the AES key for a future drand round

### Block Time Considerations

| Network | Block Time | 360 blocks = |
|---------|------------|--------------|
| Mainnet (Finney) | 12 seconds | ~72 minutes |
| Testnet | 12 seconds | ~72 minutes |
| Fast-blocks localnet | 0.25 seconds | ~90 seconds |

Always pass the correct `block_time` parameter when using fast-blocks nodes:

```python
# For fast-blocks testnet/localnet
encrypted, reveal_round = timelock.encrypt(data, n_blocks=100, block_time=0.25)
```

### Commitment Storage

- **Basic commitments**: Stored until overwritten by a new commitment
- **Revealed commitments**: Up to 10 most recent reveals stored per neuron per subnet
- **Pending commit-reveals**: Stored until reveal round passes

### MEV Protection

The `set_commitment` and `set_reveal_commitment` methods support MEV (Miner Extractable Value) protection:

```python
# With MEV protection (default)
subtensor.set_commitment(
    wallet=wallet,
    netuid=1,
    data="sensitive_data",
    mev_protection=True,  # Encrypts tx in mempool
)

# Without MEV protection (faster, visible in mempool)
subtensor.set_commitment(
    wallet=wallet,
    netuid=1,
    data="public_data",
    mev_protection=False,
)
```

---

## Common Use Cases Summary

| Use Case | Mechanism | Why |
|----------|-----------|-----|
| API endpoint discovery | Basic commitment | Validators need immediate access |
| Model hash verification | Commit-reveal | Prove model wasn't changed after test set |
| Sealed-bid auctions | Commit-reveal | Fair bidding without front-running |
| Prediction markets | Commit-reveal | Prove prediction was made before event |
| Gradual disclosure | Multiple commit-reveals | Staged information release |
| Configuration sharing | Basic commitment | Miners share runtime config |
| Version announcements | Basic commitment | Announce software versions |

---

## API Reference

### Subtensor Methods

| Method | Description |
|--------|-------------|
| `set_commitment(wallet, netuid, data)` | Write basic commitment |
| `set_reveal_commitment(wallet, netuid, data, blocks_until_reveal)` | Write commit-reveal commitment |
| `get_commitment(netuid, uid)` | Read commitment by UID |
| `get_commitment_metadata(netuid, hotkey_ss58)` | Read raw commitment by hotkey |
| `get_all_commitments(netuid)` | Read all commitments in subnet |
| `get_revealed_commitment(netuid, uid)` | Read revealed commits by UID |
| `get_revealed_commitment_by_hotkey(netuid, hotkey_ss58)` | Read revealed commits by hotkey |
| `get_all_revealed_commitments(netuid)` | Read all revealed commits in subnet |

### Timelock Functions

| Function | Description |
|----------|-------------|
| `timelock.encrypt(data, n_blocks, block_time)` | Encrypt data for future reveal |
| `timelock.decrypt(encrypted_data)` | Decrypt if reveal round passed |
| `timelock.wait_reveal_and_decrypt(encrypted_data)` | Block until decryptable, then decrypt |
| `timelock.get_latest_round()` | Get current drand round number |

---

## Best Practices

1. **Use commit-reveal for provenance**: When you need to prove something was known at a specific time without revealing it
2. **Use basic commits for discovery**: When validators need immediate access to miner information
3. **Consider reveal timing**: Choose `blocks_until_reveal` based on your use case (longer = more security, shorter = faster iteration)
4. **Handle missing commits gracefully**: Not all miners will have commitments; score them accordingly
5. **Validate commitment format**: Parse and validate commitment data defensively
6. **Use JSON for structured data**: Makes parsing easier and more portable

```python
# Good: Structured, versioned commitment
commitment = json.dumps({
    "version": 1,
    "type": "endpoint",
    "url": "https://...",
    "capabilities": ["inference", "training"]
})

# Bad: Unstructured string
commitment = "https://..."  # No metadata, hard to extend
```
