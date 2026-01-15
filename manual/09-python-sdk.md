# Python SDK Reference

This document covers the Bittensor Python SDK classes and methods.

## Installation

```bash
pip install bittensor
pip install bittensor-wallet
```

## Core Classes

### Subtensor

The main chain interface for sync operations.

```python
from bittensor import Subtensor

# Connect to networks
subtensor = Subtensor(network="finney")      # Mainnet
subtensor = Subtensor(network="test")        # Testnet
subtensor = Subtensor(network="local")       # Localnet
subtensor = Subtensor(network="ws://custom:9944")  # Custom
```

#### Query Methods

```python
# Block information
block = subtensor.get_current_block()
block_hash = subtensor.get_block_hash(block)

# Balance
balance = subtensor.get_balance(coldkey_ss58)
print(f"Balance: {balance.tao} TAO")

# Subnet information
total_subnets = subtensor.get_total_subnets()
subnet_exists = subtensor.subnet_exists(netuid=1)
owner = subtensor.get_subnet_owner(netuid=1)

# Hyperparameters
params = subtensor.get_subnet_hyperparameters(netuid=1)
print(f"Tempo: {params.tempo}")
print(f"Max UIDs: {params.max_allowed_uids}")

# Neuron information
is_registered = subtensor.is_hotkey_registered(netuid=1, hotkey_ss58="5...")
uid = subtensor.get_uid_for_hotkey_on_subnet(netuid=1, hotkey_ss58="5...")
stake = subtensor.get_stake_for_hotkey_on_subnet(netuid=1, hotkey_ss58="5...")

# Metagraph
metagraph = subtensor.metagraph(netuid=1)
```

#### Transaction Methods

```python
from bittensor_wallet import Wallet

wallet = Wallet(name="my_wallet", hotkey="my_hotkey")

# Registration
success = subtensor.register(wallet=wallet, netuid=1)  # PoW
success = subtensor.burned_register(wallet=wallet, netuid=1)  # Burn

# Subnet creation
success, netuid = subtensor.register_network(wallet=wallet)

# Weight setting
success = subtensor.set_weights(
    wallet=wallet,
    netuid=1,
    uids=[0, 1, 2],
    weights=[0.5, 0.3, 0.2],
    wait_for_inclusion=True,
    wait_for_finalization=False
)

# Commit-reveal weights
success = subtensor.commit_weights(
    wallet=wallet,
    netuid=1,
    uids=[0, 1, 2],
    weights=[0.5, 0.3, 0.2]
)

# Set basic commitment (immediate visibility)
response = subtensor.set_commitment(
    wallet=wallet,
    netuid=1,
    data="https://api.myminer.com/v1/inference"
)

# Set commit-reveal commitment (revealed after blocks_until_reveal)
response = subtensor.set_reveal_commitment(
    wallet=wallet,
    netuid=1,
    data="secret_model_hash_abc123",
    blocks_until_reveal=360,  # ~72 minutes on mainnet
    block_time=12  # 12 for mainnet, 0.25 for fast-blocks
)

# Read commitment by UID
commitment = subtensor.get_commitment(netuid=1, uid=42)

# Read all commitments in subnet
all_commits = subtensor.get_all_commitments(netuid=1)

# Read revealed commitments
revealed = subtensor.get_revealed_commitment(netuid=1, uid=42)
all_revealed = subtensor.get_all_revealed_commitments(netuid=1)

# Staking
success = subtensor.add_stake(
    wallet=wallet,
    hotkey_ss58="5...",
    amount=1.0  # TAO
)

success = subtensor.unstake(
    wallet=wallet,
    hotkey_ss58="5...",
    amount=1.0
)

# Transfer
success = subtensor.transfer(
    wallet=wallet,
    dest="5...",
    amount=1.0
)
```

### AsyncSubtensor

Async version of Subtensor for non-blocking operations.

```python
from bittensor import AsyncSubtensor
import asyncio

async def main():
    async_subtensor = AsyncSubtensor(network="finney")
    
    # Async queries
    block = await async_subtensor.get_current_block()
    balance = await async_subtensor.get_balance(coldkey_ss58)
    
    # Async transactions
    success = await async_subtensor.set_weights(
        wallet=wallet,
        netuid=1,
        uids=uids,
        weights=weights
    )

asyncio.run(main())
```

### Wallet

Key management via bittensor-wallet package.

```python
from bittensor_wallet import Wallet

# Create wallet
wallet = Wallet(name="my_wallet", hotkey="my_hotkey")

# Create keys if needed
wallet.create_if_non_existent()

# Or create explicitly
wallet.create_new_coldkey(n_words=12, use_password=True)
wallet.create_new_hotkey(n_words=12)

# Access addresses
print(f"Coldkey: {wallet.coldkey.ss58_address}")
print(f"Hotkey: {wallet.hotkey.ss58_address}")

# Sign messages
message = b"hello world"
signature = wallet.hotkey.sign(message)

# Verify
is_valid = wallet.hotkey.verify(message, signature)

# Key paths
print(f"Coldkey path: {wallet.coldkeyfile.path}")
print(f"Hotkey path: {wallet.hotkeyfile.path}")
```

### Metagraph

Subnet state snapshot.

```python
from bittensor import Metagraph, Subtensor

# Create metagraph
metagraph = Metagraph(netuid=1, network="finney")

# Sync from chain
subtensor = Subtensor(network="finney")
metagraph.sync(subtensor=subtensor)

# Or sync with lite mode (faster, no weights/bonds)
metagraph.sync(subtensor=subtensor, lite=True)

# Core attributes
print(f"Neurons: {metagraph.n}")
print(f"Block: {metagraph.block}")
print(f"NetUID: {metagraph.netuid}")

# Per-neuron arrays (indexed by UID)
stakes = metagraph.S           # Stake amounts
incentives = metagraph.I       # Incentive scores
dividends = metagraph.D        # Dividend scores
emissions = metagraph.E        # Emission amounts
trust = metagraph.T           # Trust scores
consensus = metagraph.C       # Consensus scores
ranks = metagraph.R           # Rank values
active = metagraph.active     # Activity flags
last_update = metagraph.last_update  # Last weight update block
validator_permit = metagraph.validator_permit
hotkeys = metagraph.hotkeys   # Hotkey addresses
coldkeys = metagraph.coldkeys # Coldkey addresses

# Matrices (full mode only)
weights = metagraph.W         # Weight matrix [validator][miner]
bonds = metagraph.B          # Bond matrix [validator][miner]

# Access specific neuron
uid = 0
print(f"UID {uid}: stake={metagraph.S[uid]}, incentive={metagraph.I[uid]}")

# Find UID by hotkey
hotkey = "5..."
if hotkey in metagraph.hotkeys:
    uid = metagraph.hotkeys.index(hotkey)
```

### Synapse (Deprecated)

> **⚠️ DEPRECATED**: The Synapse/Axon/Dendrite pattern is deprecated. New subnets should use custom HTTP APIs with open source designs. See Document 04 for recommended communication patterns.

Synapse was the legacy request/response data container used with Axon and Dendrite. For new subnets, use standard HTTP APIs (FastAPI, Flask, etc.) with hotkey-based signing (Epistula protocol).

---

## Common Patterns

### Registration Check and Auto-Register

```python
def ensure_registered(subtensor, wallet, netuid):
    """Register if not already registered"""
    
    if subtensor.is_hotkey_registered(
        netuid=netuid,
        hotkey_ss58=wallet.hotkey.ss58_address
    ):
        uid = subtensor.get_uid_for_hotkey_on_subnet(
            netuid=netuid,
            hotkey_ss58=wallet.hotkey.ss58_address
        )
        print(f"Already registered as UID {uid}")
        return uid
    
    print("Registering...")
    success = subtensor.burned_register(
        wallet=wallet,
        netuid=netuid
    )
    
    if success:
        uid = subtensor.get_uid_for_hotkey_on_subnet(
            netuid=netuid,
            hotkey_ss58=wallet.hotkey.ss58_address
        )
        print(f"Registered as UID {uid}")
        return uid
    
    raise Exception("Registration failed")
```

### Weight Setting with Rate Limit

```python
class WeightSetter:
    def __init__(self, subtensor, wallet, netuid):
        self.subtensor = subtensor
        self.wallet = wallet
        self.netuid = netuid
        self.last_set_block = 0
        
    async def set_weights_if_ready(self, uids: list, weights: list) -> bool:
        """Set weights respecting rate limit"""
        
        current_block = self.subtensor.get_current_block()
        
        # Get rate limit
        params = self.subtensor.get_subnet_hyperparameters(self.netuid)
        rate_limit = params.weights_rate_limit
        
        if current_block - self.last_set_block < rate_limit:
            blocks_remaining = rate_limit - (current_block - self.last_set_block)
            print(f"Rate limited. {blocks_remaining} blocks remaining.")
            return False
        
        # Normalize weights
        total = sum(weights)
        if total > 0:
            weights = [w / total for w in weights]
        
        success = self.subtensor.set_weights(
            wallet=self.wallet,
            netuid=self.netuid,
            uids=uids,
            weights=weights,
            wait_for_inclusion=True
        )
        
        if success:
            self.last_set_block = current_block
            
        return success
```

### Metagraph Sync Loop

```python
async def metagraph_sync_loop(metagraph, subtensor, interval: int = 60):
    """Keep metagraph synced"""
    
    while True:
        try:
            old_block = metagraph.block
            metagraph.sync(subtensor=subtensor, lite=True)
            
            if metagraph.block != old_block:
                print(f"Synced to block {metagraph.block}")
                
        except Exception as e:
            print(f"Sync error: {e}")
            
        await asyncio.sleep(interval)
```

### Concurrent Miner Queries (Recommended Pattern)

```python
import aiohttp
import asyncio

async def query_miners_concurrently(
    metagraph,
    endpoint: str,
    payload: dict,
    timeout: float = 12.0,
    max_concurrent: int = 50
) -> dict:
    """Query miners via HTTP with concurrency limit"""
    
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def query_one(uid: int, url: str) -> tuple[int, dict | None]:
        async with semaphore:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=timeout)
                    ) as resp:
                        if resp.status == 200:
                            return uid, await resp.json()
            except Exception:
                pass
            return uid, None
    
    # Build tasks for all registered miners
    # Miners commit their endpoint URLs to chain metadata
    tasks = []
    for uid in range(metagraph.n):
        hotkey = metagraph.hotkeys[uid]
        # Get miner's committed endpoint from chain or registry
        miner_url = get_miner_endpoint(hotkey, endpoint)
        if miner_url:
            tasks.append(query_one(uid, miner_url))
    
    # Query concurrently
    results = await asyncio.gather(*tasks)
    
    return {uid: response for uid, response in results if response}
```

---

## Logging

```python
import bittensor as bt

# Set logging level
bt.logging.set_trace()    # Most verbose
bt.logging.set_debug()
bt.logging.set_info()
bt.logging.set_warning()
bt.logging.set_error()

# Log messages
bt.logging.trace("Trace message")
bt.logging.debug("Debug message")
bt.logging.info("Info message")
bt.logging.warning("Warning message")
bt.logging.error("Error message")
```

---

## Configuration

```python
import bittensor as bt
import argparse

# Access default config
config = bt.Config()

# Add arguments
parser = argparse.ArgumentParser()
bt.Wallet.add_args(parser)
bt.Subtensor.add_args(parser)

# Parse
config = bt.Config(parser)

# Access values
print(config.wallet.name)
print(config.subtensor.network)
```
