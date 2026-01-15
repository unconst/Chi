# Local Development

This document covers setting up a local Bittensor environment for development and testing.

## Why Local Development?

- **Free**: No TAO required for registration or transactions
- **Fast**: Faster block times (~6 seconds vs 12 seconds)
- **Isolated**: No interference with mainnet/testnet
- **Controllable**: Full control over chain state
- **Debuggable**: Easy to inspect transactions and state

## Prerequisites

### System Requirements
- Docker and Docker Compose
- Python 3.10+
- 8GB+ RAM recommended
- 10GB+ disk space

### Software Installation

```bash
# Python environment
python -m venv .venv
source .venv/bin/activate

# Bittensor packages
pip install bittensor bittensor-wallet bittensor-cli

# Docker (if not installed)
# macOS: brew install docker
# Ubuntu: sudo apt install docker.io docker-compose
```

## Starting Local Subtensor

### Option 1: Docker Compose (Recommended)

Clone the subtensor repository:
```bash
git clone https://github.com/opentensor/subtensor.git
cd subtensor
```

Start localnet:
```bash
docker-compose -f docker-compose.localnet.yml up -d
```

Verify it's running:
```bash
# Check containers
docker ps | grep subtensor

# Check chain is producing blocks
curl -s -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"chain_getHeader","params":[],"id":1}' \
  http://localhost:9944 | jq .result.number
```

Stop localnet:
```bash
docker-compose -f docker-compose.localnet.yml down
```

### Option 2: BTCLI

```bash
# Start localnet
btcli local start

# Stop localnet
btcli local stop
```

### Option 3: Direct Docker Run

```bash
docker run -d \
  --name subtensor-local \
  -p 9944:9944 \
  -p 9933:9933 \
  opentensor/subtensor-localnet:latest
```

## Connecting to Localnet

**CLI:**
```bash
btcli subnet list --network local
```

**SDK:**
```python
from bittensor import Subtensor

subtensor = Subtensor(network="local")
# or
subtensor = Subtensor(network="ws://127.0.0.1:9944")

# Verify connection
block = subtensor.get_current_block()
print(f"Connected at block {block}")
```

## Wallet Setup for Localnet

### Create Test Wallets

```bash
# Owner wallet
btcli wallet new_coldkey --wallet.name owner --n_words 12 --no-password
btcli wallet new_hotkey --wallet.name owner --wallet.hotkey default

# Miner wallet
btcli wallet new_coldkey --wallet.name miner --n_words 12 --no-password
btcli wallet new_hotkey --wallet.name miner --wallet.hotkey default

# Validator wallet
btcli wallet new_coldkey --wallet.name validator --n_words 12 --no-password
btcli wallet new_hotkey --wallet.name validator --wallet.hotkey default
```

### Fund Wallets (Localnet Faucet)

Localnet has a faucet with unlimited funds:

```bash
# Fund owner
btcli wallet faucet --wallet.name owner --network local

# Fund miner
btcli wallet faucet --wallet.name miner --network local

# Fund validator
btcli wallet faucet --wallet.name validator --network local
```

**SDK:**
```python
from bittensor import Subtensor
from bittensor_wallet import Wallet

subtensor = Subtensor(network="local")

# Fund wallet using faucet
wallet = Wallet(name="owner")
success = subtensor.faucet(wallet=wallet)

# Check balance
balance = subtensor.get_balance(wallet.coldkey.ss58_address)
print(f"Balance: {balance.tao} TAO")
```

## Creating a Local Subnet

### Step 1: Create Subnet

```bash
btcli subnet create \
  --wallet.name owner \
  --network local \
  --no-prompt
```

**SDK:**
```python
from bittensor import Subtensor
from bittensor_wallet import Wallet

subtensor = Subtensor(network="local")
wallet = Wallet(name="owner")

success, netuid = subtensor.register_network(wallet=wallet)
print(f"Created subnet with netuid: {netuid}")
```

### Step 2: Configure Hyperparameters

```bash
# Set faster tempo for testing
btcli sudo set --netuid 1 --param tempo --value 10 \
  --wallet.name owner --network local

# Allow more registrations
btcli sudo set --netuid 1 --param max_allowed_uids --value 64 \
  --wallet.name owner --network local
```

### Step 3: Register Neurons

```bash
# Register miner
btcli subnet register --netuid 1 \
  --wallet.name miner --wallet.hotkey default \
  --network local

# Register validator
btcli subnet register --netuid 1 \
  --wallet.name validator --wallet.hotkey default \
  --network local
```

### Step 4: Add Stake to Validator

```bash
btcli stake add --netuid 1 \
  --wallet.name validator --wallet.hotkey default \
  --amount 1000 \
  --network local
```

## Running Local Miner

```python
from fastapi import FastAPI, Request
from bittensor_wallet import Wallet
from bittensor import Subtensor
import uvicorn

# Setup
wallet = Wallet(name="miner")
subtensor = Subtensor(network="local")

app = FastAPI()

@app.post("/compute")
async def compute(request: Request):
    data = await request.json()
    query = data.get("query", "")
    return {"response": f"Echo: {query}"}

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    # Register on subnet
    if not subtensor.is_hotkey_registered(1, wallet.hotkey.ss58_address):
        subtensor.burned_register(wallet=wallet, netuid=1)
    
    print("Miner running on port 8091")
    uvicorn.run(app, host="0.0.0.0", port=8091)
```

## Running Local Validator

```python
import aiohttp
from bittensor_wallet import Wallet
from bittensor import Subtensor, Metagraph
import asyncio

# Setup
wallet = Wallet(name="validator")
subtensor = Subtensor(network="local")
metagraph = Metagraph(netuid=1, network="local")

# For local testing, maintain a simple registry of miner endpoints
# In production, miners commit their endpoints to chain
MINER_ENDPOINTS = {}  # hotkey -> url

async def query_miner(url: str, payload: dict) -> dict | None:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload, timeout=12) as resp:
                if resp.status == 200:
                    return await resp.json()
    except Exception:
        pass
    return None

async def validate():
    while True:
        # Sync metagraph
        metagraph.sync()
        
        # Find miners with registered endpoints
        miner_uids = []
        for uid in range(metagraph.n):
            hotkey = metagraph.hotkeys[uid]
            if hotkey in MINER_ENDPOINTS:
                miner_uids.append(uid)
        
        if not miner_uids:
            print("No miners found")
            await asyncio.sleep(10)
            continue
        
        # Query miners via HTTP
        scores = {}
        for uid in miner_uids:
            hotkey = metagraph.hotkeys[uid]
            url = MINER_ENDPOINTS[hotkey]
            
            response = await query_miner(f"{url}/compute", {"query": "test"})
            
            if response and response.get("response"):
                scores[uid] = 1.0
            else:
                scores[uid] = 0.0
        
        # Set weights
        if scores:
            uids = list(scores.keys())
            weights = list(scores.values())
            
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]
            
            subtensor.set_weights(
                wallet=wallet,
                netuid=1,
                uids=uids,
                weights=weights
            )
            print(f"Set weights: {dict(zip(uids, weights))}")
        
        await asyncio.sleep(30)

asyncio.run(validate())
```

## Debugging Tools

### Check Metagraph State

```python
metagraph = bt.Metagraph(netuid=1, network="local")
metagraph.sync()

print(f"Neurons: {metagraph.n}")
for uid in range(metagraph.n):
    print(f"UID {uid}:")
    print(f"  Hotkey: {metagraph.hotkeys[uid][:16]}...")
    print(f"  Stake: {metagraph.S[uid]}")
    print(f"  Incentive: {metagraph.I[uid]}")
```

### Check Chain State

```bash
# Current block
btcli subnet show --netuid 1 --network local

# Registration status
btcli wallet overview --wallet.name miner --network local
```

### Logging

```python
import bittensor as bt

# Enable verbose logging
bt.logging.set_debug()

# Or trace level
bt.logging.set_trace()
```

### Inspect Transactions

Monitor Docker logs:
```bash
docker logs -f subtensor-local
```

## Common Issues

### "Connection refused"
- Check Docker container is running: `docker ps`
- Verify port 9944 is exposed: `netstat -an | grep 9944`
- Restart container: `docker restart subtensor-local`

### "Insufficient balance"
- Use faucet: `btcli wallet faucet --wallet.name <name> --network local`

### "Already registered"
- Expected if you've registered before
- Use same wallet/hotkey for that neuron

### "No validators found"
- Ensure validator has stake: `btcli stake show --network local`
- Check validator permit: inspect metagraph

### Slow block production
- Normal: localnet blocks every ~6 seconds
- If stuck: restart Docker container

## Full Local Test Script

```python
#!/usr/bin/env python3
"""Complete local development test"""

import bittensor as bt
from bittensor_wallet import Wallet
import asyncio
import time

bt.logging.set_debug()

async def main():
    # Connect to local
    subtensor = bt.Subtensor(network="local")
    print(f"Connected at block {subtensor.get_current_block()}")
    
    # Setup wallets
    owner = Wallet(name="owner")
    owner.create_if_non_existent()
    
    miner = Wallet(name="miner")
    miner.create_if_non_existent()
    
    validator = Wallet(name="validator")
    validator.create_if_non_existent()
    
    # Fund wallets
    for w in [owner, miner, validator]:
        subtensor.faucet(wallet=w)
        balance = subtensor.get_balance(w.coldkey.ss58_address)
        print(f"{w.name} balance: {balance.tao} TAO")
    
    # Create subnet (if not exists)
    if not subtensor.subnet_exists(netuid=1):
        success, netuid = subtensor.register_network(wallet=owner)
        print(f"Created subnet {netuid}")
    else:
        print("Subnet 1 already exists")
    
    # Register neurons
    for w, name in [(miner, "miner"), (validator, "validator")]:
        if not subtensor.is_hotkey_registered(1, w.hotkey.ss58_address):
            subtensor.burned_register(wallet=w, netuid=1)
            print(f"Registered {name}")
        else:
            print(f"{name} already registered")
    
    # Stake validator
    subtensor.add_stake(wallet=validator, hotkey_ss58=validator.hotkey.ss58_address, amount=1000)
    print("Staked validator")
    
    # Start miner (in a real test, run this in a separate process)
    # For this test script, we'll just set weights directly
    
    # Validator sets weights
    metagraph = bt.Metagraph(netuid=1, network="local")
    metagraph.sync()
    
    miner_uid = metagraph.hotkeys.index(miner.hotkey.ss58_address)
    subtensor.set_weights(
        wallet=validator,
        netuid=1,
        uids=[miner_uid],
        weights=[1.0]
    )
    print(f"Set weight on miner UID {miner_uid}")
    
    # Wait for epoch
    print("Waiting for epoch...")
    await asyncio.sleep(120)  # 10 blocks × 12 seconds
    
    # Check results
    metagraph.sync()
    print(f"Miner incentive: {metagraph.I[miner_uid]}")
    print(f"Miner emission: {metagraph.E[miner_uid]}")
    
    print("Test complete!")

if __name__ == "__main__":
    asyncio.run(main())
```
