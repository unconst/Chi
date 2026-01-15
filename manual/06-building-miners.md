# Building Miners

This document covers miner implementation across different architectural patterns.

## What Miners Do

Miners are the **producers** in Bittensor subnets. They:
1. Register on a subnet (obtain a UID)
2. Provide the subnet's commodity (compute, data, predictions, etc.)
3. Respond to validator queries (or external systems)
4. Earn TAO based on validator-assigned weights

## Important: Communication Patterns

**The Axon/Dendrite/Synapse pattern is deprecated.** New subnets should always use open source, custom communication methods. This ensures transparency, auditability, and allows miners to implement their solutions using standard tools and frameworks.

**Miners can commit any data to chain:**
- Custom API endpoints
- S3 bucket URLs  
- Database endpoints
- IP addresses for any protocol

Validators read this committed data to discover how to communicate with miners. This gives you full flexibility in your communication protocol.

**Recommended approach:**
- Use standard HTTP frameworks (FastAPI, Flask, etc.)
- Implement Epistula protocol for hotkey-based authentication
- Open source your miner for auditability
- Let the validator specification guide your implementation

## Miner Architecture Patterns

### Pattern A: Custom HTTP API
Direct HTTP server with signed headers. **Recommended for all subnets.**

### Pattern B: Socket/Orchestrator Connection
Connect to central coordinator, no public endpoint.

### Pattern C: External Activity
Perform activity on external platforms, no server needed.

---

## Pattern A: Custom HTTP API Miner

Use when: Building any new subnet (this is the recommended pattern).

### Basic Structure

```python
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import time
import hashlib
from bittensor_wallet import Wallet
from bittensor import Subtensor, Metagraph

app = FastAPI()

# Global state
wallet = Wallet(name="miner", hotkey="hotkey")
subtensor = Subtensor(network="finney")
metagraph = Metagraph(netuid=MY_NETUID, network="finney")

class CVMInfo(BaseModel):
    gpu_type: str
    vram_gb: int
    bid_per_hour: float
    attestation: str

def verify_epistula_signature(request: Request, body: bytes) -> str:
    """
    Verify Epistula signed headers.
    Returns caller hotkey if valid, raises if invalid.
    """
    timestamp = request.headers.get("X-Epistula-Timestamp")
    signature = request.headers.get("X-Epistula-Signature")
    caller_hotkey = request.headers.get("X-Epistula-Hotkey")
    
    if not all([timestamp, signature, caller_hotkey]):
        raise HTTPException(401, "Missing Epistula headers")
    
    # Check timestamp freshness (prevent replay)
    ts = int(timestamp) / 1e9
    if abs(time.time() - ts) > 60:
        raise HTTPException(401, "Timestamp too old")
    
    # Verify signature
    body_hash = hashlib.sha256(body).hexdigest()
    message = f"{timestamp}.{body_hash}"
    
    # Use substrate-interface for verification
    from substrateinterface import Keypair
    kp = Keypair(ss58_address=caller_hotkey)
    
    try:
        is_valid = kp.verify(message.encode(), bytes.fromhex(signature))
        if not is_valid:
            raise HTTPException(401, "Invalid signature")
    except Exception:
        raise HTTPException(401, "Signature verification failed")
    
    return caller_hotkey

def is_validator(hotkey: str) -> bool:
    """Check if hotkey is a registered validator with stake"""
    metagraph.sync(lite=True)
    
    if hotkey not in metagraph.hotkeys:
        return False
    
    uid = metagraph.hotkeys.index(hotkey)
    return metagraph.validator_permit[uid] and metagraph.S[uid] > 1000

@app.post("/cvm")
async def get_cvm_info(request: Request):
    """Endpoint for validators to query available compute"""
    body = await request.body()
    caller = verify_epistula_signature(request, body)
    
    if not is_validator(caller):
        raise HTTPException(403, "Not a validator")
    
    # Return our compute offering
    return CVMInfo(
        gpu_type="A100",
        vram_gb=80,
        bid_per_hour=0.50,
        attestation=get_attestation_proof()
    )

@app.get("/health")
async def health():
    return {"status": "ok"}

def get_attestation_proof() -> str:
    """Generate hardware attestation proof"""
    # Implementation depends on your TEE setup
    # Could use Intel SGX, AMD SEV, etc.
    return "attestation_proof_here"

if __name__ == "__main__":
    import uvicorn
    
    # Register on subnet first
    if not subtensor.is_hotkey_registered(MY_NETUID, wallet.hotkey.ss58_address):
        subtensor.burned_register(wallet=wallet, netuid=MY_NETUID)
    
    # Commit your endpoint URL to chain metadata
    # Validators will read this to discover how to reach you
    # (Implementation depends on your subnet's discovery mechanism)
    
    uvicorn.run(app, host="0.0.0.0", port=8091)
```

---

## Pattern B: Orchestrator-Connected Miner

Use when: No public endpoint needed, jobs routed via central system.

### Basic Structure

```python
import socketio
import asyncio
from bittensor_wallet import Wallet
from bittensor import Subtensor

class ChutesStyleMiner:
    def __init__(self, config):
        self.config = config
        self.wallet = Wallet(name=config.wallet_name, hotkey=config.hotkey_name)
        self.subtensor = Subtensor(network=config.network)
        self.sio = socketio.AsyncClient()
        
    async def setup(self):
        """Register and connect to orchestrator"""
        # Register on chain
        if not self.subtensor.is_hotkey_registered(
            self.config.netuid,
            self.wallet.hotkey.ss58_address
        ):
            self.subtensor.burned_register(
                wallet=self.wallet,
                netuid=self.config.netuid
            )
        
        # Setup socket handlers
        @self.sio.event
        async def connect():
            print("Connected to orchestrator")
            # Authenticate with hotkey signature
            await self.authenticate()
            
        @self.sio.event
        async def job(data):
            # Execute job and return result
            result = await self.execute_job(data)
            return result
            
        @self.sio.event
        async def disconnect():
            print("Disconnected, reconnecting...")
            await asyncio.sleep(5)
            await self.connect_to_orchestrator()
        
    async def authenticate(self):
        """Prove ownership of hotkey to orchestrator"""
        timestamp = str(int(time.time() * 1e9))
        message = f"auth:{self.wallet.hotkey.ss58_address}:{timestamp}"
        signature = self.wallet.hotkey.sign(message.encode()).hex()
        
        await self.sio.emit("authenticate", {
            "hotkey": self.wallet.hotkey.ss58_address,
            "timestamp": timestamp,
            "signature": signature,
            "gpu_info": self.get_gpu_info()
        })
        
    async def connect_to_orchestrator(self):
        """Connect to central job router"""
        await self.sio.connect(
            self.config.orchestrator_url,
            headers={"X-Miner-Hotkey": self.wallet.hotkey.ss58_address}
        )
        
    async def execute_job(self, data):
        """Execute compute job on local hardware"""
        job_type = data.get("type")
        payload = data.get("payload")
        
        if job_type == "inference":
            return await self.run_inference(payload)
        elif job_type == "training":
            return await self.run_training(payload)
        else:
            return {"error": "Unknown job type"}
            
    def get_gpu_info(self):
        """Report GPU capabilities"""
        import torch
        if torch.cuda.is_available():
            return {
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0),
                "vram_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
            }
        return {"gpu_count": 0}
        
    async def run(self):
        """Main loop"""
        await self.setup()
        await self.connect_to_orchestrator()
        
        # Keep running
        while True:
            await asyncio.sleep(1)

if __name__ == "__main__":
    config = MinerConfig(
        netuid=64,
        network="finney",
        wallet_name="miner",
        hotkey_name="hotkey",
        orchestrator_url="https://orchestrator.example.com"
    )
    
    miner = ChutesStyleMiner(config)
    asyncio.run(miner.run())
```

---

## Pattern C: External Activity Miner

Use when: Value comes from activity on external platforms.

### Basic Structure (Gittensor-style)

```python
from bittensor import Subtensor
from bittensor_wallet import Wallet

class GitContributorMiner:
    """
    Miner that earns by making GitHub contributions.
    No server needed - just register and contribute!
    """
    
    def __init__(self, config):
        self.config = config
        self.wallet = Wallet(name=config.wallet_name, hotkey=config.hotkey_name)
        self.subtensor = Subtensor(network=config.network)
        
    def setup(self):
        """Register on subnet and link identity"""
        # Register hotkey
        if not self.subtensor.is_hotkey_registered(
            self.config.netuid,
            self.wallet.hotkey.ss58_address
        ):
            self.subtensor.burned_register(
                wallet=self.wallet,
                netuid=self.config.netuid
            )
            
        # Set identity linking GitHub account
        # This depends on subnet's specific identity requirements
        print(f"Registered hotkey: {self.wallet.hotkey.ss58_address}")
        print(f"Link this hotkey to GitHub account: {self.config.github_username}")
        print(f"Add to PR descriptions: 'Mining $OGX - Hotkey {self.config.github_id}'")
        
    def check_status(self):
        """Check current rewards status"""
        from bittensor import Metagraph
        
        metagraph = Metagraph(netuid=self.config.netuid, network=self.config.network)
        metagraph.sync()
        
        uid = metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        
        print(f"UID: {uid}")
        print(f"Incentive: {metagraph.I[uid]}")
        print(f"Emission: {metagraph.E[uid]}")

# Usage:
# 1. Run setup once to register
# 2. Make quality GitHub contributions with required tagline
# 3. Periodically check status
# No server to run!
```

---

## Common Miner Components

### Rate Limiting
```python
from collections import defaultdict
import time

class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window = window_seconds
        self.requests = defaultdict(list)
        
    def is_allowed(self, caller_id: str) -> bool:
        now = time.time()
        cutoff = now - self.window
        
        # Clean old entries
        self.requests[caller_id] = [
            t for t in self.requests[caller_id] if t > cutoff
        ]
        
        # Check limit
        if len(self.requests[caller_id]) >= self.max_requests:
            return False
            
        self.requests[caller_id].append(now)
        return True
```

### Request Validation
```python
def validate_request(synapse) -> tuple[bool, str]:
    """Validate incoming request"""
    
    # Check required fields
    if not synapse.query:
        return False, "Missing query"
    
    # Check field constraints
    if len(synapse.query) > 10000:
        return False, "Query too long"
    
    # Check format
    if not isinstance(synapse.query, str):
        return False, "Query must be string"
    
    return True, ""
```

### Logging and Monitoring
```python
import logging
import time
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def log_request(func):
    @wraps(func)
    async def wrapper(request, *args, **kwargs):
        start = time.time()
        caller = request.headers.get("X-Epistula-Hotkey", "unknown")
        
        try:
            result = await func(request, *args, **kwargs)
            elapsed = time.time() - start
            
            logger.info(f"Request from {caller[:8]}... completed in {elapsed:.3f}s")
            return result
            
        except Exception as e:
            logger.error(f"Request from {caller[:8]}... failed: {e}")
            raise
            
    return wrapper
```

### GPU Memory Management
```python
import torch

def clear_gpu_memory():
    """Clear GPU memory after inference"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def get_available_memory():
    """Get available GPU memory in GB"""
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        return free / 1e9
    return 0
```

---

## Miner Checklist

- [ ] Register hotkey on target subnet
- [ ] Implement commodity provision logic
- [ ] Handle request validation and Epistula signature verification
- [ ] Commit endpoint info to chain (if needed)
- [ ] Add logging and monitoring
- [ ] Test with validators locally
- [ ] Open source your implementation for auditability
- [ ] Monitor incentive and emissions
