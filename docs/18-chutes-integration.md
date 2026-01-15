# Chutes Integration Guide

Chutes (SN64) is a decentralized GPU compute network on Bittensor that provides inference endpoints for AI models. This document covers how to integrate Chutes into your subnet for model serving, validation, and miner infrastructure.

## What is Chutes?

Chutes is a **capacity/uptime marketplace** where miners provide always-on GPU infrastructure. Key features:

- **Decentralized inference**: Models deployed on Chutes are served across multiple nodes
- **GPU verification**: GraVal cryptographic proofs ensure authentic hardware
- **TEE support**: Trusted Execution Environments for private/secure model deployment
- **Pay-per-use billing**: Only pay for compute actually consumed
- **No infrastructure management**: Miners or subnet developers don't need to manage Kubernetes clusters

### Chutes in the Bittensor Ecosystem

```
┌─────────────────────────────────────────────────────────────────┐
│                    YOUR SUBNET                                  │
│                                                                 │
│  ┌─────────────┐      ┌─────────────┐      ┌─────────────┐     │
│  │  VALIDATOR  │      │   MINER A   │      │   MINER B   │     │
│  │             │      │             │      │             │     │
│  │ Query LLMs  │      │ Deploy model│      │ Deploy model│     │
│  │ for scoring │      │ to Chutes   │      │ to Chutes   │     │
│  └──────┬──────┘      └──────┬──────┘      └──────┬──────┘     │
│         │                    │                    │             │
└─────────┼────────────────────┼────────────────────┼─────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                    CHUTES (SN64)                                │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                 GPU MINER NETWORK                        │   │
│  │                                                          │   │
│  │   ┌───────┐  ┌───────┐  ┌───────┐  ┌───────┐           │   │
│  │   │ H100  │  │ A100  │  │ A100  │  │ H100  │  ...      │   │
│  │   │ Node  │  │ Node  │  │ Node  │  │ Node  │           │   │
│  │   └───────┘  └───────┘  └───────┘  └───────┘           │   │
│  │                                                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  API Endpoint: https://api.chutes.ai                            │
└─────────────────────────────────────────────────────────────────┘
```

---

## Use Cases for Your Subnet

### 1. Validators Query LLMs for Scoring

Instead of running expensive GPU inference locally, validators can query models hosted on Chutes:

```python
import aiohttp

async def query_llm_for_scoring(prompt: str, api_key: str) -> str:
    """Query an LLM on Chutes for validation scoring"""
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "deepseek-ai/DeepSeek-V3-0324",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.0  # Deterministic for scoring
    }
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://llm.chutes.ai/v1/chat/completions",
            headers=headers,
            json=payload
        ) as resp:
            result = await resp.json()
            return result["choices"][0]["message"]["content"]
```

**Why use Chutes for validation?**
- Validators don't need expensive GPUs
- Consistent model behavior across all validators
- Lower barrier to running a validator
- Models are already deployed and maintained

### 2. Miners Deploy Models to Chutes

Miners can deploy their fine-tuned or custom models to Chutes, making them queryable by validators:

```python
# Miner deploys model to Chutes and commits endpoint to chain
from chutes import Chute, Image
from chutes.chute import NodeSelector

# Define the inference chute
@Chute(
    name="my-custom-model",
    image=Image(
        base_image="nvidia/cuda:12.1.0-runtime-ubuntu22.04",
        pip_packages=["torch", "transformers", "vllm"]
    ),
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb=40
    )
)
class MyModelChute:
    def __init__(self):
        from vllm import LLM
        self.model = LLM("my-org/my-model")
    
    def predict(self, prompt: str) -> str:
        outputs = self.model.generate([prompt])
        return outputs[0].outputs[0].text
```

### 3. Private Models with TEE

Miners can deploy proprietary models using Trusted Execution Environments (TEE) so that:
- Model weights remain encrypted and private
- Validators can query the model without seeing weights
- Hardware attestation proves the model is running securely

---

## Authentication & API Keys

### Creating API Keys

```bash
# Install Chutes CLI
pip install chutes

# Register (requires Bittensor wallet)
chutes register

# Create API key for programmatic access
chutes keys create --name my-subnet-key --admin
```

### API Key Types

| Key Type | Permissions | Use Case |
|----------|-------------|----------|
| `--admin` | Full access | Subnet owner/validator management |
| `--chute-ids <id>` | Specific chutes only | Access to miner's deployed models |
| `--images` | Image management | Building custom images |

### Using API Keys in Validators

```python
import os

class ChutesValidator:
    def __init__(self):
        # Load from environment for security
        self.chutes_api_key = os.getenv("CHUTES_API_KEY")
        
    async def evaluate_miner(self, miner_chute_id: str, test_prompts: list) -> float:
        """Evaluate a miner's model deployed on Chutes"""
        
        scores = []
        for prompt in test_prompts:
            response = await self._query_chute(miner_chute_id, prompt)
            score = self._score_response(response, prompt)
            scores.append(score)
        
        return sum(scores) / len(scores)
    
    async def _query_chute(self, chute_id: str, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.chutes_api_key}",
            "Content-Type": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"https://api.chutes.ai/chutes/{chute_id}/run",
                headers=headers,
                json={"prompt": prompt}
            ) as resp:
                result = await resp.json()
                return result.get("output", "")
```

---

## Validator-Free Queries for Registered Validators

Chutes can be configured to allow **free queries from registered validator hotkeys**. This is achieved through:

### 1. Hotkey-Based Authentication (Epistula)

Validators sign requests with their hotkey, which Chutes verifies against the metagraph:

```python
import time
import hashlib

def create_validator_headers(wallet, body: bytes) -> dict:
    """Create Epistula headers for validator authentication"""
    
    nonce = str(int(time.time() * 1e9))
    body_hash = hashlib.sha256(body).hexdigest()
    message = f"{nonce}.{body_hash}"
    signature = wallet.hotkey.sign(message.encode()).hex()
    
    return {
        "X-Epistula-Timestamp": nonce,
        "X-Epistula-Signature": signature,
        "X-Epistula-Hotkey": wallet.hotkey.ss58_address,
        "X-Epistula-Netuid": str(NETUID)  # Your subnet's netuid
    }
```

### 2. Subnet-Specific Free Tier

When deploying a chute for your subnet, you can configure it to:
- Check if the requester is a registered validator on your subnet
- Waive compute costs for validators (costs are covered by miner emissions)
- Rate-limit based on validator stake

```python
# In your chute deployment
@Chute(
    name="subnet-XX-model",
    # ... other config
    validator_free_access={
        "enabled": True,
        "netuid": 123,  # Your subnet
        "rate_limit_per_validator": 1000  # requests per hour
    }
)
class SubnetModel:
    # ...
```

### 3. Miner-Funded Validator Access

Miners can pre-fund credits for validator queries:

```python
# Miner funds validator access for their chute
async def fund_validator_access(chute_id: str, credits: float, api_key: str):
    """Pre-fund credits for validators to query this chute"""
    
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"https://api.chutes.ai/chutes/{chute_id}/fund-validators",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"credits": credits, "netuid": NETUID}
        ) as resp:
            return await resp.json()
```

---

## TEE (Trusted Execution Environment)

TEE allows miners to deploy **private models** while still proving they're running the correct code.

### What TEE Provides

1. **Confidentiality**: Model weights are encrypted at rest and in memory
2. **Attestation**: Cryptographic proof that specific code is running on verified hardware
3. **Integrity**: Tamper-proof execution environment

### Deploying a Private Model with TEE

```python
from chutes import Chute, Image
from chutes.chute import NodeSelector, TEEConfig

@Chute(
    name="private-model",
    image=Image(
        base_image="nvidia/cuda:12.1.0-runtime-ubuntu22.04",
        pip_packages=["torch", "transformers"]
    ),
    node_selector=NodeSelector(
        gpu_count=1,
        min_vram_gb=80,
        require_tee=True  # Require TEE-capable node
    ),
    tee_config=TEEConfig(
        enabled=True,
        attestation_required=True,
        encrypt_model_weights=True
    )
)
class PrivateModelChute:
    def __init__(self):
        # Model weights are decrypted only inside TEE
        self.model = load_encrypted_model("/secure/model.bin")
    
    def predict(self, prompt: str) -> str:
        return self.model.generate(prompt)
```

### Verifying TEE Attestation

Validators can verify that a miner's model is running in a legitimate TEE:

```python
async def verify_tee_attestation(chute_id: str) -> bool:
    """Verify that a chute is running in a valid TEE"""
    
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://api.chutes.ai/chutes/{chute_id}/attestation",
            headers={"Authorization": f"Bearer {api_key}"}
        ) as resp:
            attestation = await resp.json()
            
            # Verify attestation signature
            if not verify_tee_signature(attestation):
                return False
            
            # Check attestation is recent
            if attestation["timestamp"] < time.time() - 3600:
                return False
            
            # Verify expected code hash
            if attestation["code_hash"] != EXPECTED_CODE_HASH:
                return False
            
            return True
```

### Public vs Private Models

| Aspect | Public Model | Private (TEE) Model |
|--------|--------------|---------------------|
| Model weights | Visible/downloadable | Encrypted, only in TEE |
| Code | Open source | Open source (verifiable) |
| Hardware | Any GPU | TEE-capable GPU |
| Attestation | Optional | Required |
| Use case | Community models | Proprietary fine-tunes |

---

## Subnet Architecture Patterns

### Pattern A: Validators Use Chutes for Reference Models

Validators query a reference model on Chutes to evaluate miner submissions:

```
┌─────────────┐     Query     ┌─────────────┐
│  VALIDATOR  │──────────────►│   CHUTES    │
│             │◄──────────────│  (GPT-4/    │
│  Receives   │   Response    │  DeepSeek)  │
│  miner      │               └─────────────┘
│  submission │
│      ▼      │
│  Compares to│
│  reference  │
│      ▼      │
│  Sets score │
└─────────────┘
```

Example use case: Text generation subnet where miners submit completions, validators use a reference LLM to score quality.

### Pattern B: Miners Deploy Models to Chutes

Miners deploy their models to Chutes; validators query them directly:

```
┌─────────────┐     Query Miner's    ┌─────────────┐
│  VALIDATOR  │─────────────────────►│   CHUTES    │
│             │◄─────────────────────│             │
│  Scores     │   Model Response     │  ┌───────┐  │
│  response   │                      │  │Miner A│  │
│             │                      │  │ Model │  │
│             │                      │  └───────┘  │
│             │                      │  ┌───────┐  │
│             │                      │  │Miner B│  │
│             │                      │  │ Model │  │
│             │                      │  └───────┘  │
└─────────────┘                      └─────────────┘
```

Example use case: Model training subnet where miners fine-tune models and deploy them for evaluation.

### Pattern C: Hybrid (Reference + Miner Models)

Validators use Chutes for both reference models and miner model access:

```python
class HybridValidator:
    async def evaluate_round(self):
        # Generate test prompts using reference LLM
        test_prompts = await self.generate_test_prompts_via_chutes()
        
        # Query each miner's deployed model
        for miner_uid, miner_chute_id in self.miner_chutes.items():
            responses = []
            for prompt in test_prompts:
                response = await self.query_miner_chute(miner_chute_id, prompt)
                responses.append(response)
            
            # Score using reference model
            score = await self.score_responses_via_chutes(responses, test_prompts)
            self.scores[miner_uid] = score
```

---

## Integration Checklist

### For Validators

- [ ] Create Chutes API key with appropriate permissions
- [ ] Store API key securely (environment variable, not in code)
- [ ] Implement Epistula signing if using hotkey-based auth
- [ ] Handle rate limits and errors gracefully
- [ ] Consider caching responses where appropriate
- [ ] Verify TEE attestation for private models

### For Miners

- [ ] Install Chutes SDK: `pip install chutes`
- [ ] Register with Chutes: `chutes register`
- [ ] Deploy model as a Chute
- [ ] Commit Chute ID to chain for validator discovery
- [ ] (Optional) Enable TEE for private models
- [ ] (Optional) Pre-fund validator access credits

### For Subnet Owners

- [ ] Design scoring mechanism that uses Chutes efficiently
- [ ] Document which models/endpoints miners should deploy
- [ ] Consider validator cost subsidies from subnet treasury
- [ ] Set up monitoring for Chutes API availability
- [ ] Have fallback mechanisms if Chutes is unavailable

---

## Available Models on Chutes

Chutes provides access to various model types through templates:

| Template | Use Case | Example Models |
|----------|----------|----------------|
| `vllm` | LLM inference | DeepSeek-V3, Llama-3, Mistral |
| `sglang` | Fast LLM serving | Same as vllm, optimized routing |
| `diffusion` | Image generation | SDXL, Flux, Stable Diffusion 3 |
| `tei` | Text embeddings | BGE, E5, GTE |

### Querying Available Models

```python
async def list_available_models(api_key: str) -> list:
    """List models available on Chutes"""
    
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://api.chutes.ai/chutes/",
            headers={"Authorization": f"Bearer {api_key}"}
        ) as resp:
            return await resp.json()
```

---

## Cost Considerations

### Validator Costs

| Cost Type | Description | Mitigation |
|-----------|-------------|------------|
| Per-token inference | Pay for tokens generated | Use efficient prompts |
| Per-request | Fixed cost per API call | Batch where possible |
| Model hosting | If running custom models | Use shared models |

### Making Validation Affordable

1. **Use shared reference models**: Query existing popular models rather than deploying custom ones
2. **Batch requests**: Combine multiple evaluation prompts into single requests
3. **Cache deterministic results**: Store responses for repeated queries
4. **Stake-weighted sampling**: Evaluate high-stake miners more frequently

### Miner Costs

Miners deploying models to Chutes pay for:
- GPU compute time (hourly rate based on GPU type)
- Storage for model weights
- Network egress for responses

Miners can offset costs through:
- Subnet emissions (if their model scores well)
- Direct revenue from non-validator usage

---

## Example: LLM Scoring Validator

Complete example of a validator that uses Chutes to score miner LLM responses:

```python
import os
import asyncio
import aiohttp
import bittensor as bt
from bittensor_wallet import Wallet

CHUTES_API_KEY = os.getenv("CHUTES_API_KEY")
CHUTES_LLM_URL = "https://llm.chutes.ai/v1/chat/completions"
REFERENCE_MODEL = "deepseek-ai/DeepSeek-V3-0324"

class LLMScoringValidator:
    def __init__(self, netuid: int, wallet: Wallet, network: str = "finney"):
        self.netuid = netuid
        self.wallet = wallet
        self.subtensor = bt.Subtensor(network=network)
        self.metagraph = bt.Metagraph(netuid=netuid, network=network)
        self.scores = {}
    
    async def generate_test_prompt(self) -> str:
        """Use Chutes LLM to generate a test prompt"""
        
        payload = {
            "model": REFERENCE_MODEL,
            "messages": [
                {"role": "system", "content": "Generate a challenging but fair test prompt for an AI assistant."},
                {"role": "user", "content": "Generate one test prompt."}
            ],
            "temperature": 1.0,
            "max_tokens": 200
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                CHUTES_LLM_URL,
                headers={
                    "Authorization": f"Bearer {CHUTES_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=payload
            ) as resp:
                result = await resp.json()
                return result["choices"][0]["message"]["content"]
    
    async def score_response(self, prompt: str, response: str) -> float:
        """Use Chutes LLM to score a miner's response"""
        
        scoring_prompt = f"""Rate this response on a scale of 0-10.
        
Prompt: {prompt}

Response: {response}

Provide only a number between 0 and 10."""
        
        payload = {
            "model": REFERENCE_MODEL,
            "messages": [{"role": "user", "content": scoring_prompt}],
            "temperature": 0.0,
            "max_tokens": 10
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                CHUTES_LLM_URL,
                headers={
                    "Authorization": f"Bearer {CHUTES_API_KEY}",
                    "Content-Type": "application/json"
                },
                json=payload
            ) as resp:
                result = await resp.json()
                try:
                    score = float(result["choices"][0]["message"]["content"].strip())
                    return score / 10.0  # Normalize to 0-1
                except:
                    return 0.5  # Default on parse error
    
    async def query_miner(self, miner_endpoint: str, prompt: str) -> str:
        """Query a miner's LLM endpoint"""
        # Implementation depends on miner's API format
        pass
    
    async def run_validation_cycle(self):
        """Run one validation cycle"""
        self.metagraph.sync(subtensor=self.subtensor)
        
        # Generate test prompt
        test_prompt = await self.generate_test_prompt()
        
        # Query all miners
        for uid in range(self.metagraph.n):
            miner_endpoint = self.get_miner_endpoint(uid)
            if not miner_endpoint:
                self.scores[uid] = 0.0
                continue
            
            try:
                response = await self.query_miner(miner_endpoint, test_prompt)
                score = await self.score_response(test_prompt, response)
                self.scores[uid] = score
            except Exception as e:
                self.scores[uid] = 0.0
        
        # Set weights
        await self.set_weights()
```

---

## Troubleshooting

### Common Issues

**"API connection failed"**
- Check internet connectivity
- Verify API key is valid: `chutes keys list`
- Check Chutes status page

**"Rate limit exceeded"**
- Implement exponential backoff
- Reduce query frequency
- Request rate limit increase

**"Model not found"**
- Verify model name/ID
- Check if model is still deployed
- Ensure you have access permissions

**"TEE attestation failed"**
- Re-deploy chute to fresh TEE node
- Verify code hash matches expected
- Check attestation timestamp

### Getting Help

- Chutes Documentation: https://chutes.ai/docs
- Discord: Chutes community channel
- GitHub Issues: https://github.com/chutesai/chutes

---

## Summary

Chutes integration enables:

1. **Cheap validation**: Validators query LLMs without running GPUs
2. **Easy miner deployment**: Miners deploy models with simple SDK
3. **Privacy via TEE**: Proprietary models stay encrypted
4. **Free validator queries**: Hotkey-based auth enables cost-free validation
5. **Ecosystem synergy**: Leverage SN64's GPU network for your subnet

Start with Pattern A (validators use reference models) for simplest integration, then consider Pattern B (miners deploy to Chutes) as your subnet matures.
