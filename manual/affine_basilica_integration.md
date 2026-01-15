# Affinetes + Basilica: Docker Container Submissions

This document describes a powerful pattern where **miners submit Docker containers** that validators run via **Basilica** (a pod orchestration service). Containers can use **Chutes** for LLM inference, enabling GPU-free evaluation environments.

## Pattern Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MINER WORKFLOW                              │
│                                                                     │
│   1. Create env.py       2. Build Image        3. Push to Registry │
│   ┌──────────────┐       ┌──────────────┐      ┌──────────────────┐│
│   │ class Actor: │  ──►  │ docker build │  ──► │ docker push      ││
│   │   evaluate() │       │              │      │ myuser/env:v1    ││
│   └──────────────┘       └──────────────┘      └──────────────────┘│
│                                                         │           │
│                                                         ▼           │
│   4. Commit image reference to chain (metadata)                     │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       VALIDATOR WORKFLOW                            │
│                                                                     │
│   1. Read miner's image    2. Provision on       3. Call methods   │
│      from chain               Basilica              & score        │
│   ┌──────────────────┐    ┌──────────────────┐  ┌────────────────┐ │
│   │ metagraph.axons  │ ─► │ load_env(        │─►│ env.evaluate() │ │
│   │ → "myuser/env:v1"│    │   mode="basilica"│  │ → score        │ │
│   └──────────────────┘    │   image=...      │  └────────────────┘ │
│                           └──────────────────┘                      │
└─────────────────────────────────────────────────────────────────────┘
```

## When to Use This Pattern

| Use Case | Description |
|----------|-------------|
| **Model Evaluation Environments** | Miners submit environments that test LLM capabilities on custom benchmarks |
| **Agent Environments** | Miners provide sandboxed environments where agents solve tasks |
| **Custom Scoring Logic** | Complex evaluation that can't be expressed as a simple API call |
| **Reproducible Benchmarks** | Containerized environments guarantee consistent execution |

This pattern is ideal when you want miners to compete on **environment design** or **evaluation methodology** rather than (or in addition to) model weights.

---

## Affinetes Framework

Affinetes is a lightweight container orchestration framework that:
- Defines environments via a simple `Actor` class in `env.py`
- Automatically injects an HTTP server (no HTTP code needed)
- Supports local Docker, remote SSH, and Basilica deployment modes

### Installation

```bash
cd affinetes
pip install -e .
```

### Core API

```python
import affinetes as af_env

# Build image from environment directory
af_env.build_image_from_env(
    env_path="environments/my-env",
    image_tag="my-env:latest",
    push=True,
    registry="docker.io/myuser"
)

# Load and run environment
env = af_env.load_env(
    image="myuser/my-env:latest",
    mode="basilica",  # or "docker" for local
    env_vars={"CHUTES_API_KEY": "xxx"}
)

# Call methods defined in Actor class
result = await env.evaluate(task_id=1, model="Qwen/Qwen3-32B")

# Cleanup
await env.cleanup()
```

---

## For Miners: Creating Docker Containers

### Step 1: Define Your Environment

Create an `env.py` file with an `Actor` class:

```python
# env.py
import os
import openai
import httpx

class Actor:
    """Environment actor with evaluation methods"""
    
    def __init__(self):
        # Read API key from environment variable
        self.api_key = os.getenv("CHUTES_API_KEY")
    
    async def evaluate(
        self,
        task_id: int,
        model: str = "Qwen/Qwen3-32B",
        base_url: str = "https://llm.chutes.ai/v1",
        timeout: int = 600,
        **kwargs
    ) -> dict:
        """
        Evaluate a model on a task
        
        Args:
            task_id: Which task to run
            model: Model name on Chutes
            base_url: LLM API endpoint
            timeout: Request timeout
            
        Returns:
            Dict with score, success, and metadata
        """
        # Generate challenge for this task
        challenge = self._generate_challenge(task_id)
        
        # Query model via Chutes
        client = openai.AsyncOpenAI(
            base_url=base_url,
            api_key=self.api_key,
            timeout=httpx.Timeout(timeout)
        )
        
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": challenge.prompt}],
            temperature=0.7
        )
        
        answer = response.choices[0].message.content
        
        # Score the response
        score = self._score_response(answer, challenge)
        
        return {
            "task_id": task_id,
            "score": score,
            "success": score > 0,
            "model": model
        }
    
    def _generate_challenge(self, task_id: int):
        # Your challenge generation logic
        pass
    
    def _score_response(self, answer: str, challenge) -> float:
        # Your scoring logic
        pass
```

### Step 2: Create Dockerfile

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy environment code
COPY . .

# Affinetes injects the HTTP server automatically
# No CMD needed for function-based environments
```

### Step 3: Create requirements.txt

```
openai>=1.0.0
httpx>=0.24.0
# Add your other dependencies
```

### Step 4: Build and Push

```bash
# Using affinetes CLI
afs build my-env --tag my-env:v1 --push --registry docker.io/myuser

# Or using Python API
import affinetes as af_env

af_env.build_image_from_env(
    env_path="my-env",
    image_tag="my-env:v1",
    push=True,
    registry="docker.io/myuser"
)
```

### Step 5: Commit to Chain

Miners commit their Docker image reference to the chain so validators can discover it:

```python
# Commit image reference as metadata
subtensor.commit(
    wallet=wallet,
    netuid=netuid,
    data=b"docker.io/myuser/my-env:v1"
)
```

---

## For Validators: Running Miner Containers

### Using Basilica Mode

Basilica provisions pods on-demand with specified resources:

```python
import os
import affinetes as af_env

# Authenticate with Basilica
os.environ["BASILICA_API_TOKEN"] = "your-token"

# Load miner's environment
env = af_env.load_env(
    mode="basilica",
    image="myuser/my-env:v1",  # From miner's chain commitment
    cpu_limit="2000m",
    mem_limit="8Gi",
    ttl_buffer=600,  # Pod TTL buffer in seconds
    env_vars={
        "CHUTES_API_KEY": os.getenv("CHUTES_API_KEY")
    }
)

try:
    # Run evaluation
    result = await env.evaluate(
        task_id=42,
        model="Qwen/Qwen3-32B",
        base_url="https://llm.chutes.ai/v1",
        timeout=300,
        _timeout=360  # Client-side timeout
    )
    
    score = result.get("score", 0)
    print(f"Miner scored: {score}")
    
finally:
    await env.cleanup()
```

### Parallel Task Execution

Run multiple tasks concurrently using asyncio:

```python
import asyncio
import affinetes as af_env

async def evaluate_tasks(image: str, task_ids: list, concurrent: int = 10):
    """Run multiple tasks with concurrency control"""
    
    env = af_env.load_env(
        mode="basilica",
        image=image,
        cpu_limit="2000m",
        mem_limit="8Gi",
        env_vars={"CHUTES_API_KEY": os.getenv("CHUTES_API_KEY")}
    )
    
    semaphore = asyncio.Semaphore(concurrent)
    
    async def run_task(task_id: int):
        async with semaphore:
            return await env.evaluate(
                task_id=task_id,
                model="Qwen/Qwen3-32B",
                timeout=300
            )
    
    try:
        tasks = [run_task(tid) for tid in task_ids]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results
    finally:
        await env.cleanup()

# Run tasks 1-100 with 20 concurrent
results = asyncio.run(evaluate_tasks(
    image="myuser/my-env:v1",
    task_ids=list(range(1, 101)),
    concurrent=20
))

# Calculate total score
scores = [r["score"] for r in results if isinstance(r, dict)]
print(f"Average score: {sum(scores) / len(scores):.4f}")
```

---

## Chutes Integration

Containers use Chutes for LLM inference, which:
- Offloads GPU compute (containers run on CPU)
- Provides access to many models (DeepSeek, Qwen, Llama, etc.)
- Uses OpenAI-compatible API

### Querying Chutes from Inside Container

```python
import openai
import httpx
import os

async def query_llm(prompt: str, model: str = "Qwen/Qwen3-32B") -> str:
    """Query LLM via Chutes"""
    
    client = openai.AsyncOpenAI(
        base_url="https://llm.chutes.ai/v1",
        api_key=os.getenv("CHUTES_API_KEY"),
        timeout=httpx.Timeout(300)
    )
    
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    return response.choices[0].message.content
```

### Available Models

| Model | Use Case |
|-------|----------|
| `Qwen/Qwen3-32B` | General purpose, fast |
| `deepseek-ai/DeepSeek-V3` | Reasoning, coding |
| `meta-llama/Llama-3.1-70B` | General purpose |

See Chutes documentation for full model list.

---

## Execution Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `docker` | Local Docker daemon | Development, testing |
| `basilica` | Remote pod provisioning | Production validation |
| `url` | Connect to existing service | User-deployed environments |

### Local Development

```python
# Run locally for development
env = af_env.load_env(
    mode="docker",
    image="my-env:latest",
    env_vars={"CHUTES_API_KEY": "xxx"}
)
```

### User-Deployed Services

```python
# Connect to user-deployed environment
env = af_env.load_env(
    mode="url",
    base_url="http://your-server.com:8080"
)
```

---

## Complete Validator Example

```python
#!/usr/bin/env python3
"""Validator that runs miner Docker containers via Basilica"""

import asyncio
import os
import bittensor as bt
import affinetes as af_env

class ContainerValidator:
    def __init__(self, netuid: int, wallet):
        self.netuid = netuid
        self.wallet = wallet
        self.subtensor = bt.Subtensor()
        self.metagraph = bt.Metagraph(netuid)
        self.scores = {}
    
    async def evaluate_miner(self, uid: int, image: str) -> float:
        """Evaluate a miner's container"""
        
        try:
            env = af_env.load_env(
                mode="basilica",
                image=image,
                cpu_limit="2000m",
                mem_limit="8Gi",
                env_vars={"CHUTES_API_KEY": os.getenv("CHUTES_API_KEY")}
            )
            
            try:
                # Run evaluation tasks
                results = []
                for task_id in range(1, 11):  # 10 tasks
                    result = await env.evaluate(
                        task_id=task_id,
                        model="Qwen/Qwen3-32B",
                        _timeout=600
                    )
                    results.append(result.get("score", 0))
                
                return sum(results) / len(results)
                
            finally:
                await env.cleanup()
                
        except Exception as e:
            print(f"Miner {uid} failed: {e}")
            return 0.0
    
    async def run_validation_cycle(self):
        """Run one validation cycle"""
        self.metagraph.sync()
        
        for uid in range(self.metagraph.n):
            # Get miner's image from chain commitment
            image = self.get_miner_image(uid)
            if not image:
                self.scores[uid] = 0.0
                continue
            
            score = await self.evaluate_miner(uid, image)
            self.scores[uid] = score
        
        # Set weights
        self.set_weights()
    
    def get_miner_image(self, uid: int) -> str:
        """Read miner's Docker image from chain metadata"""
        # Implementation depends on how miners commit their data
        pass
    
    def set_weights(self):
        """Set weights based on scores"""
        # Convert scores to weights and submit
        pass
```

---

## Best Practices

### For Miners

1. **Keep containers small** - Only include necessary dependencies
2. **Use Chutes for inference** - Don't bundle GPU-heavy models in container
3. **Handle errors gracefully** - Return partial scores on failures
4. **Version your images** - Use semantic versioning (`:v1`, `:v2`, etc.)
5. **Test locally first** - Use `mode="docker"` before pushing

### For Validators

1. **Set timeouts** - Both container-level and client-level (`_timeout`)
2. **Limit concurrency** - Use semaphores to avoid overwhelming Basilica
3. **Handle failures** - Score 0 for failed containers, don't crash
4. **Cleanup always** - Use try/finally to ensure `env.cleanup()` runs
5. **Cache results** - Avoid re-evaluating unchanged container versions

---

## Summary

The Affinetes + Basilica pattern enables:

1. **Container-based submissions** - Miners submit Docker images instead of code
2. **Isolated execution** - Each miner's environment runs in its own container
3. **GPU-free containers** - Use Chutes for LLM inference
4. **Parallel evaluation** - Run thousands of tasks concurrently
5. **Reproducibility** - Containerized environments guarantee consistent results

This pattern is particularly powerful for subnets that want to measure:
- Environment design quality
- Benchmark creation
- Agent task completion
- Custom evaluation methodologies
