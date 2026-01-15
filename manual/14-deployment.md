# Deployment Guide

This document covers deploying subnets and neurons from testnet to mainnet.

## Deployment Phases

```
┌────────────┐    ┌────────────┐    ┌────────────┐
│   LOCAL    │───►│  TESTNET   │───►│  MAINNET   │
│            │    │            │    │            │
│ - Free TAO │    │ - Free TAO │    │ - Real TAO │
│ - Fast dev │    │ - Real net │    │ - Production│
│ - Isolated │    │ - Testing  │    │ - Users    │
└────────────┘    └────────────┘    └────────────┘
```

## Phase 1: Testnet Deployment

### Testnet vs Mainnet

| Aspect | Testnet | Mainnet |
|--------|---------|---------|
| TAO | Free (faucet) | Real money |
| Risk | Low | High |
| Users | Developers | Everyone |
| Speed | Same | Same |
| Purpose | Testing | Production |

### Connect to Testnet

```python
from bittensor import Subtensor

subtensor = Subtensor(network="test")
```

```bash
btcli subnet list --network test
```

### Get Testnet TAO

Request from Discord faucet or community channels.

### Deploy Subnet to Testnet

```bash
# Check cost
btcli subnet lock-cost --network test

# Create subnet
btcli subnet create \
  --wallet.name owner \
  --network test

# Configure
btcli sudo set --netuid <NETUID> --param tempo --value 100 \
  --wallet.name owner --network test
```

### Deploy Neurons to Testnet

```bash
# Register miner
btcli subnet register --netuid <NETUID> \
  --wallet.name miner --wallet.hotkey default \
  --network test

# Register validator
btcli subnet register --netuid <NETUID> \
  --wallet.name validator --wallet.hotkey default \
  --network test

# Stake validator
btcli stake add --netuid <NETUID> \
  --wallet.name validator --wallet.hotkey default \
  --amount 1000 --network test
```

### Run Services on Testnet

**Miner:**
```python
from fastapi import FastAPI
from bittensor_wallet import Wallet
from bittensor import Subtensor
import uvicorn

wallet = Wallet(name="miner")
subtensor = Subtensor(network="test")

app = FastAPI()

@app.post("/compute")
async def compute(request: dict):
    # Verify Epistula signature from validator
    # Process request and return response
    return {"result": "processed"}

# Register on subnet
if not subtensor.is_hotkey_registered(NETUID, wallet.hotkey.ss58_address):
    subtensor.burned_register(wallet=wallet, netuid=NETUID)

uvicorn.run(app, host="0.0.0.0", port=8091)
```

**Validator:**
```python
wallet = Wallet(name="validator")
subtensor = Subtensor(network="test")
metagraph = Metagraph(netuid=NETUID, network="test")

# Run validation loop using HTTP queries to miners
```

### Testnet Validation Checklist

- [ ] Subnet creates successfully
- [ ] Neurons can register
- [ ] Miner serves and responds
- [ ] Validator queries and scores correctly
- [ ] Weights set successfully
- [ ] Emissions distribute as expected
- [ ] No errors in logs after 24+ hours

---

## Phase 2: Mainnet Deployment

### Pre-Launch Checklist

**Code Quality:**
- [ ] All tests passing
- [ ] No hardcoded test values
- [ ] Error handling complete
- [ ] Logging adequate
- [ ] Security audit (if applicable)

**Operations:**
- [ ] Monitoring configured
- [ ] Alerting set up
- [ ] Backup/recovery documented
- [ ] Runbooks written
- [ ] Team trained

**Economic:**
- [ ] Have sufficient TAO for creation
- [ ] Have TAO for initial validator stake
- [ ] Understand emission economics

### Mainnet Wallet Security

**Best Practices:**
- Keep coldkey on air-gapped machine
- Use hardware wallet if possible
- Never share mnemonics
- Use strong passwords
- Backup to multiple secure locations

**Wallet Setup:**
```bash
# Create production coldkey (on secure machine)
btcli wallet new_coldkey --wallet.name production --n_words 24

# Create hotkey (can be on server)
btcli wallet new_hotkey --wallet.name production --wallet.hotkey miner_1

# BACKUP IMMEDIATELY
# Store mnemonic in multiple secure locations
```

### Create Mainnet Subnet

```bash
# Check current cost
btcli subnet lock-cost --network finney

# Create (requires TAO)
btcli subnet create \
  --wallet.name production \
  --network finney

# Record your netuid!
```

### Configure Mainnet Hyperparameters

```bash
NETUID=<your-netuid>

# Set production parameters
btcli sudo set --netuid $NETUID --param tempo --value 360 \
  --wallet.name production --network finney

btcli sudo set --netuid $NETUID --param max_allowed_uids --value 256 \
  --wallet.name production --network finney

btcli sudo set --netuid $NETUID --param immunity_period --value 7200 \
  --wallet.name production --network finney

# Enable commit-reveal if needed
btcli sudo set --netuid $NETUID --param commit_reveal_weights_enabled --value true \
  --wallet.name production --network finney
```

### Set Subnet Identity

```bash
btcli subnet identity set \
  --netuid $NETUID \
  --display "My Subnet Name" \
  --description "What my subnet does" \
  --wallet.name production \
  --network finney
```

---

## Server Deployment

### Option 1: PM2 (Process Manager)

Install PM2:
```bash
npm install -g pm2
```

Create ecosystem file `ecosystem.config.js`:
```javascript
module.exports = {
  apps: [
    {
      name: "miner",
      script: "miner.py",
      interpreter: "python3",
      cwd: "/home/user/mysubnet",
      env: {
        NETWORK: "finney",
        NETUID: "1"
      },
      max_restarts: 10,
      restart_delay: 5000
    },
    {
      name: "validator",
      script: "validator.py",
      interpreter: "python3",
      cwd: "/home/user/mysubnet",
      env: {
        NETWORK: "finney",
        NETUID: "1"
      },
      max_restarts: 10,
      restart_delay: 5000
    }
  ]
};
```

Start services:
```bash
pm2 start ecosystem.config.js
pm2 save
pm2 startup  # Enable auto-start on boot
```

Monitor:
```bash
pm2 status
pm2 logs miner
pm2 logs validator
```

### Option 2: Docker (Basic)

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Miner
CMD ["python", "miner.py"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  miner:
    build: .
    restart: unless-stopped
    environment:
      - NETWORK=finney
      - NETUID=1
    volumes:
      - ~/.bittensor/wallets:/root/.bittensor/wallets:ro
    ports:
      - "8091:8091"

  validator:
    build:
      context: .
      dockerfile: Dockerfile.validator
    restart: unless-stopped
    environment:
      - NETWORK=finney
      - NETUID=1
    volumes:
      - ~/.bittensor/wallets:/root/.bittensor/wallets:ro
```

Start:
```bash
docker-compose up -d
docker-compose logs -f
```

---

## Recommended: Docker + Watchtower Auto-Update Pattern

**This is the recommended way to run a subnet in production.** This pattern enables automatic updates for validators when the subnet owner pushes changes to their main branch.

### Why This Pattern?

1. **Automatic Updates**: Validators don't need to manually update - Watchtower pulls new images automatically
2. **Consistent Environment**: Docker ensures all validators run identical code
3. **Zero Coordination**: Subnet owners push to main, validators auto-update within minutes
4. **Rollback Safety**: Docker images are versioned, easy to rollback if needed
5. **Reduced Support Burden**: Fewer "it works on my machine" issues

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SUBNET OWNER WORKFLOW                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   1. Push to main branch                                                │
│          │                                                              │
│          ▼                                                              │
│   2. GitHub Actions triggers                                            │
│          │                                                              │
│          ▼                                                              │
│   3. Build Docker image                                                 │
│          │                                                              │
│          ▼                                                              │
│   4. Push to Docker Hub (tagged :latest)                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        VALIDATOR WORKFLOW                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Watchtower (running continuously)                                     │
│          │                                                              │
│          ▼                                                              │
│   Checks Docker Hub every 5 minutes                                     │
│          │                                                              │
│          ▼                                                              │
│   Detects new :latest image                                             │
│          │                                                              │
│          ▼                                                              │
│   Pulls new image, restarts container                                   │
│          │                                                              │
│          ▼                                                              │
│   Validator now running updated code                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### For Subnet Owners: Setting Up CI/CD

#### Step 1: Create Dockerfile

Create a `Dockerfile` in your subnet repository root:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY pyproject.toml ./
COPY requirements.txt* ./

# Install dependencies - prefer pyproject.toml if it exists
RUN pip install --no-cache-dir --upgrade pip && \
    if [ -f pyproject.toml ]; then pip install --no-cache-dir -e .; \
    elif [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; \
    fi

# Copy application code
COPY . .

# Install the package itself if using pyproject.toml
RUN if [ -f pyproject.toml ]; then pip install --no-cache-dir -e .; fi

# Default command runs the validator
# Validators should override with environment variables
CMD ["python", "-m", "validator"]
```

#### Step 2: Create GitHub Actions Workflow

Create `.github/workflows/docker-publish.yml`:

```yaml
name: Build and Push Docker Image

on:
  push:
    branches:
      - main
      - master
  # Allow manual trigger for testing
  workflow_dispatch:

env:
  REGISTRY: docker.io
  # Change this to your Docker Hub username/org and repo name
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to Docker Hub
        uses: docker/login-action@v3
        with:
          username: ${{ secrets.DOCKERHUB_USERNAME }}
          password: ${{ secrets.DOCKERHUB_TOKEN }}

      - name: Extract metadata for Docker
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
          tags: |
            type=raw,value=latest,enable={{is_default_branch}}
            type=sha,prefix=
            type=ref,event=branch

      - name: Build and push Docker image
        uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
          platforms: linux/amd64,linux/arm64
```

#### Step 3: Configure Docker Hub Secrets

In your GitHub repository, go to **Settings > Secrets and variables > Actions** and add:

- `DOCKERHUB_USERNAME`: Your Docker Hub username
- `DOCKERHUB_TOKEN`: A Docker Hub access token (create at https://hub.docker.com/settings/security)

#### Step 4: Test the Workflow

Push to main and verify:
1. GitHub Actions workflow runs successfully
2. Image appears on Docker Hub with `latest` tag
3. SHA-tagged version also appears for rollback purposes

### For Validators: Running with Watchtower

#### Step 1: Create docker-compose.yml

Validators should create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  validator:
    # Replace with the actual subnet image
    image: subnetowner/mysubnet:latest
    container_name: subnet-validator
    restart: unless-stopped
    
    environment:
      # Network configuration
      - NETWORK=finney
      - NETUID=YOUR_NETUID
      
      # Wallet configuration (names only, keys mounted as volume)
      - WALLET_NAME=validator
      - HOTKEY_NAME=default
      
      # Optional: Additional configuration
      - LOG_LEVEL=INFO
    
    volumes:
      # Mount wallet directory (read-only for security)
      - ~/.bittensor/wallets:/root/.bittensor/wallets:ro
      
      # Optional: Persistent storage for validator state
      - validator-data:/app/data
    
    # If the validator serves an HTTP endpoint, expose the port
    # ports:
    #   - "8091:8091"
    
    # Resource limits (adjust based on your hardware)
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
    
    # Health check
    healthcheck:
      test: ["CMD", "python", "-c", "print('healthy')"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s
    
    # Logging configuration
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"

  # Watchtower - automatically updates containers when new images are available
  watchtower:
    image: containrrr/watchtower:latest
    container_name: watchtower
    restart: unless-stopped
    
    environment:
      # Check for updates every 5 minutes (300 seconds)
      - WATCHTOWER_POLL_INTERVAL=300
      
      # Clean up old images after updating
      - WATCHTOWER_CLEANUP=true
      
      # Only update containers with the enable label (optional, for safety)
      # - WATCHTOWER_LABEL_ENABLE=true
      
      # Notify on updates (optional - configure as needed)
      # - WATCHTOWER_NOTIFICATIONS=slack
      # - WATCHTOWER_NOTIFICATION_SLACK_HOOK_URL=https://hooks.slack.com/...
      
      # Rolling restart - don't stop until new container is healthy
      - WATCHTOWER_ROLLING_RESTART=true
      
      # Include stopped containers
      - WATCHTOWER_INCLUDE_STOPPED=false
      
    volumes:
      # Watchtower needs access to Docker socket to manage containers
      - /var/run/docker.sock:/var/run/docker.sock
    
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

volumes:
  validator-data:
```

#### Step 2: Start the Stack

```bash
# Pull latest images and start
docker-compose pull
docker-compose up -d

# View logs
docker-compose logs -f validator

# Check status
docker-compose ps
```

#### Step 3: Verify Auto-Updates

Watchtower will automatically:
1. Check Docker Hub every 5 minutes for new images
2. Pull new images when available
3. Stop the old container gracefully
4. Start a new container with the updated image
5. Clean up the old image

To verify Watchtower is working:
```bash
# Check Watchtower logs
docker-compose logs watchtower

# You should see entries like:
# time="2024-01-15T10:30:00Z" level=info msg="Found new subnetowner/mysubnet:latest image"
# time="2024-01-15T10:30:05Z" level=info msg="Stopping /subnet-validator"
# time="2024-01-15T10:30:10Z" level=info msg="Starting /subnet-validator"
```

### Advanced: GPU Support for Validators

If your validator requires GPU access (e.g., for model inference):

```yaml
version: '3.8'

services:
  validator:
    image: subnetowner/mysubnet:latest
    container_name: subnet-validator
    restart: unless-stopped
    
    environment:
      - NETWORK=finney
      - NETUID=YOUR_NETUID
      - WALLET_NAME=validator
      - HOTKEY_NAME=default
    
    volumes:
      - ~/.bittensor/wallets:/root/.bittensor/wallets:ro
      - validator-data:/app/data
    
    # GPU configuration
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1  # or "all" for all GPUs
              capabilities: [gpu]
    
    # For older Docker versions, use runtime instead:
    # runtime: nvidia
    
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "5"

  watchtower:
    image: containrrr/watchtower:latest
    container_name: watchtower
    restart: unless-stopped
    environment:
      - WATCHTOWER_POLL_INTERVAL=300
      - WATCHTOWER_CLEANUP=true
      - WATCHTOWER_ROLLING_RESTART=true
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock

volumes:
  validator-data:
```

For GPU-enabled images, update the Dockerfile:

```dockerfile
FROM nvidia/cuda:12.1-runtime-ubuntu22.04

# Install Python
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python
RUN ln -s /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

COPY . .
RUN pip install --no-cache-dir -e .

CMD ["python", "-m", "validator"]
```

### Notifications for Updates

Configure Watchtower to notify when updates occur:

**Slack notifications:**
```yaml
watchtower:
  environment:
    - WATCHTOWER_NOTIFICATIONS=slack
    - WATCHTOWER_NOTIFICATION_SLACK_HOOK_URL=https://hooks.slack.com/services/xxx/yyy/zzz
    - WATCHTOWER_NOTIFICATION_SLACK_IDENTIFIER=my-validator-host
```

**Email notifications:**
```yaml
watchtower:
  environment:
    - WATCHTOWER_NOTIFICATIONS=email
    - WATCHTOWER_NOTIFICATION_EMAIL_FROM=watchtower@example.com
    - WATCHTOWER_NOTIFICATION_EMAIL_TO=admin@example.com
    - WATCHTOWER_NOTIFICATION_EMAIL_SERVER=smtp.example.com
    - WATCHTOWER_NOTIFICATION_EMAIL_SERVER_PORT=587
    - WATCHTOWER_NOTIFICATION_EMAIL_SERVER_USER=user
    - WATCHTOWER_NOTIFICATION_EMAIL_SERVER_PASSWORD=password
```

### Rollback Procedure

If an update causes issues:

```bash
# Stop the current container
docker-compose stop validator

# Find the previous SHA-tagged image
docker images subnetowner/mysubnet

# Edit docker-compose.yml to use specific SHA tag
# image: subnetowner/mysubnet:abc123

# Restart with pinned version
docker-compose up -d validator

# Temporarily disable Watchtower updates for this container
# Add label to docker-compose.yml:
#   labels:
#     - "com.centurylinklabs.watchtower.enable=false"
```

### Security Considerations

1. **Wallet Security**: Mount wallets as read-only (`:ro`)
2. **Docker Socket**: Watchtower needs Docker socket access - ensure your host is secure
3. **Image Verification**: Consider using Docker Content Trust for signed images
4. **Network Isolation**: Use Docker networks to isolate validator from other services
5. **Secrets Management**: Use Docker secrets or environment files for sensitive data

```yaml
# Using environment file for secrets
services:
  validator:
    env_file:
      - .env.validator  # Contains WALLET_NAME, HOTKEY_NAME, etc.
```

### Monitoring the Stack

```bash
# Check all container status
docker-compose ps

# View validator logs
docker-compose logs -f validator

# View update history
docker-compose logs watchtower | grep -E "(Found new|Stopping|Starting)"

# Check resource usage
docker stats

# Execute commands in running container
docker-compose exec validator btcli wallet overview
```

### Option 3: Systemd

Create service file `/etc/systemd/system/bittensor-miner.service`:
```ini
[Unit]
Description=Bittensor Miner
After=network.target

[Service]
Type=simple
User=bittensor
WorkingDirectory=/home/bittensor/mysubnet
Environment=NETWORK=finney
Environment=NETUID=1
ExecStart=/home/bittensor/mysubnet/.venv/bin/python miner.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable bittensor-miner
sudo systemctl start bittensor-miner
sudo systemctl status bittensor-miner
```

---

## Monitoring

### Logging Best Practices

```python
import logging
import bittensor as bt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/bittensor/miner.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Log important events
logger.info(f"Miner started on netuid {netuid}")
logger.info(f"Registered as UID {uid}")
logger.warning(f"Query timeout from {caller}")
logger.error(f"Failed to set weights: {error}")
```

### Metrics to Monitor

**Miner Metrics:**
- Requests per minute
- Average response time
- Error rate
- Incentive score
- Emission received

**Validator Metrics:**
- Queries sent per minute
- Response success rate
- Weight update frequency
- Dividend earnings
- Metagraph sync status

### Prometheus + Grafana

Export metrics:
```python
from prometheus_client import start_http_server, Counter, Histogram

REQUESTS = Counter('miner_requests_total', 'Total requests')
RESPONSE_TIME = Histogram('miner_response_seconds', 'Response time')

# In your handler
@RESPONSE_TIME.time()
def forward(synapse):
    REQUESTS.inc()
    # ...
```

### Alerting

Set up alerts for:
- Service down
- High error rate (>5%)
- Low incentive/dividends
- Registration dropped
- Stake changes

---

## Emergency Procedures

### Service Recovery

```bash
# Check status
pm2 status
# or
docker-compose ps
# or
systemctl status bittensor-miner

# Restart
pm2 restart miner
# or
docker-compose restart miner
# or
systemctl restart bittensor-miner

# Check logs
pm2 logs miner --lines 100
# or
docker-compose logs --tail 100 miner
# or
journalctl -u bittensor-miner -n 100
```

### Rollback Procedure

```bash
# If using git
cd /path/to/mysubnet
git log --oneline -5  # Find good commit
git checkout <commit-hash>

# Restart services
pm2 restart all
```

### Subnet Emergency

```bash
# Disable registrations (prevent new neurons)
btcli sudo set --netuid $NETUID --param registration_allowed --value false \
  --wallet.name production --network finney

# Transfer ownership (if compromised)
# Use SDK - no CLI command
```

---

## Upgrade Procedure

### Zero-Downtime Upgrade

1. **Prepare new version:**
   ```bash
   cd /path/to/mysubnet
   git pull origin main
   pip install -r requirements.txt
   ```

2. **Test locally:**
   ```bash
   python -m pytest tests/
   ```

3. **Rolling restart:**
   ```bash
   # Restart one at a time
   pm2 restart miner
   # Wait and verify
   pm2 restart validator
   ```

### Database Migrations

If your mechanism uses a database:
1. Backup before migration
2. Test migration on copy
3. Run migration with service stopped (brief downtime)
4. Verify and restart

---

## Operational Checklist

### Daily
- [ ] Check service status
- [ ] Review error logs
- [ ] Verify emissions received
- [ ] Check incentive/dividend trends

### Weekly
- [ ] Review metagraph changes
- [ ] Analyze competitor behavior
- [ ] Check wallet balances
- [ ] Backup any state/databases

### Monthly
- [ ] Security review
- [ ] Performance optimization
- [ ] Cost analysis
- [ ] Documentation update
