# How to Create and Run a Bittensor Subnet from This Template

This guide walks you through creating a Bittensor subnet from start to finish using this template. The template provides a complete infrastructure for building, deploying, and auto-updating subnet validators.

## Overview

The workflow is:
1. Fork this template and give it your subnet's name
2. Configure Docker Hub credentials for automatic image publishing
3. Design your incentive mechanism and write `validator.py`
4. Get validators running with docker-compose (they auto-update)
5. Register your subnet on testnet, then mainnet
6. Miners read your validator code and build their own implementations

---

## Step 1: Fork and Rename the Template

### 1.1 Fork the Repository

1. Click "Use this template" or fork this repository to your GitHub account
2. Give it a meaningful name for your subnet (e.g., `mysubnet`, `compute-network`, `data-oracle`)
3. Clone your new repository locally:

```bash
git clone https://github.com/YOUR_USERNAME/your-subnet-name.git
cd your-subnet-name
```

### 1.2 Update Project Metadata

Edit `pyproject.toml` to reflect your subnet:

```toml
[project]
name = "your-subnet-name"
version = "0.1.0"
description = "Your subnet description"
# ... update other fields
```

---

## Step 2: Set Up Docker Hub for Auto-Updates

The template includes a GitHub Actions workflow that automatically builds and publishes Docker images when you push to `main`. Validators using Watchtower will automatically pull these updates.

### 2.1 Create a Docker Hub Account and Repository

1. Go to [Docker Hub](https://hub.docker.com/) and create an account (if you don't have one)
2. Create a new repository with the same name as your GitHub repo
3. Note your Docker Hub username

### 2.2 Create a Docker Hub Access Token

1. Go to [Docker Hub Security Settings](https://hub.docker.com/settings/security)
2. Click "New Access Token"
3. Give it a description (e.g., "GitHub Actions")
4. Copy the token (you won't see it again)

### 2.3 Add Secrets to GitHub

1. Go to your GitHub repository
2. Navigate to **Settings > Secrets and variables > Actions**
3. Add two secrets:
   - `DOCKERHUB_USERNAME`: Your Docker Hub username
   - `DOCKERHUB_TOKEN`: The access token you just created

### 2.4 Verify the Workflow

The workflow is already configured in `.github/workflows/docker-publish.yml`. It will:
- Build your Docker image on every push to `main`
- Tag it as `latest` and with the git SHA (for rollbacks)
- Push to Docker Hub
- Build for both AMD64 and ARM64 architectures

---

## Step 3: Design Your Incentive Mechanism

Before writing code, design what your subnet will incentivize. Read the documentation in the `docs/` folder, especially:

- `docs/04-mechanism-patterns.md` - Different architecture patterns used in production
- `docs/07-building-validators.md` - How to build validators
- `docs/08-incentive-design.md` - Designing effective reward mechanisms
- `docs/15-validator-only-development.md` - **Critical: Validator-only development philosophy**

> Note: you can have an LLM read these docs also to write your subnet.

### Key Design Questions

1. **What commodity does your subnet produce?** (compute, data, predictions, etc.)
2. **How do miners deliver this commodity?** (HTTP API, chain commits, external data sources)
3. **How does the validator evaluate quality?** (correctness, speed, cost, uniqueness)
4. **What scoring algorithm converts quality to weights?**

### The Validator-Only Rule

**You only write `validator.py`. Never write miner code.**

- The validator defines the rules of the game
- Miners read your validator to understand what's valued
- Miners compete by implementing their own solutions
- This creates genuine competition and innovation

---

## Step 4: Write Your Validator

All your subnet logic goes into a single file: `validator.py`. The template includes a minimal example that assigns random scores.

### 4.1 Use an LLM to Help Generate Code

The `docs/` folder contains comprehensive documentation designed to be fed to an LLM coding tool. Use this process:

1. Read through the relevant docs yourself to understand the concepts
2. Describe your incentive mechanism to the LLM
3. Ask it to modify `validator.py` based on the docs and your design
4. The LLM can reference the patterns in `docs/07-building-validators.md`

### 4.2 Test Locally

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -e .

# Run the validator (will fail without wallet/registration, but tests imports)
python validator.py --help
```

---

## Step 5: Push and Build the Docker Image

Once your validator is ready:

```bash
git add -A
git commit -m "Implement incentive mechanism for my subnet"
git push origin main
```

This triggers GitHub Actions to:
1. Build the Docker image
2. Push to Docker Hub as `your-username/your-subnet-name:latest`

Check the Actions tab in GitHub to verify the build succeeded.
