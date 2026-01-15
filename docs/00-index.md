# Bittensor Subnet Development: Complete Guide

This comprehensive documentation provides everything needed to understand, create, and operate a Bittensor subnet. It synthesizes information from multiple sources and real-world production mechanisms.

## What This Documentation Covers

This guide is designed for LLMs and developers who need to:
- Understand Bittensor's architecture and incentive model
- Design effective incentive mechanisms
- Implement miners and validators using appropriate patterns
- Deploy and operate subnets on localnet, testnet, and mainnet

## Reading Order

### Foundational Concepts
1. **[01-overview.md](01-overview.md)** - What Bittensor is, key roles, and network structure
2. **[02-core-concepts.md](02-core-concepts.md)** - Subnets, neurons, keys, tokens, stake, and metagraph
3. **[03-architecture.md](03-architecture.md)** - Chain layer, SDK layer, and communication primitives

### Subnet Development
4. **[04-mechanism-patterns.md](04-mechanism-patterns.md)** - **CRITICAL**: Different mechanism architectures used in production (open source HTTP APIs)
5. **[05-subnet-lifecycle.md](05-subnet-lifecycle.md)** - Creating, activating, and configuring subnets
6. **[06-building-miners.md](06-building-miners.md)** - Implementing miners across different patterns
7. **[07-building-validators.md](07-building-validators.md)** - Scoring algorithms and weight setting
8. **[08-incentive-design.md](08-incentive-design.md)** - Designing effective reward mechanisms

### Technical Reference
9. **[09-python-sdk.md](09-python-sdk.md)** - SDK classes, methods, and usage patterns
10. **[10-btcli-reference.md](10-btcli-reference.md)** - Command-line interface reference
11. **[11-hyperparameters.md](11-hyperparameters.md)** - Configurable subnet parameters
12. **[12-epoch-mechanism.md](12-epoch-mechanism.md)** - On-chain consensus and emissions

### Deployment
13. **[13-local-development.md](13-local-development.md)** - Running localnet for development
14. **[14-deployment.md](14-deployment.md)** - Testnet and mainnet deployment

### Development Rules
15. **[15-validator-only-development.md](15-validator-only-development.md)** - **CRITICAL**: Validator-only development philosophy and single-file pattern
16. **[17-writing-a-subnet.md](17-writing-a-subnet.md)** - **START HERE**: Simplicity-first approach and "what am I measuring?"

### Getting Started with This Template
17. **[16-how-to-use-template.md](16-how-to-use-template.md)** - Step-by-step guide to creating a subnet using this repository

### External Integrations
18. **[18-chutes-integration.md](18-chutes-integration.md)** - Integrating Chutes (SN64) for LLM inference, miner model deployment, TEE, and free validator queries

## Key Insights

### Mechanism Architecture Diversity

**The Axon/Dendrite/Synapse pattern is deprecated.** New subnets should always use open source, custom communication methods. Document 04 covers the recommended patterns used in production:

- **HTTP/REST APIs** with Epistula signed headers (Targon, Chutes)
- **External data sources** validated on-chain (Gittensor, Data Universe)
- **Compute marketplaces** with attestation (Targon, Chutes)
- **Forecast/prediction markets** (Synth, Zeus)
- **Data scraping/indexing** with periodic validation (Data Universe)

Miners commit connection info to chain (API endpoints, S3 URLs, database endpoints). Validators read this to discover how to communicate with miners.

### Sybil Resistance Reality

**Coldkeys are NOT sybil-proof.** The actual sybil protection in Bittensor is:
- **256 UID slots per subnet** with **dynamic registration costs**
- When expected rewards increase, registration costs rise automatically
- This creates economic equilibrium preventing pure sybil farming
- "UID pressure" = demand for slots = registration cost

### Subnet Design Philosophy

**Keep subnets minimal.** Focus on the validator (the "referee"), leave ingenuity to miners.
- Few files, not dozens
- Clear, auditable scoring criteria
- Validators define "what is valuable"; miners compete to provide it

**NEVER write miner code.** See [15-validator-only-development.md](15-validator-only-development.md) for the full rule:
- Only write `validator.py` - miners read it to understand the interface
- Define where you look for inputs, how you evaluate, what you score
- Miners implement their own solutions based on your specification
- Single-file validators are easier to audit, understand, and maintain

### Leverage the Ecosystem

**In practice, subnets on Bittensor should always try to use other subnets.** Don't reinvent infrastructure that already exists in the network:

- **Model Training/Inference**: If miners are training models, have them upload to [Chutes (SN64)](18-chutes-integration.md) and commit the Chutes endpoint to the chain. Validators query Chutes rather than miners directly.
- **Decentralized Storage**: If you need decentralized storage, use [Hippius](https://hippius.com) rather than building your own storage layer.
- **Other Specialized Subnets**: Check what other subnets offer before building infrastructure from scratch.

This creates ecosystem synergy—your subnet benefits from the security and reliability of established subnets, and you focus only on what makes your measurement unique.

### Hyperparameter Reality

**Not all hyperparameters are changeable** even though they appear in docs:
- `tempo` - NOT changeable
- `max_allowed_uids` - NOT changeable
- Don't use commit-reveal unless absolutely necessary (weight copying is a mature subnet problem)

Choose your architecture based on your commodity type, not by copying a template.

## Quick Start Paths

### "I want to understand Bittensor"
Start with documents 01-03

### "I want to design a subnet mechanism"
Read documents 01-04, then 08

### "I want to implement a miner"
Read documents 02-04, then 06

### "I want to implement a validator"
Read documents 02-04, then 07-08

### "I want to deploy to mainnet"
Read documents 05, 10-11, then 13-14

### "I want to build a subnet"
Read document 17 first (what am I measuring?), then 15 (validator-only rule), then 04, 07-08

### "How do I make a subnet with this repository?"
Read document 16 for the complete step-by-step walkthrough

### "How do I integrate Chutes for LLM validation?"
Read document 18 for Chutes integration patterns, TEE, and validator-free queries

## Code Anchors (Where Truth Lives)

- **Chain logic**: `subtensor/pallets/subtensor/src/`
- **Python SDK**: `bittensor/bittensor/core/`
- **CLI**: `btcli/bittensor_cli/`
- **Local chain**: `subtensor/docker-compose.localnet.yml`

## Networks

| Network | Purpose | Endpoint |
|---------|---------|----------|
| Finney (Mainnet) | Production | `wss://entrypoint-finney.opentensor.ai:443` |
| Testnet | Testing | `wss://test.finney.opentensor.ai:443` |
| Local | Development | `ws://127.0.0.1:9944` |
