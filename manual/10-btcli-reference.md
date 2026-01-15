# BTCLI Reference

The Bittensor CLI (`btcli`) is the command-line interface for interacting with Bittensor.

## Installation

```bash
pip install bittensor-cli
```

## Global Options

These options work with most commands:

```
--network <network>    Network to connect to (finney, test, local, or URL)
--wallet.name <name>   Wallet name
--wallet.hotkey <key>  Hotkey name
--wallet.path <path>   Custom wallet path
--no-prompt            Skip confirmation prompts
--quiet                Minimal output
--verbose              Verbose output
```

## Command Groups

### wallet - Key Management

```bash
# List wallets
btcli wallet list

# Create new coldkey
btcli wallet new_coldkey --wallet.name <name> --n_words 12

# Create new hotkey
btcli wallet new_hotkey --wallet.name <name> --wallet.hotkey <hotkey>

# Show wallet info
btcli wallet overview --wallet.name <name>

# Get balance
btcli wallet balance --wallet.name <name>

# Regenerate from mnemonic
btcli wallet regen_coldkey --wallet.name <name> --mnemonic "word1 word2..."
btcli wallet regen_hotkey --wallet.name <name> --wallet.hotkey <hotkey> --mnemonic "..."
```

### subnet - Subnet Operations

```bash
# List all subnets
btcli subnet list --network finney

# Show subnet details
btcli subnet show --netuid <NETUID> --network finney

# Get subnet hyperparameters
btcli subnet hyperparameters --netuid <NETUID> --network finney

# Get metagraph
btcli subnet metagraph --netuid <NETUID> --network finney

# Check registration cost
btcli subnet lock-cost --network finney

# Create new subnet
btcli subnet create --wallet.name <name> --network finney

# Register on subnet (burn)
btcli subnet register --netuid <NETUID> --wallet.name <name> --wallet.hotkey <hotkey>

# Register on subnet (PoW)
btcli subnet pow_register --netuid <NETUID> --wallet.name <name> --wallet.hotkey <hotkey>
```

### stake - Staking Operations

```bash
# Show stake
btcli stake show --wallet.name <name>

# Add stake
btcli stake add --wallet.name <name> --wallet.hotkey <hotkey> --amount <TAO>

# Remove stake
btcli stake remove --wallet.name <name> --wallet.hotkey <hotkey> --amount <TAO>

# Delegate to another hotkey
btcli stake delegate --wallet.name <name> --delegate_ss58 <address> --amount <TAO>

# Undelegate
btcli stake undelegate --wallet.name <name> --delegate_ss58 <address> --amount <TAO>
```

### root - Root Network Operations

```bash
# Show root network
btcli root show --network finney

# List root neurons
btcli root list --network finney

# Register on root
btcli root register --wallet.name <name> --wallet.hotkey <hotkey>

# Set root weights
btcli root weights --wallet.name <name> --wallet.hotkey <hotkey>
```

### sudo - Subnet Owner Operations

```bash
# Set hyperparameter
btcli sudo set --netuid <NETUID> --param <param> --value <value> --wallet.name <name>

# Get hyperparameter
btcli sudo get --netuid <NETUID> --param <param>

# Available parameters:
#   tempo, max_allowed_uids, immunity_period, weights_rate_limit,
#   max_weights_limit, min_allowed_weights, difficulty, burn,
#   commit_reveal_weights_enabled, commit_reveal_period, ...
```

### weights - Weight Operations

```bash
# Commit weights (for commit-reveal)
btcli weights commit --netuid <NETUID> --wallet.name <name> --wallet.hotkey <hotkey>

# Reveal weights
btcli weights reveal --netuid <NETUID> --wallet.name <name> --wallet.hotkey <hotkey>
```

### transfer - TAO Transfers

```bash
# Transfer TAO
btcli wallet transfer --dest <address> --amount <TAO> --wallet.name <name>
```

---

## Common Workflows

### 1. Initial Setup

```bash
# Create wallet with coldkey and hotkey
btcli wallet new_coldkey --wallet.name my_wallet
btcli wallet new_hotkey --wallet.name my_wallet --wallet.hotkey miner

# Check balance
btcli wallet balance --wallet.name my_wallet
```

### 2. Register on Existing Subnet

```bash
# Check registration cost
btcli subnet show --netuid 1 --network finney

# Register via burn
btcli subnet register \
  --netuid 1 \
  --wallet.name my_wallet \
  --wallet.hotkey miner \
  --network finney
```

### 3. Create New Subnet

```bash
# Check creation cost
btcli subnet lock-cost --network finney

# Create subnet
btcli subnet create \
  --wallet.name my_wallet \
  --network finney

# Note the assigned netuid from output
```

### 4. Configure Subnet Hyperparameters

```bash
# Set tempo
btcli sudo set \
  --netuid <NETUID> \
  --param tempo \
  --value 100 \
  --wallet.name my_wallet

# Set max neurons
btcli sudo set \
  --netuid <NETUID> \
  --param max_allowed_uids \
  --value 512 \
  --wallet.name my_wallet

# Enable commit-reveal
btcli sudo set \
  --netuid <NETUID> \
  --param commit_reveal_weights_enabled \
  --value true \
  --wallet.name my_wallet
```

### 5. Monitor Subnet

```bash
# Get metagraph
btcli subnet metagraph --netuid <NETUID> --network finney

# Check hyperparameters
btcli subnet hyperparameters --netuid <NETUID> --network finney

# Show stake distribution
btcli stake show --wallet.name my_wallet --all
```

### 6. Stake Management

```bash
# Add stake to your miner
btcli stake add \
  --wallet.name my_wallet \
  --wallet.hotkey miner \
  --amount 100.0

# Delegate to someone else's validator
btcli stake delegate \
  --wallet.name my_wallet \
  --delegate_ss58 5ValidatorHotkey... \
  --amount 50.0
```

---

## Network Endpoints

| Network | Endpoint | Usage |
|---------|----------|-------|
| `finney` | `wss://entrypoint-finney.opentensor.ai:443` | Mainnet |
| `test` | `wss://test.finney.opentensor.ai:443` | Testnet |
| `local` | `ws://127.0.0.1:9944` | Local development |

```bash
# Use specific network
btcli subnet list --network finney
btcli subnet list --network test
btcli subnet list --network local

# Use custom endpoint
btcli subnet list --network ws://custom-node:9944
```

---

## Configuration File

BTCLI can use a config file at `~/.bittensor/config.yaml`:

```yaml
wallet:
  name: default
  hotkey: default
  path: ~/.bittensor/wallets

subtensor:
  network: finney
```

Override with command line flags:
```bash
btcli subnet list --network test  # Overrides config
```

---

## Useful Commands

### Quick Status Check

```bash
# Balance
btcli wallet balance --wallet.name my_wallet

# Registration status
btcli wallet overview --wallet.name my_wallet

# Subnet emissions
btcli subnet show --netuid 1
```

### Debugging

```bash
# Verbose output
btcli subnet metagraph --netuid 1 --verbose

# Check specific neuron
btcli subnet metagraph --netuid 1 | grep "5HotkeyAddress..."
```

### Batch Operations

```bash
# Export wallet addresses
btcli wallet list | grep -E "coldkey|hotkey"

# Monitor all subnets
for i in {1..64}; do
  btcli subnet show --netuid $i --network finney 2>/dev/null
done
```

---

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | General error |
| 2 | Invalid arguments |
| 3 | Wallet error |
| 4 | Network error |
