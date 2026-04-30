# Lemma Subnet (Bittensor)

Lemma is a Bittensor subnet for formal theorem proving in Lean 4.

- Miners receive deterministic theorem challenges.
- Miners return strict Lean 4 proof code over HTTP (`POST /solve`).
- Validators compile proofs with Lean.
- Only valid proofs are eligible for rewards.
- Rewards prioritize elegant proofs: shorter code and fewer imports.

## What this repository contains

- `validator.py`: Lemma validator loop (challenge selection, miner querying, Lean verification, scoring, weights).
- `TESTNET_RUNBOOK.md`: 72-hour deployment checklist and go/hold process.
- `CHECKLIST.md`: live operator task checklist for the 72-hour run.
- `data/round_results.jsonl`: append-only proof/scoring telemetry log.
- `knowledge/`: mechanism and subnet design references.
- `docker-compose.yml` + `Dockerfile`: containerized validator runtime.

## Quick start

1. Copy `env.example` to `.env` and set wallet + `NETUID`.
2. Install Lean 4 on the validator host (the validator compiles submissions locally).
3. Set `MINER_ENDPOINTS` in `.env`:
   - Example: `MINER_ENDPOINTS=2=http://10.0.0.12:8080,5=http://10.0.0.13:8080`
   - Overrides take precedence over on-chain commitments for matching UIDs.
4. Run locally:
   - `uv run python validator.py --network finney --netuid <your-netuid>`
5. Or run with Docker:
   - `docker compose up --build -d`

## Preflight Check

- Script: `scripts/preflight_check.py`
- Purpose: verify environment and filesystem readiness before validator launch.
- Example:
  - `python3 scripts/preflight_check.py --root . --env-file .env --require-lean`
- Optional metrics check:
  - `python3 scripts/preflight_check.py --metrics-url http://127.0.0.1:9109/metrics.json`
- Exit code:
  - `0` when all required checks pass
  - `2` when any required check fails

## Local integration test

1. Start a stub miner:
   - `python3 miner_stub.py`
2. Point validator to it:
   - `MINER_ENDPOINTS=2=http://127.0.0.1:8080`
3. Run validator with your validator wallet:
   - `uv run python validator.py --network finney --netuid <your-netuid>`

### Multi-miner local stack

- Start 3 stub miners:
  - `docker compose -f docker-compose.miner.yml up --build -d`
- Default local ports:
  - miner-1: `http://127.0.0.1:8081`
  - miner-2: `http://127.0.0.1:8082`
  - miner-3: `http://127.0.0.1:8083`
- Example validator mapping:
  - `MINER_ENDPOINTS=2=http://127.0.0.1:8081,3=http://127.0.0.1:8082,4=http://127.0.0.1:8083`

## Miner API contract

- Endpoint: `POST /solve`
- Request fields:
  - `challenge_id`
  - `statement`
  - `lean_header`
  - `lean_goal`
  - `max_response_chars`
- Response fields:
  - `proof_code` (required)
  - `metadata` (optional)
  - Epistula response headers are expected for verification:
    - `X-Epistula-Timestamp`
    - `X-Epistula-Signature`
    - `X-Epistula-Hotkey`
- Validator request headers (Epistula):
  - `X-Epistula-Timestamp`
  - `X-Epistula-Signature`
  - `X-Epistula-Hotkey`
  - Enabled by default via `EPISTULA_SIGN_REQUESTS=true`
  - Response verification defaults:
    - `EPISTULA_VERIFY_RESPONSES=true`
    - `EPISTULA_STRICT_VERIFY=true`
    - `EPISTULA_MAX_SKEW_SECONDS=60`

## Miner endpoint commitments

- Primary discovery path: on-chain commitment per miner UID.
- Supported commitment payloads:
  - plain URL string (e.g. `https://miner.example.com`)
  - JSON with `endpoint`, `url`, or `base_url`
- Validator refresh cadence: `COMMITMENT_REFRESH_BLOCKS` (default `20`).
- Local `MINER_ENDPOINTS` remains available as override/fallback.

## Scoring and hardening details

- Challenge generation:
  - extracted module: `challenge_engine.py`
  - deterministic seeded challenge batch each round (`CHALLENGE_BATCH_SIZE`, default `6`)
  - multiple theorem families to reduce overfitting to a tiny static bank
  - each challenge carries a difficulty weight
- Round score formula:
  - `valid_rate = weighted mean(compiled_flag, challenge_difficulty)`
  - `avg_elegance = weighted mean(elegance_score, challenge_difficulty)`
  - `instant_score = valid_rate * avg_elegance`
  - `ema_score = EMA_ALPHA * instant_score + (1-EMA_ALPHA) * previous`
  - toggle with `DIFFICULTY_WEIGHTING_ENABLED=true`
- Submission constraints:
  - `MAX_RESPONSE_CHARS`
  - `MAX_PROOF_LINES`
  - `MAX_IMPORT_LINES`
  - forbidden tokens (`unsafe`, `IO.`, `open scoped`, `set_option`)
- Lean sandboxing:
  - timeout via `LEAN_TIMEOUT_SECONDS`
  - process memory cap via `LEAN_MEMORY_LIMIT_MB`
- Round telemetry:
  - each `(uid, challenge)` attempt is written to `ROUND_LOG_PATH` as JSONL

## Validator Metrics Endpoint

- Built into `validator.py` (no extra dependencies).
- Enabled with:
  - `METRICS_ENABLED=true`
  - `METRICS_HOST=0.0.0.0`
  - `METRICS_PORT=9109`
- Endpoints:
  - Prometheus text: `GET /metrics`
  - JSON snapshot: `GET /metrics.json`
- Key exported signals:
  - attempt/valid/invalid counters
  - Epistula verification failures
  - timeout and HTTP error counters
  - set_weights success/failure counters
  - per-UID instant score, EMA score, valid rate

### Alert Checker Script

- Script: `scripts/check_alerts.py`
- Purpose: poll `/metrics.json` and fail fast on threshold breaches (cron/CI-friendly).
- Example:
  - `python3 scripts/check_alerts.py --url http://127.0.0.1:9109/metrics.json`
- JSON output:
  - `python3 scripts/check_alerts.py --json`
- Exit code:
  - `0` when healthy
  - `2` when any threshold is breached

### Staged Tuning Report

- Script: `scripts/tuning_report.py`
- Purpose: synthesize concentration, optional metrics alerts, and optional validator disagreement into a single `GO`/`HOLD`.
- Example (single validator log):
  - `python3 scripts/tuning_report.py --log data/round_results.jsonl --tail 5000`
- Example (with metrics + second validator):
  - `python3 scripts/tuning_report.py --log data/validator_a.jsonl --validator-b-log data/validator_b.jsonl --metrics-url http://127.0.0.1:9109/metrics.json --tail 5000`
- JSON output:
  - `python3 scripts/tuning_report.py --json`

### One-Command Tuning Cycle

- Script: `scripts/run_tuning_cycle.py`
- Purpose: run staged report + difficulty patch generation and store timestamped artifacts.
- Example:
  - `python3 scripts/run_tuning_cycle.py --log data/round_results.jsonl --tail 5000`
- With second validator + live metrics:
  - `python3 scripts/run_tuning_cycle.py --log data/validator_a.jsonl --validator-b-log data/validator_b.jsonl --metrics-url http://127.0.0.1:9109/metrics.json --tail 5000`
- Output directory:
  - `artifacts/tuning/<timestamp>/`
  - includes `tuning_report.json`, `difficulty.patch`, and `summary.json`

## Testnet Drill Playbook

- Deploy 2-3 validators and at least 3 miners for 48-72 hours.
- Monitor:
  - emission concentration (top miners share over time)
  - invalid-proof rate per miner
  - validator disagreement in effective weights
  - Epistula verification failure rates
- Tune gradually:
  - `CHALLENGE_BATCH_SIZE`, `EMA_ALPHA`, proof/import limits
  - only one or two knobs per day to preserve attribution.

## Log Analysis Script

- Script: `scripts/analyze_round_logs.py`
- Default input: `data/round_results.jsonl`
- Human-readable report:
  - `python3 scripts/analyze_round_logs.py --log data/round_results.jsonl`
- Analyze only latest rows:
  - `python3 scripts/analyze_round_logs.py --tail 2000`
- JSON output (for dashboards/automation):
  - `python3 scripts/analyze_round_logs.py --json`
- Report includes:
  - per-UID valid rate, avg elegance, avg instant score, dominant failure reason
  - concentration indicators: `top_1_share`, `top_3_share`, `hhi`

## Validator Disagreement Script

- Script: `scripts/compare_validator_logs.py`
- Compare two validator logs:
  - `python3 scripts/compare_validator_logs.py --left data/validator_a.jsonl --right data/validator_b.jsonl`
- Compare latest window only:
  - `python3 scripts/compare_validator_logs.py --left data/validator_a.jsonl --right data/validator_b.jsonl --tail 3000`
- JSON output:
  - `python3 scripts/compare_validator_logs.py --left data/validator_a.jsonl --right data/validator_b.jsonl --json`
- Report includes:
  - overlap coverage (`shared_rows`, `left_only_rows`, `right_only_rows`)
  - `valid_flag_agreement`
  - mean absolute deltas for elegance and instant score
  - UID-level disagreement counts and top mismatch rows

## Difficulty Calibration Script

- Script: `scripts/calibrate_difficulty.py`
- Purpose: estimate per-family empirical solve rates and recommend updated difficulty weights.
- Run:
  - `python3 scripts/calibrate_difficulty.py --log data/round_results.jsonl`
- Latest-window calibration:
  - `python3 scripts/calibrate_difficulty.py --tail 5000 --target-solve-rate 0.60`
- JSON output:
  - `python3 scripts/calibrate_difficulty.py --json`
- Output includes:
  - per-family sample count and solve rate
  - multiplier suggestion (clamped by `--min-multiplier` / `--max-multiplier`)
  - suggested difficulty value to apply in `challenge_engine.py`

### Difficulty Patch Generator

- Script: `scripts/generate_difficulty_patch.py`
- Purpose: create a reviewable unified diff for `challenge_engine.py` from calibration output.
- Print patch to stdout:
  - `python3 scripts/generate_difficulty_patch.py --log data/round_results.jsonl --tail 5000`
- Save patch file:
  - `python3 scripts/generate_difficulty_patch.py --output-patch difficulty.patch`
- Optionally apply in place (explicit opt-in):
  - `python3 scripts/generate_difficulty_patch.py --apply`

### Difficulty Rollback

- Script: `scripts/rollback_difficulty.sh`
- Purpose: reverse an applied `difficulty.patch` safely.
- Example:
  - `./scripts/rollback_difficulty.sh artifacts/tuning/<timestamp>/difficulty.patch`

## Replayable Simulation Harness

- Module: `simulation_harness.py`
- Purpose: deterministic simulation of validator scoring/EMA/weights against scripted miner behaviors.
- Includes:
  - configurable miner profiles (`base_valid_rate`, `base_elegance`, `jitter`)
  - seeded challenge batches from `challenge_engine.py`
  - difficulty-weighted instant scoring + EMA + final weight normalization
- Regression tests:
  - `test_simulation_harness.py` asserts deterministic replay and expected rank ordering (`honest > noisy > adversarial`)

### Simulation CLI

- Script: `scripts/run_simulation.py`
- Run preset baseline:
  - `python3 scripts/run_simulation.py --scenario baseline`
- Try alternate scenarios:
  - `python3 scripts/run_simulation.py --scenario tight_competition`
  - `python3 scripts/run_simulation.py --scenario adversarial_heavy`
- JSON output:
  - `python3 scripts/run_simulation.py --scenario baseline --json`
- Tunable knobs:
  - `--seed`, `--rounds`, `--batch-size`, `--ema-alpha`, `--start-block`

## One-Command Local Ops

- Start validator + metrics watch + tuning loop (+ optional stub miners):
  - `./scripts/start_local_ops.sh`
- Stop all local ops processes:
  - `./scripts/stop_local_ops.sh`
- Useful env overrides:
  - `NETWORK`, `NETUID`
  - `TAIL_ROWS` (default `5000`)
  - `TUNING_SLEEP_SECONDS` (default `3600`)
  - `START_MINERS=true|false`

### Runtime Retention

- Script: `scripts/prune_runtime_artifacts.sh`
- Purpose: remove old tuning artifacts/logs to keep disk usage bounded.
- Example:
  - `KEEP_TUNING_DAYS=14 KEEP_LOG_DAYS=7 ./scripts/prune_runtime_artifacts.sh`

## Post-Soak Go/No-Go

- Script: `scripts/post_soak_report.py`
- Purpose: evaluate last N minutes of logs + live metrics and return pass/fail.
- Example:
  - `python3 scripts/post_soak_report.py --log data/round_results.jsonl --metrics-url http://127.0.0.1:9109/metrics.json --minutes 60`
- JSON output:
  - `python3 scripts/post_soak_report.py --json`
- Exit code:
  - `0` when pass
  - `2` when fail (useful for CI/cron gates)
