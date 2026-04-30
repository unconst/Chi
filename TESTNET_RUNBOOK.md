# Lemma 72-Hour Testnet Runbook

This runbook is an execution checklist for the first controlled testnet cycle.

## Objectives

- Validate end-to-end correctness of the Lemma subnet loop.
- Measure validator agreement and miner ranking stability.
- Establish initial operating thresholds for safe parameter tuning.

## Topology

- Validators: 2-3 independent validator instances.
- Miners: at least 3 active miners (mix of stronger and weaker behavior).
- Duration: 72 hours.
- Observation cadence: every 12 hours (minimum), daily formal review.

## Preflight (T-24h to T-1h)

- Confirm all nodes run the same code revision.
- Ensure all validators expose metrics:
  - `/metrics`
  - `/metrics.json`
- Confirm miner endpoints are discoverable:
  - on-chain commitment and/or `MINER_ENDPOINTS` override.
- Verify alert checker baseline:
  - `python3 scripts/check_alerts.py --url http://127.0.0.1:9109/metrics.json`
- Run deterministic simulation sanity:
  - `python3 scripts/run_simulation.py --scenario baseline`
  - `python3 scripts/run_simulation.py --scenario adversarial_heavy --rounds 120`

## Day 0 (Launch)

1. Start validators and miners.
2. Confirm each validator completes at least one weight set cycle.
3. Check metrics health on each validator:
   - no persistent timeout or http error spikes
   - set_weights success counter increments
4. Record initial snapshot:
   - current config values
   - validator UID list
   - miner UID list/endpoints

## Day 1 (First 24h Review)

Run on primary analysis host:

- Per-validator quality summary:
  - `python3 scripts/analyze_round_logs.py --log data/validator_a.jsonl --tail 5000`
- Validator disagreement:
  - `python3 scripts/compare_validator_logs.py --left data/validator_a.jsonl --right data/validator_b.jsonl --tail 5000`
- Staged go/hold:
  - `python3 scripts/tuning_report.py --log data/validator_a.jsonl --validator-b-log data/validator_b.jsonl --metrics-url http://127.0.0.1:9109/metrics.json --tail 5000`

Decision rule:

- If `HOLD`: do not tune parameters, investigate root cause.
- If `GO`: generate but do not yet apply difficulty patch:
  - `python3 scripts/generate_difficulty_patch.py --log data/validator_a.jsonl --tail 5000 --output-patch day1_difficulty.patch`

## Day 2 (24-48h Review)

- Re-run all Day 1 analyses.
- Compare trend direction:
  - invalid rate stable/down
  - epistula fail rate stable/down
  - validator agreement stable/up
  - concentration not worsening
- If two consecutive `GO` windows and no severe alerts:
  - apply one scoped change only (recommended: difficulty patch only)
  - redeploy validators
  - record exact change in ops notes

## Day 3 (48-72h Final Review)

- Run full tuning cycle artifact generation:
  - `python3 scripts/run_tuning_cycle.py --log data/validator_a.jsonl --validator-b-log data/validator_b.jsonl --metrics-url http://127.0.0.1:9109/metrics.json --tail 10000`
- Archive artifact bundle from:
  - `artifacts/tuning/<timestamp>/`
- Produce final decision:
  - continue testnet with one additional incremental change, or
  - freeze config for broader rollout candidate.

## Initial Thresholds (Suggested)

Treat these as starting defaults, not immutable constants:

- `valid_flag_agreement >= 0.95` between validators.
- `top_1_share <= 0.45` sustained.
- `epistula_fail_rate <= 0.05`.
- `timeout_rate <= 0.10`.
- `http_error_rate <= 0.10`.
- `weight_fail_rate <= 0.20`.

If any critical threshold is breached for 2 consecutive checks, set status to `HOLD`.

## Change Management Rules

- Never change more than one parameter family per review window.
- Prioritize order:
  1. difficulty weights
  2. challenge batch size
  3. EMA alpha
- Require one full observation window after each change before further changes.
- Keep all patches reviewable; avoid unreviewed auto-apply in production.

## Incident Triggers

Immediate incident mode if any occurs:

- set_weights failures across all validators for >2 cycles.
- validator disagreement drops below 0.85.
- epistula failures spike above 15%.
- invalid rate doubles from previous daily baseline.

Incident actions:

- freeze parameter changes.
- capture logs and metrics snapshots.
- revert last tuning patch if regression-correlated.

## Minimal Operator Command Set

- Health check:
  - `python3 scripts/check_alerts.py --url http://127.0.0.1:9109/metrics.json`
- Analyze logs:
  - `python3 scripts/analyze_round_logs.py --log data/round_results.jsonl --tail 5000`
- Compare validators:
  - `python3 scripts/compare_validator_logs.py --left data/validator_a.jsonl --right data/validator_b.jsonl --tail 5000`
- Calibrate difficulty:
  - `python3 scripts/calibrate_difficulty.py --log data/round_results.jsonl --tail 5000`
- Generate patch:
  - `python3 scripts/generate_difficulty_patch.py --log data/round_results.jsonl --tail 5000 --output-patch difficulty.patch`
- Full tuning cycle:
  - `python3 scripts/run_tuning_cycle.py --log data/validator_a.jsonl --validator-b-log data/validator_b.jsonl --metrics-url http://127.0.0.1:9109/metrics.json --tail 5000`
