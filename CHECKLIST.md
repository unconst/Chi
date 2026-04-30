# Lemma 72-Hour Testnet Checklist

Use this as the live execution sheet during the first controlled run.

## Preflight (T-24h to T-1h)

- [ ] All validators on same code revision.
- [ ] All validators configured with same core hyperparameters.
- [ ] `METRICS_ENABLED=true` and `/metrics.json` reachable for each validator.
- [ ] Miner endpoints discoverable (commitments and/or `MINER_ENDPOINTS`).
- [ ] Alert check baseline passes on each validator:
  - [ ] `python3 scripts/check_alerts.py --url http://127.0.0.1:9109/metrics.json`
- [ ] Simulation sanity checks pass:
  - [ ] `python3 scripts/run_simulation.py --scenario baseline`
  - [ ] `python3 scripts/run_simulation.py --scenario adversarial_heavy --rounds 120`
- [ ] Operator has write access to `artifacts/tuning/`.

## Day 0 (Launch)

- [ ] Validators started successfully.
- [ ] Miners started successfully.
- [ ] Each validator completed at least one weight set cycle.
- [ ] `set_weights` success counter increments on all validators.
- [ ] No immediate sustained spike in:
  - [ ] timeout rate
  - [ ] http error rate
  - [ ] epistula fail rate
- [ ] Initial snapshot captured (UIDs, endpoints, config values).

## Day 1 Review (0-24h)

- [ ] Per-validator summary run:
  - [ ] `python3 scripts/analyze_round_logs.py --log data/validator_a.jsonl --tail 5000`
- [ ] Validator disagreement run:
  - [ ] `python3 scripts/compare_validator_logs.py --left data/validator_a.jsonl --right data/validator_b.jsonl --tail 5000`
- [ ] Staged go/hold report run:
  - [ ] `python3 scripts/tuning_report.py --log data/validator_a.jsonl --validator-b-log data/validator_b.jsonl --metrics-url http://127.0.0.1:9109/metrics.json --tail 5000`
- [ ] Decision recorded:
  - [ ] `GO`
  - [ ] `HOLD`
- [ ] If `GO`, patch generated but not auto-applied:
  - [ ] `python3 scripts/generate_difficulty_patch.py --log data/validator_a.jsonl --tail 5000 --output-patch day1_difficulty.patch`

## Day 2 Review (24-48h)

- [ ] Re-ran Day 1 command set.
- [ ] Trend check completed:
  - [ ] invalid rate stable/down
  - [ ] epistula fail rate stable/down
  - [ ] validator agreement stable/up
  - [ ] concentration not worsening
- [ ] If two consecutive `GO` windows:
  - [ ] exactly one parameter family selected for change
  - [ ] change reviewed
  - [ ] validators redeployed
  - [ ] change logged with timestamp

## Day 3 Final Review (48-72h)

- [ ] Full tuning cycle executed:
  - [ ] `python3 scripts/run_tuning_cycle.py --log data/validator_a.jsonl --validator-b-log data/validator_b.jsonl --metrics-url http://127.0.0.1:9109/metrics.json --tail 10000`
- [ ] Artifacts archived from `artifacts/tuning/<timestamp>/`.
- [ ] Final decision recorded:
  - [ ] continue testnet with one incremental change
  - [ ] freeze config for rollout candidate

## Threshold Guardrail Checks

- [ ] `valid_flag_agreement >= 0.95`
- [ ] `top_1_share <= 0.45`
- [ ] `epistula_fail_rate <= 0.05`
- [ ] `timeout_rate <= 0.10`
- [ ] `http_error_rate <= 0.10`
- [ ] `weight_fail_rate <= 0.20`

If any critical threshold is breached for two consecutive checks:

- [ ] Status set to `HOLD`.
- [ ] Parameter changes frozen.
- [ ] Incident review started.

## Incident Trigger Checklist

Immediate incident mode if any true:

- [ ] set_weights failures across all validators for >2 cycles
- [ ] validator agreement < 0.85
- [ ] epistula fail rate > 0.15
- [ ] invalid rate > 2x previous daily baseline

Incident response:

- [ ] freeze tuning changes
- [ ] capture logs + metrics snapshots
- [ ] assess rollback of last tuning patch
- [ ] document root cause + mitigation
