#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDS_DIR="$ROOT_DIR/.pids"
LOGS_DIR="$ROOT_DIR/.logs"

NETWORK="${NETWORK:-finney}"
NETUID="${NETUID:-1}"
TAIL_ROWS="${TAIL_ROWS:-5000}"
TUNING_SLEEP_SECONDS="${TUNING_SLEEP_SECONDS:-3600}"
START_MINERS="${START_MINERS:-true}"

mkdir -p "$PIDS_DIR" "$LOGS_DIR"

echo "Running preflight check..."
(cd "$ROOT_DIR" && python3 scripts/preflight_check.py --root . --env-file .env || true)

if [[ "${START_MINERS}" == "true" ]]; then
  echo "Starting local stub miners..."
  (cd "$ROOT_DIR" && docker compose -f docker-compose.miner.yml up --build -d)
fi

echo "Starting validator..."
(
  cd "$ROOT_DIR"
  nohup uv run python validator.py --network "$NETWORK" --netuid "$NETUID" \
    > "$LOGS_DIR/validator.log" 2>&1 &
  echo $! > "$PIDS_DIR/validator.pid"
)

echo "Starting metrics watcher..."
(
  cd "$ROOT_DIR"
  nohup bash -lc "while true; do date; curl -s http://127.0.0.1:9109/metrics.json | python3 -m json.tool; echo; sleep 15; done" \
    > "$LOGS_DIR/metrics_watch.log" 2>&1 &
  echo $! > "$PIDS_DIR/metrics_watch.pid"
)

echo "Starting tuning cycle loop..."
(
  cd "$ROOT_DIR"
  nohup bash -lc "while true; do python3 scripts/run_tuning_cycle.py --log data/round_results.jsonl --tail $TAIL_ROWS; sleep $TUNING_SLEEP_SECONDS; done" \
    > "$LOGS_DIR/tuning_cycle.log" 2>&1 &
  echo $! > "$PIDS_DIR/tuning_cycle.pid"
)

echo "Local ops started."
echo "Logs:"
echo "  $LOGS_DIR/validator.log"
echo "  $LOGS_DIR/metrics_watch.log"
echo "  $LOGS_DIR/tuning_cycle.log"
echo "Use scripts/stop_local_ops.sh to stop."
