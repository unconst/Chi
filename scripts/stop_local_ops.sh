#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDS_DIR="$ROOT_DIR/.pids"
STOP_MINERS="${STOP_MINERS:-true}"

stop_pid_file() {
  local pid_file="$1"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file")"
    if [[ -n "${pid}" ]] && kill -0 "$pid" 2>/dev/null; then
      echo "Stopping PID $pid from $(basename "$pid_file")"
      kill "$pid" || true
    fi
    rm -f "$pid_file"
  fi
}

stop_pid_file "$PIDS_DIR/validator.pid"
stop_pid_file "$PIDS_DIR/metrics_watch.pid"
stop_pid_file "$PIDS_DIR/tuning_cycle.pid"

if [[ "${STOP_MINERS}" == "true" ]]; then
  echo "Stopping local stub miners..."
  (cd "$ROOT_DIR" && docker compose -f docker-compose.miner.yml down)
fi

echo "Local ops stopped."
