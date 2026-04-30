#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

KEEP_TUNING_DAYS="${KEEP_TUNING_DAYS:-14}"
KEEP_LOG_DAYS="${KEEP_LOG_DAYS:-7}"

echo "Pruning runtime artifacts under $ROOT_DIR"
echo "Retaining tuning artifacts for $KEEP_TUNING_DAYS days"
echo "Retaining logs for $KEEP_LOG_DAYS days"

if [[ -d "$ROOT_DIR/artifacts/tuning" ]]; then
  find "$ROOT_DIR/artifacts/tuning" -mindepth 1 -maxdepth 1 -type d -mtime +"$KEEP_TUNING_DAYS" -exec rm -rf {} +
fi

for log_dir in "$ROOT_DIR/.logs" "$ROOT_DIR/.ops-cron"; do
  if [[ -d "$log_dir" ]]; then
    find "$log_dir" -type f -mtime +"$KEEP_LOG_DAYS" -delete
  fi
done

echo "Prune complete."
