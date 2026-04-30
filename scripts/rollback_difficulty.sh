#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_FILE="${TARGET_FILE:-$ROOT_DIR/challenge_engine.py}"

PATCH_FILE="${1:-}"
if [[ -z "$PATCH_FILE" ]]; then
  echo "Usage: $0 <patch-file>"
  echo "Example: $0 artifacts/tuning/20260430T000000Z/difficulty.patch"
  exit 1
fi

if [[ ! -f "$PATCH_FILE" ]]; then
  echo "Patch file not found: $PATCH_FILE"
  exit 1
fi

if [[ ! -f "$TARGET_FILE" ]]; then
  echo "Target file not found: $TARGET_FILE"
  exit 1
fi

echo "Reversing difficulty patch: $PATCH_FILE"
cd "$ROOT_DIR"
git apply -R --check "$PATCH_FILE"
git apply -R "$PATCH_FILE"
echo "Rollback applied to $TARGET_FILE"
