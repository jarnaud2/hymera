#!/usr/bin/env bash
# ctest kin_la_sampler: LargeAngleCollisionFast against LargeAngleCollision in law.
set -euo pipefail
EXE="$1"
PYTHON="${HARNESS_PYTHON:-$HOME/.venv-hymera/bin/python}"
OUT="$(mktemp -d)"
"$EXE" "$OUT/la.bin" 200000 777 > "$OUT/time.txt"
cat "$OUT/time.txt"
exec "$PYTHON" "$(dirname "${BASH_SOURCE[0]}")/la_sampler_check.py" "$OUT/la.bin"
