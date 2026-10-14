#!/usr/bin/env bash
# ctest entry for check.py: uses the harness Python (HARNESS_PYTHON or ~/.venv-hymera).
set -euo pipefail
PYTHON="${HARNESS_PYTHON:-$HOME/.venv-hymera/bin/python}"
"$PYTHON" -c 'import numpy' 2>/dev/null ||
  { echo "need numpy in $PYTHON; run tests/regression/setup_env.sh" >&2; exit 1; }
exec "$PYTHON" "$(dirname "${BASH_SOURCE[0]}")/check.py" "$@"
