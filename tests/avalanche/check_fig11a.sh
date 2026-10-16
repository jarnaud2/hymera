#!/usr/bin/env bash
# ctest kin_avalanche_fig11a: runs a few rows of the fig11a table on the analytic
# field and checks the fitted growth rate against fig11a.jld2 (10%).
#
#   check_fig11a.sh --exe build/bin/avalanche [--rows "0 9"] [--threads N] [--keep DIR]
#   check_fig11a.sh ... --set "Simulation/EnableSmallAngleCollisions=0"   (negative test)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${HARNESS_PYTHON:-$HOME/.venv-hymera/bin/python}"
"$PYTHON" -c 'import numpy, h5py' 2>/dev/null ||
  { echo "need numpy and h5py in $PYTHON; run tests/regression/setup_env.sh" >&2; exit 1; }
EXE=""; ROWS="0 9"; THREADS="${OMP_NUM_THREADS:-$(nproc)}"; KEEP=""; SET=""
while [ $# -gt 0 ]; do
  case "$1" in
    --exe) EXE="$2"; shift 2 ;;
    --rows) ROWS="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --keep) KEEP="$2"; shift 2 ;;
    --set) SET="$2"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
[ -n "$EXE" ] || { echo "--exe is required" >&2; exit 2; }
OUT="${KEEP:-$(mktemp -d)}"
OMP_NUM_THREADS="$THREADS" PYTHON="$PYTHON" "$HERE/run_fig11a.sh" "$EXE" "$OUT" --rows "$ROWS" --set "$SET"
exec "$PYTHON" "$HERE/fig11a.py" "$OUT" --label "check"
