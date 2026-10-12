#!/usr/bin/env bash
# Restart check for the diagnostics file (<problem_id>.diag.h5).
#
#   run_restart_diag.sh <conservation-binary>
#
# Runs the conservation deck for 4 steps with collisions off (so the run is
# deterministic across a restart: the random-number state is not restarted),
# writing a restart file every 2 steps. Then restarts from the mid-run restart
# file, which reopens the diagnostics file and must drop the rows after the
# restart time before appending. Passes when the restarted run's time series
# are bit-identical to the uninterrupted run's.
set -euo pipefail

EXE="${1:?usage: run_restart_diag.sh <conservation-binary>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
PYTHON="${HARNESS_PYTHON:-$HOME/.venv-hymera/bin/python}"
EXE="$(cd "$(dirname "$EXE")" && pwd)/$(basename "$EXE")"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
RUN="$WORK/r1/r2"
mkdir -p "$RUN"
ln -s "$REPO/inputs" "$WORK/inputs"
cp "$HERE/fixtures/fields.h5" "$RUN/"

LAUNCH=()
if command -v mpirun >/dev/null; then LAUNCH=(mpirun -n 1); fi
COMMON=(Simulation/EnableSmallAngleCollisions=0 Simulation/EnableLargeAngleCollisions=0)

cd "$RUN"
OMP_NUM_THREADS=1 "${LAUNCH[@]}" "$EXE" -i "$HERE/decks/conservation.input" \
  parthenon/time/nlim=4 parthenon/output8/file_type=rst parthenon/output8/dt=0.000330966 \
  "${COMMON[@]}" > full.log 2>&1
cp fig11a_70.diag.h5 full.diag.h5
OMP_NUM_THREADS=1 "${LAUNCH[@]}" "$EXE" -r fig11a_70.out8.00001.rhdf parthenon/time/nlim=4 \
  > restart.log 2>&1

"$PYTHON" - <<'EOF'
import sys
import h5py
import numpy as np

full = h5py.File("full.diag.h5", "r")["timeseries/conservation"]
rest = h5py.File("fig11a_70.diag.h5", "r")["timeseries/conservation"]
ok = True
for c in full:
    a, b = full[c][()], rest[c][()]
    same = a.shape == b.shape and np.array_equal(a, b)
    print(f"  {c}: {len(a)} rows uninterrupted, {len(b)} after restart, identical: {same}")
    ok &= same
t = rest["t"][()]
mono = bool(np.all(np.diff(t) > 0))
print(f"  t strictly increasing after restart: {mono}")
ok &= mono
print("PASS: diagnostics restart" if ok else "FAIL: diagnostics restart")
sys.exit(0 if ok else 1)
EOF
