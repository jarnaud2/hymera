#!/usr/bin/env bash
# Particle-push timing benchmark.
#
# Runs the `profile` executable (particle push only, no MHD solve; fields come
# from the harness fixture fields.h5) on the conservation regression deck with
# more particles, and prints the wall time per MHD step from Parthenon's
# wsec_step lines.
#
#   bench_push.sh [options] PROFILE_EXE
#
#   --label NAME          name for the table (default: from the options)
#   --parallel-push 0|1   Simulation/parallel_push (default 1). On a host build,
#                         0 is the serial loop and 1 the threaded one. On a CUDA
#                         build both run the same GPU kernel.
#   --threads "1 4 16"    OMP_NUM_THREADS values to sweep (default "1")
#   --npb N               particles per mesh block; the deck has 8 blocks
#                         (default 2048, i.e. 16,384 particles)
#   --nlim N              MHD steps per run (default 3); the first is not timed
#   --launch "CMD"        launcher prefix (default: "mpirun -n 1 --bind-to none"
#                         if mpirun is found). Bind the rank to as many cores as
#                         threads, or all threads share one core.
#                         The string "{threads}" is replaced by the thread count.
#   --keep DIR            keep the run directories under DIR
#   --set "B/k=v ..."     extra input-deck overrides, e.g.
#                         "Simulation/integrator=dopri5 Simulation/EnableSmallAngleCollisions=0"
#
# Output: one line per thread count,
#   label  threads  particles  s/step  alive  mean_p
# mean_p (alive particles, final output) needs python3 with h5py; it is a sanity
# check that the runs did the same physics, not part of the timing.
set -euo pipefail

usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-2}"; }
LABEL=""; PP=1; THREADS="1"; NPB=2048; NLIM=3; KEEP=""; SET=""
# --bind-to none: Open MPI otherwise pins the one rank to one core, and every
# OpenMP thread then shares it (no speed-up from threads at all).
LAUNCH_DEFAULT=""; command -v mpirun >/dev/null && LAUNCH_DEFAULT="mpirun -n 1 --bind-to none"
LAUNCH="$LAUNCH_DEFAULT"
while [ $# -gt 0 ]; do
  case "$1" in
    --label) LABEL="$2"; shift 2 ;;
    --parallel-push) PP="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --npb) NPB="$2"; shift 2 ;;
    --nlim) NLIM="$2"; shift 2 ;;
    --launch) LAUNCH="$2"; shift 2 ;;
    --keep) KEEP="$2"; shift 2 ;;
    --set) SET="$2"; shift 2 ;;
    -h|--help) usage 0 ;;
    -*) echo "unknown option: $1" >&2; usage ;;
    *) [ -z "${EXE_ARG:-}" ] || usage; EXE_ARG="$1"; shift ;;
  esac
done
[ -n "${EXE_ARG:-}" ] || usage
EXE="$(cd "$(dirname "$EXE_ARG")" && pwd)/$(basename "$EXE_ARG")"
[ -x "$EXE" ] || { echo "not executable: $EXE" >&2; exit 1; }
[ -n "$LABEL" ] || LABEL="parallel_push=$PP"

# The repository: HYMERA_REPO, or the directory this script lives in (scripts/..).
REPO="${HYMERA_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DECK="$REPO/tests/regression/kinetic/decks/conservation.input"
FIELDS="$REPO/tests/regression/kinetic/fixtures/fields.h5"
PYTHON="${HARNESS_PYTHON:-python3}"
"$PYTHON" -c 'import h5py' 2>/dev/null || PYTHON=""

BASE="${KEEP:-$(mktemp -d)}"
[ -n "$KEEP" ] || trap 'rm -rf "$BASE"' EXIT
mkdir -p "$BASE"

printf '%-24s %7s %9s %10s %12s %7s %10s\n' label threads particles "s/step" "push ns/slot" alive mean_p
for T in $THREADS; do
  WORK="$BASE/${LABEL//[^A-Za-z0-9_=.-]/_}_t$T"
  rm -rf "$WORK"; mkdir -p "$WORK/r1/r2"
  ln -s "$REPO/inputs" "$WORK/inputs"
  cp "$FIELDS" "$WORK/r1/r2/"
  CMD="${LAUNCH//\{threads\}/$T}"
  # shellcheck disable=SC2086
  if ! (cd "$WORK/r1/r2" && OMP_NUM_THREADS=$T OMP_PROC_BIND=spread OMP_PLACES=threads \
        $CMD "$EXE" -i "$DECK" parthenon/time/nlim="$NLIM" \
        ParticleSeed/num_particles_per_block="$NPB" Simulation/parallel_push="$PP" \
        Simulation/verbosity=quiet Simulation/push_timing=1 $SET > log.txt 2>&1); then
    echo "run failed ($LABEL, $T threads); see $WORK/r1/r2/log.txt" >&2
    tail -5 "$WORK/r1/r2/log.txt" >&2
    exit 1
  fi
  # Mean wsec_step over cycles 2..NLIM (cycle 1 includes first-touch costs).
  STEP=$(grep -E '^cycle=[0-9]+ ' "$WORK/r1/r2/log.txt" |
         awk '{for(i=1;i<=NF;i++) if($i ~ /^cycle=/){split($i,c,"=")} ; for(i=1;i<=NF;i++) if($i ~ /^wsec_step=/){split($i,w,"=")} ; if (c[2] >= 2) {s+=w[2]; n++}} END {if (n) printf "%.3f", s/n; else print "n/a"}')
  ALIVE="-"; MEANP="-"; PUSHNS="-"
  if [ -n "$PYTHON" ]; then
    # Push-kernel time per particle slot per large-angle step, outputs after the
    # first (which holds the set-up step): /timeseries/push of the diagnostics file.
    PUSHNS=$("$PYTHON" - "$WORK/r1/r2" <<'EOF'
import glob, sys
import h5py
f = sorted(glob.glob(sys.argv[1] + "/*.diag.h5"))
g = h5py.File(f[0], "r").get("timeseries/push") if f else None
if g is None or len(g["push_s"]) < 2:
    print("-"); sys.exit()
s, n = g["push_s"][1:].sum(), g["slots"][1:].sum()
print(f"{s / n * 1e9:.2f}" if n else "-")
EOF
)
    read -r ALIVE MEANP < <("$PYTHON" - "$WORK/r1/r2" <<'EOF'
import glob, sys
import h5py
f = sorted(glob.glob(sys.argv[1] + "/*.out2.final.phdf"))
if not f:
    print("- -"); sys.exit()
s = h5py.File(f[0], "r")["particles/SwarmVars"]
alive = (s["particle.status"][()] & 2) != 0
p = s["particle.p"][()][alive]
print(int(alive.sum()), f"{p.mean():.6f}" if len(p) else "-")
EOF
)
  fi
  printf '%-24s %7s %9s %10s %12s %7s %10s\n' "$LABEL" "$T" "$((NPB * 8))" "$STEP" "$PUSHNS" "$ALIVE" "$MEANP"
done
