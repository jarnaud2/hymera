#!/bin/bash
# fig11a avalanche benchmark sweep on the analytic field.
#
#   run_fig11a.sh EXE OUTDIR [--rows "0 1 2"] [--launch "mpirun -n 1"] [--set "B/k=v ..."]
#                 [--deck fig11a_mhd.input] [--copy FILE]   (FILE copied into the run directory)
#
# One run per row i of the reference table (fig11a.jld2): Avalanche/E_0 = ES[i],
# ParticleSeed/r_0 = R0s[i], tlim = tlims[i], dt_force = output2/dt = tlim/100,
# into OUTDIR/fig11a_<i>/ (log.txt, time.txt, <problem_id>.diag.h5). Rows run one
# after another; to run rows in parallel start several copies with disjoint --rows.
# Then: python3 fig11a.py OUTDIR
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${HYMERA_REPO:-$(cd "$HERE/../.." && pwd)}"
EXE="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; shift
OUT="$1"; shift
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"
ROWS=""; LAUNCH=""; SET=""; DECK="$HERE/fig11a.input"; COPY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --rows) ROWS="$2"; shift 2 ;;
    --launch) LAUNCH="$2"; shift 2 ;;
    --set) SET="$2"; shift 2 ;;
    --deck) DECK="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"; shift 2 ;;
    --copy) COPY="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
PY="${PYTHON:-python3}"
TABLE=$("$PY" - "$HERE/fig11a.jld2" <<'EOF'
import sys, h5py
f = h5py.File(sys.argv[1], "r")
ES, R0s, T = (f[k][()].ravel() for k in ("ES", "R0s", "tlims"))
for i in range(len(ES)):
    print(i, repr(float(ES[i])), repr(float(R0s[i])), repr(float(T[i])), repr(float(T[i]) / 100))
EOF
)
[ -n "$ROWS" ] || ROWS=$(echo "$TABLE" | awk '{print $1}')
for i in $ROWS; do
  read -r _ E0 R0 TLIM DT <<< "$(echo "$TABLE" | awk -v i="$i" '$1 == i')"
  W="$OUT/fig11a_$i"; rm -rf "$W"; mkdir -p "$W"
  # The deck reads ../../inputs relative to the run directory.
  mkdir -p "$W/r1/r2"; ln -s "$REPO/inputs" "$W/inputs"
  [ -z "$COPY" ] || cp "$COPY" "$W/r1/r2/"
  T0=$(date +%s.%N)
  ( cd "$W/r1/r2" && $LAUNCH "$EXE" -i "$DECK" \
      parthenon/job/problem_id="$(basename "$DECK" .input)_$i" Avalanche/E_0="$E0" ParticleSeed/r_0="$R0" \
      parthenon/time/tlim="$TLIM" parthenon/time/dt_force="$DT" parthenon/output2/dt="$DT" \
      $SET > "$W/log.txt" 2>&1 < /dev/null ) || echo "row $i: run failed (see $W/log.txt)"
  awk -v a="$T0" -v b="$(date +%s.%N)" 'BEGIN{printf "%.1f\n", b - a}' > "$W/time.txt"
  # Keep the time series and the first output (parameters); drop the other phdf.
  mv "$W"/r1/r2/*.diag.h5 "$W/" 2>/dev/null || true
  ls "$W"/r1/r2/*.out2.000000.phdf > /dev/null 2>&1 && mv "$W"/r1/r2/*.out2.000000.phdf "$W/"
  rm -rf "$W/r1" "$W/inputs"
  echo "row $i: E_0 $E0 r_0 $R0 tlim $TLIM: $(cat "$W/time.txt" 2>/dev/null || echo ?) s"
done
