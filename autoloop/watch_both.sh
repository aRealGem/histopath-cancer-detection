#!/usr/bin/env bash
# Poll BOTH compute pools; exit (waking the agent) on the first actionable event:
#   * the Kaggle e2cnn kernel finishes (complete/error) -> collect + gate
#   * a p4m job (job2/5/6) newly reports status=done on the bus -> gate that member
set -u
export KAGGLE_CONFIG_DIR="$HOME/.kaggle"
KG="$HOME/.venvs/kaggle/bin/kaggle"
KID="jackiemartindale/histopath-e2cnn"
OUT_DS="jackiemartindale/histopath-colab-out"
TRACK="job2 job5 job6"
MAX_MIN=75
POLL=150
start=$(date +%s)
prev_done=""

while :; do
  el=$(( ($(date +%s) - start) / 60 ))
  # -- Kaggle kernel --
  ks=$("$KG" kernels status "$KID" 2>/dev/null | grep -o 'KernelWorkerStatus\.[A-Z]*' | head -1)
  case "$ks" in
    *COMPLETE*) echo "[both +${el}m] KAGGLE_DONE ($ks) — collect + gate e2cnn"; exit 0 ;;
    *ERROR*|*CANCEL*) echo "[both +${el}m] KAGGLE_FAILED ($ks) — check kernel log"; exit 0 ;;
  esac
  # -- p4m bus --
  D=$(mktemp -d)
  "$KG" datasets download -d "$OUT_DS" -p "$D" --unzip --force >/dev/null 2>&1
  line=""; now_done=""
  for j in $TRACK; do
    st=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('status','?'))" "$D/job_${j}.json" 2>/dev/null || echo missing)
    line="$line ${j}=${st}"; [ "$st" = "done" ] && now_done="$now_done $j"
  done
  rm -rf "$D"
  echo "[both +${el}m] kaggle=${ks:-?} |$line"
  for j in $now_done; do
    case " $prev_done " in *" $j "*) ;; *) echo "P4M_DONE: $j (${line# }) — gate it"; exit 0 ;; esac
  done
  prev_done="$now_done"
  if [ "$el" -ge "$MAX_MIN" ]; then echo "TIMEOUT ${el}m: kaggle=${ks:-?} |$line"; exit 0; fi
  sleep $POLL
done
