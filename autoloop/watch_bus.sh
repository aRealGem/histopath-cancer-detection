#!/usr/bin/env bash
# Poll the colab-out bus; exit as soon as a tracked p4m job (job2/job5/job6)
# newly reports status=done (a member ready to gate), or all three are resolved,
# or a max wall-clock is hit. Re-invokes the agent on exit.
set -u
export KAGGLE_CONFIG_DIR=~/.kaggle
KG=~/.venvs/kaggle/bin/kaggle
OUT_DS=jackiemartindale/histopath-colab-out
TRACK="job2 job5 job6"
MAX_MIN=80
POLL=180
start=$(date +%s)
prev_done=""

status_of() { python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('status','?'))" "$1" 2>/dev/null || echo missing; }

while :; do
  D=$(mktemp -d)
  $KG datasets download -d "$OUT_DS" -p "$D" --unzip --force >/dev/null 2>&1
  now_done=""; line=""
  for j in $TRACK; do
    st=$(status_of "$D/job_${j}.json")
    line="$line ${j}=${st}"
    [ "$st" = "done" ] && now_done="$now_done $j"
  done
  rm -rf "$D"
  el=$(( ($(date +%s) - start) / 60 ))
  echo "[watch +${el}m]$line"
  # New done since last poll?
  for j in $now_done; do
    case " $prev_done " in *" $j "*) ;; *)
      echo "NEW_DONE: $j (statuses:$line) — gate it"; exit 0 ;;
    esac
  done
  prev_done="$now_done"
  # All three resolved (done or a terminal error/needs_human twice)?
  if echo "$line" | grep -q 'job2=done' && echo "$line" | grep -q 'job5=done' && echo "$line" | grep -q 'job6=done'; then
    echo "ALL_DONE:$line"; exit 0
  fi
  if [ $el -ge $MAX_MIN ]; then echo "TIMEOUT after ${el}m:$line"; exit 0; fi
  sleep $POLL
done
