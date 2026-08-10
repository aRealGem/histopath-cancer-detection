#!/usr/bin/env bash
# Collect the e2cnn Kaggle-kernel output and merge it into the colab-out bus, where
# autoloop/process.py decorr+LB-gates it like any other member. The kernel is a pure
# producer (no bus creds); the Pi (which has creds) does the bus write here.
set -euo pipefail
export KAGGLE_CONFIG_DIR="$HOME/.kaggle"
KG="$HOME/.venvs/kaggle/bin/kaggle"
KID="jackiemartindale/histopath-e2cnn"
OUT_DS="jackiemartindale/histopath-colab-out"

st=$("$KG" kernels status "$KID" 2>&1 | tr -d '\r')
echo "kernel status: $st"
echo "$st" | grep -qi complete || { echo "not complete yet -> nothing to collect."; exit 1; }

T=$(mktemp -d)
"$KG" kernels output "$KID" -p "$T" >/dev/null 2>&1
for f in oof_e2cnn.csv sub_e2cnn.csv job_job3.json; do
  [ -f "$T/$f" ] || { echo "MISSING $f in kernel output"; exit 1; }
done

# cumulative merge into the bus (mirror push_out's anti-wipe guard)
M=$(mktemp -d)
"$KG" datasets download -d "$OUT_DS" -p "$M" --unzip --force >/dev/null 2>&1
n=$(find "$M" -type f ! -name dataset-metadata.json | wc -l)
[ "$n" -ge 3 ] || { echo "ABORT: bus download empty ($n files) -> refusing to push a wipe"; exit 1; }
cp "$T/oof_e2cnn.csv" "$T/sub_e2cnn.csv" "$T/job_job3.json" "$M/"
python3 - "$M" "$OUT_DS" <<'PY'
import json, sys, os
d, ds = sys.argv[1], sys.argv[2]
json.dump({"title": "histopath-colab-out", "id": ds, "licenses": [{"name": "CC0-1.0"}]},
          open(os.path.join(d, "dataset-metadata.json"), "w"))
PY
"$KG" datasets version -p "$M" -m "collect e2cnn (job3) from Kaggle GPU kernel -> bus" --dir-mode zip 2>&1 | tail -3
echo "merged e2cnn into the bus. Next: python3 autoloop/process.py  (decorr+LB gate)."
