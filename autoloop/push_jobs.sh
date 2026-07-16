#!/usr/bin/env bash
# Publish queue.json to the Kaggle dataset jackiemartindale/histopath-jobs (the job-queue
# channel the Colab poll-loop reads). Creates the dataset on first run, versions it after.
# Usage: push_jobs.sh ["commit message"]
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KAGGLE="$HOME/.venvs/kaggle/bin/kaggle"
export KAGGLE_CONFIG_DIR="$HOME/.kaggle"
DS="jackiemartindale/histopath-jobs"
STAGE="$BASE/jobs_ds"
MSG="${1:-update queue $(date -u +%Y-%m-%dT%H:%M:%SZ)}"

mkdir -p "$STAGE"
cp "$BASE/queue.json" "$STAGE/queue.json"
cat > "$STAGE/dataset-metadata.json" <<JSON
{
  "title": "histopath-jobs",
  "id": "$DS",
  "licenses": [{"name": "CC0-1.0"}]
}
JSON

# Validate JSON before publishing.
python3 -c "import json,sys; json.load(open('$STAGE/queue.json')); print('queue.json valid')"

if "$KAGGLE" datasets files -d "$DS" >/dev/null 2>&1; then
  echo "versioning existing $DS ..."
  "$KAGGLE" datasets version -p "$STAGE" -m "$MSG" --dir-mode zip
else
  echo "creating $DS ..."
  "$KAGGLE" datasets create -p "$STAGE" --dir-mode zip
fi
echo "done: $DS <- $MSG"
