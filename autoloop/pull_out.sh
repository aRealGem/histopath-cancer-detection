#!/usr/bin/env bash
# Pull the Colab handoff dataset jackiemartindale/histopath-colab-out into inbox/.
# (process.py also does this each tick; this is the standalone/manual version.)
set -euo pipefail

BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KAGGLE="$HOME/.venvs/kaggle/bin/kaggle"
export KAGGLE_CONFIG_DIR="$HOME/.kaggle"
DS="jackiemartindale/histopath-colab-out"
INBOX="$BASE/inbox"

mkdir -p "$INBOX"
"$KAGGLE" datasets download -d "$DS" -p "$INBOX" --unzip --force
echo "pulled $DS -> $INBOX"
ls -1 "$INBOX"
