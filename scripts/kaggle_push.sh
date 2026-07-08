#!/usr/bin/env bash
# Push the runner notebook to Kaggle as a GPU kernel and run it there.
# SandboxPi orchestrates only — the actual training happens on Kaggle's T4.
#
# Prereqs (one-time):
#   1. pip install "kaggle>=1.6,<2"
#   2. Kaggle API token at ~/.kaggle/kaggle.json (chmod 600) — git-ignored, never committed.
#   3. Accept the competition rules:
#        https://www.kaggle.com/competitions/histopathologic-cancer-detection/rules
#   4. Edit kernel-metadata.json: set "id" to "<your-kaggle-username>/histopath-mobilenetv3-baseline".
#
# The kernel needs the src/ code. Two supported routes:
#   A) Make this repo public and have the runner notebook `git clone` it (enable_internet=true).
#   B) Attach the repo as a Kaggle dataset/utility and add it to "dataset_sources".
#
# Kaggle picks the T4 accelerator from the notebook's Settings/metadata (enable_gpu=true);
# the brief's `--accelerator NvidiaTeslaT4` is the equivalent selection.
set -euo pipefail
cd "$(dirname "$0")/.."

command -v kaggle >/dev/null 2>&1 || { echo "kaggle CLI not found: pip install 'kaggle>=1.6,<2'"; exit 1; }
[[ -f "$HOME/.kaggle/kaggle.json" ]] || { echo "Missing ~/.kaggle/kaggle.json (chmod 600)"; exit 1; }
grep -q "YOUR_KAGGLE_USERNAME" kernel-metadata.json && \
  { echo "Edit kernel-metadata.json: replace YOUR_KAGGLE_USERNAME with your Kaggle username."; exit 1; }

echo "Pushing kernel to Kaggle ..."
kaggle kernels push -p .
echo "Pushed. Track status with:  kaggle kernels status <id-from-metadata>"
echo "Fetch outputs (submission.csv, roc_val.png) with:  kaggle kernels output <id> -p ./artifacts"
