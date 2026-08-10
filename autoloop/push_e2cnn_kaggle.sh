#!/usr/bin/env bash
# Push (and thereby RUN) the e2cnn member on a Kaggle GPU kernel = the second compute
# pool, in parallel with the Colab A100 p4m chain. THIS SPENDS Kaggle GPU quota.
# Confirm the weekly GPU quota has capacity (Kaggle Notebooks UI meter) before running.
set -euo pipefail
export KAGGLE_CONFIG_DIR="$HOME/.kaggle"
KG="$HOME/.venvs/kaggle/bin/kaggle"
HERE="$(cd "$(dirname "$0")" && pwd)"
KDIR="$HERE/kaggle_e2cnn"

# stage the kernel code next to its metadata (single source of truth = notebooks/)
cp "$HERE/../notebooks/e2cnn_kaggle_kernel.py" "$KDIR/e2cnn_kaggle_kernel.py"
echo "pushing kernel (this queues a GPU run):"
"$KG" kernels push -p "$KDIR"
echo
echo "watch:   $KG kernels status jackiemartindale/histopath-e2cnn"
echo "collect: $HERE/collect_e2cnn_kaggle.sh   (once status=complete)"
