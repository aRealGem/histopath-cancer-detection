#!/usr/bin/env bash
# Download + extract the competition data for LOCAL/Colab use.
# On Kaggle notebooks the data is already mounted at
#   /kaggle/input/histopathologic-cancer-detection  — skip this script there.
#
# Prereq: Kaggle API token at ~/.kaggle/kaggle.json (chmod 600), and you must
# have clicked "I Understand and Accept" on the competition Rules tab first:
#   https://www.kaggle.com/competitions/histopathologic-cancer-detection/rules
set -euo pipefail

DEST="${1:-./data}"
COMP="histopathologic-cancer-detection"

mkdir -p "$DEST"
echo "Downloading $COMP into $DEST ..."
kaggle competitions download -c "$COMP" -p "$DEST"
echo "Extracting ..."
unzip -q -o "$DEST/$COMP.zip" -d "$DEST"
rm -f "$DEST/$COMP.zip"

# WSI mapping enables leakage-free grouped splits (see README "Leakage" note).
# The full map ships gzipped in the repo at data/wsi/patch_id_wsi_full.csv.gz, so
# normally there's nothing to fetch. This guarded refresh re-pulls it from the
# Kaggle forum attachment only if the bundled copy is missing.
WSI_GZ="data/wsi/patch_id_wsi_full.csv.gz"
if [[ -f "$WSI_GZ" ]]; then
  echo "WSI map already bundled at $WSI_GZ — skipping fetch."
else
  echo "WSI map missing; fetching from Kaggle forum attachment ..."
  mkdir -p "$(dirname "$WSI_GZ")"
  wget -q -O "$DEST/patch_id_wsi_full.zip" \
    https://storage.googleapis.com/kaggle-forum-message-attachments/496876/11666/patch_id_wsi_full.zip
  unzip -q -o "$DEST/patch_id_wsi_full.zip" -d "$DEST"
  rm -f "$DEST/patch_id_wsi_full.zip"
  # Normalize to the gzipped path the config points at.
  gzip -c "$DEST/patch_id_wsi_full.csv" > "$WSI_GZ"
  echo "Wrote $WSI_GZ"
fi

echo "Done. Set data.root: $DEST in configs/baseline.yaml"
ls -la "$DEST"
