#!/usr/bin/env bash
# Usage:
#   bash scripts/fetch_yale_no_kaggle.sh <DIRECT_URL_TO_ARCHIVE>
# If no URL arg is provided, you'll be prompted to paste one (from Yale/UCSD mirror).
set -euo pipefail

URL="${1:-}"
if [ -z "${URL}" ]; then
  echo "Paste the direct download URL for the Yale Face Database (e.g., a .zip/.tgz/.tar.gz):"
  read -r URL
fi

mkdir -p data/team_faces/yalefaces
TMP="./.tmp_yale"
mkdir -p "${TMP}"

echo "[Yale no-Kaggle] Downloading: ${URL}"
FNAME="${TMP}/yale_download"
curl -L "${URL}" -o "${FNAME}"

# Try common archive types
echo "[Yale no-Kaggle] Extracting..."
if file "${FNAME}" | grep -qi "Zip"; then
  unzip -o "${FNAME}" -d "${TMP}" >/dev/null
elif file "${FNAME}" | grep -qiE "gzip compressed data|tar archive"; then
  mkdir -p "${TMP}/extract"
  tar -xzf "${FNAME}" -C "${TMP}/extract" || tar -xof "${FNAME}" -C "${TMP}/extract" || true
else
  echo "Unknown archive format; attempting unzip then tar..."
  unzip -o "${FNAME}" -d "${TMP}" || tar -xof "${FNAME}" -C "${TMP}" || true
fi

# Find the folder with yale images (pgm/gif) and move to data/team_faces/yalefaces
FOUND_DIR=""
for d in "${TMP}" "${TMP}/extract" "${TMP}"/**; do
  if [ -d "${d}" ] && find "${d}" -maxdepth 2 -type f \( -iname "*.pgm" -o -iname "*.gif" -o -iname "*.png" -o -iname "*.jpg" \) | head -n 1 >/dev/null; then
    FOUND_DIR="${d}"
    break
  fi
done

if [ -z "${FOUND_DIR}" ]; then
  echo "Could not locate extracted Yale image folder. Please inspect ${TMP} manually."
  exit 1
fi

# Copy into our expected location
rsync -a "${FOUND_DIR}/" "data/team_faces/yalefaces/"

# Reorganize into subject01/... folders
python scripts/reorg_yale.py

# Align faces (optional but recommended for consistency)
python -m src.preprocessing.align --in data/team_faces --out data/team_faces || true

echo "[Yale no-Kaggle] Done -> data/team_faces/subject01, subject02, ..."
