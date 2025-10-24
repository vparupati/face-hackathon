#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/att_faces
echo "[ORL] Downloading AT&T (ORL) faces via Kaggle..."
kaggle datasets download -d kasikrit/att-database-of-faces -p data/att_faces --force
echo "[ORL] Unzipping..."
unzip -o data/att_faces/att-database-of-faces.zip -d data/att_faces > /dev/null
if [ -d "data/att_faces/att_faces" ]; then
  rsync -a data/att_faces/att_faces/ data/att_faces/
  rm -rf data/att_faces/att_faces
fi
echo "[ORL] (Optional) Aligning (ORL already cropped)"
python -m src.preprocessing.align --in data/att_faces --out data/att_faces || true
echo "[ORL] Done -> data/att_faces"
