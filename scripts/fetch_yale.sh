#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/team_faces
echo "[Yale] Downloading Yale Faces via Kaggle..."
kaggle datasets download -d kasikrit/yalefaces -p data/team_faces --force
echo "[Yale] Unzipping..."
unzip -o data/team_faces/yalefaces.zip -d data/team_faces > /dev/null
python scripts/reorg_yale.py
echo "[Yale] Aligning faces..."
python -m src.preprocessing.align --in data/team_faces --out data/team_faces || true
echo "[Yale] Done -> data/team_faces/subject01, subject02, ..."
