#!/usr/bin/env bash
set -euo pipefail
mkdir -p data/expressions
echo "[FER-2013] Downloading FER-2013 via Kaggle..."
kaggle datasets download -d msambare/fer2013 -p data/expressions --force
echo "[FER-2013] Unzipping..."
unzip -o data/expressions/fer2013.zip -d data/expressions > /dev/null

if [ -d "data/expressions/train" ]; then
  echo "[FER-2013] Using data/expressions/train/* as classes"
  ls data/expressions/train | sort > models/expressions.classes.txt || true
else
  echo "[FER-2013] 'train' folder not found; leaving structure as-is"
fi

echo "[FER-2013] Aligning (optional)..."
python -m src.preprocessing.align --in data/expressions --out data/expressions || true
echo "[FER-2013] Done -> data/expressions"
