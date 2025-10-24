# Kaggle CLI setup (one-time)
1) Create API token on Kaggle: Account → "Create New API Token" (downloads `kaggle.json`).
2) Move it to your machine:
```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```
3) Install CLI (if needed):
```bash
pip install kaggle
```
4) If not using `~/.kaggle`, set:
```bash
export KAGGLE_CONFIG_DIR=/path/to/folder/with/kaggle.json
```
