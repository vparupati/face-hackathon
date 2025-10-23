# Face Hackathon — Siamese Similarity, Recognition, Expressions

End-to-end scaffold to deliver all 3 challenges:
1) **Similarity score** (Siamese network with contrastive loss)
2) **Recognition of ≥3 people** (embeddings + kNN/SVM)
3) **Expression classification** (transfer learning)

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 1) Face Detection & Alignment
Put raw images under `data/raw/` in per-person folders (and per-expression subfolders for the expressions task).
Then run:

```bash
python -m src.preprocessing.align --in data/raw --out data/aligned --save-landmarks
```

This creates aligned 160×160 crops under `data/aligned/` mirroring your folder hierarchy.

### 2) Train Siamese (Challenge 1)
Folder structure expected for training (each subfolder = person id):
```
data/att_faces/
  person_01/ img1.jpg ...
  person_02/ ...
```

```bash
python -m src.models.siamese --data data/att_faces --epochs 10 --out models/siamese_encoder.pt --onnx models/siamese_encoder.onnx
```

Evaluate threshold and export ROC plot + threshold recommendation:
```bash
python -m src.models.siamese --data data/att_faces --eval --encoder models/siamese_encoder.pt --report reports/siamese_eval.json
```

### 3) Recognition (Challenge 2)
Embed your **team faces** using the Siamese encoder:
```bash
python -m src.inference.embed --in data/team_faces --encoder models/siamese_encoder.pt --out data/team_embeddings.npz
```

Train a recognition classifier (kNN & LinearSVC; picks best by validation):
```bash
python -m src.models.classifier --embeds data/team_embeddings.npz --out models/recognizer.joblib
```

### 4) Expression Model (Challenge 3)
Expect structure: `data/expressions/<person>/<expression>/*.jpg` (e.g., happy/angry/sad/neutral).
```bash
python -m src.models.expression_cnn --data data/expressions --epochs 10 --out models/expressions.onnx
```

### 5) Serve API
```bash
uvicorn src.api.app:app --reload
```

Endpoints:
- `GET /health`
- `POST /similarity` (multipart: `image_a`, `image_b`)
- `POST /recognize` (multipart: `image`)
- `POST /expression` (multipart: `image`)

### 6) Demo UI (Streamlit)
```bash
streamlit run demo_app/app.py
```

The UI calls the local FastAPI server by default (`http://127.0.0.1:8000`).

## Data Notes
- For **Challenge 1** (similarity), we recommend AT&T (ORL) faces.
- For **Challenge 2**, collect ≥20 images/person for at least 3 people.
- For **Challenge 3**, collect ≥30–50 images/class overall (happy/angry/sad/neutral) with multiple people.

## Security & Privacy
Store only aligned faces & embeddings for the hackathon; encrypt retained data if re-used. Obtain consent.

## Repo Layout
See `face-hackathon/` for full structure.
