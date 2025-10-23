import io, os, base64
from typing import Optional
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
import joblib
import onnxruntime as ort

from src.models.siamese import Encoder

app = FastAPI(title="Face Hackathon API")

# Lazy-loaded globals
_encoder_pt = None
_encoder_session = None
_recognizer = None
_expr_session = None

IMG_SIZE = 160

tf_basic = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

def read_image(file: UploadFile) -> Image.Image:
    return Image.open(io.BytesIO(file.file.read())).convert("RGB")

def load_encoder_pt():
    global _encoder_pt
    if _encoder_pt is None:
        path = os.environ.get("SIAMESE_ENCODER_PT", "models/siamese_encoder.pt")
        model = Encoder()
        if os.path.exists(path):
            try:
                model.load_state_dict(torch.load(path, map_location="cpu"))
                print(f"Loaded Siamese encoder from {path}")
            except Exception as e:
                print(f"Warning: Failed to load Siamese encoder from {path}: {e}")
                print("Using untrained model")
        else:
            print(f"Warning: Siamese encoder not found at {path}, using untrained model")
        model.eval()
        _encoder_pt = model
    return _encoder_pt

def load_expr_onnx():
    global _expr_session
    if _expr_session is None:
        path = os.environ.get("EXPRESSIONS_ONNX", "models/expressions.onnx")
        if os.path.exists(path):
            try:
                _expr_session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                print(f"Loaded expression model from {path}")
            except Exception as e:
                print(f"Warning: Failed to load expression model from {path}: {e}")
                _expr_session = None
        else:
            print(f"Warning: Expression model not found at {path}")
    return _expr_session

def load_recognizer():
    global _recognizer
    if _recognizer is None:
        path = os.environ.get("RECOGNIZER_PATH", "models/recognizer.joblib")
        if os.path.exists(path):
            try:
                _recognizer = joblib.load(path)
                print(f"Loaded recognizer from {path}")
            except Exception as e:
                print(f"Warning: Failed to load recognizer from {path}: {e}")
                _recognizer = None
        else:
            print(f"Warning: Recognizer not found at {path}")
    return _recognizer

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/similarity")
def similarity(image_a: UploadFile = File(...), image_b: UploadFile = File(...)):
    try:
        model = load_encoder_pt()
        ia = tf_basic(read_image(image_a)).unsqueeze(0)
        ib = tf_basic(read_image(image_b)).unsqueeze(0)
        with torch.no_grad():
            za = model(ia)
            zb = model(ib)
            dist = torch.sqrt(((za - zb)**2).sum(dim=1)).item()
        # Convert distance to similarity score in [0,1] with a simple mapping
        score = float(np.exp(-dist))
        verdict = "similar" if score >= 0.6 else "dissimilar"  # threshold can be tuned from eval report
        return {"score": score, "verdict": verdict, "distance": float(dist)}
    except Exception as e:
        return {"error": f"Failed to process similarity: {str(e)}"}

@app.post("/recognize")
def recognize(image: UploadFile = File(...)):
    try:
        model = load_encoder_pt()
        recog = load_recognizer()
        if recog is None:
            return {"error": "Recognizer not found. Train and save to models/recognizer.joblib"}
        x = tf_basic(read_image(image)).unsqueeze(0)
        with torch.no_grad():
            z = model(x).numpy()
        pred = recog["model"].predict(z)[0]
        # Confidence proxy using kNN distance or SVM decision function
        conf = 0.5
        try:
            if recog["type"] == "knn":
                d, idx = recog["model"].kneighbors(z, n_neighbors=1, return_distance=True)
                conf = float(np.exp(-d[0][0]))
            else:
                df = recog["model"].decision_function(z)
                conf = float(1/(1+np.exp(-np.max(df))))
        except Exception:
            pass
        return {"label": str(pred), "confidence": conf}
    except Exception as e:
        return {"error": f"Failed to process recognition: {str(e)}"}

@app.post("/expression")
def expression(image: UploadFile = File(...)):
    try:
        sess = load_expr_onnx()
        if sess is None:
            return {"error": "Expression ONNX not found. Train and save to models/expressions.onnx"}
        img = tf_basic(read_image(image)).unsqueeze(0).numpy()
        logits = sess.run(None, {"image": img})[0]
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        idx = int(np.argmax(probs, axis=1)[0])
        classes_path = os.environ.get("EXPRESSIONS_CLASSES", "models/expressions.classes.txt")
        if os.path.exists(classes_path):
            with open(classes_path, "r") as f:
                classes = [l.strip() for l in f if l.strip()]
        else:
            classes = [f"class_{i}" for i in range(probs.shape[1])]
        return {"expression": classes[idx], "probs": {classes[i]: float(probs[0,i]) for i in range(len(classes))}}
    except Exception as e:
        return {"error": f"Failed to process expression: {str(e)}"}
