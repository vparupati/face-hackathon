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
EXPR_IMG_SIZE = 100  # RAF-DB model uses 100x100 RGB

tf_basic = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Transform for RAF-DB expression model (100x100 RGB)
tf_expr = transforms.Compose([
    transforms.Resize((EXPR_IMG_SIZE, EXPR_IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

def read_image(file: UploadFile) -> Image.Image:
    return Image.open(io.BytesIO(file.file.read())).convert("RGB")

def load_encoder_pt():
    global _encoder_pt
    if _encoder_pt is None:
        path = os.environ.get("SIAMESE_ENCODER_PT", "models/siamese_encoder_v2.pt")
        model = Encoder()
        if os.path.exists(path):
            try:
                # Use MPS on Mac, CUDA on NVIDIA, or CPU
                if torch.cuda.is_available():
                    map_location = "cuda"
                elif torch.backends.mps.is_available():
                    map_location = "mps"
                else:
                    map_location = "cpu"
                model.load_state_dict(torch.load(path, map_location=map_location))
                print(f"Loaded Siamese encoder from {path}")
            except Exception as e:
                print(f"Warning: Failed to load Siamese encoder from {path}: {e}")
                print("Using untrained model")
        else:
            print(f"Warning: Siamese encoder not found at {path}, using untrained model")
        # Move model to appropriate device
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        model.to(device)
        model.eval()
        _encoder_pt = model
    return _encoder_pt

def load_expr_onnx():
    global _expr_session
    if _expr_session is None:
        # Use the active RAF-DB model (versioned)
        path = os.environ.get("EXPRESSIONS_ONNX", "models/rafdb_expressions_active.onnx")
        if os.path.exists(path):
            try:
                _expr_session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                print(f"Loaded RAF-DB expression model from {path} (Latest v20251025)")
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
        
        # Move tensors to the same device as the model
        device = next(model.parameters()).device
        ia = tf_basic(read_image(image_a)).unsqueeze(0).to(device)
        ib = tf_basic(read_image(image_b)).unsqueeze(0).to(device)
        
        with torch.no_grad():
            za = model(ia)
            zb = model(ib)
            
            # Debug: Check embedding norms
            norm_a = torch.norm(za).item()
            norm_b = torch.norm(zb).item()
            print(f"Embedding norms - A: {norm_a:.4f}, B: {norm_b:.4f}")
            
            # Use cosine similarity instead of euclidean distance
            # Cosine similarity ranges from -1 to 1, higher is more similar
            cos_sim = torch.nn.functional.cosine_similarity(za, zb, dim=1).item()
            print(f"Cosine similarity: {cos_sim:.4f}")
            
            # Convert cosine similarity to distance (0 to 2, lower is more similar)
            dist = 1 - cos_sim
            
            # Use cosine similarity as the score (0 to 1, higher is more similar)
            score = (cos_sim + 1) / 2  # Convert from [-1,1] to [0,1]
        
        # Use a more strict threshold for cosine similarity
        # 0.85+ cosine similarity = very similar faces only
        verdict = "similar" if cos_sim >= 0.85 else "dissimilar"
        return {"score": score, "verdict": verdict, "distance": float(dist)}
    except Exception as e:
        print(f"Error in similarity endpoint: {str(e)}")
        return {"error": f"Failed to process similarity: {str(e)}"}

@app.post("/recognize")
def recognize(image: UploadFile = File(...)):
    try:
        model = load_encoder_pt()
        recog = load_recognizer()
        if recog is None:
            return {"error": "Recognizer not found. Train and save to models/recognizer.joblib"}
        # Move tensor to the same device as the model
        device = next(model.parameters()).device
        x = tf_basic(read_image(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            z = model(x).cpu().numpy()
        pred = recog["model"].predict(z)[0]
        # Confidence proxy using kNN distance or SVM decision function
        conf = 0.5
        try:
            if recog["type"] == "knn":
                d, idx = recog["model"].kneighbors(z, n_neighbors=1, return_distance=True)
                conf = float(np.exp(-d[0][0]))
                # If distance is too high, mark as unknown
                if d[0][0] > 0.5:  # Threshold for "unknown" face
                    pred = "Unknown"
                    conf = 0.1
            else:
                df = recog["model"].decision_function(z)
                conf = float(1/(1+np.exp(-np.max(df))))
                # If decision function is too low, mark as unknown
                if np.max(df) < 0.5:
                    pred = "Unknown"
                    conf = 0.1
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
            return {"error": "Expression ONNX not found. Train and save to models/resnet_expressions.onnx"}
        
        # Use RAF-DB specific transform (100x100 RGB)
        img = tf_expr(read_image(image)).unsqueeze(0).numpy()
        logits = sess.run(None, {"image": img})[0]
        probs = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        idx = int(np.argmax(probs, axis=1)[0])
        
        # Load RAF-DB classes
        classes_path = os.environ.get("EXPRESSIONS_CLASSES", "models/rafdb_expressions_active.classes.txt")
        if os.path.exists(classes_path):
            with open(classes_path, "r") as f:
                classes = [l.strip() for l in f if l.strip()]
        else:
            classes = [f"class_{i}" for i in range(probs.shape[1])]
        
        return {
            "expression": classes[idx], 
            "confidence": float(probs[0, idx]),
            "probs": {classes[i]: float(probs[0,i]) for i in range(len(classes))},
            "model": "ResNet18 RAF-DB v20251025 (Latest)"
        }
    except Exception as e:
        return {"error": f"Failed to process expression: {str(e)}"}
