import os, argparse, random, json, math, sys
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.metrics import roc_curve, auc
from .c3_cnn import C3CNN, C3SiameseEncoder

# --- Dataset that yields positive/negative pairs from a folder-of-folders (class per subfolder) ---
class PairDataset(Dataset):
    def __init__(self, root: str, img_size: int = 160, pairs_per_class: int = 20, split: str = "train", val_ratio: float = 0.2, seed: int = 42):
        self.root = Path(root)
        self.img_size = img_size
        self.seed = seed
        self.rng = random.Random(seed)

        classes = sorted([p for p in self.root.iterdir() if p.is_dir()])
        self.class_to_imgs = {c.name: sorted([str(x) for x in c.glob("*") if x.suffix.lower() in (".jpg",".jpeg",".png",".bmp",".pgm")]) for c in classes}
        self.classes = [c for c, imgs in self.class_to_imgs.items() if len(imgs) >= 2]

        # Split each class into train/val
        self.split_paths: Dict[str, Dict[str, List[str]]] = {}
        for c in self.classes:
            imgs = self.class_to_imgs[c]
            self.rng.shuffle(imgs)
            n_val = max(1, int(len(imgs)*val_ratio))
            val = imgs[:n_val]
            tr = imgs[n_val:]
            if len(tr) < 1 or len(val) < 1:
                continue
            self.split_paths[c] = {"train": tr, "val": val}

        self.split = split
        self.pairs: List[Tuple[str, str, int]] = []
        for c in self.split_paths:
            imgs = self.split_paths[c][split]
            # positive pairs
            for _ in range(pairs_per_class):
                a, b = self.rng.sample(imgs, 2)
                self.pairs.append((a, b, 1))
            # negative pairs
            other = [cc for cc in self.split_paths if cc != c]
            for _ in range(pairs_per_class):
                a = self.rng.choice(imgs)
                cc2 = self.rng.choice(other)
                b = self.rng.choice(self.split_paths[cc2][split])
                self.pairs.append((a, b, 0))

        # Add data augmentation for better robustness
        if split == "train":
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomRotation(5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            self.tf = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        a, b, y = self.pairs[idx]
        ia = self.tf(Image.open(a).convert("RGB"))
        ib = self.tf(Image.open(b).convert("RGB"))
        return ia, ib, torch.tensor([y], dtype=torch.float32)

# --- Simple encoder (can be replaced by ResNet18 backbone or C3 CNN) ---
class Encoder(nn.Module):
    def __init__(self, embedding_dim: int = 128, backbone: str = "resnet18"):
        super().__init__()
        self.backbone_type = backbone
        
        if backbone == "c3":
            self.backbone = C3CNN(embedding_dim=embedding_dim)
        elif backbone == "resnet18":
            self.backbone = models.resnet18(weights=None)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, embedding_dim)
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

    def forward(self, x):
        if self.backbone_type == "c3":
            z = self.backbone(x)  # C3CNN already includes L2 normalization
        else:
            z = self.backbone(x)
            z = F.normalize(z, p=2, dim=1)  # L2 norm for ResNet18
        return z

def contrastive_loss(distances, labels, margin=0.5):
    # labels: 1 same, 0 different
    pos = labels * distances.pow(2)
    neg = (1 - labels) * torch.clamp(margin - distances, min=0).pow(2)
    return (pos + neg).mean()

def train_epoch(model, loader, opt, device):
    model.train()
    total = 0.0
    for ia, ib, y in tqdm(loader, desc="train", leave=False):
        ia, ib, y = ia.to(device), ib.to(device), y.to(device)
        za = model(ia)
        zb = model(ib)
        distances = torch.sqrt(((za - zb) ** 2).sum(dim=1, keepdim=True) + 1e-8)
        loss = contrastive_loss(distances, y, margin=0.5)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item()*ia.size(0)
    return total/len(loader.dataset)

@torch.no_grad()
def eval_pairs(model, loader, device):
    model.eval()
    ys, ds = [], []
    for ia, ib, y in tqdm(loader, desc="eval", leave=False):
        ia, ib = ia.to(device), ib.to(device)
        za, zb = model(ia), model(ib)
        distances = torch.sqrt(((za - zb) ** 2).sum(dim=1) + 1e-8).cpu().numpy()
        ds.extend(distances.tolist())
        ys.extend(y.squeeze(1).numpy().tolist())
    fpr, tpr, th = roc_curve(ys, [-d for d in ds])  # higher is more similar => use -distance
    return {
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "thresholds": th.tolist(),
        "auc": float(auc(fpr, tpr)),
        "distances": ds,
        "labels": ys,
    }

def save_onnx(model, path, img_size=160):
    model.eval()
    x = torch.randn(1,3,img_size,img_size)
    torch.onnx.export(model, x, path, input_names=["image"], output_names=["embedding"], opset_version=13)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=False, help="Root with class subfolders for pairs")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--img", type=int, default=160)
    ap.add_argument("--encoder", help="Path to a trained encoder (.pt) for eval mode")
    ap.add_argument("--out", default="models/siamese_encoder.pt")
    ap.add_argument("--onnx", default="models/siamese_encoder.onnx")
    ap.add_argument("--eval", action="store_true", help="Run evaluation only")
    ap.add_argument("--report", default="reports/siamese_eval.json")
    ap.add_argument("--backbone", choices=["resnet18", "c3"], default="resnet18", help="Backbone architecture")
    args = ap.parse_args()

    # Use MPS (Metal Performance Shaders) on Mac, CUDA on NVIDIA GPUs, or CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    os.makedirs(os.path.dirname(args.onnx), exist_ok=True)
    os.makedirs(os.path.dirname(args.report), exist_ok=True)

    model = Encoder(backbone=args.backbone)
    model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Using backbone: {args.backbone}")
    print(f"Total trainable parameters: {total_params:,}")

    if args.eval:
        assert args.encoder, "--encoder required for --eval"
        model.load_state_dict(torch.load(args.encoder, map_location=device))
        val_ds = PairDataset(args.data, img_size=args.img, split="val")
        val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)
        metrics = eval_pairs(model, val_ld, device)
        with open(args.report, "w") as f:
            json.dump(metrics, f, indent=2)
        # recommend threshold at closest point to (0,1)
        fpr = np.array(metrics["fpr"]); tpr = np.array(metrics["tpr"]); thr = np.array(metrics["thresholds"])
        idx = np.argmin((fpr-0)**2 + (tpr-1)**2)
        print(f"Recommended threshold (similarity score) = {-thr[idx]:.4f} (based on -distance) AUC={metrics['auc']:.3f}")
        sys.exit(0)

    # Train mode
    train_ds = PairDataset(args.data, img_size=args.img, split="train")
    val_ds = PairDataset(args.data, img_size=args.img, split="val")
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    for ep in range(1, args.epochs+1):
        tl = train_epoch(model, train_ld, opt, device)
        metrics = eval_pairs(model, val_ld, device)
        print(f"Epoch {ep}: train_loss={tl:.4f} AUC={metrics['auc']:.3f}")

    torch.save(model.state_dict(), args.out)
    save_onnx(model, args.onnx, img_size=args.img)
    print(f"Saved encoder to {args.out} and ONNX to {args.onnx}")

if __name__ == "__main__":
    main()
