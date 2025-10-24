import os, argparse
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from .c3_expression import C3ExpressionCNN, C3ExpressionTrainer

def train_model(data_dir: str, out_path: str, epochs: int = 10, img_size: int = 160, batch: int = 32, lr: float = 1e-3, model_type: str = "mobilenet"):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    tf_train = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.1,0.1,0.1,0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    tf_val = transforms.Compose([
        transforms.Resize((img_size,img_size)), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=tf_train)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=tf_val)

    train_ld = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=2)
    val_ld = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=2)

    num_classes = len(train_ds.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {train_ds.classes}")

    # Create model based on type
    if model_type == "c3":
        model = C3ExpressionCNN(num_classes=num_classes, input_size=img_size)
        print(f"Using C3 Expression CNN with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    elif model_type == "mobilenet":
        model = models.mobilenet_v2(weights=None)
        in_feat = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_feat, num_classes)
        print(f"Using MobileNetV2 with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()

    best_acc = 0.0
    for ep in range(1, epochs+1):
        model.train(); tl=0.0
        for x,y in tqdm(train_ld, desc=f"train ep{ep}", leave=False):
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out,y)
            loss.backward()
            opt.step()
            tl += loss.item()*x.size(0)

        model.eval(); correct=0; tot=0
        with torch.no_grad():
            for x,y in val_ld:
                x,y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred==y).sum().item()
                tot += y.numel()
        acc = correct/tot if tot else 0
        print(f"Epoch {ep} train_loss={tl/max(1,len(train_ds)):.4f} val_acc={acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out_path.replace(".onnx",".pt"))

    # Export ONNX
    dummy = torch.randn(1,3,img_size,img_size).to(device)
    torch.onnx.export(model, dummy, out_path, input_names=["image"], output_names=["logits"], opset_version=13)
    
    # Save class names
    classes_path = out_path.replace(".onnx", ".classes.txt")
    with open(classes_path, "w") as f:
        for cls in train_ds.classes:
            f.write(f"{cls}\n")
    
    print(f"Saved expression model: {out_path}  best_val_acc={best_acc:.3f}")
    print(f"Saved class names: {classes_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Root with train/test subfolders containing expression classes")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default="models/expressions.onnx")
    ap.add_argument("--img", type=int, default=160)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--model", choices=["c3", "mobilenet"], default="c3", help="Model architecture to use")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    train_model(args.data, args.out, epochs=args.epochs, img_size=args.img, batch=args.batch, lr=args.lr, model_type=args.model)

if __name__ == "__main__":
    main()
