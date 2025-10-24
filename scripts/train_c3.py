#!/usr/bin/env python3
"""
Training script for C3 CNN model on face recognition task.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.siamese import main as siamese_main

def main():
    parser = argparse.ArgumentParser(description="Train C3 CNN for face recognition")
    parser.add_argument("--data", required=True, help="Path to face dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--img", type=int, default=160, help="Image size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--out", default="models/c3_siamese_encoder.pt", help="Output model path")
    parser.add_argument("--onnx", default="models/c3_siamese_encoder.onnx", help="ONNX export path")
    parser.add_argument("--report", default="reports/c3_siamese_eval.json", help="Evaluation report path")
    
    args = parser.parse_args()
    
    # Prepare arguments for siamese training
    sys.argv = [
        "train_c3.py",
        "--data", args.data,
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--img", str(args.img),
        "--out", args.out,
        "--onnx", args.onnx,
        "--report", args.report,
        "--backbone", "c3"
    ]
    
    print("=" * 60)
    print("Training C3 CNN for Face Recognition")
    print("=" * 60)
    print(f"Dataset: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.img}")
    print(f"Learning rate: {args.lr}")
    print(f"Output model: {args.out}")
    print("=" * 60)
    
    # Run the siamese training with C3 backbone
    siamese_main()

if __name__ == "__main__":
    main()
