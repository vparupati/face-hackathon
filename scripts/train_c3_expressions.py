#!/usr/bin/env python3
"""
Training script for C3 CNN model on expression recognition task.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.expression_cnn import main as expression_main

def main():
    parser = argparse.ArgumentParser(description="Train C3 CNN for expression recognition")
    parser.add_argument("--data", required=True, help="Path to expression dataset (with train/test subfolders)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--img", type=int, default=160, help="Image size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--out", default="models/c3_expressions.onnx", help="Output model path")
    parser.add_argument("--model", choices=["c3", "mobilenet"], default="c3", help="Model architecture")
    
    args = parser.parse_args()
    
    # Prepare arguments for expression training
    sys.argv = [
        "train_c3_expressions.py",
        "--data", args.data,
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--img", str(args.img),
        "--lr", str(args.lr),
        "--out", args.out,
        "--model", args.model
    ]
    
    print("=" * 60)
    print("Training C3 CNN for Expression Recognition")
    print("=" * 60)
    print(f"Dataset: {args.data}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.img}")
    print(f"Learning rate: {args.lr}")
    print(f"Output model: {args.out}")
    print("=" * 60)
    
    # Run the expression training
    expression_main()

if __name__ == "__main__":
    main()
