#!/usr/bin/env python3
"""
Training script for ResNet expression recognition model with M4 Max optimizations.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.resnet_expression_cnn import main as resnet_main

def main():
    parser = argparse.ArgumentParser(description="Train ResNet expression model with M4 Max optimizations")
    parser.add_argument("--data", required=True, help="Path to expression dataset (with train/test subfolders)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=64, help="Batch size (optimized for M4 Max)")
    parser.add_argument("--img", type=int, default=48, help="Image size (keep 48 for FER2013)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--out", default="models/resnet_expressions.onnx", help="Output model path")
    parser.add_argument("--model", choices=["resnet"], default="resnet", help="Model architecture")
    
    args = parser.parse_args()
    
    # Prepare arguments for ResNet training
    sys.argv = [
        "train_resnet_expressions.py",
        "--data", args.data,
        "--epochs", str(args.epochs),
        "--batch", str(args.batch),
        "--img", str(args.img),
        "--lr", str(args.lr),
        "--out", args.out,
        "--model", args.model
    ]
    
    print("🚀 ResNet Expression Training with M4 Max Optimizations")
    print("=" * 70)
    print(f"Dataset: {args.data}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.img}")
    print(f"Learning rate: {args.lr}")
    print(f"Output model: {args.out}")
    print("=" * 70)
    print("Key Improvements:")
    print("✅ ResNet18 architecture (better for 48x48)")
    print("✅ M4 Max GPU optimizations")
    print("✅ Focal loss for class imbalance")
    print("✅ Advanced data augmentation")
    print("✅ Learning rate scheduling")
    print("✅ Early stopping")
    print("✅ Grayscale-optimized preprocessing")
    print("✅ Pretrained ImageNet weights")
    print("=" * 70)
    
    # Run the ResNet training
    resnet_main()

if __name__ == "__main__":
    main()
