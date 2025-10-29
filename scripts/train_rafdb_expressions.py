#!/usr/bin/env python3
"""
Training script for ResNet expression recognition on RAF-DB dataset with M4 Max optimizations.
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.resnet_expression_cnn import train_resnet_model

def main():
    parser = argparse.ArgumentParser(description="Train ResNet expression model on RAF-DB with M4 Max optimizations")
    parser.add_argument("--data", default="data/rafdb", help="Path to RAF-DB dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=64, help="Batch size (optimized for M4 Max)")
    parser.add_argument("--img", type=int, default=100, help="Image size (100 for RAF-DB)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--out", default="models/rafdb_expressions_v2.onnx", help="Output model path")
    parser.add_argument("--model", choices=["resnet"], default="resnet", help="Model architecture")
    
    args = parser.parse_args()
    
    print("🚀 RAF-DB ResNet Expression Training with M4 Max Optimizations")
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
    print("✅ RAF-DB dataset (100x100 RGB, 20K+ images)")
    print("✅ ResNet18 architecture")
    print("✅ M4 Max GPU optimizations")
    print("✅ Focal loss for class imbalance")
    print("✅ Advanced data augmentation")
    print("✅ Learning rate scheduling")
    print("✅ Early stopping")
    print("✅ RGB preprocessing (vs grayscale FER2013)")
    print("✅ Expected 80-90% accuracy!")
    print("=" * 70)
    
    # Run the RAF-DB training
    train_resnet_model(
        data_dir=args.data,
        out_path=args.out,
        epochs=args.epochs,
        img_size=args.img,
        batch=args.batch,
        lr=args.lr,
        model_type=args.model,
        is_grayscale=False  # RAF-DB is RGB
    )

if __name__ == "__main__":
    main()
