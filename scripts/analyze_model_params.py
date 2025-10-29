#!/usr/bin/env python3
"""
Script to analyze model parameters and architecture details.
"""

import os
import sys
import onnx
import numpy as np
from pathlib import Path
import argparse

def analyze_onnx_model(model_path):
    """Analyze ONNX model to get parameter count and architecture details"""
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load ONNX model
    model = onnx.load(model_path)
    
    total_params = 0
    layer_details = []
    
    print(f"🔍 Analyzing ONNX Model: {os.path.basename(model_path)}")
    print("="*60)
    
    # Analyze each node in the graph
    for node in model.graph.node:
        if node.op_type in ['Conv', 'Gemm', 'BatchNormalization']:
            # Find corresponding initializer (weights)
            for initializer in model.graph.initializer:
                if initializer.name in node.input:
                    param_count = np.prod(initializer.dims) if initializer.dims else 1
                    total_params += param_count
                    
                    layer_info = {
                        'name': node.name,
                        'type': node.op_type,
                        'params': param_count,
                        'shape': list(initializer.dims) if initializer.dims else [1]
                    }
                    layer_details.append(layer_info)
    
    return total_params, layer_details

def analyze_pytorch_model(pt_path):
    """Analyze PyTorch model to get detailed parameter count"""
    
    if not os.path.exists(pt_path):
        print(f"⚠️  PyTorch model not found: {pt_path}")
        return None, None
    
    try:
        import torch
        from src.models.resnet_expression_cnn import ResNetExpression
        
        # Load model architecture
        model = ResNetExpression(num_classes=7, input_channels=3)
        
        # Load weights
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
            
        model.load_state_dict(torch.load(pt_path, map_location=device))
        model.eval()
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"🔍 Analyzing PyTorch Model: {os.path.basename(pt_path)}")
        print("="*60)
        
        # Detailed layer analysis
        layer_details = []
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                param_count = module.weight.numel()
                if hasattr(module, 'bias') and module.bias is not None:
                    param_count += module.bias.numel()
                
                layer_details.append({
                    'name': name,
                    'type': type(module).__name__,
                    'params': param_count,
                    'shape': list(module.weight.shape) if hasattr(module, 'weight') else []
                })
        
        return total_params, trainable_params, layer_details
        
    except Exception as e:
        print(f"❌ Error analyzing PyTorch model: {e}")
        return None, None, None

def print_model_summary(total_params, layer_details, model_type="ONNX"):
    """Print detailed model summary"""
    
    print(f"\n📊 MODEL SUMMARY ({model_type})")
    print("="*60)
    print(f"Total Parameters: {total_params:,}")
    print(f"Model Size: {total_params * 4 / (1024**2):.2f} MB (assuming float32)")
    
    print(f"\n🏗️ ARCHITECTURE BREAKDOWN:")
    print("-"*60)
    print(f"{'Layer Name':<30} {'Type':<15} {'Parameters':<12} {'Shape'}")
    print("-"*60)
    
    for layer in layer_details:
        shape_str = str(layer['shape']) if layer['shape'] else 'N/A'
        print(f"{layer['name']:<30} {layer['type']:<15} {layer['params']:<12,} {shape_str}")
    
    # Group by layer type
    type_counts = {}
    for layer in layer_details:
        layer_type = layer['type']
        if layer_type not in type_counts:
            type_counts[layer_type] = {'count': 0, 'params': 0}
        type_counts[layer_type]['count'] += 1
        type_counts[layer_type]['params'] += layer['params']
    
    print(f"\n📈 PARAMETERS BY LAYER TYPE:")
    print("-"*60)
    for layer_type, stats in sorted(type_counts.items(), key=lambda x: x[1]['params'], reverse=True):
        percentage = (stats['params'] / total_params) * 100
        print(f"{layer_type:<20} {stats['count']:<5} layers, {stats['params']:<12,} params ({percentage:.1f}%)")

def compare_with_other_models():
    """Compare parameter count with other common models"""
    
    print(f"\n🔍 PARAMETER COMPARISON WITH OTHER MODELS:")
    print("="*60)
    
    comparisons = {
        "Our ResNet18 RAF-DB": "~11.2M",
        "ResNet18 (ImageNet)": "11.7M",
        "ResNet50 (ImageNet)": "25.6M", 
        "EfficientNet-B0": "5.3M",
        "EfficientNet-B3": "12M",
        "MobileNetV2": "3.4M",
        "VGG16": "138M",
        "C3 CNN (Our Baseline)": "1.2M"
    }
    
    for model, params in comparisons.items():
        print(f"{model:<25} {params}")

def main():
    parser = argparse.ArgumentParser(description="Analyze model parameters")
    parser.add_argument("--onnx-model", default="models/rafdb_expressions_active.onnx")
    parser.add_argument("--pt-model", default="models/rafdb_expressions_active.pt")
    
    args = parser.parse_args()
    
    print("🎯 Model Parameter Analysis")
    print("="*50)
    
    # Analyze ONNX model
    try:
        onnx_params, onnx_layers = analyze_onnx_model(args.onnx_model)
        print_model_summary(onnx_params, onnx_layers, "ONNX")
    except Exception as e:
        print(f"❌ Error analyzing ONNX model: {e}")
    
    # Analyze PyTorch model (if available)
    try:
        pt_result = analyze_pytorch_model(args.pt_model)
        if pt_result[0] is not None:
            pt_params, trainable_params, pt_layers = pt_result
            print_model_summary(pt_params, pt_layers, "PyTorch")
            print(f"\n🔧 Trainable Parameters: {trainable_params:,}")
    except Exception as e:
        print(f"❌ Error analyzing PyTorch model: {e}")
    
    # Compare with other models
    compare_with_other_models()
    
    print(f"\n🎉 Analysis complete!")

if __name__ == "__main__":
    main()
