#!/usr/bin/env python3
"""
Script to compare different model versions and benchmark performance.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import onnxruntime as ort
from torchvision import transforms
import time
import pandas as pd
import argparse
import glob

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def load_model_and_classes(model_path, classes_path):
    """Load ONNX model and class names"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"Classes file not found: {classes_path}")
    
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    
    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]
    
    return session, classes

def preprocess_image(image_path, img_size=100):
    """Preprocess image for model"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).numpy()
    return tensor

def benchmark_model(session, test_images, img_size=100, num_runs=3):
    """Benchmark model performance"""
    
    # Warmup
    for _ in range(5):
        dummy_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
        session.run(None, {"image": dummy_input})
    
    # Benchmark inference time
    inference_times = []
    for _ in range(num_runs):
        start_time = time.time()
        for img_path, _, _ in test_images[:100]:  # Use first 100 images
            try:
                image_tensor = preprocess_image(img_path, img_size)
                session.run(None, {"image": image_tensor})
            except:
                continue
        end_time = time.time()
        inference_times.append(end_time - start_time)
    
    avg_inference_time = np.mean(inference_times)
    std_inference_time = np.std(inference_times)
    
    return avg_inference_time, std_inference_time

def evaluate_model_accuracy(session, test_images, classes, img_size=100):
    """Evaluate model accuracy"""
    
    correct = 0
    total = 0
    
    for img_path, true_label, class_name in test_images:
        try:
            image_tensor = preprocess_image(img_path, img_size)
            logits = session.run(None, {"image": image_tensor})[0]
            predicted_class = np.argmax(logits, axis=1)[0]
            
            if predicted_class == true_label:
                correct += 1
            total += 1
            
        except Exception as e:
            continue
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total

def get_model_info(model_path):
    """Extract model information from file"""
    
    # Get file size
    file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
    
    # Extract timestamp from filename if available
    import re
    timestamp_match = re.search(r'v(\d{8}_\d{4})', os.path.basename(model_path))
    timestamp = timestamp_match.group(1) if timestamp_match else "unknown"
    
    return {
        'file_size_mb': file_size,
        'timestamp': timestamp
    }

def compare_models(models_dir="models", test_dir="data/rafdb/test", max_samples=500):
    """Compare all available models"""
    
    print("🔍 Finding all RAF-DB models...")
    
    # Find all RAF-DB models
    model_patterns = [
        "rafdb_expressions*.onnx",
        "rafdb_expressions_active.onnx"
    ]
    
    models = []
    for pattern in model_patterns:
        models.extend(glob.glob(os.path.join(models_dir, pattern)))
    
    # Remove duplicates and non-existent files
    models = list(set([m for m in models if os.path.exists(m)]))
    
    if not models:
        print("❌ No RAF-DB models found!")
        return
    
    print(f"📊 Found {len(models)} models to compare")
    
    # Get test images
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    test_images = []
    
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(test_dir, class_name)
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img_file in images:
                test_images.append((os.path.join(class_dir, img_file), class_idx, class_name))
    
    if max_samples and len(test_images) > max_samples:
        import random
        test_images = random.sample(test_images, max_samples)
    
    print(f"📊 Using {len(test_images)} test images for comparison")
    
    # Compare each model
    results = []
    
    for model_path in models:
        print(f"\n🔍 Evaluating: {os.path.basename(model_path)}")
        
        try:
            # Find corresponding classes file
            classes_path = model_path.replace('.onnx', '.classes.txt')
            if not os.path.exists(classes_path):
                print(f"⚠️  Classes file not found: {classes_path}")
                continue
            
            # Load model
            session, model_classes = load_model_and_classes(model_path, classes_path)
            
            # Get model info
            model_info = get_model_info(model_path)
            
            # Evaluate accuracy
            accuracy, correct, total = evaluate_model_accuracy(session, test_images, model_classes)
            
            # Benchmark performance
            avg_time, std_time = benchmark_model(session, test_images)
            
            # Calculate throughput
            throughput = len(test_images) / avg_time
            
            results.append({
                'model_name': os.path.basename(model_path),
                'timestamp': model_info['timestamp'],
                'file_size_mb': model_info['file_size_mb'],
                'accuracy': accuracy,
                'correct_predictions': correct,
                'total_predictions': total,
                'avg_inference_time': avg_time,
                'std_inference_time': std_time,
                'throughput_images_per_sec': throughput
            })
            
            print(f"  ✅ Accuracy: {accuracy:.3f}")
            print(f"  ⏱️  Avg Time: {avg_time:.2f}s")
            print(f"  🚀 Throughput: {throughput:.1f} img/s")
            
        except Exception as e:
            print(f"  ❌ Error evaluating {model_path}: {e}")
            continue
    
    return results

def plot_model_comparison(results, save_path=None):
    """Plot comprehensive model comparison"""
    
    if not results:
        print("No results to plot!")
        return
    
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Accuracy comparison
    bars = axes[0, 0].bar(range(len(df)), df['accuracy'], color='skyblue', alpha=0.7)
    axes[0, 0].set_xlabel('Model')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_xticks(range(len(df)))
    axes[0, 0].set_xticklabels([name.replace('rafdb_expressions_', '').replace('.onnx', '') 
                               for name in df['model_name']], rotation=45, ha='right')
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, df['accuracy']):
        axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # 2. Throughput comparison
    bars = axes[0, 1].bar(range(len(df)), df['throughput_images_per_sec'], color='lightgreen', alpha=0.7)
    axes[0, 1].set_xlabel('Model')
    axes[0, 1].set_ylabel('Throughput (images/sec)')
    axes[0, 1].set_title('Model Throughput Comparison')
    axes[0, 1].set_xticks(range(len(df)))
    axes[0, 1].set_xticklabels([name.replace('rafdb_expressions_', '').replace('.onnx', '') 
                               for name in df['model_name']], rotation=45, ha='right')
    
    # Add throughput values on bars
    for bar, throughput in zip(bars, df['throughput_images_per_sec']):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(df['throughput_images_per_sec'])*0.01,
                       f'{throughput:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. File size comparison
    bars = axes[1, 0].bar(range(len(df)), df['file_size_mb'], color='orange', alpha=0.7)
    axes[1, 0].set_xlabel('Model')
    axes[1, 0].set_ylabel('File Size (MB)')
    axes[1, 0].set_title('Model File Size Comparison')
    axes[1, 0].set_xticks(range(len(df)))
    axes[1, 0].set_xticklabels([name.replace('rafdb_expressions_', '').replace('.onnx', '') 
                               for name in df['model_name']], rotation=45, ha='right')
    
    # Add file size values on bars
    for bar, size in zip(bars, df['file_size_mb']):
        axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(df['file_size_mb'])*0.01,
                       f'{size:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Accuracy vs Throughput scatter plot
    scatter = axes[1, 1].scatter(df['throughput_images_per_sec'], df['accuracy'], 
                               s=df['file_size_mb']*10, alpha=0.7, c=range(len(df)), cmap='viridis')
    axes[1, 1].set_xlabel('Throughput (images/sec)')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Accuracy vs Throughput (bubble size = file size)')
    
    # Add model labels
    for i, row in df.iterrows():
        axes[1, 1].annotate(row['model_name'].replace('rafdb_expressions_', '').replace('.onnx', ''),
                           (row['throughput_images_per_sec'], row['accuracy']),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Model comparison saved to: {save_path}")
    
    plt.show()

def print_comparison_table(results):
    """Print detailed comparison table"""
    
    if not results:
        print("No results to display!")
        return
    
    df = pd.DataFrame(results)
    
    print("\n" + "="*100)
    print("📊 MODEL COMPARISON RESULTS")
    print("="*100)
    
    # Sort by accuracy
    df_sorted = df.sort_values('accuracy', ascending=False)
    
    print(f"{'Model':<25} {'Accuracy':<10} {'Throughput':<12} {'File Size':<10} {'Timestamp':<12}")
    print("-" * 100)
    
    for _, row in df_sorted.iterrows():
        model_name = row['model_name'].replace('rafdb_expressions_', '').replace('.onnx', '')
        print(f"{model_name:<25} {row['accuracy']:<10.3f} {row['throughput_images_per_sec']:<12.1f} "
              f"{row['file_size_mb']:<10.1f} {row['timestamp']:<12}")
    
    print("-" * 100)
    print(f"{'BEST MODEL':<25} {df_sorted.iloc[0]['accuracy']:<10.3f} "
          f"{df_sorted.iloc[0]['throughput_images_per_sec']:<12.1f} "
          f"{df_sorted.iloc[0]['file_size_mb']:<10.1f} {df_sorted.iloc[0]['timestamp']:<12}")
    
    print("\n" + "="*100)
    print("📈 PERFORMANCE INSIGHTS")
    print("="*100)
    
    best_accuracy = df_sorted.iloc[0]
    best_throughput = df.loc[df['throughput_images_per_sec'].idxmax()]
    smallest_model = df.loc[df['file_size_mb'].idxmin()]
    
    print(f"🏆 Highest Accuracy: {best_accuracy['model_name']} ({best_accuracy['accuracy']:.3f})")
    print(f"🚀 Fastest Throughput: {best_throughput['model_name']} ({best_throughput['throughput_images_per_sec']:.1f} img/s)")
    print(f"📦 Smallest Model: {smallest_model['model_name']} ({smallest_model['file_size_mb']:.1f} MB)")
    
    # Calculate improvements
    if len(df) > 1:
        accuracy_range = df['accuracy'].max() - df['accuracy'].min()
        throughput_range = df['throughput_images_per_sec'].max() - df['throughput_images_per_sec'].min()
        
        print(f"\n📊 Performance Ranges:")
        print(f"   Accuracy: {accuracy_range:.3f} ({df['accuracy'].min():.3f} - {df['accuracy'].max():.3f})")
        print(f"   Throughput: {throughput_range:.1f} img/s ({df['throughput_images_per_sec'].min():.1f} - {df['throughput_images_per_sec'].max():.1f})")

def main():
    parser = argparse.ArgumentParser(description="Compare RAF-DB model versions")
    parser.add_argument("--models-dir", default="models", help="Directory containing models")
    parser.add_argument("--test-dir", default="data/rafdb/test", help="Test dataset directory")
    parser.add_argument("--output", default="model_comparison.png", help="Output plot path")
    parser.add_argument("--max-samples", type=int, default=500, help="Max test samples")
    
    args = parser.parse_args()
    
    print("🎯 RAF-DB Model Comparison Tool")
    print("="*50)
    print(f"Models Directory: {args.models_dir}")
    print(f"Test Directory: {args.test_dir}")
    print(f"Max Samples: {args.max_samples}")
    print("="*50)
    
    try:
        # Compare models
        results = compare_models(args.models_dir, args.test_dir, args.max_samples)
        
        if results:
            # Generate comparison plot
            plot_model_comparison(results, args.output)
            
            # Print detailed table
            print_comparison_table(results)
            
            print(f"\n🎉 Model comparison complete!")
            print(f"📁 Comparison plot saved to: {args.output}")
        else:
            print("❌ No models found for comparison!")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
