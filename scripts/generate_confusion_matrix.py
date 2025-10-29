#!/usr/bin/env python3
"""
Script to generate confusion matrix for the latest RAF-DB expression model.
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
from sklearn.metrics import confusion_matrix, classification_report
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

def load_model_and_classes(model_path, classes_path):
    """Load ONNX model and class names"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(classes_path):
        raise FileNotFoundError(f"Classes file not found: {classes_path}")
    
    # Load ONNX model
    session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    
    # Load class names
    with open(classes_path, 'r') as f:
        classes = [line.strip() for line in f if line.strip()]
    
    return session, classes

def preprocess_image(image_path, img_size=100):
    """Preprocess image for RAF-DB model"""
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).numpy()
    return tensor

def predict_emotion(session, image_tensor):
    """Predict emotion using ONNX model"""
    logits = session.run(None, {"image": image_tensor})[0]
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    predicted_class = np.argmax(probabilities, axis=1)[0]
    confidence = probabilities[0, predicted_class]
    return predicted_class, confidence

def evaluate_test_set(test_dir, session, classes, img_size=100, max_samples=None):
    """Evaluate model on test set and return predictions"""
    print(f"🔍 Evaluating model on test set: {test_dir}")
    
    true_labels = []
    predicted_labels = []
    confidences = []
    
    # Get all test images
    test_images = []
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(test_dir, class_name)
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img_file in images:
                test_images.append((os.path.join(class_dir, img_file), class_idx, class_name))
    
    print(f"📊 Found {len(test_images)} test images")
    
    # Limit samples if specified
    if max_samples and len(test_images) > max_samples:
        import random
        test_images = random.sample(test_images, max_samples)
        print(f"🎯 Using {max_samples} random samples for evaluation")
    
    # Process each image
    for i, (img_path, true_label, class_name) in enumerate(test_images):
        try:
            # Preprocess image
            image_tensor = preprocess_image(img_path, img_size)
            
            # Predict emotion
            pred_label, confidence = predict_emotion(session, image_tensor)
            
            # Store results
            true_labels.append(true_label)
            predicted_labels.append(pred_label)
            confidences.append(confidence)
            
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(test_images)} images...")
                
        except Exception as e:
            print(f"⚠️  Error processing {img_path}: {e}")
            continue
    
    print(f"✅ Evaluation complete! Processed {len(true_labels)} images")
    return np.array(true_labels), np.array(predicted_labels), np.array(confidences)

def plot_confusion_matrix(y_true, y_pred, classes, save_path=None):
    """Plot confusion matrix with detailed statistics"""
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate accuracy per class
    class_accuracies = cm.diagonal() / cm.sum(axis=1)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=ax1)
    ax1.set_title('Confusion Matrix - RAF-DB Expression Model', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Emotion', fontsize=12)
    ax1.set_ylabel('True Emotion', fontsize=12)
    
    # Add accuracy text
    total_accuracy = np.trace(cm) / np.sum(cm)
    ax1.text(0.02, 0.98, f'Overall Accuracy: {total_accuracy:.3f}', 
             transform=ax1.transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Plot 2: Per-Class Accuracy
    bars = ax2.bar(range(len(classes)), class_accuracies, color='skyblue', alpha=0.7)
    ax2.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Emotion Class', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_xticks(range(len(classes)))
    ax2.set_xticklabels(classes, rotation=45, ha='right')
    ax2.set_ylim(0, 1)
    
    # Add accuracy values on bars
    for i, (bar, acc) in enumerate(zip(bars, class_accuracies)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Add grid
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to: {save_path}")
    
    plt.show()
    
    return cm, class_accuracies, total_accuracy

def print_detailed_report(y_true, y_pred, classes, confidences):
    """Print detailed classification report"""
    
    print("\n" + "="*60)
    print("📊 DETAILED CLASSIFICATION REPORT")
    print("="*60)
    
    # Overall statistics
    total_samples = len(y_true)
    correct_predictions = np.sum(y_true == y_pred)
    overall_accuracy = correct_predictions / total_samples
    avg_confidence = np.mean(confidences)
    
    print(f"Total Test Samples: {total_samples}")
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Overall Accuracy: {overall_accuracy:.3f}")
    print(f"Average Confidence: {avg_confidence:.3f}")
    
    print("\n" + "-"*60)
    print("PER-CLASS PERFORMANCE:")
    print("-"*60)
    
    # Per-class statistics
    for i, class_name in enumerate(classes):
        class_mask = y_true == i
        class_samples = np.sum(class_mask)
        
        if class_samples > 0:
            class_correct = np.sum((y_true == i) & (y_pred == i))
            class_accuracy = class_correct / class_samples
            class_confidences = confidences[class_mask]
            avg_class_confidence = np.mean(class_confidences)
            
            print(f"{class_name:12s}: {class_samples:4d} samples, "
                  f"{class_correct:4d} correct, "
                  f"acc={class_accuracy:.3f}, "
                  f"conf={avg_class_confidence:.3f}")
    
    print("\n" + "-"*60)
    print("CONFUSION ANALYSIS:")
    print("-"*60)
    
    # Find most confused pairs
    cm = confusion_matrix(y_true, y_pred)
    confusion_pairs = []
    
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i, j] > 0:
                confusion_pairs.append((cm[i, j], classes[i], classes[j]))
    
    confusion_pairs.sort(reverse=True)
    
    print("Most Common Misclassifications:")
    for count, true_class, pred_class in confusion_pairs[:10]:
        percentage = count / np.sum(cm[classes.index(true_class), :]) * 100
        print(f"  {true_class} → {pred_class}: {count} times ({percentage:.1f}%)")

def main():
    parser = argparse.ArgumentParser(description="Generate confusion matrix for RAF-DB model")
    parser.add_argument("--model", default="models/rafdb_expressions_active.onnx", 
                       help="Path to ONNX model")
    parser.add_argument("--classes", default="models/rafdb_expressions_active.classes.txt",
                       help="Path to classes file")
    parser.add_argument("--test-dir", default="data/rafdb/test",
                       help="Path to test dataset")
    parser.add_argument("--output", default="confusion_matrix.png",
                       help="Output path for confusion matrix plot")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of samples to evaluate (for quick testing)")
    parser.add_argument("--img-size", type=int, default=100,
                       help="Image size for preprocessing")
    
    args = parser.parse_args()
    
    print("🎯 RAF-DB Model Confusion Matrix Generator")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Classes: {args.classes}")
    print(f"Test Directory: {args.test_dir}")
    print(f"Output: {args.output}")
    print(f"Image Size: {args.img_size}")
    if args.max_samples:
        print(f"Max Samples: {args.max_samples}")
    print("="*50)
    
    try:
        # Load model and classes
        print("📥 Loading model and classes...")
        session, classes = load_model_and_classes(args.model, args.classes)
        print(f"✅ Loaded model with {len(classes)} classes: {classes}")
        
        # Evaluate test set
        y_true, y_pred, confidences = evaluate_test_set(
            args.test_dir, session, classes, args.img_size, args.max_samples
        )
        
        # Generate confusion matrix plot
        print("📊 Generating confusion matrix...")
        cm, class_accuracies, overall_accuracy = plot_confusion_matrix(
            y_true, y_pred, classes, args.output
        )
        
        # Print detailed report
        print_detailed_report(y_true, y_pred, classes, confidences)
        
        print(f"\n🎉 Confusion matrix analysis complete!")
        print(f"📊 Overall Accuracy: {overall_accuracy:.3f}")
        print(f"📁 Plot saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
