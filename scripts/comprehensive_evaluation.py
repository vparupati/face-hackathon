#!/usr/bin/env python3
"""
Script to generate comprehensive model evaluation metrics and visualizations.
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
from sklearn.metrics import (
    confusion_matrix, classification_report, precision_recall_curve,
    roc_curve, auc, precision_score, recall_score, f1_score
)
import pandas as pd
import argparse

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
    return predicted_class, confidence, probabilities[0]

def evaluate_model_comprehensive(test_dir, session, classes, img_size=100, max_samples=None):
    """Comprehensive model evaluation"""
    print(f"🔍 Comprehensive evaluation on: {test_dir}")
    
    true_labels = []
    predicted_labels = []
    confidences = []
    all_probabilities = []
    
    # Get all test images
    test_images = []
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(test_dir, class_name)
        if os.path.exists(class_dir):
            images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img_file in images:
                test_images.append((os.path.join(class_dir, img_file), class_idx, class_name))
    
    print(f"📊 Found {len(test_images)} test images")
    
    if max_samples and len(test_images) > max_samples:
        import random
        test_images = random.sample(test_images, max_samples)
        print(f"🎯 Using {max_samples} random samples")
    
    # Process each image
    for i, (img_path, true_label, class_name) in enumerate(test_images):
        try:
            image_tensor = preprocess_image(img_path, img_size)
            pred_label, confidence, probabilities = predict_emotion(session, image_tensor)
            
            true_labels.append(true_label)
            predicted_labels.append(pred_label)
            confidences.append(confidence)
            all_probabilities.append(probabilities)
            
            if (i + 1) % 500 == 0:
                print(f"  Processed {i + 1}/{len(test_images)} images...")
                
        except Exception as e:
            print(f"⚠️  Error processing {img_path}: {e}")
            continue
    
    print(f"✅ Evaluation complete! Processed {len(true_labels)} images")
    return (np.array(true_labels), np.array(predicted_labels), 
            np.array(confidences), np.array(all_probabilities))

def plot_confidence_distribution(y_true, y_pred, confidences, classes, save_path=None):
    """Plot confidence distribution for correct vs incorrect predictions"""
    
    correct_mask = y_true == y_pred
    correct_confidences = confidences[correct_mask]
    incorrect_confidences = confidences[~correct_mask]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Overall confidence distribution
    ax1.hist(correct_confidences, bins=30, alpha=0.7, label='Correct', color='green')
    ax1.hist(incorrect_confidences, bins=30, alpha=0.7, label='Incorrect', color='red')
    ax1.set_xlabel('Confidence Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Confidence Distribution: Correct vs Incorrect Predictions')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Per-class confidence distribution
    for i, class_name in enumerate(classes):
        class_mask = y_true == i
        if np.sum(class_mask) > 0:
            class_confidences = confidences[class_mask]
            ax2.hist(class_confidences, bins=20, alpha=0.6, label=class_name)
    
    ax2.set_xlabel('Confidence Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Confidence Distribution by Emotion Class')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confidence distribution saved to: {save_path}")
    
    plt.show()

def plot_precision_recall_curves(y_true, all_probabilities, classes, save_path=None):
    """Plot precision-recall curves for each class"""
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, class_name in enumerate(classes):
        if i < len(axes):
            # Binary classification for this class
            y_binary = (y_true == i).astype(int)
            y_scores = all_probabilities[:, i]
            
            precision, recall, _ = precision_recall_curve(y_binary, y_scores)
            pr_auc = auc(recall, precision)
            
            axes[i].plot(recall, precision, linewidth=2, label=f'PR-AUC = {pr_auc:.3f}')
            axes[i].set_xlabel('Recall')
            axes[i].set_ylabel('Precision')
            axes[i].set_title(f'{class_name}\n(PR-AUC: {pr_auc:.3f})')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    # Remove empty subplot
    if len(classes) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.suptitle('Precision-Recall Curves by Emotion Class', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Precision-Recall curves saved to: {save_path}")
    
    plt.show()

def plot_roc_curves(y_true, all_probabilities, classes, save_path=None):
    """Plot ROC curves for each class"""
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, class_name in enumerate(classes):
        if i < len(axes):
            # Binary classification for this class
            y_binary = (y_true == i).astype(int)
            y_scores = all_probabilities[:, i]
            
            fpr, tpr, _ = roc_curve(y_binary, y_scores)
            roc_auc = auc(fpr, tpr)
            
            axes[i].plot(fpr, tpr, linewidth=2, label=f'AUC = {roc_auc:.3f}')
            axes[i].plot([0, 1], [0, 1], 'k--', alpha=0.5)
            axes[i].set_xlabel('False Positive Rate')
            axes[i].set_ylabel('True Positive Rate')
            axes[i].set_title(f'{class_name}\n(AUC: {roc_auc:.3f})')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
    
    # Remove empty subplot
    if len(classes) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.suptitle('ROC Curves by Emotion Class', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 ROC curves saved to: {save_path}")
    
    plt.show()

def plot_class_imbalance_analysis(test_dir, classes, save_path=None):
    """Analyze and visualize class distribution"""
    
    class_counts = []
    for class_name in classes:
        class_dir = os.path.join(test_dir, class_name)
        if os.path.exists(class_dir):
            count = len([f for f in os.listdir(class_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            class_counts.append(count)
        else:
            class_counts.append(0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Bar plot
    bars = ax1.bar(range(len(classes)), class_counts, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Emotion Class')
    ax1.set_ylabel('Number of Images')
    ax1.set_title('Class Distribution in Test Set')
    ax1.set_xticks(range(len(classes)))
    ax1.set_xticklabels(classes, rotation=45, ha='right')
    
    # Add count labels on bars
    for bar, count in zip(bars, class_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(class_counts)*0.01,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # Pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
    wedges, texts, autotexts = ax2.pie(class_counts, labels=classes, autopct='%1.1f%%', 
                                       colors=colors, startangle=90)
    ax2.set_title('Class Distribution (Percentage)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Class imbalance analysis saved to: {save_path}")
    
    plt.show()
    
    return class_counts

def plot_error_analysis(y_true, y_pred, confidences, classes, save_path=None):
    """Analyze prediction errors in detail"""
    
    errors = []
    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            errors.append({
                'true_class': classes[y_true[i]],
                'predicted_class': classes[y_pred[i]],
                'confidence': confidences[i]
            })
    
    error_df = pd.DataFrame(errors)
    
    if len(error_df) == 0:
        print("No errors to analyze!")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Most common error types
    error_counts = error_df.groupby(['true_class', 'predicted_class']).size().reset_index(name='count')
    top_errors = error_counts.nlargest(10, 'count')
    
    axes[0, 0].barh(range(len(top_errors)), top_errors['count'])
    axes[0, 0].set_yticks(range(len(top_errors)))
    axes[0, 0].set_yticklabels([f"{row['true_class']} → {row['predicted_class']}" 
                               for _, row in top_errors.iterrows()])
    axes[0, 0].set_xlabel('Number of Errors')
    axes[0, 0].set_title('Most Common Misclassifications')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Confidence distribution of errors
    axes[0, 1].hist(error_df['confidence'], bins=30, alpha=0.7, color='red')
    axes[0, 1].set_xlabel('Confidence Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Confidence Distribution of Errors')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Errors by true class
    error_by_class = error_df.groupby('true_class').size().sort_values(ascending=True)
    axes[1, 0].barh(range(len(error_by_class)), error_by_class.values)
    axes[1, 0].set_yticks(range(len(error_by_class)))
    axes[1, 0].set_yticklabels(error_by_class.index)
    axes[1, 0].set_xlabel('Number of Errors')
    axes[1, 0].set_title('Errors by True Class')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Average confidence by error type
    avg_conf_by_error = error_df.groupby(['true_class', 'predicted_class'])['confidence'].mean().reset_index()
    avg_conf_by_error['error_type'] = avg_conf_by_error['true_class'] + ' → ' + avg_conf_by_error['predicted_class']
    top_conf_errors = avg_conf_by_error.nlargest(10, 'confidence')
    
    axes[1, 1].barh(range(len(top_conf_errors)), top_conf_errors['confidence'])
    axes[1, 1].set_yticks(range(len(top_conf_errors)))
    axes[1, 1].set_yticklabels(top_conf_errors['error_type'])
    axes[1, 1].set_xlabel('Average Confidence')
    axes[1, 1].set_title('High-Confidence Errors (Most Concerning)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle('Detailed Error Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Error analysis saved to: {save_path}")
    
    plt.show()

def generate_comprehensive_report(y_true, y_pred, all_probabilities, classes, save_path=None):
    """Generate comprehensive classification report"""
    
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True)
    
    # Convert to DataFrame for better visualization
    report_df = pd.DataFrame(report).transpose()
    
    # Add additional metrics
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for i, class_name in enumerate(classes):
        y_binary = (y_true == i).astype(int)
        y_scores = all_probabilities[:, i]
        
        precision_scores.append(precision_score(y_binary, (y_scores > 0.5).astype(int), zero_division=0))
        recall_scores.append(recall_score(y_binary, (y_scores > 0.5).astype(int), zero_division=0))
        f1_scores.append(f1_score(y_binary, (y_scores > 0.5).astype(int), zero_division=0))
    
    # Create detailed report
    detailed_report = pd.DataFrame({
        'Class': classes,
        'Precision': precision_scores,
        'Recall': recall_scores,
        'F1-Score': f1_scores,
        'Support': [np.sum(y_true == i) for i in range(len(classes))]
    })
    
    print("\n" + "="*80)
    print("📊 COMPREHENSIVE CLASSIFICATION REPORT")
    print("="*80)
    print(detailed_report.to_string(index=False))
    
    if save_path:
        detailed_report.to_csv(save_path, index=False)
        print(f"📊 Detailed report saved to: {save_path}")
    
    return detailed_report

def main():
    parser = argparse.ArgumentParser(description="Generate comprehensive model evaluation")
    parser.add_argument("--model", default="models/rafdb_expressions_active.onnx")
    parser.add_argument("--classes", default="models/rafdb_expressions_active.classes.txt")
    parser.add_argument("--test-dir", default="data/rafdb/test")
    parser.add_argument("--output-dir", default="evaluation_results")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=100)
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🎯 Comprehensive Model Evaluation")
    print("="*50)
    print(f"Model: {args.model}")
    print(f"Classes: {args.classes}")
    print(f"Test Directory: {args.test_dir}")
    print(f"Output Directory: {args.output_dir}")
    print("="*50)
    
    try:
        # Load model and classes
        session, classes = load_model_and_classes(args.model, args.classes)
        print(f"✅ Loaded model with {len(classes)} classes: {classes}")
        
        # Comprehensive evaluation
        y_true, y_pred, confidences, all_probabilities = evaluate_model_comprehensive(
            args.test_dir, session, classes, args.img_size, args.max_samples
        )
        
        # Generate all visualizations
        print("\n📊 Generating visualizations...")
        
        # 1. Confidence distribution
        plot_confidence_distribution(
            y_true, y_pred, confidences, classes,
            os.path.join(args.output_dir, "confidence_distribution.png")
        )
        
        # 2. Precision-Recall curves
        plot_precision_recall_curves(
            y_true, all_probabilities, classes,
            os.path.join(args.output_dir, "precision_recall_curves.png")
        )
        
        # 3. ROC curves
        plot_roc_curves(
            y_true, all_probabilities, classes,
            os.path.join(args.output_dir, "roc_curves.png")
        )
        
        # 4. Class imbalance analysis
        class_counts = plot_class_imbalance_analysis(
            args.test_dir, classes,
            os.path.join(args.output_dir, "class_imbalance.png")
        )
        
        # 5. Error analysis
        plot_error_analysis(
            y_true, y_pred, confidences, classes,
            os.path.join(args.output_dir, "error_analysis.png")
        )
        
        # 6. Comprehensive report
        detailed_report = generate_comprehensive_report(
            y_true, y_pred, all_probabilities, classes,
            os.path.join(args.output_dir, "detailed_report.csv")
        )
        
        print(f"\n🎉 Comprehensive evaluation complete!")
        print(f"📁 All results saved to: {args.output_dir}")
        print(f"📊 Overall Accuracy: {np.mean(y_true == y_pred):.3f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
