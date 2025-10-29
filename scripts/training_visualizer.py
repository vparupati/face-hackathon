#!/usr/bin/env python3
"""
Script to visualize training progress from log files.
"""

import os
import sys
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import glob

def parse_training_log(log_file):
    """Parse training log file and extract metrics"""
    
    epochs = []
    train_losses = []
    val_accuracies = []
    learning_rates = []
    
    with open(log_file, 'r') as f:
        for line in f:
            # Parse epoch line: "Epoch 1: train_loss=0.1234 train_acc=0.567 val_loss=0.1234 val_acc=0.567 lr=1.23e-04"
            epoch_match = re.search(r'Epoch (\d+): train_loss=([\d.]+) train_acc=([\d.]+) val_loss=([\d.]+) val_acc=([\d.]+) lr=([\d.e-]+)', line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
                train_loss = float(epoch_match.group(2))
                train_acc = float(epoch_match.group(3))
                val_loss = float(epoch_match.group(4))
                val_acc = float(epoch_match.group(5))
                lr = float(epoch_match.group(6))
                
                epochs.append(epoch)
                train_losses.append(train_loss)
                val_accuracies.append(val_acc)
                learning_rates.append(lr)
    
    return {
        'epochs': epochs,
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'learning_rates': learning_rates
    }

def plot_training_progress(data, save_path=None):
    """Plot training progress metrics"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Training Loss
    axes[0, 0].plot(data['epochs'], data['train_losses'], 'b-', linewidth=2, label='Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # 2. Validation Accuracy
    axes[0, 1].plot(data['epochs'], data['val_accuracies'], 'g-', linewidth=2, label='Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Validation Accuracy Over Time')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Add best accuracy annotation
    best_acc_idx = np.argmax(data['val_accuracies'])
    best_acc = data['val_accuracies'][best_acc_idx]
    best_epoch = data['epochs'][best_acc_idx]
    axes[0, 1].annotate(f'Best: {best_acc:.3f} @ Epoch {best_epoch}',
                       xy=(best_epoch, best_acc), xytext=(best_epoch+2, best_acc-0.05),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       fontsize=10, color='red', fontweight='bold')
    
    # 3. Learning Rate Schedule
    axes[1, 0].plot(data['epochs'], data['learning_rates'], 'r-', linewidth=2, label='Learning Rate')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # 4. Loss vs Accuracy
    axes[1, 1].scatter(data['train_losses'], data['val_accuracies'], 
                      c=data['epochs'], cmap='viridis', alpha=0.7, s=50)
    axes[1, 1].set_xlabel('Training Loss')
    axes[1, 1].set_ylabel('Validation Accuracy')
    axes[1, 1].set_title('Training Loss vs Validation Accuracy')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
    cbar.set_label('Epoch')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training progress saved to: {save_path}")
    
    plt.show()

def plot_multiple_training_runs(log_files, save_path=None):
    """Plot multiple training runs for comparison"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(log_files)))
    
    for i, log_file in enumerate(log_files):
        data = parse_training_log(log_file)
        run_name = os.path.basename(log_file).replace('.log', '')
        
        # Plot validation accuracy
        axes[0].plot(data['epochs'], data['val_accuracies'], 
                    color=colors[i], linewidth=2, label=run_name)
        
        # Plot training loss
        axes[1].plot(data['epochs'], data['train_losses'], 
                    color=colors[i], linewidth=2, label=run_name)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Accuracy')
    axes[0].set_title('Validation Accuracy Comparison')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Training Loss')
    axes[1].set_title('Training Loss Comparison')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training comparison saved to: {save_path}")
    
    plt.show()

def print_training_summary(data, log_file):
    """Print training summary statistics"""
    
    print(f"\n📊 Training Summary: {os.path.basename(log_file)}")
    print("="*60)
    
    if not data['epochs']:
        print("❌ No training data found in log file!")
        return
    
    # Basic statistics
    total_epochs = len(data['epochs'])
    final_train_loss = data['train_losses'][-1]
    best_val_acc = max(data['val_accuracies'])
    best_epoch = data['epochs'][np.argmax(data['val_accuracies'])]
    final_lr = data['learning_rates'][-1]
    
    print(f"Total Epochs: {total_epochs}")
    print(f"Final Training Loss: {final_train_loss:.4f}")
    print(f"Best Validation Accuracy: {best_val_acc:.3f} (Epoch {best_epoch})")
    print(f"Final Learning Rate: {final_lr:.2e}")
    
    # Training progress
    if len(data['val_accuracies']) > 1:
        acc_improvement = data['val_accuracies'][-1] - data['val_accuracies'][0]
        print(f"Accuracy Improvement: {acc_improvement:+.3f}")
    
    # Learning rate range
    lr_range = max(data['learning_rates']) - min(data['learning_rates'])
    print(f"Learning Rate Range: {min(data['learning_rates']):.2e} - {max(data['learning_rates']):.2e}")
    
    # Convergence analysis
    if len(data['val_accuracies']) >= 10:
        recent_accs = data['val_accuracies'][-10:]
        acc_std = np.std(recent_accs)
        print(f"Recent Accuracy Stability (last 10 epochs): {acc_std:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Visualize training progress from log files")
    parser.add_argument("--log-file", help="Specific log file to analyze")
    parser.add_argument("--log-dir", default=".", help="Directory containing log files")
    parser.add_argument("--output", default="training_progress.png", help="Output plot path")
    parser.add_argument("--compare", action="store_true", help="Compare multiple training runs")
    
    args = parser.parse_args()
    
    print("📈 Training Progress Visualizer")
    print("="*40)
    
    try:
        if args.log_file:
            # Analyze single log file
            if not os.path.exists(args.log_file):
                print(f"❌ Log file not found: {args.log_file}")
                return
            
            print(f"📊 Analyzing: {args.log_file}")
            data = parse_training_log(args.log_file)
            
            if data['epochs']:
                plot_training_progress(data, args.output)
                print_training_summary(data, args.log_file)
            else:
                print("❌ No training data found in log file!")
        
        elif args.compare:
            # Compare multiple log files
            log_files = glob.glob(os.path.join(args.log_dir, "*.log"))
            
            if not log_files:
                print("❌ No log files found!")
                return
            
            print(f"📊 Found {len(log_files)} log files for comparison")
            plot_multiple_training_runs(log_files, args.output)
            
            # Print summary for each
            for log_file in log_files:
                data = parse_training_log(log_file)
                print_training_summary(data, log_file)
        
        else:
            # Find and analyze latest log file
            log_files = glob.glob(os.path.join(args.log_dir, "*.log"))
            
            if not log_files:
                print("❌ No log files found!")
                return
            
            # Use most recent log file
            latest_log = max(log_files, key=os.path.getmtime)
            print(f"📊 Analyzing latest log: {latest_log}")
            
            data = parse_training_log(latest_log)
            
            if data['epochs']:
                plot_training_progress(data, args.output)
                print_training_summary(data, latest_log)
            else:
                print("❌ No training data found in log file!")
        
        print(f"\n🎉 Analysis complete!")
        print(f"📁 Plot saved to: {args.output}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
