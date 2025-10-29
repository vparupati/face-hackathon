#!/usr/bin/env python3
"""
Train RAF-DB model with real-time progress monitoring.
"""

import subprocess
import sys
import time
import os
from datetime import datetime

def run_training_with_progress():
    """Run training with real-time progress monitoring"""
    
    print("🚀 RAF-DB Training with Real-time Progress")
    print("=" * 60)
    print("Starting training...")
    print("Progress will be shown every 30% completion")
    print("Press Ctrl+C to stop monitoring (training continues in background)")
    print("=" * 60)
    
    # Generate versioned filename with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_name = f"models/rafdb_expressions_v{timestamp}.onnx"
    
    print(f"📁 Saving model as: {model_name}")
    
    # Start training in background
    cmd = [
        "python3", "scripts/train_rafdb_expressions.py",
        "--epochs", "50",
        "--batch", "64", 
        "--img", "100",
        "--lr", "1e-3",
        "--out", model_name
    ]
    
    # Start the training process
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    # Monitor progress
    last_reported_epoch = 0
    start_time = time.time()
    
    try:
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            
            # Look for epoch completion
            if "Epoch" in line and "val_acc=" in line:
                try:
                    # Extract epoch number and accuracy
                    parts = line.split()
                    epoch_idx = parts.index("Epoch")
                    epoch_num = int(parts[epoch_idx + 1].replace(":", ""))
                    
                    acc_idx = parts.index("val_acc=")
                    accuracy = float(parts[acc_idx + 1])
                    
                    # Check if we've reached a 30% milestone
                    progress_percent = (epoch_num / 50) * 100
                    milestone = int(progress_percent // 30) * 30
                    
                    if milestone >= 30 and epoch_num > last_reported_epoch:
                        elapsed = time.time() - start_time
                        elapsed_min = elapsed / 60
                        
                        print(f"\n📊 Progress Update - {milestone}% Complete")
                        print(f"   Epoch: {epoch_num}/50")
                        print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
                        print(f"   Elapsed: {elapsed_min:.1f} minutes")
                        print(f"   Time: {datetime.now().strftime('%H:%M:%S')}")
                        
                        # Estimate remaining time
                        if progress_percent > 0:
                            estimated_total = elapsed / (progress_percent / 100)
                            remaining = estimated_total - elapsed
                            remaining_min = remaining / 60
                            print(f"   Estimated remaining: {remaining_min:.1f} minutes")
                        
                        print("-" * 50)
                        last_reported_epoch = epoch_num
                
                except (ValueError, IndexError):
                    pass
            
            # Show all training output
            print(line)
            
            # Check if training completed
            if "Training completed!" in line:
                print(f"\n🎉 Training Complete!")
                print(f"   Total Time: {(time.time() - start_time)/60:.1f} minutes")
                break
                
    except KeyboardInterrupt:
        print(f"\n⏹️  Monitoring stopped by user")
        print(f"   Training continues in background (PID: {process.pid})")
        print(f"   Check progress with: tail -f rafdb_training.log")
        return process.pid
    
    # Wait for process to complete
    process.wait()
    return None

if __name__ == "__main__":
    pid = run_training_with_progress()
    if pid:
        print(f"\n💡 To check progress later:")
        print(f"   tail -f rafdb_training.log")
        print(f"   ps aux | grep {pid}")
    else:
        print(f"\n✅ Training completed successfully!")
