#!/usr/bin/env python3
"""
Monitor RAF-DB training progress and show updates every 30% completion.
"""

import time
import os
import re
from datetime import datetime

def parse_training_log(log_file):
    """Parse training log to extract progress information"""
    if not os.path.exists(log_file):
        return None, None, None, None
    
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Extract epoch information
    epoch_matches = re.findall(r'Epoch\s+(\d+):.*?val_acc=([\d.]+)', content)
    
    if not epoch_matches:
        return None, None, None, None
    
    # Get latest epoch and accuracy
    latest_epoch = int(epoch_matches[-1][0])
    latest_accuracy = float(epoch_matches[-1][1])
    
    # Extract total epochs
    total_epochs_match = re.search(r'Starting training for (\d+) epochs', content)
    total_epochs = int(total_epochs_match.group(1)) if total_epochs_match else 50
    
    # Calculate progress percentage
    progress_percent = (latest_epoch / total_epochs) * 100
    
    return latest_epoch, total_epochs, latest_accuracy, progress_percent

def monitor_training():
    """Monitor training progress and show updates"""
    log_file = "rafdb_training.log"
    
    print("🚀 RAF-DB Training Progress Monitor")
    print("=" * 50)
    print("Monitoring: rafdb_training.log")
    print("Updates every 30% completion")
    print("=" * 50)
    
    last_reported_percent = 0
    start_time = time.time()
    
    while True:
        try:
            epoch, total_epochs, accuracy, progress_percent = parse_training_log(log_file)
            
            if epoch is None:
                print(f"⏳ Waiting for training to start... ({datetime.now().strftime('%H:%M:%S')})")
                time.sleep(10)
                continue
            
            # Check if we've reached a 30% milestone
            current_milestone = int(progress_percent // 30) * 30
            
            if current_milestone > last_reported_percent and current_milestone >= 30:
                elapsed_time = time.time() - start_time
                elapsed_minutes = elapsed_time / 60
                
                print(f"\n📊 Progress Update - {current_milestone}% Complete")
                print(f"   Epoch: {epoch}/{total_epochs}")
                print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
                print(f"   Elapsed: {elapsed_minutes:.1f} minutes")
                print(f"   Time: {datetime.now().strftime('%H:%M:%S')}")
                
                # Estimate remaining time
                if progress_percent > 0:
                    estimated_total = elapsed_time / (progress_percent / 100)
                    remaining_time = estimated_total - elapsed_time
                    remaining_minutes = remaining_time / 60
                    print(f"   Estimated remaining: {remaining_minutes:.1f} minutes")
                
                print("-" * 50)
                last_reported_percent = current_milestone
            
            # Check if training is complete
            if epoch >= total_epochs:
                print(f"\n🎉 Training Complete!")
                print(f"   Final Epoch: {epoch}/{total_epochs}")
                print(f"   Final Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
                print(f"   Total Time: {(time.time() - start_time)/60:.1f} minutes")
                break
            
            time.sleep(30)  # Check every 30 seconds
            
        except KeyboardInterrupt:
            print("\n⏹️  Monitoring stopped by user")
            break
        except Exception as e:
            print(f"❌ Error monitoring: {e}")
            time.sleep(30)

if __name__ == "__main__":
    monitor_training()
