#!/usr/bin/env python3
"""
Monitor the current training process in real-time.
"""

import time
import os
import subprocess
from datetime import datetime

def get_process_info(pid):
    """Get process information"""
    try:
        result = subprocess.run(['ps', '-p', str(pid), '-o', 'pid,etime,pcpu,pmem,command'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if len(lines) > 1:
                return lines[1].split()
        return None
    except:
        return None

def check_model_files():
    """Check for new model files"""
    models = []
    for file in os.listdir('models'):
        if file.startswith('rafdb_expressions_v') and file.endswith('.onnx'):
            models.append(file)
    return sorted(models, reverse=True)

def monitor_training(pid=70383):
    """Monitor the training process"""
    print("🚀 Training Progress Monitor")
    print("=" * 50)
    print(f"Monitoring Process ID: {pid}")
    print("Press Ctrl+C to stop monitoring")
    print("=" * 50)
    
    start_time = time.time()
    last_check = 0
    
    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Check if process is still running
            process_info = get_process_info(pid)
            if not process_info:
                print(f"\n❌ Process {pid} not found - training may have completed!")
                break
            
            # Check for new model files
            models = check_model_files()
            
            # Display status every 30 seconds
            if current_time - last_check >= 30:
                print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')} - Elapsed: {elapsed/60:.1f} minutes")
                print(f"🔄 Process: {'Running' if process_info else 'Stopped'}")
                
                if process_info:
                    print(f"   CPU: {process_info[2]}% | Memory: {process_info[3]}%")
                    print(f"   Runtime: {process_info[1]}")
                
                if models:
                    print(f"✅ ONNX Models Found: {len(models)}")
                    for model in models[:3]:  # Show latest 3
                        size = os.path.getsize(f'models/{model}') / (1024*1024)
                        mod_time = datetime.fromtimestamp(os.path.getmtime(f'models/{model}'))
                        print(f"   📁 {model} ({size:.1f}MB) - {mod_time.strftime('%H:%M:%S')}")
                else:
                    print("⏳ ONNX export in progress...")
                
                print("-" * 50)
                last_check = current_time
            
            time.sleep(5)  # Check every 5 seconds
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Monitoring stopped by user")
        print(f"💡 Check manually with: ls -la models/rafdb_expressions_v*.onnx")

if __name__ == "__main__":
    monitor_training()
