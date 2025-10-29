#!/usr/bin/env python3
"""
Model management script for versioned models.
"""

import os
import glob
import argparse
from datetime import datetime

def list_models():
    """List all available model versions"""
    print("📋 Available Model Versions:")
    print("=" * 50)
    
    # Find all RAF-DB models
    rafdb_models = glob.glob("models/rafdb_expressions_v*.onnx")
    rafdb_models.sort(reverse=True)  # Newest first
    
    if not rafdb_models:
        print("❌ No RAF-DB models found")
        return
    
    for i, model in enumerate(rafdb_models, 1):
        # Extract version info
        basename = os.path.basename(model)
        version = basename.replace("rafdb_expressions_", "").replace(".onnx", "")
        
        # Get file info
        stat = os.stat(model)
        size_mb = stat.st_size / (1024 * 1024)
        mod_time = datetime.fromtimestamp(stat.st_mtime)
        
        print(f"{i:2d}. {basename}")
        print(f"    Version: {version}")
        print(f"    Size: {size_mb:.1f} MB")
        print(f"    Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print()

def compare_models():
    """Compare model accuracies from training logs"""
    print("📊 Model Performance Comparison:")
    print("=" * 50)
    
    # Find all training logs
    log_files = glob.glob("*training.log")
    log_files.sort(reverse=True)
    
    if not log_files:
        print("❌ No training logs found")
        return
    
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                content = f.read()
            
            # Extract final accuracy
            if "Best validation accuracy:" in content:
                acc_line = [line for line in content.split('\n') if "Best validation accuracy:" in line][-1]
                accuracy = float(acc_line.split(":")[-1].strip())
                
                # Extract model info
                if "rafdb_expressions_v" in content:
                    model_match = [line for line in content.split('\n') if "rafdb_expressions_v" in line and ".onnx" in line]
                    if model_match:
                        model_name = model_match[0].split()[-1]
                    else:
                        model_name = "Unknown"
                else:
                    model_name = "Original"
                
                print(f"📈 {model_name}")
                print(f"   Accuracy: {accuracy:.3f} ({accuracy*100:.1f}%)")
                print(f"   Log: {log_file}")
                print()
        except Exception as e:
            print(f"❌ Error reading {log_file}: {e}")

def set_active_model(model_path):
    """Set a model as the active one for the API"""
    if not os.path.exists(model_path):
        print(f"❌ Model not found: {model_path}")
        return
    
    # Create symlink to the active model
    active_path = "models/rafdb_expressions_active.onnx"
    active_classes = "models/rafdb_expressions_active.classes.txt"
    
    try:
        # Remove existing symlink
        if os.path.exists(active_path):
            os.remove(active_path)
        if os.path.exists(active_classes):
            os.remove(active_classes)
        
        # Create new symlinks
        os.symlink(os.path.basename(model_path), active_path)
        
        # Also link the classes file
        classes_path = model_path.replace(".onnx", ".classes.txt")
        if os.path.exists(classes_path):
            os.symlink(os.path.basename(classes_path), active_classes)
        
        print(f"✅ Set active model: {model_path}")
        print(f"   API will now use: {active_path}")
        
    except Exception as e:
        print(f"❌ Error setting active model: {e}")

def cleanup_old_models(keep=5):
    """Keep only the latest N models, delete the rest"""
    print(f"🧹 Cleaning up old models (keeping latest {keep})...")
    
    # Find all RAF-DB models
    rafdb_models = glob.glob("models/rafdb_expressions_v*.onnx")
    rafdb_models.sort(key=os.path.getmtime, reverse=True)  # Newest first
    
    if len(rafdb_models) <= keep:
        print(f"✅ Only {len(rafdb_models)} models found, no cleanup needed")
        return
    
    # Delete old models
    to_delete = rafdb_models[keep:]
    for model in to_delete:
        try:
            # Delete model and classes file
            os.remove(model)
            classes_file = model.replace(".onnx", ".classes.txt")
            if os.path.exists(classes_file):
                os.remove(classes_file)
            print(f"🗑️  Deleted: {os.path.basename(model)}")
        except Exception as e:
            print(f"❌ Error deleting {model}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Manage model versions")
    parser.add_argument("--list", action="store_true", help="List all model versions")
    parser.add_argument("--compare", action="store_true", help="Compare model performance")
    parser.add_argument("--set-active", help="Set a model as active for API")
    parser.add_argument("--cleanup", type=int, metavar="N", help="Keep only latest N models")
    
    args = parser.parse_args()
    
    if args.list:
        list_models()
    elif args.compare:
        compare_models()
    elif args.set_active:
        set_active_model(args.set_active)
    elif args.cleanup:
        cleanup_old_models(args.cleanup)
    else:
        print("🔧 Model Management Tool")
        print("=" * 30)
        print("Usage:")
        print("  python3 scripts/manage_models.py --list")
        print("  python3 scripts/manage_models.py --compare")
        print("  python3 scripts/manage_models.py --set-active models/rafdb_expressions_v20241025_1430.onnx")
        print("  python3 scripts/manage_models.py --cleanup 5")

if __name__ == "__main__":
    main()
