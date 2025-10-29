#!/usr/bin/env python3
"""
Download and prepare RAF-DB dataset for facial expression recognition.
"""

import os
import sys
import argparse
from pathlib import Path
import requests
import zipfile
from PIL import Image
import pandas as pd
from tqdm import tqdm
import shutil

def download_file(url, filename, chunk_size=8192):
    """Download a file with progress bar"""
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=filename) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    print(f"Downloaded {filename}")

def setup_rafdb_dataset(data_dir="data/rafdb"):
    """Set up RAF-DB dataset structure"""
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")
    
    for split_dir in [train_dir, test_dir]:
        os.makedirs(split_dir, exist_ok=True)
        # Create emotion subdirectories
        emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        for emotion in emotions:
            os.makedirs(os.path.join(split_dir, emotion), exist_ok=True)
    
    print(f"Created RAF-DB directory structure in {data_dir}")
    return train_dir, test_dir

def download_rafdb_from_huggingface(data_dir="data/rafdb"):
    """Download RAF-DB from Hugging Face"""
    
    try:
        from datasets import load_dataset
        print("Loading RAF-DB from Hugging Face...")
        
        # Load the dataset
        dataset = load_dataset("deanngkl/raf-db-7emotions")
        
        # Map emotion labels to folder names
        emotion_map = {
            0: 'angry',
            1: 'disgust', 
            2: 'fear',
            3: 'happy',
            4: 'neutral',
            5: 'sad',
            6: 'surprise'
        }
        
        # Set up directories
        train_dir, test_dir = setup_rafdb_dataset(data_dir)
        
        # Process training set
        print("Processing training images...")
        train_data = dataset['train']
        for i, item in enumerate(tqdm(train_data, desc="Training")):
            image = item['image']
            label = item['label']
            emotion = emotion_map[label]
            
            # Save image to appropriate folder
            filename = f"train_{i:05d}.jpg"
            image_path = os.path.join(train_dir, emotion, filename)
            image.save(image_path, 'JPEG')
        
        # Process test set
        print("Processing test images...")
        test_data = dataset['test']
        for i, item in enumerate(tqdm(test_data, desc="Test")):
            image = item['image']
            label = item['label']
            emotion = emotion_map[label]
            
            # Save image to appropriate folder
            filename = f"test_{i:05d}.jpg"
            image_path = os.path.join(test_dir, emotion, filename)
            image.save(image_path, 'JPEG')
        
        print(f"✅ RAF-DB dataset downloaded and organized in {data_dir}")
        
        # Print statistics
        print("\n📊 Dataset Statistics:")
        for split in ['train', 'test']:
            split_dir = os.path.join(data_dir, split)
            total_images = 0
            for emotion in emotion_map.values():
                emotion_dir = os.path.join(split_dir, emotion)
                count = len([f for f in os.listdir(emotion_dir) if f.endswith('.jpg')])
                print(f"  {split}/{emotion}: {count} images")
                total_images += count
            print(f"  {split} total: {total_images} images")
        
        return True
        
    except ImportError:
        print("❌ Hugging Face datasets library not found.")
        print("Please install it with: pip install datasets")
        return False
    except Exception as e:
        print(f"❌ Error downloading RAF-DB: {e}")
        return False

def download_rafdb_manual(data_dir="data/rafdb"):
    """Manual download instructions for RAF-DB"""
    
    print("📋 Manual RAF-DB Download Instructions:")
    print("=" * 50)
    print("1. Visit: https://www.whdeng.cn/raf/model1.html")
    print("2. Register and request access to RAF-DB")
    print("3. Download the dataset (usually a ZIP file)")
    print("4. Extract to a temporary folder")
    print("5. Run this script with --manual-path pointing to extracted folder")
    print("=" * 50)
    
    return False

def process_manual_rafdb(manual_path, data_dir="data/rafdb"):
    """Process manually downloaded RAF-DB"""
    
    if not os.path.exists(manual_path):
        print(f"❌ Manual path not found: {manual_path}")
        return False
    
    print(f"Processing manual RAF-DB from: {manual_path}")
    
    # Set up directories
    train_dir, test_dir = setup_rafdb_dataset(data_dir)
    
    # Look for common RAF-DB file structures
    emotion_map = {
        0: 'angry',
        1: 'disgust', 
        2: 'fear',
        3: 'happy',
        4: 'neutral',
        5: 'sad',
        6: 'surprise'
    }
    
    # Try to find annotation files
    annotation_files = []
    for root, dirs, files in os.walk(manual_path):
        for file in files:
            if 'list' in file.lower() and file.endswith('.txt'):
                annotation_files.append(os.path.join(root, file))
    
    if not annotation_files:
        print("❌ No annotation files found. Please check the manual download.")
        return False
    
    print(f"Found annotation files: {annotation_files}")
    
    # Process each annotation file
    for ann_file in annotation_files:
        print(f"Processing {ann_file}...")
        
        with open(ann_file, 'r') as f:
            lines = f.readlines()
        
        for line in tqdm(lines, desc="Processing"):
            parts = line.strip().split()
            if len(parts) >= 2:
                image_path = parts[0]
                label = int(parts[1])
                
                if label in emotion_map:
                    emotion = emotion_map[label]
                    
                    # Determine if train or test (you may need to adjust this logic)
                    is_train = 'train' in ann_file.lower() or 'Train' in ann_file
                    target_dir = train_dir if is_train else test_dir
                    
                    # Copy image to appropriate folder
                    source_path = os.path.join(manual_path, image_path)
                    if os.path.exists(source_path):
                        filename = os.path.basename(image_path)
                        target_path = os.path.join(target_dir, emotion, filename)
                        
                        # Ensure target directory exists
                        os.makedirs(os.path.dirname(target_path), exist_ok=True)
                        
                        # Copy image
                        shutil.copy2(source_path, target_path)
    
    print(f"✅ RAF-DB processed and organized in {data_dir}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Download and prepare RAF-DB dataset")
    parser.add_argument("--data-dir", default="data/rafdb", help="Directory to save RAF-DB")
    parser.add_argument("--manual-path", help="Path to manually downloaded RAF-DB")
    parser.add_argument("--method", choices=["huggingface", "manual"], default="huggingface", 
                       help="Download method")
    
    args = parser.parse_args()
    
    print("🚀 RAF-DB Dataset Downloader")
    print("=" * 40)
    print(f"Target directory: {args.data_dir}")
    print(f"Method: {args.method}")
    print("=" * 40)
    
    if args.method == "huggingface":
        success = download_rafdb_from_huggingface(args.data_dir)
    elif args.method == "manual":
        if args.manual_path:
            success = process_manual_rafdb(args.manual_path, args.data_dir)
        else:
            success = download_rafdb_manual(args.data_dir)
    
    if success:
        print("\n🎉 RAF-DB dataset ready for training!")
        print(f"📁 Dataset location: {args.data_dir}")
        print("🚀 You can now train with: python3 scripts/train_resnet_expressions.py --data data/rafdb --img 100")
    else:
        print("\n❌ Failed to download RAF-DB dataset")
        print("💡 Try the manual method or check your internet connection")

if __name__ == "__main__":
    main()
