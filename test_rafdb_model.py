#!/usr/bin/env python3
"""
Test the RAF-DB model to ensure it's working correctly.
"""

import requests
import io
from PIL import Image
import numpy as np

def test_rafdb_model():
    """Test the RAF-DB model with a sample image"""
    
    # Create a test image (100x100 RGB)
    test_image = Image.new('RGB', (100, 100), color='red')
    
    # Save to bytes
    img_bytes = io.BytesIO()
    test_image.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    # Test the API
    try:
        print("🧪 Testing RAF-DB Model...")
        print("=" * 40)
        
        files = {"image": ("test.jpg", img_bytes, "image/jpeg")}
        response = requests.post("http://localhost:8000/expression", files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API Response Successful!")
            print(f"   Expression: {result.get('expression', 'N/A')}")
            print(f"   Confidence: {result.get('confidence', 'N/A'):.3f}")
            print(f"   Model: {result.get('model', 'N/A')}")
            print(f"   Probabilities: {result.get('probs', {})}")
            return True
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_rafdb_model()
    if success:
        print("\n🎉 RAF-DB model is working correctly!")
    else:
        print("\n❌ RAF-DB model test failed!")
