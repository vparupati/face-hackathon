#!/usr/bin/env python3
"""
Test script for C3 Expression CNN model.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_c3_model():
    """Test the C3 Expression CNN model."""
    try:
        from models.c3_expression import C3ExpressionCNN, count_parameters
        import torch
        
        print("Testing C3 Expression CNN...")
        
        # Create model
        model = C3ExpressionCNN(num_classes=7, input_size=160)
        param_count = count_parameters(model)
        print(f"✓ C3 Expression CNN created successfully")
        print(f"✓ Total parameters: {param_count:,}")
        
        # Test forward pass
        x = torch.randn(2, 3, 160, 160)
        output = model(x)
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output.shape}")
        
        # Test predictions
        probs = model.predict_proba(x)
        preds = model.predict(x)
        print(f"✓ Predictions successful")
        print(f"  Probabilities shape: {probs.shape}")
        print(f"  Predictions: {preds}")
        print(f"  Probability sums: {probs.sum(dim=1)}")
        
        # Test feature maps
        features = model.get_feature_maps(x)
        print(f"✓ Feature extraction successful")
        for name, feat in features.items():
            print(f"  {name} shape: {feat.shape}")
        
        print("\n🎉 All tests passed! C3 Expression CNN is working correctly.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure PyTorch is installed: pip install torch torchvision")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_c3_model()
    sys.exit(0 if success else 1)
