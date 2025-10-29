#!/usr/bin/env python3
"""
Script to demonstrate and explain confidence calculation methods in our expression recognition system.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def explain_confidence_calculation():
    """Explain how confidence is calculated in our system"""
    
    print("🎯 CONFIDENCE CALCULATION EXPLAINED")
    print("="*60)
    
    print("\n📊 STEP-BY-STEP PROCESS:")
    print("-"*40)
    
    # Step 1: Model Output (Logits)
    print("1️⃣ MODEL OUTPUT (LOGITS):")
    print("   • Model produces raw scores (logits) for each class")
    print("   • Example logits: [2.1, -0.5, 1.8, 0.2, -1.1, 3.2, 0.8]")
    
    # Simulate example logits
    example_logits = np.array([2.1, -0.5, 1.8, 0.2, -1.1, 3.2, 0.8])
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    print(f"   • Logits: {example_logits}")
    print(f"   • Classes: {classes}")
    
    # Step 2: Softmax Calculation
    print("\n2️⃣ SOFTMAX CALCULATION:")
    print("   • Convert logits to probabilities using softmax")
    print("   • Formula: P(class_i) = exp(logit_i) / sum(exp(all_logits))")
    
    # Calculate softmax manually
    exp_logits = np.exp(example_logits - example_logits.max())  # Numerical stability
    probabilities = exp_logits / exp_logits.sum()
    
    print(f"   • exp(logits): {exp_logits}")
    print(f"   • Probabilities: {probabilities}")
    
    # Step 3: Confidence Selection
    print("\n3️⃣ CONFIDENCE SELECTION:")
    print("   • Confidence = probability of predicted class")
    print("   • Predicted class = argmax(probabilities)")
    
    predicted_class_idx = np.argmax(probabilities)
    confidence = probabilities[predicted_class_idx]
    
    print(f"   • Predicted class: {classes[predicted_class_idx]} (index {predicted_class_idx})")
    print(f"   • Confidence: {confidence:.3f}")
    
    # Step 4: Complete Example
    print("\n4️⃣ COMPLETE EXAMPLE:")
    print("-"*40)
    for i, (class_name, prob) in enumerate(zip(classes, probabilities)):
        marker = "👑" if i == predicted_class_idx else "  "
        print(f"   {marker} {class_name:<10}: {prob:.3f}")
    
    return example_logits, probabilities, predicted_class_idx, confidence

def demonstrate_confidence_scenarios():
    """Demonstrate different confidence scenarios"""
    
    print("\n\n🎭 CONFIDENCE SCENARIOS")
    print("="*60)
    
    scenarios = [
        {
            "name": "High Confidence",
            "logits": [0.1, -2.0, 0.5, -1.0, -0.5, 4.5, 0.2],
            "description": "Clear winner - model is very sure"
        },
        {
            "name": "Medium Confidence", 
            "logits": [1.2, 0.8, 0.5, 1.5, 0.2, 2.1, 0.9],
            "description": "Moderate certainty - some competition"
        },
        {
            "name": "Low Confidence",
            "logits": [0.8, 0.7, 0.9, 0.6, 0.5, 1.1, 0.8],
            "description": "Uncertain - multiple classes similar"
        },
        {
            "name": "Very Low Confidence",
            "logits": [0.1, 0.2, 0.0, 0.3, 0.1, 0.4, 0.2],
            "description": "Very uncertain - model struggles"
        }
    ]
    
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
    
    for scenario in scenarios:
        print(f"\n📊 {scenario['name']}:")
        print(f"   Description: {scenario['description']}")
        
        logits = np.array(scenario['logits'])
        exp_logits = np.exp(logits - logits.max())
        probabilities = exp_logits / exp_logits.sum()
        
        predicted_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_idx]
        
        print(f"   Logits: {logits}")
        print(f"   Predicted: {classes[predicted_idx]} (confidence: {confidence:.3f})")
        
        # Show top 3 predictions
        top3_indices = np.argsort(probabilities)[-3:][::-1]
        print(f"   Top 3 predictions:")
        for i, idx in enumerate(top3_indices):
            print(f"     {i+1}. {classes[idx]}: {probabilities[idx]:.3f}")

def explain_confidence_interpretation():
    """Explain how to interpret confidence scores"""
    
    print("\n\n📈 CONFIDENCE INTERPRETATION")
    print("="*60)
    
    print("🎯 CONFIDENCE RANGES:")
    print("-"*30)
    print("• 0.90 - 1.00: Very High Confidence (Model is very sure)")
    print("• 0.70 - 0.89: High Confidence (Model is confident)")
    print("• 0.50 - 0.69: Medium Confidence (Model is somewhat sure)")
    print("• 0.30 - 0.49: Low Confidence (Model is uncertain)")
    print("• 0.00 - 0.29: Very Low Confidence (Model is very uncertain)")
    
    print("\n🔍 WHAT CONFIDENCE MEANS:")
    print("-"*30)
    print("• Confidence = Probability that the prediction is correct")
    print("• Based on model's internal uncertainty")
    print("• Higher confidence = model has seen similar patterns")
    print("• Lower confidence = model encounters unfamiliar patterns")
    
    print("\n⚠️ CONFIDENCE LIMITATIONS:")
    print("-"*30)
    print("• Not calibrated to true accuracy")
    print("• Model can be confidently wrong")
    print("• Depends on training data distribution")
    print("• May not reflect real-world uncertainty")

def demonstrate_confidence_vs_accuracy():
    """Show relationship between confidence and accuracy"""
    
    print("\n\n📊 CONFIDENCE vs ACCURACY ANALYSIS")
    print("="*60)
    
    print("🔍 FROM OUR CONFUSION MATRIX RESULTS:")
    print("-"*40)
    
    # Real data from our confusion matrix
    class_data = [
        {"class": "neutral", "accuracy": 0.996, "avg_confidence": 0.991, "samples": 1027},
        {"class": "surprise", "accuracy": 0.809, "avg_confidence": 0.778, "samples": 324},
        {"class": "disgust", "accuracy": 0.778, "avg_confidence": 0.680, "samples": 176},
        {"class": "sad", "accuracy": 0.730, "avg_confidence": 0.661, "samples": 492},
        {"class": "happy", "accuracy": 0.669, "avg_confidence": 0.722, "samples": 1192},
        {"class": "fear", "accuracy": 0.606, "avg_confidence": 0.767, "samples": 71},
        {"class": "angry", "accuracy": 0.394, "avg_confidence": 0.572, "samples": 815}
    ]
    
    print(f"{'Class':<10} {'Accuracy':<10} {'Confidence':<12} {'Samples':<8} {'Calibration'}")
    print("-"*60)
    
    for data in class_data:
        calibration = "Well" if abs(data['accuracy'] - data['avg_confidence']) < 0.1 else "Poor"
        print(f"{data['class']:<10} {data['accuracy']:<10.3f} {data['avg_confidence']:<12.3f} "
              f"{data['samples']:<8} {calibration}")
    
    print("\n💡 KEY INSIGHTS:")
    print("-"*20)
    print("• Neutral: Well-calibrated (99.6% accuracy, 99.1% confidence)")
    print("• Angry: Poorly calibrated (39.4% accuracy, 57.2% confidence)")
    print("• Fear: Overconfident (60.6% accuracy, 76.7% confidence)")
    print("• Happy: Well-calibrated (66.9% accuracy, 72.2% confidence)")

def show_confidence_calculation_code():
    """Show the actual code used for confidence calculation"""
    
    print("\n\n💻 CONFIDENCE CALCULATION CODE")
    print("="*60)
    
    print("🐍 PYTHON IMPLEMENTATION:")
    print("-"*30)
    
    code = '''
# Step 1: Get model output (logits)
logits = session.run(None, {"image": img})[0]

# Step 2: Convert logits to probabilities (softmax)
# Numerical stability: subtract max to prevent overflow
probs = np.exp(logits - logits.max(axis=1, keepdims=True))
probs = probs / probs.sum(axis=1, keepdims=True)

# Step 3: Get predicted class and confidence
predicted_class_idx = np.argmax(probs, axis=1)[0]
confidence = probs[0, predicted_class_idx]

# Step 4: Return results
return {
    "expression": classes[predicted_class_idx],
    "confidence": float(confidence),
    "probs": {classes[i]: float(probs[0,i]) for i in range(len(classes))}
}
'''
    
    print(code)
    
    print("🔧 KEY TECHNICAL DETAILS:")
    print("-"*30)
    print("• Numerical Stability: Subtract max logit before exp()")
    print("• Softmax Normalization: Divide by sum to get probabilities")
    print("• Confidence = Max probability (highest softmax value)")
    print("• All probabilities sum to 1.0")

def main():
    """Main function to run all explanations"""
    
    # Run all explanation sections
    explain_confidence_calculation()
    demonstrate_confidence_scenarios()
    explain_confidence_interpretation()
    demonstrate_confidence_vs_accuracy()
    show_confidence_calculation_code()
    
    print("\n\n🎉 CONFIDENCE CALCULATION SUMMARY")
    print("="*60)
    print("✅ Confidence = Softmax probability of predicted class")
    print("✅ Range: 0.0 to 1.0 (higher = more confident)")
    print("✅ Based on model's internal uncertainty")
    print("✅ Used for decision making and user feedback")
    print("✅ Not always calibrated to true accuracy")

if __name__ == "__main__":
    main()
