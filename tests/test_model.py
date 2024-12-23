import torch
import sys
import os
# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
from src.model import MNISTClassifier, count_parameters
from src.train import train_model

def test_model_architecture():
    """
    Comprehensive test of model architecture and constraints
    """
    model = MNISTClassifier()
    
    # Test input size and processing
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    
    # Detailed parameter analysis
    total_params = count_parameters(model)
    print("\n--- Model Parameter Analysis ---")
    print(f"Total trainable parameters: {total_params}")
    
    # Strict parameter constraint
    assert total_params < 25000, f"Model has {total_params} parameters (should be < 25000)"
    assert total_params > 0, "Model should have some parameters"
    
    # Verify output characteristics
    assert output.shape[1] == 10, "Model should have 10 output classes"
    print("Output shape verified: 10 classes")
    
    # Optional: Layer-wise parameter breakdown
    param_breakdown = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_breakdown[name] = param.numel()
    
    print("\nLayer-wise Parameter Breakdown:")
    for layer, params in param_breakdown.items():
        print(f"{layer}: {params} parameters")

def test_model_training_stability():
    """
    Test model training stability and accuracy requirements
    """
    _, final_accuracy = train_model()
    assert final_accuracy >= 95.0, f"Model accuracy {final_accuracy:.2f}% is below required 95%"
    print(f"\nFinal training accuracy: {final_accuracy:.2f}%")
    print("Training accuracy test: PASSED")

def main():
    test_model_architecture()
    test_model_training_stability()
    print("\n✅ All model tests passed successfully!")

if __name__ == "__main__":
    main()