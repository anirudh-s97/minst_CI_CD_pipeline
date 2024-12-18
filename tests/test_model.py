import torch
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.model import MNISTClassifier, count_parameters

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
    Perform a quick training stability test
    """
    import torch.nn as nn
    import torch.optim as optim
    
    model = MNISTClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Create dummy data
    dummy_input = torch.randn(32, 1, 28, 28)
    dummy_target = torch.randint(0, 10, (32,))
    
    # Ensure no errors in forward and backward pass
    try:
        # Forward pass
        output = model(dummy_input)
        loss = criterion(output, dummy_target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print("\nTraining stability test: PASSED")
    except Exception as e:
        print(f"\nTraining stability test: FAILED\nError: {e}")
        raise

def main():
    test_model_architecture()
    test_model_training_stability()
    print("\n✅ All model tests passed successfully!")

if __name__ == "__main__":
    main()