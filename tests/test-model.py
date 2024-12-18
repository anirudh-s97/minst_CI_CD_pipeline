import torch
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MNISTClassifier

def test_model_architecture():
    """
    Test model architecture and parameter constraints
    """
    model = MNISTClassifier()
    
    # Test input size
    test_input = torch.randn(1, 1, 28, 28)
    output = model(test_input)
    
    # Check input handling
    assert output.shape[1] == 10, "Model should have 10 output classes"
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    assert total_params < 25000, "Model should have less than 25000 parameters"

def test_model_training():
    """
    Perform a quick training test to check basic functionality
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    model = MNISTClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    
    # Create dummy data
    dummy_input = torch.randn(32, 1, 28, 28)
    dummy_target = torch.randint(0, 10, (32,))
    
    # Forward pass
    output = model(dummy_input)
    loss = criterion(output, dummy_target)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Ensure no catastrophic errors occur during training

def main():
    test_model_architecture()
    test_model_training()
    print("All tests passed successfully!")

if __name__ == "__main__":
    main()