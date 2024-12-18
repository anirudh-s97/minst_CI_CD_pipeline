import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MNISTClassifier, count_parameters
from datetime import datetime
import os
import sys

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the augmentation utilities
from augmentation import save_augmented_images, get_train_transforms

def train_model():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Define transformations with augmentations
    transform = get_train_transforms()
    
    # Download MNIST dataset
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    
    # Save augmented images for validation
    save_augmented_images(train_dataset, transform.transforms[-1])
    
    # Create DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True
    )
    
    # Initialize model, loss, and optimizer
    model = MNISTClassifier()
    
    # Print parameter count
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop (1 epoch)
    model.train()
    best_accuracy = 0
    for epoch in range(1):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}, Loss: {total_loss/len(train_loader)}, Accuracy: {accuracy}%')
        
        # Update best accuracy
        best_accuracy = max(best_accuracy, accuracy)
    
    # Verify accuracy meets requirements
    assert best_accuracy > 95, f"Model accuracy {best_accuracy}% should be > 95%"
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Add timestamp to model filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'models/mnist_model_{timestamp}.pth'
    
    # Save the model
    torch.save(model.state_dict(), model_path)
    print(f'Model saved to {model_path}')
    
    return model, best_accuracy

if __name__ == "__main__":
    train_model()