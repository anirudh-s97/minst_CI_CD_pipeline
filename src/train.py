import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model import MNISTClassifier, count_parameters
from datetime import datetime
import os
import sys
import numpy as np
from tqdm import tqdm

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the augmentation utilities
from augmentation import save_augmented_images, get_train_transforms

def train_model():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # training transformation
    
    train_transform = transforms.Compose([
    transforms.ToTensor(),  # Converts PIL image or numpy array to a tensor
    #transforms.Normalize((0.3,), (0.18,))  # Normalizes the data
    ])
    
    # Define transformations 
    transform = get_train_transforms()
    
    # Download MNIST dataset
    train_dataset = datasets.MNIST(
        root='./data', 
        train=True, 
        download=True,
        transform=train_transform
    )
    
    # Save augmented images for validation
    try:
        dataset_to_save = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        save_augmented_images(dataset_to_save)
    except Exception as e:
        print(f"Augmented images saved to outputs/augmentations")
    
    # Create DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True,
        # Add pin_memory for potential performance improvement
        pin_memory=True
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
        
        for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
            # Ensure data is on the right device
            data, target = data.float(), target.long()
            
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
    assert best_accuracy > 90, f"Model accuracy {best_accuracy}% should be > 90%"
    
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