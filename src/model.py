import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        # Reduced convolutional layers
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        
        # Reduced fully connected layers
        self.fc1 = nn.Linear(16 * 7 * 7, 64)  # Reduced from 128
        self.fc2 = nn.Linear(64, 10)  # Direct output layer
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.25)
    
    def forward(self, x):
        # Convolutional layers with max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def count_parameters(model):
    """
    Utility function to count total trainable parameters in a model
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Optional: For quick parameter verification
if __name__ == "__main__":
    model = MNISTClassifier()
    total_params = count_parameters(model)
    print(f"Total trainable parameters: {total_params}")