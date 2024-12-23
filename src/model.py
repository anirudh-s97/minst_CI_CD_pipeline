import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        
        # Block 1: Stronger initial feature extraction
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),    # 144 + 16 = 160 params
            nn.BatchNorm2d(16),                            # 32 params
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),   # 2,304 + 16 = 2,320 params
            nn.BatchNorm2d(16),                            # 32 params
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                            # 28x28 -> 14x14
        )
        
        # Block 2: Maintain width but add depth
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),   # 4,608 + 32 = 4,640 params
            nn.BatchNorm2d(32),                            # 64 params
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=3),             # 512 + 16 = 528 params (1x1 conv for channel reduction)
            nn.BatchNorm2d(16),                            # 32 params
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                            # 14x14 -> 7x7
        )
        
        # Block 3: Efficient feature refinement
        self.block3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),   # 3,456 + 24 = 3,480 params
            nn.BatchNorm2d(32),                            # 48 params
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                            # 7x7 -> 3x3
        )
        
        # More efficient classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # Reduced dropout for better training
            nn.Linear(32 * 3 * 3, 24),                     # 10,368 + 48 = 10,416 params
            nn.ReLU(),
            nn.Linear(24, 10)                              # 480 + 10 = 490 params
        )
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(-1, 32 * 3 * 3)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = MNISTClassifier()
    
    print("\nParameter count per layer:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: {param.numel():,}")
    
    total_params = count_parameters(model)
    print(f"\nTotal trainable parameters: {total_params:,}")
    
    dummy_input = torch.randn(1, 1, 28, 28)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")