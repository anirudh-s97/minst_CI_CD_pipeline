import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image

def get_augmentations():
    """
    Create a set of image augmentations for MNIST dataset
    """
    augmentations = transforms.Compose([
        # Ensure conversion to tensor first
        transforms.ToTensor(),
        
        # Custom augmentation using torchvision functional transforms
        transforms.Lambda(lambda x: F.rotate(x, angle=10)),
        transforms.Lambda(lambda x: F.affine(x, angle=0, translate=(0.1, 0.1), scale=1.1, shear=0)),
        
        # Normalize
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    return augmentations

def save_augmented_images(dataset, num_samples=9):
    """
    Apply augmentations and save sample images
    
    Args:
        dataset: Original dataset
        num_samples: Number of samples to save
    """
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs/augmentations', exist_ok=True)
    
    # Prepare figure
    fig, axes = plt.subplots(3, num_samples, figsize=(15, 6))
    fig.suptitle('Image Augmentation Comparison', fontsize=16)
    
    # Get a batch of images
    for i in range(min(num_samples, len(dataset))):
        # Get original image
        original_img, _ = dataset[i]
        
        # Convert tensor to numpy for plotting
        if isinstance(original_img, torch.Tensor):
            original_img = original_img.squeeze().numpy()
        elif isinstance(original_img, Image.Image):
            original_img = np.array(original_img)
        
        # Plot original image
        axes[0, i].imshow(original_img, cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, 0].set_title('Original')
        
        # Apply and plot augmentations
        try:
            # Rotation augmentation
            rotated = F.rotate(torch.from_numpy(original_img).unsqueeze(0), angle=10)
            axes[1, i].imshow(rotated.squeeze().numpy(), cmap='gray')
            axes[1, i].axis('off')
            if i == 0:
                axes[1, 0].set_title('Rotation')
            
            # Affine augmentation
            affined = F.affine(torch.from_numpy(original_img).unsqueeze(0), angle=0, translate=(2, 2), scale=1.1, shear=0)
            axes[2, i].imshow(affined.squeeze().numpy(), cmap='gray')
            axes[2, i].axis('off')
            if i == 0:
                axes[2, 0].set_title('Affine')
        
        except Exception as e:
            print(f"Error processing image {i}: {e}")
    
    # Save and close
    plt.tight_layout()
    plt.savefig('outputs/augmentations/mnist_augmentations.png')
    plt.close()
    
    print("Augmented images saved to outputs/augmentations/mnist_augmentations.png")

def get_train_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])