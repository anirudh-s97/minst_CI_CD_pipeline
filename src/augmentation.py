import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
import os
import matplotlib.pyplot as plt

def get_augmentations():
    """
    Create a set of image augmentations for MNIST dataset
    """
    augmentations = transforms.Compose([
        # 1. Random Rotation
        transforms.RandomRotation(10, fill=0),
        
        # 2. Random Affine Transformation
        transforms.RandomAffine(
            degrees=0, 
            translate=(0.1, 0.1),  # 10% translation
            scale=(0.9, 1.1),       # Slight scaling
            fill=0
        ),
        
        # 3. Elastic Deformation (using random perspective as a simpler alternative)
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5, fill=0)
    ])
    
    return augmentations

def save_augmented_images(dataset, augmentations, num_samples=9):
    """
    Apply augmentations and save sample images
    
    Args:
        dataset: Original dataset
        augmentations: Augmentation transforms
        num_samples: Number of samples to save
    """
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs/augmentations', exist_ok=True)
    
    # Select first batch of images
    images, labels = next(iter(torch.utils.data.DataLoader(dataset, batch_size=num_samples)))
    
    # Create a figure to plot original and augmented images
    fig, axes = plt.subplots(3, num_samples, figsize=(15, 6))
    fig.suptitle('Image Augmentation Comparison', fontsize=16)
    
    for i in range(num_samples):
        # Original image
        original_img = images[i].squeeze()
        axes[0, i].imshow(original_img, cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, 0].set_title('Original')
        
        # Apply augmentations
        augmented_img = augmentations(images[i].unsqueeze(0)).squeeze()
        
        # Plot augmented images
        axes[1, i].imshow(augmented_img, cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, 0].set_title('Augmentation 1 (Rotation)')
        
        # Additional augmentation
        augmented_img2 = transforms.RandomAffine(
            degrees=0, 
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            fill=0
        )(images[i].unsqueeze(0)).squeeze()
        
        axes[2, i].imshow(augmented_img2, cmap='gray')
        axes[2, i].axis('off')
        if i == 0:
            axes[2, 0].set_title('Augmentation 2 (Affine)')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('outputs/augmentations/mnist_augmentations.png')
    plt.close()
    
    print("Augmented images saved to outputs/augmentations/mnist_augmentations.png")

# Modify the existing transforms in src/train.py to include these augmentations
def get_train_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        get_augmentations()  # Add the augmentations
    ])
