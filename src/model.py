import tensorflow as tf
from tensorflow.keras import layers, models

def create_mnist_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Create a 3-layer Deep Neural Network for MNIST classification
    
    Args:
        input_shape (tuple): Input image dimensions
        num_classes (int): Number of output classes
    
    Returns:
        tf.keras.Model: Compiled neural network model
    """
    model = models.Sequential([
        # Convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        
        # Flatten and fully connected layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
