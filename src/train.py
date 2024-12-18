import os
import datetime
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from src.model import create_mnist_model

def load_mnist_data():
    """Load and preprocess MNIST dataset"""
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Reshape and normalize
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    
    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

def train_model():
    """Train MNIST model and save with timestamp"""
    # Load data
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    
    # Create model
    model = create_mnist_model()
    
    # Train model
    model.fit(x_train, y_train, epochs=1, validation_split=0.2, verbose=1)
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy * 100:.2f}%")
    
    # Create models directory if not exists
    os.makedirs('models', exist_ok=True)
    
    # Generate timestamp for model filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f'models/mnist_model_{timestamp}.h5'
    
    # Save model
    model.save(model_filename)
    print(f"Model saved to {model_filename}")
    
    return model, test_accuracy

if __name__ == "__main__":
    train_model()
