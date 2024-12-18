import tensorflow as tf
import numpy as np
from src.model import create_mnist_model

def test_model_architecture():
    """
    Test key characteristics of the model
    """
    model = create_mnist_model()
    
    # 1. Check total parameters
    total_params = model.count_params()
    print(f"Total parameters: {total_params}")
    assert total_params < 25000, f"Model has too many parameters: {total_params}"
    
    # 2. Check input shape
    input_shape = model.input_shape
    assert input_shape == (None, 28, 28, 1), f"Incorrect input shape: {input_shape}"
    
    # 3. Check output shape
    output_shape = model.output_shape
    assert output_shape == (None, 10), f"Incorrect output shape: {output_shape}"
    
    # 4. Train and check accuracy
    from tensorflow.keras.datasets import mnist
    from tensorflow.keras.utils import to_categorical
    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1).astype('float32') / 255
    x_test = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255
    
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    model.fit(x_train, y_train, epochs=1, verbose=0)
    
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    assert accuracy > 0.90, f"Accuracy too low: {accuracy}"
    
    print("All model tests passed successfully!")

if __name__ == "__main__":
    test_model_architecture()
