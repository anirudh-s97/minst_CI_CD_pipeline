# MNIST Classification CI/CD Pipeline

## Project Overview
This project demonstrates a machine learning CI/CD pipeline for MNIST digit classification using PyTorch.

## Local Setup
1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running Tests
```bash
python -m tests.test_model
```

## Training the Model
```bash
python src/train.py
```

## GitHub Actions
This project includes a GitHub Actions workflow that:
- Installs dependencies
- Runs model tests
- Trains the model
- Verifies model training

## Model Specifications
- Architecture: 3-Layer Deep Neural Network
- Input: 28x28 MNIST images
- Output: 10 classes (digits 0-9)
- Parameters: <25,000

# Train Accuracy 

![image](https://github.com/user-attachments/assets/ed0e34ca-35bb-4da8-82b3-6dc291f4eb07)
