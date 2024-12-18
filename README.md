# ML Model CI/CD Pipeline

## Project Setup

### Local Development

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

4. Run tests:
```bash
python -m tests.test_model
```

5. Train the model:
```bash
python src/train.py
```

### GitHub Actions

This project uses GitHub Actions for:
- Dependency installation
- Model testing
- Model training

Each push and pull request will trigger the workflow.
