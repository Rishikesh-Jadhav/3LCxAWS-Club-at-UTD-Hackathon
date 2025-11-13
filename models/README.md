# Models Directory

This folder stores your trained model checkpoints.

## What Gets Saved Here

When you run the training scripts, the best model will be automatically saved to:
- `best_model.pth` - Best model based on validation accuracy

## Model Contents

Each saved model includes:
- Model state dictionary (weights and biases)
- Optimizer state (for resuming training)
- Validation accuracy
- Training epoch number
- Class names

## Loading a Saved Model

```python
import torch
from src.train import SimpleCNN

# Load checkpoint
checkpoint = torch.load('models/best_model.pth')

# Create model and load weights
model = SimpleCNN()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"Loaded model with {checkpoint['val_acc']:.2f}% validation accuracy")
```

## Note

Model files are large (.pth files) and are **excluded from Git** by `.gitignore`.
You'll need to train your own model using the provided scripts.

## Expected Performance

- **Baseline Model:** ~83% validation accuracy
- **Your Goal:** Beat the baseline using data-centric AI techniques!

