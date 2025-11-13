"""
Baseline Training Script
Hackathon: 3LC x AWS Cloud @ UT Dallas - Chihuahua vs Muffin Challenge

This script trains a simple CNN classifier on the Chihuahua vs Muffin dataset.
Baseline accuracy: ~83%

Your goal: Improve this using data-centric AI techniques with 3LC!

Usage:
    python train.py
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from tqdm import tqdm

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class SimpleCNN(nn.Module):
    """
    Simple CNN architecture for binary image classification
    Architecture:
        - 3 Convolutional blocks (Conv -> ReLU -> MaxPool)
        - 2 Fully connected layers
        - Output: 2 classes (chihuahua, muffin)
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        # After 3 pooling layers: 128x128 -> 64x64 -> 32x32 -> 16x16
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Activation
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # Convolutional blocks
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(-1, 128 * 16 * 16)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def get_data_loaders(data_dir, batch_size=16):
    """
    Create training and validation data loaders
    
    Args:
        data_dir: Path to data directory containing train/
        batch_size: Number of images per batch
    
    Returns:
        train_loader, val_loader
    """
    
    # Data augmentation and normalization for training
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Just normalization for validation
    val_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load full dataset
    full_dataset = datasets.ImageFolder(
        root=str(data_dir / "train"),
        transform=train_transform
    )
    
    # Split into train/validation (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply validation transform to validation set
    val_dataset.dataset.transform = val_transform
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Dataset loaded:")
    print(f"  Training samples:   {len(train_dataset)}")
    print(f"  Validation samples: {len(val_dataset)}")
    print(f"  Classes: {full_dataset.classes}")
    
    return train_loader, val_loader, full_dataset.classes

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{running_loss/len(pbar):.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    val_loss = running_loss / len(val_loader)
    val_acc = 100. * correct / total
    
    return val_loss, val_acc

def main():
    """Main training function"""
    
    print("=" * 60)
    print("Baseline CNN Training - Chihuahua vs Muffin Challenge")
    print("=" * 60)
    
    # Configuration
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.001
    
    # Paths (relative to script location)
    base_path = Path(__file__).parent.parent
    data_path = base_path / "data"
    model_path = base_path / "models"
    
    # Create models directory if it doesn't exist
    model_path.mkdir(exist_ok=True)
    
    # Check if data exists
    if not (data_path / "train").exists():
        print("[ERROR]: Training data not found!")
        print(f"Expected path: {data_path / 'train'}")
        print("Please download the dataset first.")
        return
    
    # Load data
    print("\n[Loading dataset...]")
    train_loader, val_loader, classes = get_data_loaders(data_path, BATCH_SIZE)
    
    # Create model
    print(f"\n[Creating model...]")
    model = SimpleCNN().to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    print(f"\n[Starting training for {NUM_EPOCHS} epochs...]")
    print("=" * 60)
    
    best_val_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Print results
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'classes': classes
            }, model_path / "best_model.pth")
            print(f"  [OK] Saved best model (Val Acc: {val_acc:.2f}%)")
    
    print("\n" + "=" * 60)
    print(f"[OK] Training complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {model_path / 'best_model.pth'}")
    print("=" * 60)
    
    print("\n[Next Steps]:")
    print("1. Open 3LC Dashboard to analyze your results")
    print("2. Identify misclassified samples")
    print("3. Apply data-centric improvements")
    print("4. Retrain and iterate!")
    print("\nGoal: Beat the baseline accuracy of ~83%!")

if __name__ == "__main__":
    main()

