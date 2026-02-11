import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from pathlib import Path
import time

# ============================================================================
# MODEL DEFINITION (Same as prediction.py)
# ============================================================================

class DeepfakeCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(DeepfakeCNN, self).__init__()
        self.backbone = models.resnet50(pretrained=False)
        num_features = self.backbone.fc.in_features
        
        self.backbone.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# ============================================================================
# DATASET CLASS
# ============================================================================

class DeepfakeDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load FAKE images (label = 0)
        fake_dir = self.data_dir / 'FAKE'
        if fake_dir.exists():
            for img_path in fake_dir.glob('*.jpg'):
                self.images.append(str(img_path))
                self.labels.append(0)
            for img_path in fake_dir.glob('*.png'):
                self.images.append(str(img_path))
                self.labels.append(0)
        
        # Load REAL images (label = 1)
        real_dir = self.data_dir / 'REAL'
        if real_dir.exists():
            for img_path in real_dir.glob('*.jpg'):
                self.images.append(str(img_path))
                self.labels.append(1)
            for img_path in real_dir.glob('*.png'):
                self.images.append(str(img_path))
                self.labels.append(1)
        
        print(f"[DATA] Loaded {len(self.images)} images", file=sys.stderr)
        print(f"[DATA] FAKE: {self.labels.count(0)}, REAL: {self.labels.count(1)}", file=sys.stderr)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            if self.transform:
                image = self.transform(image)
            
            return image, label
        except Exception as e:
            print(f"[ERROR] Failed to load {img_path}: {e}", file=sys.stderr)
            # Return a black image as fallback
            return torch.zeros(3, 224, 224), label

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(data_dir, model_path, epochs=10, batch_size=32, learning_rate=0.001):
    """
    Train the deepfake detection model
    
    Args:
        data_dir: Path to training_data folder
        model_path: Path to save/load model weights
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
    
    Returns:
        dict: Training results
    """
    
    print("\n" + "="*60, file=sys.stderr)
    print("🎓 TRAINING STARTED", file=sys.stderr)
    print("="*60, file=sys.stderr)
    
    start_time = time.time()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DEVICE] Using: {device}", file=sys.stderr)
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load dataset
    print("[DATA] Loading dataset...", file=sys.stderr)
    dataset = DeepfakeDataset(data_dir, transform=transform)
    
    if len(dataset) == 0:
        error_msg = "No training data found"
        print(f"[ERROR] {error_msg}", file=sys.stderr)
        return {
            'success': False,
            'error': error_msg
        }
    
    # Split into train/validation (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    print(f"[DATA] Train: {train_size}, Val: {val_size}", file=sys.stderr)
    
    # Data loaders
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
    
    # Model
    print("[MODEL] Initializing model...", file=sys.stderr)
    model = DeepfakeCNN(num_classes=2)
    
    # Load existing weights if available
    if os.path.exists(model_path):
        print(f"[MODEL] Loading existing weights from {model_path}", file=sys.stderr)
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print("[MODEL] ✅ Loaded existing weights", file=sys.stderr)
            else:
                model.load_state_dict(checkpoint)
                print("[MODEL] ✅ Loaded existing weights", file=sys.stderr)
        except Exception as e:
            print(f"[MODEL] ⚠️ Could not load weights: {e}", file=sys.stderr)
            print("[MODEL] Starting with random weights", file=sys.stderr)
    else:
        print("[MODEL] No existing weights, starting fresh", file=sys.stderr)
    
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\n[TRAIN] Starting training...", file=sys.stderr)
    best_val_acc = 0.0
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        epoch_time = time.time() - epoch_start
        
        # Log progress
        print(f"[EPOCH {epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s", 
              file=sys.stderr)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"[SAVE] ✅ New best accuracy: {val_acc:.2f}%", file=sys.stderr)
    
    # Save final model
    print(f"\n[SAVE] Saving model to {model_path}", file=sys.stderr)
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_architecture': 'DeepfakeCNN',
        'num_classes': 2,
        'class_names': ['FAKE', 'REAL'],
        'accuracy': best_val_acc,
        'history': history,
        'epochs_trained': epochs,
        'timestamp': time.time()
    }
    
    torch.save(checkpoint, model_path)
    print("[SAVE] ✅ Model saved successfully", file=sys.stderr)
    
    total_time = time.time() - start_time
    
    print("\n" + "="*60, file=sys.stderr)
    print("✅ TRAINING COMPLETED", file=sys.stderr)
    print("="*60, file=sys.stderr)
    print(f"Total time: {total_time:.1f}s", file=sys.stderr)
    print(f"Best validation accuracy: {best_val_acc:.2f}%", file=sys.stderr)
    print("="*60 + "\n", file=sys.stderr)
    
    return {
        'success': True,
        'accuracy': round(best_val_acc, 2),
        'train_accuracy': round(history['train_acc'][-1], 2),
        'val_accuracy': round(history['val_acc'][-1], 2),
        'final_train_loss': round(history['train_loss'][-1], 4),
        'final_val_loss': round(history['val_loss'][-1], 4),
        'epochs': epochs,
        'duration': round(total_time, 1),
        'samples_trained': len(dataset)
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Deepfake Detection Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to training_data folder')
    parser.add_argument('--model_path', type=str, required=True, help='Path to save model')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    
    args = parser.parse_args()
    
    result = train_model(
        data_dir=args.data_dir,
        model_path=args.model_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Output JSON result
    print(json.dumps(result), flush=True)

if __name__ == '__main__':
    main()