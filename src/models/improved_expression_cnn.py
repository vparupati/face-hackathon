import os, argparse
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
from sklearn.utils.class_weight import compute_class_weight

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        # Handle alpha weights
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha[targets]
            focal_loss = alpha_t * (1-pt)**self.gamma * ce_loss
        else:
            focal_loss = (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class EfficientNetExpression(nn.Module):
    """EfficientNet-B0 adapted for 48x48 grayscale expression recognition"""
    
    def __init__(self, num_classes=7, input_channels=1):
        super(EfficientNetExpression, self).__init__()
        
        # Load EfficientNet-B0 pretrained weights (use None to avoid hash issues)
        self.backbone = models.efficientnet_b0(weights=None)
        
        # Modify first layer for grayscale input
        self.backbone.features[0][0] = nn.Conv2d(
            input_channels, 32, kernel_size=3, stride=2, padding=1, bias=False
        )
        
        # Modify classifier for our number of classes
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Process grayscale input directly (no conversion needed)
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.backbone.classifier(x)
        return x

class ImprovedExpressionTrainer:
    """Advanced trainer with M4 Max optimizations"""
    
    def __init__(self, model, device, num_classes=7, learning_rate=1e-3, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes
        
        # Use AdamW optimizer
        self.optimizer = optim.AdamW(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Mixed precision scaler for M4 Max (MPS doesn't support autocast yet)
        self.scaler = GradScaler() if device.type == 'cuda' else None
        
        # Focal loss for class imbalance
        self.criterion = FocalLoss(alpha=1, gamma=2)
        
        # Class weights for balancing
        self.class_weights = None
        
    def set_class_weights(self, class_weights):
        """Set class weights for balanced training"""
        self.class_weights = class_weights
        if class_weights is not None:
            self.criterion = FocalLoss(alpha=class_weights, gamma=2)
    
    def train_epoch(self, train_loader):
        """Train one epoch with mixed precision"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass (MPS doesn't support autocast yet)
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        return total_loss / len(train_loader), correct / total
    
    def evaluate(self, val_loader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        return total_loss / len(val_loader), correct / total

class GrayscaleToSingleChannel:
    """Convert grayscale to single channel for FER2013"""
    def __call__(self, x):
        return x[0:1] if x.size(0) > 1 else x

def get_advanced_transforms(img_size=48):
    """Advanced data augmentation for 48x48 grayscale images"""
    
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        GrayscaleToSingleChannel(),  # Keep only grayscale channel
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize grayscale
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        GrayscaleToSingleChannel(),  # Keep only grayscale channel
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    return train_transform, val_transform

def compute_class_weights(dataset):
    """Compute class weights for imbalanced dataset"""
    targets = [dataset[i][1] for i in range(len(dataset))]
    class_weights = compute_class_weight(
        'balanced', 
        classes=np.unique(targets), 
        y=targets
    )
    return torch.FloatTensor(class_weights)

def train_improved_model(data_dir: str, out_path: str, epochs: int = 50, img_size: int = 48, 
                        batch: int = 64, lr: float = 1e-3, model_type: str = "efficientnet"):
    """Train improved expression model with M4 Max optimizations"""
    
    # Detect best device for M4 Max
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using M4 Max GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Advanced transforms
    train_transform, val_transform = get_advanced_transforms(img_size)
    
    # Load datasets
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "test"), transform=val_transform)
    
    # Compute class weights for imbalanced dataset
    class_weights = compute_class_weights(train_ds)
    print(f"Class weights: {class_weights}")
    
    # Create weighted sampler for balanced training
    class_counts = [len([x for x in train_ds.targets if x == i]) for i in range(len(train_ds.classes))]
    weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    sample_weights = weights[train_ds.targets]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
    
    # Data loaders with M4 Max optimizations
    pin_memory = device.type == 'cuda'  # Only use pin_memory for CUDA
    num_workers = 0 if device.type == 'mps' else 2  # Reduce workers for MPS
    train_ld = DataLoader(
        train_ds, 
        batch_size=batch, 
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    val_ld = DataLoader(
        val_ds, 
        batch_size=batch, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0
    )
    
    num_classes = len(train_ds.classes)
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {train_ds.classes}")
    
    # Create model
    if model_type == "efficientnet":
        model = EfficientNetExpression(num_classes=num_classes, input_channels=1)
        print(f"Using EfficientNet-B0 with {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create trainer
    trainer = ImprovedExpressionTrainer(model, device, num_classes, lr)
    trainer.set_class_weights(class_weights.to(device))
    
    # Training loop
    best_acc = 0.0
    patience = 10
    patience_counter = 0
    
    print(f"\nStarting training for {epochs} epochs...")
    print("=" * 60)
    
    for ep in range(1, epochs + 1):
        # Train
        train_loss, train_acc = trainer.train_epoch(train_ld)
        
        # Validate
        val_loss, val_acc = trainer.evaluate(val_ld)
        
        # Update learning rate
        trainer.scheduler.step()
        current_lr = trainer.optimizer.param_groups[0]['lr']
        
        print(f"Epoch {ep:2d}: train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} lr={current_lr:.2e}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), out_path.replace(".onnx", ".pt"))
            print(f"  → New best model saved! (val_acc={val_acc:.3f})")
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping at epoch {ep} (patience={patience})")
            break
    
    print("=" * 60)
    print(f"Training completed! Best validation accuracy: {best_acc:.3f}")
    
    # Load best model for export
    model.load_state_dict(torch.load(out_path.replace(".onnx", ".pt")))
    
    # Export ONNX with M4 Max optimizations
    model.eval()
    dummy_input = torch.randn(1, 1, img_size, img_size).to(device)
    
    print("Exporting ONNX model...")
    torch.onnx.export(
        model, 
        dummy_input, 
        out_path, 
        input_names=["image"], 
        output_names=["logits"], 
        opset_version=13,
        dynamic_axes={
            'image': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )
    
    # Save class names
    classes_path = out_path.replace(".onnx", ".classes.txt")
    with open(classes_path, "w") as f:
        for cls in train_ds.classes:
            f.write(f"{cls}\n")
    
    print(f"Model exported to: {out_path}")
    print(f"Class names saved to: {classes_path}")

def main():
    parser = argparse.ArgumentParser(description="Train improved expression model")
    parser.add_argument("--data", required=True, help="Path to expression dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--out", default="models/improved_expressions.onnx", help="Output model path")
    parser.add_argument("--img", type=int, default=48, help="Image size (keep 48 for FER2013)")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--model", choices=["efficientnet"], default="efficientnet", help="Model architecture")
    
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    print("🚀 Improved Expression Training with M4 Max Optimizations")
    print("=" * 60)
    print(f"Dataset: {args.data}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch}")
    print(f"Image size: {args.img}")
    print(f"Learning rate: {args.lr}")
    print(f"Output: {args.out}")
    print("=" * 60)
    
    train_improved_model(
        args.data, 
        args.out, 
        epochs=args.epochs, 
        img_size=args.img, 
        batch=args.batch, 
        lr=args.lr, 
        model_type=args.model
    )

if __name__ == "__main__":
    main()
