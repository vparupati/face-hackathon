import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List


class C3ExpressionCNN(nn.Module):
    """
    C3 CNN for Expression Classification: A 3-layer Convolutional Neural Network 
    optimized for facial expression recognition.
    
    Architecture:
    - Conv2d(3, 32, 3x3) + ReLU + MaxPool2d(2x2) + BatchNorm
    - Conv2d(32, 64, 3x3) + ReLU + MaxPool2d(2x2) + BatchNorm
    - Conv2d(64, 128, 3x3) + ReLU + MaxPool2d(2x2) + BatchNorm
    - AdaptiveAvgPool2d(1x1) + Flatten
    - Dropout(0.5)
    - Linear(128, num_classes)
    """
    
    def __init__(self, num_classes: int = 7, input_size: int = 160, dropout_rate: float = 0.5):
        super(C3ExpressionCNN, self).__init__()
        self.num_classes = num_classes
        self.input_size = input_size
        self.dropout_rate = dropout_rate
        
        # Convolutional layers with batch normalization
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final classification layer
        self.classifier = nn.Linear(128, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the C3 Expression CNN.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        # First convolutional block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        
        # Second convolutional block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        # Third convolutional block
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Apply dropout
        x = self.dropout(x)
        
        # Final classification layer
        x = self.classifier(x)
        
        return x
    
    def get_feature_maps(self, x):
        """
        Get intermediate feature maps for visualization.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Dictionary containing feature maps from each layer
        """
        features = {}
        
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        features['conv1'] = x
        x = self.pool(x)
        
        # Second conv block
        x = F.relu(self.bn2(self.conv2(x)))
        features['conv2'] = x
        x = self.pool(x)
        
        # Third conv block
        x = F.relu(self.bn3(self.conv3(x)))
        features['conv3'] = x
        x = self.pool(x)
        
        # Global pooling and final layer
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        features['logits'] = x
        
        return features
    
    def predict_proba(self, x):
        """
        Get probability predictions.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Probability tensor of shape (batch_size, num_classes)
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)
    
    def predict(self, x):
        """
        Get class predictions.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Predicted class indices of shape (batch_size,)
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)


class C3ExpressionTrainer:
    """
    Training utility class for C3 Expression CNN.
    """
    
    def __init__(self, model, device, learning_rate=1e-3, weight_decay=1e-4):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
        
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
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
        """Evaluate the model."""
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


def create_c3_expression_model(num_classes: int = 7, input_size: int = 160, dropout_rate: float = 0.5) -> C3ExpressionCNN:
    """
    Factory function to create a C3 Expression CNN model.
    
    Args:
        num_classes: Number of expression classes (default: 7 for FER2013)
        input_size: Input image size (assumed square)
        dropout_rate: Dropout rate for regularization
        
    Returns:
        C3ExpressionCNN model instance
    """
    return C3ExpressionCNN(
        num_classes=num_classes, 
        input_size=input_size, 
        dropout_rate=dropout_rate
    )


def count_parameters(model):
    """
    Count the number of trainable parameters in the model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the C3 Expression CNN model
    model = C3ExpressionCNN(num_classes=7, input_size=160)
    print(f"C3 Expression CNN Model:")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 160, 160)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test predictions
    probs = model.predict_proba(x)
    preds = model.predict(x)
    print(f"Probabilities shape: {probs.shape}")
    print(f"Predictions: {preds}")
    
    # Test feature maps
    features = model.get_feature_maps(x)
    for name, feat in features.items():
        print(f"{name} shape: {feat.shape}")
