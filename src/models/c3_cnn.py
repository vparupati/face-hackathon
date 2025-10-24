import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class C3CNN(nn.Module):
    """
    C3 CNN: A 3-layer Convolutional Neural Network for face recognition.
    
    Architecture:
    - Conv2d(3, 32, 3x3) + ReLU + MaxPool2d(2x2)
    - Conv2d(32, 64, 3x3) + ReLU + MaxPool2d(2x2) 
    - Conv2d(64, 128, 3x3) + ReLU + MaxPool2d(2x2)
    - AdaptiveAvgPool2d(1x1) + Flatten
    - Linear(128, embedding_dim)
    """
    
    def __init__(self, embedding_dim: int = 128, input_size: int = 160):
        super(C3CNN, self).__init__()
        self.embedding_dim = embedding_dim
        self.input_size = input_size
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Final embedding layer
        self.fc = nn.Linear(128, embedding_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        """
        Forward pass through the C3 CNN.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Normalized embeddings of shape (batch_size, embedding_dim)
        """
        # First convolutional block
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        
        # Second convolutional block
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        
        # Third convolutional block
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        # Global average pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        
        # Apply dropout
        x = self.dropout(x)
        
        # Final embedding layer
        x = self.fc(x)
        
        # L2 normalization for cosine similarity
        x = F.normalize(x, p=2, dim=1)
        
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
        x = F.relu(self.conv1(x))
        features['conv1'] = x
        x = self.pool(x)
        
        # Second conv block
        x = F.relu(self.conv2(x))
        features['conv2'] = x
        x = self.pool(x)
        
        # Third conv block
        x = F.relu(self.conv3(x))
        features['conv3'] = x
        x = self.pool(x)
        
        # Global pooling and final layer
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        
        features['embedding'] = x
        
        return features


class C3SiameseEncoder(nn.Module):
    """
    Siamese network using C3 CNN as the backbone encoder.
    """
    
    def __init__(self, embedding_dim: int = 128, input_size: int = 160):
        super(C3SiameseEncoder, self).__init__()
        self.encoder = C3CNN(embedding_dim=embedding_dim, input_size=input_size)
        
    def forward(self, x):
        """
        Forward pass through the C3 encoder.
        
        Args:
            x: Input tensor of shape (batch_size, 3, height, width)
            
        Returns:
            Normalized embeddings of shape (batch_size, embedding_dim)
        """
        return self.encoder(x)
    
    def forward_pair(self, x1, x2):
        """
        Forward pass for a pair of images.
        
        Args:
            x1, x2: Input tensors of shape (batch_size, 3, height, width)
            
        Returns:
            Tuple of (embedding1, embedding2, distance)
        """
        z1 = self.forward(x1)
        z2 = self.forward(x2)
        
        # Compute Euclidean distance
        distance = torch.sqrt(((z1 - z2) ** 2).sum(dim=1, keepdim=True) + 1e-8)
        
        return z1, z2, distance


def create_c3_model(embedding_dim: int = 128, input_size: int = 160) -> C3CNN:
    """
    Factory function to create a C3 CNN model.
    
    Args:
        embedding_dim: Dimension of the output embedding
        input_size: Input image size (assumed square)
        
    Returns:
        C3CNN model instance
    """
    return C3CNN(embedding_dim=embedding_dim, input_size=input_size)


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
    # Test the C3 CNN model
    model = C3CNN(embedding_dim=128, input_size=160)
    print(f"C3 CNN Model:")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 160, 160)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output norm (should be ~1.0): {torch.norm(output, dim=1)}")
    
    # Test feature maps
    features = model.get_feature_maps(x)
    for name, feat in features.items():
        print(f"{name} shape: {feat.shape}")
