"""
Sparse 3D Convolutional Neural Network for shape classification.
Uses MinkowskiEngine for efficient sparse convolutions.
"""

import torch
import torch.nn as nn
import MinkowskiEngine as ME


class SparseShapeClassifier(nn.Module):
    """
    Sparse 3D CNN for classifying point clouds into 5 shape categories.
    Uses sparse convolutions from MinkowskiEngine for efficiency.
    """
    
    def __init__(self, in_channels: int = 1, num_classes: int = 5):
        super(SparseShapeClassifier, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        
        # Sparse convolution layers
        self.conv1 = ME.MinkowskiConvolution(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=3,
            stride=1,
            dimension=3
        )
        self.bn1 = ME.MinkowskiBatchNorm(32)
        self.relu1 = ME.MinkowskiReLU()
        
        self.conv2 = ME.MinkowskiConvolution(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            dimension=3
        )
        self.bn2 = ME.MinkowskiBatchNorm(64)
        self.relu2 = ME.MinkowskiReLU()
        
        self.conv3 = ME.MinkowskiConvolution(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2,
            dimension=3
        )
        self.bn3 = ME.MinkowskiBatchNorm(128)
        self.relu3 = ME.MinkowskiReLU()
        
        self.conv4 = ME.MinkowskiConvolution(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            stride=2,
            dimension=3
        )
        self.bn4 = ME.MinkowskiBatchNorm(256)
        self.relu4 = ME.MinkowskiReLU()
        
        # Global pooling
        self.global_pool = ME.MinkowskiGlobalPooling()
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x: ME.SparseTensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Sparse tensor with point cloud data
            
        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        # Sparse convolution blocks
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        
        # Global pooling
        x = self.global_pool(x)
        
        # Dense layers
        x = x.F  # Extract features from sparse tensor
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        
        return x


def create_model(num_classes: int = 5) -> SparseShapeClassifier:
    """Factory function to create the model."""
    return SparseShapeClassifier(in_channels=1, num_classes=num_classes)


if __name__ == '__main__':
    # Test model creation and forward pass
    print("Testing SparseShapeClassifier...")
    
    model = create_model(num_classes=5)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Create dummy input
    coords = torch.tensor([
        [0, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [1, 1, 1, 1],
    ], dtype=torch.int32)
    
    features = torch.randn(6, 1)
    
    sparse_input = ME.SparseTensor(
        features=features,
        coordinates=coords,
        device='cpu'
    )
    
    output = model(sparse_input)
    print(f"Output shape: {output.shape}")
    print(f"Output: {output}")
    print("Model test passed!")
