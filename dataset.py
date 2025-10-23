"""
Dataset class for 3D shape point clouds.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from typing import Tuple, List
import MinkowskiEngine as ME

from generate_shapes import generate_shape


class ShapePointCloudDataset(Dataset):
    """Dataset for 3D shape point clouds."""
    
    def __init__(
        self,
        num_samples_per_class: int = 100,
        num_points: int = 1000,
        voxel_size: float = 0.05,
        augment: bool = True
    ):
        """
        Initialize the dataset.
        
        Args:
            num_samples_per_class: Number of samples to generate per class
            num_points: Number of points per point cloud
            voxel_size: Voxel size for quantization
            augment: Whether to apply data augmentation
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_points = num_points
        self.voxel_size = voxel_size
        self.augment = augment
        
        self.shapes = ['sphere', 'cube', 'cylinder', 'cone', 'pyramid']
        self.num_classes = len(self.shapes)
        
        # Pre-generate all samples
        self.samples = []
        for shape_type in self.shapes:
            for _ in range(num_samples_per_class):
                points, label = generate_shape(shape_type, num_points)
                self.samples.append((points, label))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        """Get a single sample."""
        points, label = self.samples[idx]
        points = points.copy()
        
        if self.augment:
            # Random rotation around Y axis
            angle = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rotation_matrix = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ], dtype=np.float32)
            points = points @ rotation_matrix.T
            
            # Random scaling
            scale = np.random.uniform(0.8, 1.2)
            points = points * scale
            
            # Random jitter
            points += np.random.normal(0, 0.02, points.shape).astype(np.float32)
        
        return points, label
    
    def collate_fn(self, batch: List[Tuple[np.ndarray, int]]):
        """
        Collate function for DataLoader to create sparse tensors.
        
        Args:
            batch: List of (points, label) tuples
            
        Returns:
            Tuple of (sparse_tensor, labels)
        """
        coords_list = []
        feats_list = []
        labels_list = []
        
        for batch_idx, (points, label) in enumerate(batch):
            # Quantize points to voxel grid
            coords = np.floor(points / self.voxel_size).astype(np.int32)
            
            # Add batch index as first column
            batch_coords = np.hstack([
                np.full((len(coords), 1), batch_idx, dtype=np.int32),
                coords
            ])
            
            # Use constant features (can be modified to use colors, normals, etc.)
            feats = np.ones((len(coords), 1), dtype=np.float32)
            
            coords_list.append(batch_coords)
            feats_list.append(feats)
            labels_list.append(label)
        
        # Concatenate all coordinates and features
        coords_batch = np.vstack(coords_list)
        feats_batch = np.vstack(feats_list)
        
        # Remove duplicate coordinates (keep only unique voxels)
        coords_batch, unique_map = ME.utils.sparse_quantize(
            coordinates=coords_batch,
            return_index=True
        )
        feats_batch = feats_batch[unique_map]
        
        # Convert to tensors
        coords_batch = torch.from_numpy(coords_batch).int()
        feats_batch = torch.from_numpy(feats_batch).float()
        labels_batch = torch.tensor(labels_list, dtype=torch.long)
        
        # Create sparse tensor
        sparse_tensor = ME.SparseTensor(
            features=feats_batch,
            coordinates=coords_batch
        )
        
        return sparse_tensor, labels_batch


def create_dataloaders(
    batch_size: int = 8,
    num_train_samples: int = 100,
    num_val_samples: int = 20,
    num_points: int = 1000,
    voxel_size: float = 0.05,
    num_workers: int = 0
):
    """
    Create training and validation dataloaders.
    
    Args:
        batch_size: Batch size for training
        num_train_samples: Number of training samples per class
        num_val_samples: Number of validation samples per class
        num_points: Number of points per point cloud
        voxel_size: Voxel size for quantization
        num_workers: Number of workers for data loading
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = ShapePointCloudDataset(
        num_samples_per_class=num_train_samples,
        num_points=num_points,
        voxel_size=voxel_size,
        augment=True
    )
    
    val_dataset = ShapePointCloudDataset(
        num_samples_per_class=num_val_samples,
        num_points=num_points,
        voxel_size=voxel_size,
        augment=False
    )
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        num_workers=num_workers
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=val_dataset.collate_fn,
        num_workers=num_workers
    )
    
    return train_loader, val_loader


if __name__ == '__main__':
    # Test dataset
    print("Testing ShapePointCloudDataset...")
    
    dataset = ShapePointCloudDataset(num_samples_per_class=2, num_points=500)
    print(f"Dataset size: {len(dataset)}")
    
    points, label = dataset[0]
    print(f"Sample shape: {points.shape}, label: {label}")
    
    # Test dataloader
    train_loader, val_loader = create_dataloaders(
        batch_size=4,
        num_train_samples=2,
        num_val_samples=1,
        num_points=500
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    for sparse_tensor, labels in train_loader:
        print(f"Batch - Sparse tensor shape: {sparse_tensor.shape}, Labels: {labels}")
        break
    
    print("Dataset test passed!")
