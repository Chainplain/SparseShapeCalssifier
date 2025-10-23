"""
Inference script for the Sparse 3D Shape Classifier.
"""

import torch
import numpy as np
import argparse
import MinkowskiEngine as ME

from model import create_model
from generate_shapes import generate_shape, get_shape_name


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load a trained model from checkpoint."""
    model = create_model(num_classes=5)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model


def preprocess_point_cloud(points: np.ndarray, voxel_size: float = 0.05) -> ME.SparseTensor:
    """
    Preprocess a point cloud for inference.
    
    Args:
        points: Point cloud array of shape (N, 3)
        voxel_size: Voxel size for quantization
        
    Returns:
        Sparse tensor ready for model input
    """
    # Quantize points to voxel grid
    coords = np.floor(points / voxel_size).astype(np.int32)
    
    # Add batch index (0 for single sample)
    batch_coords = np.hstack([
        np.zeros((len(coords), 1), dtype=np.int32),
        coords
    ])
    
    # Use constant features
    feats = np.ones((len(coords), 1), dtype=np.float32)
    
    # Remove duplicate coordinates
    coords_unique, unique_map = ME.utils.sparse_quantize(
        coordinates=batch_coords,
        return_index=True
    )
    feats_unique = feats[unique_map]
    
    # Convert to tensors
    coords_tensor = torch.from_numpy(coords_unique).int()
    feats_tensor = torch.from_numpy(feats_unique).float()
    
    # Create sparse tensor
    sparse_tensor = ME.SparseTensor(
        features=feats_tensor,
        coordinates=coords_tensor
    )
    
    return sparse_tensor


def predict(
    model: torch.nn.Module,
    points: np.ndarray,
    device: torch.device,
    voxel_size: float = 0.05
) -> tuple:
    """
    Predict the shape class for a point cloud.
    
    Args:
        model: Trained model
        points: Point cloud array of shape (N, 3)
        device: Device to run inference on
        voxel_size: Voxel size for quantization
        
    Returns:
        Tuple of (predicted_class, class_probabilities)
    """
    # Preprocess
    sparse_tensor = preprocess_point_cloud(points, voxel_size)
    sparse_tensor = sparse_tensor.to(device)
    
    # Inference
    with torch.no_grad():
        outputs = model(sparse_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        class_probs = probabilities[0].cpu().numpy()
    
    return predicted_class, class_probs


def evaluate_on_test_shapes(
    checkpoint_path: str,
    num_samples: int = 10,
    device: str = 'cuda'
):
    """
    Evaluate the model on randomly generated test shapes.
    
    Args:
        checkpoint_path: Path to model checkpoint
        num_samples: Number of test samples per class
        device: Device to run on
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, device)
    
    # Test on each shape
    shapes = ['sphere', 'cube', 'cylinder', 'cone', 'pyramid']
    
    print(f"\nEvaluating on {num_samples} samples per class...\n")
    
    total_correct = 0
    total_samples = 0
    
    for shape_type in shapes:
        correct = 0
        
        for i in range(num_samples):
            # Generate test sample
            points, true_label = generate_shape(shape_type, num_points=1000)
            
            # Predict
            pred_label, probs = predict(model, points, device)
            
            if pred_label == true_label:
                correct += 1
            
            # Print first sample details
            if i == 0:
                print(f"{shape_type.upper()} (True label: {true_label})")
                print(f"  Predicted: {get_shape_name(pred_label)} (label: {pred_label})")
                print(f"  Probabilities:")
                for j, prob in enumerate(probs):
                    print(f"    {get_shape_name(j)}: {prob:.4f}")
                print()
        
        accuracy = 100.0 * correct / num_samples
        print(f"{shape_type}: {correct}/{num_samples} correct ({accuracy:.2f}%)")
        
        total_correct += correct
        total_samples += num_samples
    
    overall_accuracy = 100.0 * total_correct / total_samples
    print(f"\nOverall Accuracy: {total_correct}/{total_samples} ({overall_accuracy:.2f}%)")


def predict_single_shape(
    checkpoint_path: str,
    shape_type: str,
    num_points: int = 1000,
    device: str = 'cuda'
):
    """
    Predict a single generated shape.
    
    Args:
        checkpoint_path: Path to model checkpoint
        shape_type: Type of shape to generate and predict
        num_points: Number of points in the point cloud
        device: Device to run on
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, device)
    
    # Generate shape
    print(f"Generating {shape_type} with {num_points} points...")
    points, true_label = generate_shape(shape_type, num_points)
    
    # Predict
    print("Running inference...")
    pred_label, probs = predict(model, points, device)
    
    # Print results
    print("\n" + "="*50)
    print(f"True Shape: {shape_type} (label: {true_label})")
    print(f"Predicted Shape: {get_shape_name(pred_label)} (label: {pred_label})")
    print("="*50)
    print("\nClass Probabilities:")
    for i, prob in enumerate(probs):
        print(f"  {get_shape_name(i):10s}: {prob:.4f} {'<-- PREDICTED' if i == pred_label else ''}")
    print()


def main():
    parser = argparse.ArgumentParser(description='Inference for Sparse 3D Shape Classifier')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, choices=['evaluate', 'single'], default='evaluate',
                        help='Inference mode')
    parser.add_argument('--shape', type=str, choices=['sphere', 'cube', 'cylinder', 'cone', 'pyramid'],
                        help='Shape type for single prediction mode')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of test samples per class for evaluate mode')
    parser.add_argument('--num-points', type=int, default=1000,
                        help='Number of points per point cloud')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    if args.mode == 'evaluate':
        evaluate_on_test_shapes(
            checkpoint_path=args.checkpoint,
            num_samples=args.num_samples,
            device=args.device
        )
    elif args.mode == 'single':
        if args.shape is None:
            parser.error("--shape is required for single prediction mode")
        predict_single_shape(
            checkpoint_path=args.checkpoint,
            shape_type=args.shape,
            num_points=args.num_points,
            device=args.device
        )


if __name__ == '__main__':
    main()
