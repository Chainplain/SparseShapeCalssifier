"""
Demo script to showcase the 3D shape classifier capabilities.
Demonstrates shape generation without requiring GPU/MinkowskiEngine installation.
"""

import numpy as np
from generate_shapes import generate_shape, get_shape_name


def visualize_point_cloud_stats(points: np.ndarray, shape_name: str):
    """Print statistics about a point cloud."""
    print(f"\n{shape_name.upper()} Statistics:")
    print(f"  Number of points: {points.shape[0]}")
    print(f"  Point cloud bounds:")
    print(f"    X: [{points[:, 0].min():.3f}, {points[:, 0].max():.3f}]")
    print(f"    Y: [{points[:, 1].min():.3f}, {points[:, 1].max():.3f}]")
    print(f"    Z: [{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    print(f"  Centroid: ({points.mean(axis=0)[0]:.3f}, "
          f"{points.mean(axis=0)[1]:.3f}, {points.mean(axis=0)[2]:.3f})")
    
    # Calculate some shape-specific metrics
    distances = np.linalg.norm(points, axis=1)
    print(f"  Distance from origin: mean={distances.mean():.3f}, "
          f"std={distances.std():.3f}")


def demonstrate_all_shapes():
    """Generate and display statistics for all shape types."""
    print("="*70)
    print("3D Shape Classifier - Shape Generation Demo")
    print("="*70)
    
    shapes = ['sphere', 'cube', 'cylinder', 'cone', 'pyramid']
    
    for shape_type in shapes:
        points, label = generate_shape(shape_type, num_points=1000)
        visualize_point_cloud_stats(points, shape_type)
        print(f"  Class label: {label} -> '{get_shape_name(label)}'")
    
    print("\n" + "="*70)


def demonstrate_shape_variation():
    """Show how the same shape varies across different generations."""
    print("\n" + "="*70)
    print("Demonstrating Shape Variation")
    print("="*70)
    
    shape_type = 'sphere'
    print(f"\nGenerating 3 different {shape_type}s with 500 points each:\n")
    
    for i in range(3):
        points, label = generate_shape(shape_type, num_points=500)
        centroid = points.mean(axis=0)
        radius = np.linalg.norm(points, axis=1).mean()
        print(f"  {shape_type} {i+1}: centroid=({centroid[0]:.3f}, "
              f"{centroid[1]:.3f}, {centroid[2]:.3f}), avg_radius={radius:.3f}")


def demonstrate_classification_task():
    """Simulate what the classifier does."""
    print("\n" + "="*70)
    print("Simulated Classification Task")
    print("="*70)
    
    print("\nThe classifier learns to distinguish between these shapes:")
    print("\nClass Mapping:")
    for i in range(5):
        print(f"  {i}: {get_shape_name(i)}")
    
    print("\nGenerating test samples:")
    test_shapes = ['sphere', 'pyramid', 'cylinder']
    
    for shape in test_shapes:
        points, true_label = generate_shape(shape, num_points=800)
        print(f"\n  Input: Point cloud with {points.shape[0]} points")
        print(f"  True label: {true_label} ({get_shape_name(true_label)})")
        print(f"  Model would predict: {get_shape_name(true_label)}")
        print(f"  → Classification {'✓ CORRECT' if True else '✗ WRONG'}")


def main():
    """Run the demo."""
    demonstrate_all_shapes()
    demonstrate_shape_variation()
    demonstrate_classification_task()
    
    print("\n" + "="*70)
    print("Demo completed!")
    print("\nTo train the actual classifier, run:")
    print("  python train.py --epochs 50 --batch-size 8")
    print("\nNote: Training requires PyTorch and MinkowskiEngine with CUDA support")
    print("="*70)


if __name__ == '__main__':
    main()
