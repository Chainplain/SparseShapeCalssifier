"""
Basic tests for shape generation without requiring CUDA/MinkowskiEngine.
"""

import numpy as np
from generate_shapes import generate_shape, get_shape_name


def test_shape_generation():
    """Test that all shapes can be generated correctly."""
    shapes = ['sphere', 'cube', 'cylinder', 'cone', 'pyramid']
    
    print("Testing shape generation...")
    for shape in shapes:
        for num_points in [100, 500, 1000]:
            points, label = generate_shape(shape, num_points=num_points)
            
            # Check shape dimensions
            assert points.shape[1] == 3, f"Expected 3D points, got {points.shape}"
            
            # Allow small tolerance (within 1 point)
            assert abs(points.shape[0] - num_points) <= 1, \
                f"{shape}: Expected ~{num_points} points, got {points.shape[0]}"
            
            # Check label is valid
            assert label in range(5), f"Invalid label {label}"
            assert get_shape_name(label) == shape, \
                f"Label {label} maps to {get_shape_name(label)}, expected {shape}"
            
            # Check points are finite
            assert np.all(np.isfinite(points)), f"Points contain non-finite values"
            
    print("✓ Shape generation tests passed")


def test_label_consistency():
    """Test that labels are consistent across generations."""
    shapes = ['sphere', 'cube', 'cylinder', 'cone', 'pyramid']
    expected_labels = {
        'sphere': 0,
        'cube': 1,
        'cylinder': 2,
        'cone': 3,
        'pyramid': 4
    }
    
    print("\nTesting label consistency...")
    for shape in shapes:
        for _ in range(5):
            _, label = generate_shape(shape)
            assert label == expected_labels[shape], \
                f"Label mismatch for {shape}: got {label}, expected {expected_labels[shape]}"
    
    print("✓ Label consistency tests passed")


def test_point_cloud_properties():
    """Test that generated point clouds have reasonable properties."""
    print("\nTesting point cloud properties...")
    
    # Sphere should have points roughly at distance 1 from origin
    points, _ = generate_shape('sphere', num_points=1000)
    distances = np.linalg.norm(points, axis=1)
    assert np.abs(np.mean(distances) - 1.0) < 0.1, \
        f"Sphere points not at expected radius: mean distance {np.mean(distances)}"
    
    # Cube should have points within [-1, 1] range
    points, _ = generate_shape('cube', num_points=1000)
    assert np.all(points >= -1.01) and np.all(points <= 1.01), \
        "Cube points outside expected bounds"
    
    print("✓ Point cloud property tests passed")


def test_shape_name_mapping():
    """Test bidirectional shape name <-> label mapping."""
    print("\nTesting shape name mapping...")
    
    shapes = ['sphere', 'cube', 'cylinder', 'cone', 'pyramid']
    for i, shape in enumerate(shapes):
        # Generate shape and get label
        _, label = generate_shape(shape)
        # Check label to name mapping
        assert get_shape_name(label) == shape, \
            f"get_shape_name({label}) returned {get_shape_name(label)}, expected {shape}"
    
    print("✓ Shape name mapping tests passed")


def main():
    """Run all tests."""
    print("="*60)
    print("Running Basic Tests for 3D Shape Classifier")
    print("="*60)
    print()
    
    test_shape_generation()
    test_label_consistency()
    test_point_cloud_properties()
    test_shape_name_mapping()
    
    print("\n" + "="*60)
    print("All basic tests passed! ✓")
    print("="*60)


if __name__ == '__main__':
    main()
