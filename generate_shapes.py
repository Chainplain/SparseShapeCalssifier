"""
Generate 3D point clouds for different shapes.
Supports: sphere, cube, cylinder, cone, pyramid
"""

import numpy as np
from typing import Tuple


def generate_sphere(num_points: int = 1000, radius: float = 1.0) -> np.ndarray:
    """Generate points on a sphere surface using Fibonacci sphere algorithm."""
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1
        radius_at_y = np.sqrt(1 - y * y) * radius  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius_at_y
        z = np.sin(theta) * radius_at_y
        y = y * radius

        points.append([x, y, z])

    return np.array(points, dtype=np.float32)


def generate_cube(num_points: int = 1000, side_length: float = 2.0) -> np.ndarray:
    """Generate points on a cube surface."""
    points = []
    half_side = side_length / 2.0
    points_per_face = num_points // 6
    remainder = num_points - (points_per_face * 6)

    # Generate points on each face
    for face in range(6):
        # Add extra point to first faces if there's a remainder
        face_points = points_per_face + (1 if face < remainder else 0)
        
        for _ in range(face_points):
            u = np.random.uniform(-half_side, half_side)
            v = np.random.uniform(-half_side, half_side)

            if face == 0:  # Front face (z = half_side)
                points.append([u, v, half_side])
            elif face == 1:  # Back face (z = -half_side)
                points.append([u, v, -half_side])
            elif face == 2:  # Right face (x = half_side)
                points.append([half_side, u, v])
            elif face == 3:  # Left face (x = -half_side)
                points.append([-half_side, u, v])
            elif face == 4:  # Top face (y = half_side)
                points.append([u, half_side, v])
            elif face == 5:  # Bottom face (y = -half_side)
                points.append([u, -half_side, v])

    return np.array(points, dtype=np.float32)


def generate_cylinder(num_points: int = 1000, radius: float = 1.0, height: float = 2.0) -> np.ndarray:
    """Generate points on a cylinder surface."""
    points = []
    half_height = height / 2.0

    # Points on the side surface
    side_points = int(num_points * 0.7)
    for _ in range(side_points):
        theta = np.random.uniform(0, 2 * np.pi)
        y = np.random.uniform(-half_height, half_height)
        x = radius * np.cos(theta)
        z = radius * np.sin(theta)
        points.append([x, y, z])

    # Points on top and bottom circles
    cap_points = (num_points - side_points) // 2
    for _ in range(cap_points):
        r = np.random.uniform(0, radius)
        theta = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        z = r * np.sin(theta)
        points.append([x, half_height, z])  # Top
        points.append([x, -half_height, z])  # Bottom

    return np.array(points, dtype=np.float32)


def generate_cone(num_points: int = 1000, radius: float = 1.0, height: float = 2.0) -> np.ndarray:
    """Generate points on a cone surface."""
    points = []
    half_height = height / 2.0

    # Points on the side surface
    side_points = int(num_points * 0.8)
    for _ in range(side_points):
        # Height from bottom to top
        y = np.random.uniform(-half_height, half_height)
        # Radius decreases linearly from base to apex
        r = radius * (1 - (y + half_height) / height)
        theta = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        z = r * np.sin(theta)
        points.append([x, y, z])

    # Points on the base circle
    base_points = num_points - side_points
    for _ in range(base_points):
        r = np.random.uniform(0, radius)
        theta = np.random.uniform(0, 2 * np.pi)
        x = r * np.cos(theta)
        z = r * np.sin(theta)
        points.append([x, -half_height, z])

    return np.array(points, dtype=np.float32)


def generate_pyramid(num_points: int = 1000, base_size: float = 2.0, height: float = 2.0) -> np.ndarray:
    """Generate points on a pyramid surface (square base)."""
    points = []
    half_base = base_size / 2.0
    half_height = height / 2.0

    # Points on the base
    base_points = int(num_points * 0.2)
    for _ in range(base_points):
        x = np.random.uniform(-half_base, half_base)
        z = np.random.uniform(-half_base, half_base)
        points.append([x, -half_height, z])

    # Points on the four triangular faces
    face_points = (num_points - base_points) // 4
    for face in range(4):
        for _ in range(face_points):
            # Parameter along the edge (0 to 1)
            t = np.random.uniform(0, 1)
            # Parameter from base to apex (0 to 1)
            s = np.random.uniform(0, 1)

            if face == 0:  # Front face
                base_x = half_base
                base_z = np.random.uniform(-half_base, half_base)
                x = base_x * (1 - s)
                z = base_z * (1 - s)
            elif face == 1:  # Back face
                base_x = -half_base
                base_z = np.random.uniform(-half_base, half_base)
                x = base_x * (1 - s)
                z = base_z * (1 - s)
            elif face == 2:  # Right face
                base_x = np.random.uniform(-half_base, half_base)
                base_z = half_base
                x = base_x * (1 - s)
                z = base_z * (1 - s)
            else:  # Left face
                base_x = np.random.uniform(-half_base, half_base)
                base_z = -half_base
                x = base_x * (1 - s)
                z = base_z * (1 - s)

            y = -half_height + s * height
            points.append([x, y, z])

    return np.array(points, dtype=np.float32)


def generate_shape(shape_type: str, num_points: int = 1000) -> Tuple[np.ndarray, int]:
    """
    Generate a point cloud for a specific shape type.
    
    Args:
        shape_type: One of 'sphere', 'cube', 'cylinder', 'cone', 'pyramid'
        num_points: Number of points to generate
        
    Returns:
        Tuple of (point_cloud, label) where label is an integer class ID
    """
    shape_generators = {
        'sphere': (generate_sphere, 0),
        'cube': (generate_cube, 1),
        'cylinder': (generate_cylinder, 2),
        'cone': (generate_cone, 3),
        'pyramid': (generate_pyramid, 4)
    }
    
    if shape_type not in shape_generators:
        raise ValueError(f"Unknown shape type: {shape_type}. Must be one of {list(shape_generators.keys())}")
    
    generator, label = shape_generators[shape_type]
    points = generator(num_points)
    
    return points, label


def get_shape_name(label: int) -> str:
    """Convert label ID to shape name."""
    label_to_name = {
        0: 'sphere',
        1: 'cube',
        2: 'cylinder',
        3: 'cone',
        4: 'pyramid'
    }
    return label_to_name.get(label, 'unknown')


if __name__ == '__main__':
    # Test shape generation
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    shapes = ['sphere', 'cube', 'cylinder', 'cone', 'pyramid']
    
    fig = plt.figure(figsize=(15, 3))
    for i, shape in enumerate(shapes):
        points, label = generate_shape(shape, num_points=500)
        
        ax = fig.add_subplot(1, 5, i + 1, projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=1)
        ax.set_title(f'{shape} (label={label})')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
    plt.tight_layout()
    plt.savefig('shapes_visualization.png')
    print("Shape visualization saved to shapes_visualization.png")
