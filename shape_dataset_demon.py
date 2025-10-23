# shape_dataset_demo.py
# Demo: visualize 8 synthetic samples (4 spheres, 4 cubes) using make_8_test_cases
# in a 2x4 subplot grid. Each subplot shows the voxelized active coordinates.

import math
import random
from dataclasses import dataclass

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables 3D)

# ---------- Synthetic shape samplers ----------

def sample_sphere_surface(n_pts: int, radius=0.5, noise=0.0):
    p = torch.randn(n_pts, 3)
    p = p / (p.norm(dim=1, keepdim=True) + 1e-8)
    p = radius * p
    if noise > 0:
        p = p + noise * torch.randn_like(p)
    return p

def sample_cube_surface(n_pts: int, side=1.0, noise=0.0):
    s = side / 2.0
    pts = []
    per_face = max(1, math.ceil(n_pts / 6))
    for face in range(6):
        uv = torch.rand(per_face, 2) * side - s  # in [-s, s]
        xyz = torch.zeros(per_face, 3)
        if face == 0:
            xyz[:, 0] = s;        xyz[:, 1:] = uv
        elif face == 1:
            xyz[:, 0] = -s;       xyz[:, 1:] = uv
        elif face == 2:
            xyz[:, 1] = s;        xyz[:, [0,2]] = uv
        elif face == 3:
            xyz[:, 1] = -s;       xyz[:, [0,2]] = uv
        elif face == 4:
            xyz[:, 2] = s;        xyz[:, :2] = uv
        else:
            xyz[:, 2] = -s;       xyz[:, :2] = uv
        pts.append(xyz)
    pts = torch.cat(pts, dim=0)[:n_pts]
    if noise > 0:
        pts = pts + noise * torch.randn_like(pts)
    return pts

def rand_rotation_matrix():
    axis = torch.randn(3)
    axis = axis / (axis.norm() + 1e-8)
    angle = torch.rand(1).item() * 2 * math.pi
    K = torch.tensor([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], dtype=torch.float32)
    I = torch.eye(3)
    R = I + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
    return R

def voxelize(points_xyz: torch.Tensor, grid_size=(32,32,32), pad=1e-6):
    """Map points to voxel grid [0, G-1]^3 and return unique voxel coordinates (int32)."""
    Gx, Gy, Gz = grid_size
    max_abs = torch.max(points_xyz.abs()).item()
    if max_abs < 1e-8:
        points_norm = points_xyz.clone()
    else:
        points_norm = points_xyz / (2.0*max_abs + pad)  # ~[-0.5,0.5]
    points_norm = points_norm + 0.5  # -> [0,1]
    grid = torch.stack([
        (points_norm[:, 0] * (Gx - 1)).clamp(0, Gx - 1),
        (points_norm[:, 1] * (Gy - 1)).clamp(0, Gy - 1),
        (points_norm[:, 2] * (Gz - 1)).clamp(0, Gz - 1),
    ], dim=1).floor().to(torch.int32)  # [N,3]
    coords_u = torch.unique(grid, dim=0)
    return coords_u

# ---------- Use make_8_test_cases from previous answer ----------

def make_8_test_cases(n_points, noise):
    """
    Returns: points_list, gt_labels, names
    0=sphere, 1=cube
    """
    cases = []
    names = []
    labels = []

    # 4 spheres with different radii/rotations/noise
    radii = [0.25, 0.35, 0.5, 0.65]
    for r in radii:
        pts = sample_sphere_surface(n_points, radius=r, noise=noise)
        R = rand_rotation_matrix()
        pts = (R @ pts.t()).t()
        cases.append(pts)
        labels.append(0)
        names.append(f"sphere r={r:.2f}")

    # 4 cubes with different scales/rotations/noise
    sides = [0.6, 0.8, 1.0, 1.2]
    for s in sides:
        pts = sample_cube_surface(n_points, side=s, noise=noise)
        R = rand_rotation_matrix()
        pts = (R @ pts.t()).t()
        cases.append(pts)
        labels.append(1)
        names.append(f"cube s={s:.2f}")

    return cases, torch.tensor(labels, dtype=torch.long), names

# ---------- Visualization: 8 subplots (2x4) ----------

def show_8_samples(points_list, names, grid_size=(32,32,32)):
    """
    Plot RAW point sets directly (no voxelization, no normalization) in a 2x4 grid.
    All subplots use independent axis limits and aspect ratios.
    Signature unchanged; `grid_size` is unused but kept for compatibility.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(16, 8))
    rows, cols = 2, 4
    for i, pts in enumerate(points_list):
        if isinstance(pts, torch.Tensor):
            p = pts.detach().cpu().numpy()
        else:
            p = np.asarray(pts)
        if p.size == 0:
            p = np.zeros((1, 3), dtype=float)

        xs, ys, zs = p[:, 0], p[:, 1], p[:, 2]

        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.scatter(xs, ys, zs, s=6, depthshade=True)
        ax.set_title(names[i])

        # Independent limits for each subplot
        ax.set_xlim(xs.min(), xs.max())
        ax.set_ylim(ys.min(), ys.max())
        ax.set_zlim(zs.min(), zs.max())

        # Equal aspect for 3D (Matplotlib >= 3.3)
        try:
            ax.set_box_aspect([1, 1, 1])
        except Exception:
            pass  # older Matplotlib, limits above still ensure same numeric scale

        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.grid(True)

    plt.tight_layout()
    plt.show()



# ---------- Main ----------

if __name__ == "__main__":
    # Reproducibility
    random.seed(0); np.random.seed(0); torch.manual_seed(0)

    # Generate 8 test cases (4 spheres, 4 cubes) and visualize them
    n_points = 2048
    noise = 0.02
    grid_size = (32, 32, 32)

    points_list, gt_labels, names = make_8_test_cases(n_points=n_points, noise=noise)
    show_8_samples(points_list, names, grid_size=grid_size)
