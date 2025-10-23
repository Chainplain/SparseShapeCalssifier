# sparse_shape_classifier.py
# Classify shapes (sphere, cube, cylinder, cone, pyramid) from point distributions using spconv (occupancy only).
# After training, builds 10 test cases (2 per class) and shows a 2x5 subplot grid
# with each sample's points, predicted class, and confidence.

import os
import math
import random
import argparse
from dataclasses import dataclass

# --- Plotting (safe for headless) ---
if os.environ.get("DISPLAY", "") == "":
    # Use non-interactive backend when no display is present
    import matplotlib
    matplotlib.use("Agg")

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import spconv.pytorch as spconv
except Exception as e:
    raise RuntimeError(
        "Failed to import spconv.pytorch. Install a CUDA-enabled build (spconv >= 2.x) "
        "compatible with your PyTorch/CUDA.\n"
        "Example: pip install spconv-cu121  (or the cuNNN wheel matching your CUDA)"
    ) from e

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# --------- Utils: randomness, rotations, sampling ---------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def rand_rotation_matrix():
    """Random SO(3) via axis-angle."""
    axis = torch.randn(3)
    axis = axis / (axis.norm() + 1e-12)
    angle = torch.rand(1).item() * 2 * math.pi
    K = torch.tensor([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]], dtype=torch.float32)
    I = torch.eye(3)
    R = I + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
    return R

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
        if face == 0:   # x = +s
            xyz[:, 0] = s
            xyz[:, 1:] = uv
        elif face == 1: # x = -s
            xyz[:, 0] = -s
            xyz[:, 1:] = uv
        elif face == 2: # y = +s
            xyz[:, 1] = s
            xyz[:, [0,2]] = uv
        elif face == 3: # y = -s
            xyz[:, 1] = -s
            xyz[:, [0,2]] = uv
        elif face == 4: # z = +s
            xyz[:, 2] = s
            xyz[:, :2] = uv
        else:           # z = -s
            xyz[:, 2] = -s
            xyz[:, :2] = uv
        pts.append(xyz)
    pts = torch.cat(pts, dim=0)[:n_pts]
    if noise > 0:
        pts = pts + noise * torch.randn_like(pts)
    return pts

# --------- NEW SHAPE SAMPLERS ---------

def sample_cylinder_surface(n_pts: int, radius=0.5, height=1.0, noise=0.0):
    r, h = radius, height
    side_area = 2 * math.pi * r * h
    caps_area = 2 * math.pi * r * r
    ps = side_area / (side_area + caps_area)
    n_side = int(round(ps * n_pts))
    n_caps = max(0, n_pts - n_side)

    # Lateral surface
    theta = torch.rand(n_side) * 2 * math.pi
    z = torch.rand(n_side) * h - h / 2
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    side = torch.stack([x, y, z], dim=1) if n_side > 0 else torch.zeros(0, 3)

    # Caps (uniform disk via sqrt(u))
    n_top = n_caps // 2
    n_bot = n_caps - n_top

    def sample_disk(n):
        if n <= 0:
            return torch.zeros(0), torch.zeros(0)
        th = torch.rand(n) * 2 * math.pi
        rad = r * torch.sqrt(torch.rand(n))
        return rad * torch.cos(th), rad * torch.sin(th)

    xt, yt = sample_disk(n_top)
    top = torch.stack([xt, yt, torch.full_like(xt, h / 2)], dim=1) if n_top > 0 else torch.zeros(0, 3)
    xb, yb = sample_disk(n_bot)
    bot = torch.stack([xb, yb, torch.full_like(xb, -h / 2)], dim=1) if n_bot > 0 else torch.zeros(0, 3)

    pts = torch.cat([side, top, bot], dim=0)
    if noise > 0 and pts.numel() > 0:
        pts = pts + noise * torch.randn_like(pts)
    return pts

def sample_cone_surface(n_pts: int, radius=0.5, height=1.0, noise=0.0):
    r, h = radius, height
    s = math.sqrt(r*r + h*h)              # slant height
    lat_area = math.pi * r * s
    base_area = math.pi * r * r
    pl = lat_area / (lat_area + base_area)
    n_lat = int(round(pl * n_pts))
    n_base = max(0, n_pts - n_lat)

    # Lateral surface: v ~ 1 - sqrt(1-u)
    if n_lat > 0:
        u = torch.rand(n_lat)
        v = 1 - torch.sqrt(1 - u)
        theta = torch.rand(n_lat) * 2 * math.pi
        rv = r * (1 - v)
        x = rv * torch.cos(theta)
        y = rv * torch.sin(theta)
        z = -h/2 + v * h
        lat = torch.stack([x, y, z], dim=1)
    else:
        lat = torch.zeros(0, 3)

    # Base disk at z=-h/2
    if n_base > 0:
        th = torch.rand(n_base) * 2 * math.pi
        rad = r * torch.sqrt(torch.rand(n_base))
        xb = rad * torch.cos(th)
        yb = rad * torch.sin(th)
        base = torch.stack([xb, yb, torch.full_like(xb, -h/2)], dim=1)
    else:
        base = torch.zeros(0, 3)

    pts = torch.cat([lat, base], dim=0)
    if noise > 0 and pts.numel() > 0:
        pts = pts + noise * torch.randn_like(pts)
    return pts

def sample_pyramid_surface(n_pts: int, base_side=1.0, height=1.0, noise=0.0):
    s = base_side
    h = height
    half = s / 2
    apex = torch.tensor([0.0, 0.0, h/2])

    v0 = torch.tensor([-half, -half, -h/2])
    v1 = torch.tensor([ half, -half, -h/2])
    v2 = torch.tensor([ half,  half, -h/2])
    v3 = torch.tensor([-half,  half, -h/2])

    def tri_area(a, b, c):
        return 0.5 * torch.linalg.norm(torch.cross(b - a, c - a)).item()

    tri_faces = [(apex, v0, v1), (apex, v1, v2), (apex, v2, v3), (apex, v3, v0)]
    tri_areas = [tri_area(*f) for f in tri_faces]
    base_area = s * s
    total_area = sum(tri_areas) + base_area

    # Allocate samples by area
    n_tris = [int(round(n_pts * (A / total_area))) for A in tri_areas]
    # Fix rounding
    n_base = max(0, n_pts - sum(n_tris))

    def sample_triangle(a, b, c, n):
        if n <= 0:
            return torch.zeros(0, 3)
        u = torch.sqrt(torch.rand(n)).unsqueeze(1)
        v = torch.rand(n, 1)
        w = 1 - u
        vv = u * (1 - v)
        ww = u * v
        pts = w * a + vv * b + ww * c
        return pts

    parts = [sample_triangle(a, b, c, n) for (a, b, c), n in zip(tri_faces, n_tris)]

    if n_base > 0:
        xy = torch.rand(n_base, 2) * s - half
        base = torch.cat([xy, torch.full((n_base, 1), -h/2)], dim=1)
        parts.append(base)

    pts = torch.cat(parts, dim=0) if parts else torch.zeros(0, 3)
    if noise > 0 and pts.numel() > 0:
        pts = pts + noise * torch.randn_like(pts)
    return pts

def voxelize(points_xyz: torch.Tensor, grid_size=(32,32,32), pad=1e-6):
    """
    Normalize points to fit in [-0.5,0.5]^3 then map to [0, G-1] grid.
    Return unique voxel coords [N_active, 3] (int32).
    """
    if points_xyz.numel() == 0:
        return torch.zeros(0, 3, dtype=torch.int32)

    Gx, Gy, Gz = grid_size
    max_abs = float(points_xyz.abs().max().item())
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
    return coords_u  # int32

# --------- Dataset ---------

@dataclass
class ShapeCfg:
    n_points: int = 2048
    grid_size: tuple = (32, 32, 32)
    noise: float = 0.01
    rot_aug: bool = True
    dropout: float = 0.1
    classes: tuple = ("sphere", "cube", "cylinder", "cone", "pyramid")

class ShapeDataset(Dataset):
    def __init__(self, split="train", size=20000, cfg: ShapeCfg = ShapeCfg()):
        self.split = split
        self.size = size
        self.cfg = cfg
        self.num_classes = len(cfg.classes)

    def __len__(self):
        return self.size

    def _sample_shape(self, label: int, n: int):
        if label == 0:  # sphere
            r = 0.2 + torch.rand(1).item() * 0.5
            return sample_sphere_surface(n, radius=r, noise=self.cfg.noise)
        elif label == 1:  # cube
            s = 0.6 + torch.rand(1).item() * 0.6
            return sample_cube_surface(n, side=s, noise=self.cfg.noise)
        elif label == 2:  # cylinder
            r = 0.25 + torch.rand(1).item() * 0.35
            h = 0.6 + torch.rand(1).item() * 0.8
            return sample_cylinder_surface(n, radius=r, height=h, noise=self.cfg.noise)
        elif label == 3:  # cone
            r = 0.25 + torch.rand(1).item() * 0.35
            h = 0.6 + torch.rand(1).item() * 0.8
            return sample_cone_surface(n, radius=r, height=h, noise=self.cfg.noise)
        else:  # pyramid
            s = 0.6 + torch.rand(1).item() * 0.6
            h = 0.6 + torch.rand(1).item() * 0.8
            return sample_pyramid_surface(n, base_side=s, height=h, noise=self.cfg.noise)

    def __getitem__(self, idx):
        label = torch.tensor(idx % self.num_classes, dtype=torch.long)
        n = self.cfg.n_points

        pts = self._sample_shape(label.item(), n)

        if self.cfg.rot_aug:
            R = rand_rotation_matrix()
            pts = (R @ pts.t()).t()
        if self.cfg.dropout > 0 and pts.shape[0] > 0:
            mask = torch.rand(pts.shape[0]) > self.cfg.dropout
            # ensure at least one point remains
            if mask.sum().item() == 0:
                mask[random.randrange(pts.shape[0])] = True
            pts = pts[mask]

        coords = voxelize(pts, self.cfg.grid_size)   # [N_active, 3] int32
        feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)  # occupancy
        return feats, coords, label, torch.tensor(self.cfg.grid_size, dtype=torch.int32)

def collate_sparse(batch):
    feats_list, coords_list, labels, grid_sizes = zip(*batch)
    B = len(batch)
    spatial_shape = grid_sizes[0].tolist()

    feats_cat = torch.cat(feats_list, dim=0) if feats_list else torch.zeros(0,1)
    batch_col = []
    for b, coords in enumerate(coords_list):
        if coords.numel() == 0:
            continue
        bcol = torch.full((coords.shape[0], 1), b, dtype=torch.int32)
        batch_col.append(torch.cat([bcol, coords.to(torch.int32)], dim=1))
    indices_cat = torch.cat(batch_col, dim=0) if batch_col else torch.zeros(0, 4, dtype=torch.int32)
    labels = torch.stack(labels, dim=0)
    return feats_cat, indices_cat, labels, spatial_shape, B

# --------- Model (unchanged backbone; head size adapts to classes) ---------

class SparseClassifier(nn.Module):
    def __init__(self, in_ch=1, num_classes=2):
        super().__init__()
        self.backbone = spconv.SparseSequential(
            spconv.SubMConv3d(in_ch, 32, kernel_size=3, padding=1, indice_key="s0"),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),

            spconv.SparseConv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),

            spconv.SubMConv3d(64, 64, kernel_size=3, padding=1, indice_key="s1"),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),

            spconv.SparseConv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),

            spconv.SubMConv3d(128, 128, kernel_size=3, padding=1, indice_key="s2"),
            nn.BatchNorm1d(128), nn.ReLU(inplace=True),
        )
        self.to_dense = spconv.ToDense()
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x_sparse: spconv.SparseConvTensor):
        x = self.backbone(x_sparse)
        x = self.to_dense(x)   # [B, C, D, H, W]
        return self.head(x)    # [B, num_classes]

# --------- Train / Eval ---------

def run_epoch(model, loader, device, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total, correct, total_loss = 0, 0, 0.0
    criterion = nn.CrossEntropyLoss()

    for feats, indices, labels, spatial_shape, B in loader:
        # skip empty sparse batches (shouldn't happen often but be robust)
        if indices.numel() == 0:
            continue

        feats = feats.to(device, non_blocking=True)
        indices = indices.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        x_sparse = spconv.SparseConvTensor(
            features=feats,
            indices=indices,                 # int32, [sumN, 1+3]
            spatial_shape=spatial_shape,     # [X,Y,Z]
            batch_size=B
        )

        logits = model(x_sparse)
        loss = criterion(logits, labels)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            correct += (pred == labels).sum().item()
            total += labels.numel()
            total_loss += loss.item() * labels.numel()

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc

# --------- Demo helpers ---------

def build_sparse_batch_from_points(points_list, grid_size, device):
    feats_all, indices_all = [], []
    for b, pts in enumerate(points_list):
        coords = voxelize(pts, grid_size)  # [N_active,3] int32
        if coords.numel() == 0:
            continue
        feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
        bcol = torch.full((coords.shape[0], 1), b, dtype=torch.int32)
        inds = torch.cat([bcol, coords.to(torch.int32)], dim=1)
        feats_all.append(feats)
        indices_all.append(inds)
    if len(indices_all) == 0:
        return torch.zeros(0,1), torch.zeros(0,4, dtype=torch.int32), list(grid_size), len(points_list)
    feats_cat = torch.cat(feats_all, dim=0).to(device)
    indices_cat = torch.cat(indices_all, dim=0).to(device)
    return feats_cat, indices_cat, list(grid_size), len(points_list)

def make_test_cases_10(n_points, noise):
    """
    Build 10 cases (2 per class): sphere, cube, cylinder, cone, pyramid.
    Returns: points_list, gt_labels, names
    labels: 0=sphere,1=cube,2=cylinder,3=cone,4=pyramid
    """
    cases, names, labels = [], [], []

    # 2 spheres
    for r in [0.3, 0.6]:
        pts = sample_sphere_surface(n_points, radius=r, noise=noise)
        pts = (rand_rotation_matrix() @ pts.t()).t()
        cases.append(pts); labels.append(0); names.append(f"sphere r={r:.2f}")

    # 2 cubes
    for s in [0.7, 1.1]:
        pts = sample_cube_surface(n_points, side=s, noise=noise)
        pts = (rand_rotation_matrix() @ pts.t()).t()
        cases.append(pts); labels.append(1); names.append(f"cube s={s:.2f}")

    # 2 cylinders
    for r, h in [(0.35, 0.8), (0.5, 1.2)]:
        pts = sample_cylinder_surface(n_points, radius=r, height=h, noise=noise)
        pts = (rand_rotation_matrix() @ pts.t()).t()
        cases.append(pts); labels.append(2); names.append(f"cyl r={r:.2f},h={h:.2f}")

    # 2 cones
    for r, h in [(0.35, 0.8), (0.5, 1.2)]:
        pts = sample_cone_surface(n_points, radius=r, height=h, noise=noise)
        pts = (rand_rotation_matrix() @ pts.t()).t()
        cases.append(pts); labels.append(3); names.append(f"cone r={r:.2f},h={h:.2f}")

    # 2 pyramids
    for s, h in [(0.8, 0.8), (1.1, 1.2)]:
        pts = sample_pyramid_surface(n_points, base_side=s, height=h, noise=noise)
        pts = (rand_rotation_matrix() @ pts.t()).t()
        cases.append(pts); labels.append(4); names.append(f"pyr s={s:.2f},h={h:.2f}")

    return cases, torch.tensor(labels, dtype=torch.long), names

def visualize_cases(points_list, preds, probs, names, class_names, save_path=None):
    N = len(points_list)
    cols = min(5, N)
    rows = math.ceil(N / cols)
    fig = plt.figure(figsize=(4 * cols, 4 * rows))

    all_points = torch.cat([p if p.numel() > 0 else torch.zeros(1,3) for p in points_list], dim=0)
    x_min, y_min, z_min = all_points.min(dim=0).values
    x_max, y_max, z_max = all_points.max(dim=0).values

    for i, pts in enumerate(points_list):
        if pts.numel() == 0:
            pts = torch.zeros(1, 3)
        xs = pts[:, 0].cpu().numpy()
        ys = pts[:, 1].cpu().numpy()
        zs = pts[:, 2].cpu().numpy()

        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        ax.scatter(xs, ys, zs, s=6, depthshade=True)
        pred_name = class_names[preds[i]]
        ax.set_title(f"{names[i]}\nPred: {pred_name}  (p={probs[i]:.2f})")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.grid(True)

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=150)
    # In headless mode, just saving is fine; show() won't crash due to Agg, but it's optional.
    plt.show()

# --------- Main ---------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--points", type=int, default=2048)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--rot_aug", action="store_true", default=True, help="Enable rotation augmentation")
    parser.add_argument("--no_rot_aug", action="store_true", help="Disable rotation augmentation")
    parser.add_argument("--quick", action="store_true", help="Use fewer epochs and a small dataset for speed")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader workers (0 is safest across OSes)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for spconv in this example.\n"
            "Enable a CUDA GPU and install matching CUDA builds of PyTorch + spconv."
        )

    device = torch.device("cuda")
    set_seed(args.seed)

    rot_aug = False if args.no_rot_aug else args.rot_aug

    cfg = ShapeCfg(
        n_points=args.points,
        grid_size=(args.grid, args.grid, args.grid),
        noise=args.noise,
        rot_aug=rot_aug,
        dropout=args.dropout,
        classes=("sphere", "cube", "cylinder", "cone", "pyramid"),
    )

    # Optionally shrink training for a quick demo
    train_size = 4000 if args.quick else 20000
    val_size   = 800  if args.quick else 2000
    epochs     = 2    if args.quick else args.epochs

    train_ds = ShapeDataset(split="train", size=train_size, cfg=cfg)
    val_ds   = ShapeDataset(split="val",   size=val_size,   cfg=cfg)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True, collate_fn=collate_sparse
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=max(0, args.workers // 2), pin_memory=True, collate_fn=collate_sparse
    )

    model = SparseClassifier(in_ch=1, num_classes=len(cfg.classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"Model is on device: {next(model.parameters()).device}")

    best_acc, best_ep = 0.0, -1
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, device, optimizer=None)
        if va_acc > best_acc:
            best_acc, best_ep = va_acc, epoch
            torch.save(model.state_dict(), "best_sparse_shape_cls.pt")
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f} | best {best_acc:.3f} @ {best_ep}")

    print("Saved best model to best_sparse_shape_cls.pt")

    # --------- Build test cases, run the model, visualize ----------
    model.eval()
    if os.path.exists("best_sparse_shape_cls.pt"):
        model.load_state_dict(torch.load("best_sparse_shape_cls.pt", map_location=device))

    test_points, gt_labels, names = make_test_cases_10(n_points=args.points, noise=args.noise)

    feats, indices, spatial_shape, B = build_sparse_batch_from_points(test_points, cfg.grid_size, device)
    if indices.numel() == 0:
        raise RuntimeError("All test point sets were empty after voxelization. Try lowering dropout or increasing points.")

    x_sparse = spconv.SparseConvTensor(
        features=feats,
        indices=indices,
        spatial_shape=spatial_shape,
        batch_size=B
    )

    with torch.no_grad():
        logits = model(x_sparse)  # [B, num_classes]
        probs = torch.softmax(logits, dim=1)
        pred = probs.argmax(dim=1).cpu()
        conf = probs.max(dim=1).values.cpu().tolist()

    # Print a table of results
    print(f"\n=== Test Cases ({B}) ===")
    classes = cfg.classes
    for i in range(B):
        gt = classes[gt_labels[i].item()]
        pd = classes[pred[i].item()]
        print(f"{i:02d}: {names[i]:22s} | GT: {gt:8s} | Pred: {pd:8s} | p={conf[i]:.3f}")

    visualize_cases(test_points, pred.tolist(), conf, names, class_names=classes, save_path="demo_grid.png")
    print("Saved visualization to demo_grid.png")

if __name__ == "__main__":
    # Optional: enable spconv debug dumps (not required)
    # os.environ["SPCONV_DEBUG_SAVE_PATH"] = "/tmp/spconv_debug"
    main()
