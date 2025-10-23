import open3d as o3d

# spconv_latent_demo.py
# Minimal demo: 3D point cloud -> spconv SparseConvTensor -> tiny sparse CNN -> compact latent

import argparse, math, numpy as np, torch, torch.nn as nn

def make_cloud(n=20000, seed=42):
    rng = np.random.default_rng(seed)
    # points on a sphere surface
    phi = rng.uniform(0, 2*math.pi, size=n)
    cost = rng.uniform(-1, 1, size=n)
    theta = np.arccos(cost); r = 0.9
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    xyz = np.stack([x, y, z], axis=1).astype(np.float32)
    feats = np.ones((xyz.shape[0], 1), np.float32)  # in_channels=1
    return torch.from_numpy(xyz), torch.from_numpy(feats)

def voxelize_for_spconv(xyz, feats=None, voxel_size=0.05, batch_size=1):
    """
    xyz: (N,3) meters (float32). feats: (N,C) float32 or None->ones.
    Returns:
      indices  : (M, 4) int32  [x, y, z, batch]  (non-negative)
      features : (M, C) float32
      spatial_shape: (Dx, Dy, Dz) ints
      batch_size   : int
    """
    if feats is None:
        feats = torch.ones((xyz.size(0), 1), dtype=torch.float32)

    # 1) metric -> integer voxel coords
    coords = torch.floor(xyz / voxel_size).to(torch.int32)          # (N,3)

    # 2) shift coords so min is 0 in each axis (spconv needs non-negative indices)
    mins = coords.min(dim=0).values
    coords_shift = coords - mins                                    # (N,3) >= 0

    # 3) unique voxels (pick-first aggregation)
    uniq, inv = torch.unique(coords_shift, dim=0, return_inverse=True)
    _, sel = torch.unique(inv, sorted=True, return_index=True)
    uniq = uniq[sel]                                                # (M,3) unique, stable order
    features = feats[sel].to(torch.float32)                         # (M,C)

    # 4) build indices [x,y,z,b] and spatial shape
    # NOTE: spconv expects [x, y, z, b] with b in last column.
    batch_col = torch.zeros((uniq.size(0), 1), dtype=torch.int32)
    indices = torch.cat([uniq, batch_col], dim=1).contiguous()      # (M,4)

    # spatial shape is max index + 1 (size along each dim)
    spatial_shape = tuple((uniq.max(dim=0).values + 1).tolist())    # (Dx, Dy, Dz)

    return indices, features, spatial_shape, batch_size

class TinySparseBackbone(nn.Module):
    """
    Lightweight sparse CNN:
      SubMConv3d -> BN -> ReLU x 3 (all stride=1)
    Produces per-voxel embedding; we will global-average it to get a compact latent.
    """
    def __init__(self, in_ch=1, width=64, out_ch=128):
        super().__init__()
        import spconv.pytorch as spconv
        self.net = spconv.SparseSequential(
            spconv.SubMConv3d(in_ch, width, kernel_size=3, indice_key='subm'),
            nn.BatchNorm1d(width), nn.ReLU(True),
            spconv.SubMConv3d(width, width, kernel_size=3, indice_key='subm'),
            nn.BatchNorm1d(width), nn.ReLU(True),
            spconv.SubMConv3d(width, out_ch, kernel_size=3, indice_key='subm'),
        )
    def forward(self, x):  # x: spconv.SparseConvTensor
        return self.net(x)

def global_mean_sparse(tensor):
    """
    Mean-pool SparseConvTensor features per batch item.
    tensor.features: (M, C), tensor.indices: (M, 4) with batch in last column.
    """
    bcol = tensor.indices[:, 3]
    B = int(bcol.max().item()) + 1
    outs = []
    for b in range(B):
        m = (bcol == b)
        outs.append(tensor.features[m].mean(dim=0, keepdim=True))
    return torch.cat(outs, dim=0)  # (B, C)

def main():
    import spconv.pytorch as spconv

    ap = argparse.ArgumentParser()
    ap.add_argument('--voxel_size', type=float, default=0.05)
    ap.add_argument('--latent_dim', type=int, default=128)
    ap.add_argument('--num_points', type=int, default=20000)
    ap.add_argument('--cpu', action='store_true')
    args = ap.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print('Using device:', device)

    # 1) Make synthetic cloud
    xyz, feats = make_cloud(args.num_points)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(xyz.numpy())
    # print("Showing point cloud in Open3D viewer...")
    # o3d.visualization.draw_geometries([pcd])
    
    
    # print(xyz)
    
    # # 2) Voxelize for spconv (handles negative coords by shifting to 0)
    indices, features, spatial_shape, batch_size = voxelize_for_spconv(xyz, feats, args.voxel_size, 1)
    print(f'Points: {xyz.shape[0]:,} → Active voxels: {indices.shape[0]:,}, spatial_shape={spatial_shape}')

    # # 3) Build SparseConvTensor
    # st = spconv.SparseConvTensor(
    #     features=features.to(device),   # (M, C)
    #     indices=indices.to(device),     # (M, 4) int32, [x,y,z,b]
    #     spatial_shape=spatial_shape,    # tuple of 3 ints
    #     batch_size=batch_size
    # )

    # # 4) Backbone and global pooling
    # net = TinySparseBackbone(in_ch=features.shape[1], width=64, out_ch=args.latent_dim).to(device)
    # net.eval()
    # with torch.no_grad():
    #     y = net(st)                 # SparseConvTensor with per-voxel features (M, latent_dim)
    #     z = global_mean_sparse(y)   # (B, latent_dim), here B=1

    # print('Per-voxel features:', tuple(y.features.shape))
    # print('Global latent:', tuple(z.shape), f'(expected [1, {args.latent_dim}])')
    # print('Latent (first 10 dims):', z[0, :10].cpu().numpy())

if __name__ == '__main__':
    main()
