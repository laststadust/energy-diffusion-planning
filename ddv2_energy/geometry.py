"""Vehicle-footprint road energy in the planning-time rear-axle frame.

x points forward, y points left, positions are in metres, and yaw is in radians."""
import math
import torch
import torch.nn.functional as F

def road_energy(traj, sdf, origin, resolution, length=5.176, width=2.297, rear_axle_to_center=1.461, nx=9, ny=5):
    """Return mean squared road-boundary violations for each candidate.

    traj contains xy [B,T,2] and yaw [B,T]. sdf is [B,1,Ny,Nx], or a
    single field broadcast across candidates. Positive distances denote road
    interior. origin gives the (x,y) centre of pixel (0,0); columns follow x.
    resolution, vehicle dimensions and rear_axle_to_center are in metres.

    A regular footprint grid covers the vehicle; its covering radius is
    subtracted from the minimum sampled SDF. This radius is distinct from
    rasterization error. Outside the observed field, positive border values
    are clipped before subtracting the distance to the field boundary.
    """
    if nx < 2 or ny < 2 or width <= 0 or (length <= 0) or (resolution <= 0):
        raise ValueError('road grid parameters must be positive')
    xy = traj['xy']
    yaw = traj['yaw']
    h, w = sdf.shape[-2:]
    if h < 2 or w < 2 or sdf.ndim != 4 or (sdf.shape[1] != 1):
        raise ValueError('SDF shape must be [B,1,Ny,Nx]')
    vx = torch.linspace(-length / 2, length / 2, nx, device=xy.device, dtype=xy.dtype) + rear_axle_to_center
    vy = torch.linspace(-width / 2, width / 2, ny, device=xy.device, dtype=xy.dtype)
    x = vx.repeat_interleave(ny)
    y = vy.repeat(nx)
    cs = yaw.cos()[..., None]
    sn = yaw.sin()[..., None]
    sample = torch.stack((xy[..., 0, None] + cs * x - sn * y, xy[..., 1, None] + sn * x + cs * y), -1)
    o = torch.as_tensor(origin, dtype=xy.dtype, device=xy.device).reshape(-1, 1, 1, 2)
    pixel = (sample - o) / resolution
    hi = xy.new_tensor([w - 1, h - 1])
    grid = 2 * pixel / hi - 1
    sdf = sdf.to(xy).expand(xy.shape[0], -1, -1, -1)
    field = F.grid_sample(sdf, grid, mode='bilinear', padding_mode='border', align_corners=True)[:, 0]
    outside = torch.maximum(-pixel, pixel - hi).clamp_min(0)
    gap = torch.linalg.vector_norm(outside, dim=-1) * resolution
    field = torch.where(gap > 0, torch.minimum(field, torch.zeros_like(field)) - gap, field)
    cover = math.hypot(length / (nx - 1), width / (ny - 1)) * 0.5
    margin = field.amin(-1) - cover
    return F.relu(-margin / width).square().mean(-1)
