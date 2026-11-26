"""Trajectory representation shared by road and interaction evaluation."""
import math
import torch

def pose_traj(poses, v0, dt=0.5):
    """Convert candidate rear-axle poses [K,H,3] to graph input.

    Prepend zero position/yaw at the current planning time. v0 is the measured
    nonnegative speed of this scene. Future speeds are interval displacement
    magnitudes divided by dt, not independently predicted instantaneous speeds.
    No future observation is used. Returns xy [K,H+1,2] and yaw/speed [K,H+1].
    """
    if poses.ndim != 3 or poses.shape[-1] != 3 or min(poses.shape[:2]) < 1:
        raise ValueError('expected nonempty [K,H,3] native poses')
    if not torch.is_floating_point(poses) or not math.isfinite(dt) or dt <= 0:
        raise ValueError('expected floating-point native poses and positive finite dt')
    measured = torch.as_tensor(v0, dtype=poses.dtype, device=poses.device)
    if measured.numel() != 1 or not bool(torch.isfinite(measured).all()) or measured.item() < 0:
        raise ValueError('v0 must be the finite nonnegative measured speed of this scene')
    k = poses.shape[0]
    xy = torch.cat((poses.new_zeros(k, 1, 2), poses[..., :2]), dim=1)
    yaw = torch.cat((poses.new_zeros(k, 1), poses[..., 2]), dim=1)
    speed = torch.cat((measured.detach().reshape(1, 1).expand(k, 1), torch.linalg.vector_norm(xy[:, 1:] - xy[:, :-1], dim=-1) / dt), 1)
    return {'xy': xy, 'yaw': yaw, 'speed': speed}
