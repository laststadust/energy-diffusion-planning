"""Candidate-conditioned contour and vehicle-safety graphs.

All coordinates use the planning-time rear axle: x forward, y left, metres,
seconds and radians. Actor fields are [cx, cy, yaw, length, width, vx, vy,
is_vehicle, class_id, heading_reliable]; the last two fields are optional.
Class IDs default to 0=unknown, 1=vehicle, 2=pedestrian, 3=bicycle.
Graph evaluation extrapolates current observations at constant velocity.
Future recorded objects are accepted only by label_cost for supervision.
"""
import math
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

@dataclass(frozen=True)
class InteractionConfig:
    """Vehicle geometry, graph dimensions and directional-distance parameters.

Default dimensions describe the nuPlan Pacifica rear-axle convention."""
    ego_length: float = 5.176
    ego_width: float = 2.297
    rear_axle_to_center: float = 1.461
    rho: float = 0.7
    reaction_accel: float = 1.5
    brake_min: float = 3.0
    brake_max: float = 6.0
    lateral_accel: float = 0.4
    lateral_brake: float = 1.0
    lateral_buffer: float = 0.1
    contour_buffer: float = 0.5
    heading_max: float = math.pi / 4
    proximity: float = 4.0
    hidden: int = 32
    layers: int = 2
    heads: int = 4
    classes: int = 16

    def __post_init__(self):
        if not 0 < self.brake_min <= self.brake_max or self.lateral_brake <= 0 or self.rho < 0 or (min(self.ego_length, self.ego_width, self.heading_max, self.proximity) <= 0) or (min(self.reaction_accel, self.lateral_accel, self.lateral_buffer, self.contour_buffer) < 0) or (self.layers < 1) or (self.heads < 1) or self.hidden % self.heads or (self.classes < 3):
            raise ValueError('invalid interaction configuration')

def _axis(yaw):
    return torch.stack((yaw.cos(), yaw.sin()), -1)

def _left(u):
    return torch.stack((-u[..., 1], u[..., 0]), -1)

def rect_corners(center, yaw, length, width):
    """Return four corners in consistent perimeter order for broadcastable rectangles."""
    f = _axis(yaw) * length[..., None] * 0.5
    l = _left(_axis(yaw)) * width[..., None] * 0.5
    return center[..., None, :] + torch.stack((f + l, -f + l, -f - l, f - l), -2)

def _vertex_segment(a, b):
    v = b.roll(-1, dims=-2) - b
    rel = a[..., :, None, :] - b[..., None, :, :]
    t = ((rel * v[..., None, :, :]).sum(-1) / v.square().sum(-1).clamp_min(1e-12)[..., None, :]).clamp(0, 1)
    dist = torch.linalg.vector_norm(rel - t[..., None] * v[..., None, :, :], dim=-1)
    return dist.amin(dim=(-1, -2))

def rect_signed_distance(a, b):
    """Signed separation of broadcastable [...,4,2] rectangles.

    Separated boxes use Euclidean contour clearance; overlapping boxes use
    the negative minimum separating translation, including containment.
    Ties and contact use the selected PyTorch subgradient.
    """
    a, b = torch.broadcast_tensors(a, b)
    axes = torch.stack((a[..., 1, :] - a[..., 0, :], a[..., 2, :] - a[..., 1, :], b[..., 1, :] - b[..., 0, :], b[..., 2, :] - b[..., 1, :]), -2)
    axes = F.normalize(axes, dim=-1, eps=1e-09)
    pa, pb = ((a[..., :, None, :] * axes[..., None, :, :]).sum(-1), (b[..., :, None, :] * axes[..., None, :, :]).sum(-1))
    gap = torch.maximum(pa.amin(-2) - pb.amax(-2), pb.amin(-2) - pa.amax(-2)).amax(-1)
    sep = torch.minimum(_vertex_segment(a, b), _vertex_segment(b, a))
    return torch.where(gap > 0, sep, gap)

def _relations(traj, states, valid, cfg):
    """Build contour clearance and eight directional safety features.

    states is [B,T,A,D], valid is [B,T,A]. Invalid padded values are cleared
    before arithmetic. The directional features are longitudinal/lateral
    applicability, actual gaps, required gaps and gap deficits. Discrete
    relation masks have no gradient; geometric quantities retain gradients.
    """
    xy, yaw, speed = (traj['xy'], traj['yaw'], traj['speed'])
    ego_f = _axis(yaw)
    ego_center = xy + ego_f * cfg.rear_axle_to_center
    ego_v = ego_f * speed[..., None]
    ce = rect_corners(ego_center, yaw, torch.full_like(yaw, cfg.ego_length), torch.full_like(yaw, cfg.ego_width))
    s = torch.where(valid[..., None], states, torch.zeros_like(states))
    co = rect_corners(s[..., :2], s[..., 2], s[..., 3].clamp_min(0.0001), s[..., 4].clamp_min(0.0001))
    ce1 = ce.unsqueeze(-3)
    d = rect_signed_distance(ce1, co)
    ax = _axis(s[..., 2])
    lat = _left(ax)
    delta = s[..., :2] - ego_center[:, :, None]

    def project(u):
        pe = ((ce1 - ego_center[:, :, None, None, :]) * u[..., None, :]).sum(-1)
        po = ((co - ego_center[:, :, None, None, :]) * u[..., None, :]).sum(-1)
        dc = (delta * u).sum(-1)
        gap = torch.where(dc > 0, po.amin(-1) - pe.amax(-1), pe.amin(-1) - po.amax(-1))
        return (dc, gap)
    dcx, gx = project(ax)
    dcy, gy = project(lat)
    vex = (ego_v[:, :, None] * ax).sum(-1)
    vox = (s[..., 5:7] * ax).sum(-1)
    vey = (ego_v[:, :, None] * lat).sum(-1)
    voy = (s[..., 5:7] * lat).sum(-1)
    ok = valid & (s[..., 7] > 0.5) & ((ego_f[:, :, None] * ax).sum(-1) > math.cos(cfg.heading_max)) & (vex >= 0) & (vox >= 0)
    if s.shape[-1] > 9:
        ok = ok & (s[..., 9] > 0.5)
    ml = ok & (gy <= 0) & (dcx.abs() > 1e-05)
    mt = ok & (gx <= 0) & (dcy.abs() > 1e-05)
    vr = torch.where(dcx > 0, vex, vox)
    vf = torch.where(dcx > 0, vox, vex)
    dsx = (vr * cfg.rho + 0.5 * cfg.reaction_accel * cfg.rho ** 2 + (vr + cfg.reaction_accel * cfg.rho).square() / (2 * cfg.brake_min) - vf.square() / (2 * cfg.brake_max)).clamp_min(0.0)
    uL = (-torch.where(dcy > 0, voy, vey)).clamp_min(0.0)
    uR = torch.where(dcy > 0, vey, voy).clamp_min(0.0)

    def stopping(u):
        return u * cfg.rho + 0.5 * cfg.lateral_accel * cfg.rho ** 2 + (u + cfg.lateral_accel * cfg.rho).square() / (2 * cfg.lateral_brake)
    dsy = cfg.lateral_buffer + stopping(uL) + stopping(uR)
    r = torch.stack((ml.to(s.dtype), mt.to(s.dtype), gx * ml, gy * mt, dsx * ml, dsy * mt, (dsx - gx) * ml, (dsy - gy) * mt), -1)
    return (ce, co, d, ego_v, s, r)

def label_cost(traj, future, future_mask, cfg=None, **kwargs):
    """Candidate supervision from matched future observations.

    Take maximum object risk at each future time, then average over h=1..H.
    Empty scenes have zero risk. The caller must reject incomplete matching;
    a missing future observation must not be interpreted as a safe object.
    """
    cfg = cfg or InteractionConfig(**kwargs)
    if future.shape[2] == 0:
        return traj['xy'].sum((1, 2)) * 0.0
    m = future_mask.bool()
    _, _, d, _, _, r = _relations(traj, future, m, cfg)
    c = ((cfg.contour_buffer - d) / cfg.ego_width).clamp_min(0.0).square()
    c = c + (r[..., 6:8] / cfg.ego_width).clamp_min(0.0).square().amax(-1)
    return torch.where(m, c, torch.zeros_like(c)).amax(-1)[:, 1:].mean(1)

def _norm_weights(logits, mask, dim):
    """Normalize nonnegative scores, returning zero for empty masked neighbourhoods."""
    mask = mask.bool()
    return (F.softplus(logits) + 1e-06) * mask / ((F.softplus(logits) + 1e-06) * mask).sum(dim, keepdim=True).clamp_min(1e-08)

class _CornerLayer(nn.Module):
    """Sparse attention messages with relative position, four edge flags and clearance."""

    def __init__(self, d):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(2 * d + 7, d), nn.SiLU(), nn.Linear(d, 1))
        self.message = nn.Sequential(nn.Linear(d + 7, d), nn.SiLU(), nn.Linear(d, d))
        self.update = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, x, src, dst, edge):
        xb = x[src]
        xa = x[dst]
        w = F.softplus(self.score(torch.cat((xa, xb, edge), -1))) + 1e-06
        m = self.message(torch.cat((xb, edge), -1))
        agg = torch.zeros_like(x).index_add(0, dst, w * m)
        den = x.new_zeros((x.shape[0], 1)).index_add(0, dst, w)
        msg = agg / den.clamp_min(1e-08)
        return self.norm(x + self.update(torch.cat((x, msg), -1)))

def _edges(pos, alive, clearance, proximity):
    """Construct shape, temporal, proximity and direct ego-object edges.

    Overlapping proximity/direct relations share one edge with both flags.
    Adjacency is discrete; relative positions and contour clearance retain
    gradients. Frame chunks bound adjacency memory without discarding edges.
    """
    B, T, M, _, _ = pos.shape
    device = pos.device
    ids = torch.arange(B * T * M * 4, device=device).reshape(B, T, M, 4)
    srcs = []
    dsts = []
    attrs = []
    flat = pos.reshape(-1, 2)

    def add(src, dst, typ, close=None, direct=None, dist=None):
        src, dst = (src.reshape(-1), dst.reshape(-1))
        srcs.append(src)
        dsts.append(dst)
        flags = flat.new_zeros((src.shape[0], 4))
        if typ is not None:
            flags[:, typ] = 1.0
        else:
            flags[:, 2] = close.to(flags.dtype)
            flags[:, 3] = direct.to(flags.dtype)
        v = flat.new_zeros((src.shape[0], 1)) if dist is None else dist[:, None]
        attrs.append(torch.cat((flat[src] - flat[dst], flags, v), -1))
    for n in (-1, 1):
        add(ids.roll(n, -1)[alive], ids[alive], 0)
    temporal = alive[:, 1:] & alive[:, :-1]
    for src, dst in ((ids[:, 1:], ids[:, :-1]), (ids[:, :-1], ids[:, 1:])):
        add(src[temporal], dst[temporal], 1)
    fpos = pos.reshape(B * T, M * 4, 2)
    obj = torch.arange(M * 4, device=device) // 4
    inter = obj[:, None] != obj[None, :]
    ego = ((obj[:, None] == 0) | (obj[None, :] == 0)) & inter
    present = alive.reshape(B * T, M).repeat_interleave(4, dim=-1)
    chunk = max(1, min(B * T, 1048576 // (M * 4) ** 2))
    for lo in range(0, B * T, chunk):
        with torch.no_grad():
            v = fpos[lo:lo + chunk].detach()
            near = ((v[:, :, None] - v[:, None, :]).square().sum(-1) <= proximity ** 2) & inter
            keep = present[lo:lo + chunk]
            f, i, j = ((near | ego) & keep[:, :, None] & keep[:, None, :]).nonzero(as_tuple=True)
            direct = ego[i, j]
            spatial = near[f, i, j]
            a = torch.maximum(obj[i], obj[j]) - 1
        frame = lo + f
        sd = clearance.reshape(B * T, M - 1)[frame, a.clamp_min(0)] * direct
        add(frame * (M * 4) + i, frame * (M * 4) + j, None, spatial, direct, sd)
    return (torch.cat(srcs), torch.cat(dsts), torch.cat(attrs))

class InteractionEnergy(nn.Module):
    """Fuse contour and safety features into one nonnegative candidate energy.

    forward expects trajectory xy [B,T,2], yaw/speed [B,T], current actors
    [B,A,D], and actor mask [B,A]. T includes the current time. The graph uses
    only current actors, with constant-velocity extrapolation at interval dt.
    The output is [B]. Parameter names are stable for graph checkpoints.
    """

    def __init__(self, cfg=None, **kwargs):
        super().__init__()
        self.cfg = cfg or InteractionConfig(**kwargs)
        d = self.cfg.hidden
        self.category = nn.Embedding(self.cfg.classes, 8)
        self.node = nn.Linear(13, d)
        self.layers = nn.ModuleList([_CornerLayer(d) for _ in range(self.cfg.layers)])
        self.encode_s = nn.Sequential(nn.Linear(8, d), nn.SiLU(), nn.Linear(d, d))
        self.attn_s = nn.Linear(d, 1)
        self.z_s = nn.Linear(2 * d, d)
        self.obj = nn.Linear(2 * d + 2, d)
        self.q = nn.Linear(6, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.f = nn.Linear(d, d)
        self.g = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))

    def forward(self, traj, actors, mask, dt=0.5):
        cfg = self.cfg
        B, T = traj['yaw'].shape
        A = actors.shape[1]
        if A == 0:
            return traj['xy'].sum((1, 2)) * 0.0 + self.g[-1].bias.sum() * 0.0
        times = torch.arange(T, device=actors.device, dtype=actors.dtype) * dt
        s = actors[:, None].expand(-1, T, -1, -1)
        extrap = torch.cat((s[..., :2] + times[None, :, None, None] * s[..., 5:7], s[..., 2:]), -1)
        alive = mask.bool()[:, None].expand(-1, T, -1)
        ce, co, dd, ve, s, r = _relations(traj, extrap, alive, cfg)
        corners = torch.cat((ce[:, :, None], co), dim=2)
        diff = corners - traj['xy'][:, :, None, None]
        axis = _axis(traj['yaw'])[:, :, None, None]
        pos = torch.stack(((diff * axis).sum(-1), (diff * _left(axis)).sum(-1)), -1)
        vel = torch.cat((torch.zeros_like(ve[:, :, None]), s[..., 5:7] - ve[:, :, None]), 2)
        vr = torch.stack(((vel * axis[..., 0, :]).sum(-1), (vel * _left(axis[..., 0, :])).sum(-1)), -1)
        cat = (s[..., 8].long() if s.shape[-1] > 8 else torch.where(s[..., 7] > 0.5, 1, 2)).clamp(0, cfg.classes - 1)
        cat = torch.cat((torch.zeros_like(cat[:, :, :1]), cat), -1)
        ego_flag = torch.zeros_like(cat, dtype=actors.dtype)
        ego_flag[:, :, 0] = 1.0
        pair = torch.cat((vr, ego_flag[..., None], self.category(cat)), -1)
        x = self.node(torch.cat((pos, pair[:, :, :, None].expand(-1, -1, -1, 4, -1)), -1)).reshape(-1, cfg.hidden)
        all_alive = torch.cat((torch.ones_like(alive[:, :, :1]), alive), -1)
        src, dst, rba = _edges(pos, all_alive, dd, cfg.proximity)
        for layer in self.layers:
            x = layer(x, src, dst, rba)
        zi = x.reshape(B, T, A + 1, 4, -1).mean(3)[:, :, 1:]
        rel_ok = r[..., :2].bool().any(-1)
        bi = self.encode_s(r)
        omega = _norm_weights(self.attn_s(bi).squeeze(-1), rel_ok, -1)
        ch = (bi * omega[..., None]).sum(2, keepdim=True).expand_as(bi)
        zs = self.z_s(torch.cat((bi, ch), -1)) * rel_ok[..., None]
        u = self.obj(torch.cat((zi, zs, r[..., :2]), -1))
        qin = torch.cat((traj['xy'], _axis(traj['yaw']), traj['speed'][..., None], times[None, :, None].expand(B, -1, -1)), dim=-1)
        heads = cfg.heads
        dim = cfg.hidden // heads
        q = self.q(qin).reshape(B, T, 1, heads, dim)
        k = self.k(u).reshape(B, T, A, heads, dim)
        logits = (q * k).sum(-1) / math.sqrt(dim)
        weights = _norm_weights(logits, alive[..., None], 2)
        mixed = (weights[..., None] * self.v(u).reshape(B, T, A, heads, dim)).sum(2).reshape(B, T, cfg.hidden)
        fh = self.f(mixed) * alive.any(2, keepdim=True)
        en = F.softplus(self.g(fh[:, 1:].mean(1)).squeeze(-1))
        return torch.where(alive.any(2).any(1), en, torch.zeros_like(en))
