"""Training context and supervision in the planning-time rear-axle frame.

Current objects and map fields support reward evaluation during training.
Future recorded objects are kept separately for graph supervision. These
utilities do not supply privileged inputs to the inference policy.
"""
import collections
import numpy as np
CLASSES = {'vehicle': 1, 'pedestrian': 2, 'bicycle': 3, 'traffic_cone': 4, 'barrier': 5, 'czone_sign': 6, 'generic_object': 7}
STATIC_CLASSES = {'traffic_cone', 'barrier', 'czone_sign', 'generic_object'}

def rot(yaw):
    """Return the counter-clockwise two-dimensional rotation matrix."""
    c, s = (np.cos(yaw), np.sin(yaw))
    return np.array([[c, -s], [s, c]])

def pose(f):
    """Extract translation and yaw from a NAVSIM frame using its WXYZ quaternion convention."""
    q = np.asarray(f['ego2global_rotation'], dtype=np.float64)
    norm = np.linalg.norm(q)
    w, x, y, z = q / norm if norm > 1e-12 else (np.nan,) * 4
    return (np.asarray(f['ego2global_translation'], dtype=np.float64)[:2], np.arctan2(2 * (w * z - x * y), 1 - 2 * (y * y + z * z)))

def wrap(a):
    """Wrap angles to [-pi, pi]."""
    return np.arctan2(np.sin(a), np.cos(a))

def scene_sdf(api, p, yaw):
    """Rasterize a local drivable map into a signed-distance field [1,193,193].

    The grid origin is (-32,-32) metres with 0.5 metre resolution. Rows follow
    y and columns follow x. Polygon holes remain non-drivable. Return None
    when no drivable polygon is present. Requires OpenCV, SciPy and nuPlan.
    """
    import cv2
    from scipy.ndimage import distance_transform_edt
    from nuplan.common.actor_state.state_representation import Point2D
    from nuplan.common.maps.maps_datatypes import SemanticMapLayer
    org, res, size = (-32.0, 0.5, 193)
    mask = np.zeros((size, size), np.uint8)
    layers = [SemanticMapLayer.LANE, SemanticMapLayer.INTERSECTION, SemanticMapLayer.CARPARK_AREA]
    objects = api.get_proximal_map_objects(Point2D(*p), 96.0, layers)

    def pix(ring):
        return np.rint(((np.asarray(ring.coords)[:, :2] - p) @ rot(yaw) - org) / res).astype(np.int32)
    for group in objects.values():
        for obj in group:
            geom = obj.polygon
            for poly in getattr(geom, 'geoms', [geom]):
                area = np.zeros_like(mask)
                cv2.fillPoly(area, [pix(poly.exterior)], 1)
                for ring in poly.interiors:
                    cv2.fillPoly(area, [pix(ring)], 0)
                mask |= area
    if not mask.any():
        return None
    padded = np.pad(mask, 1)
    return ((distance_transform_edt(padded) - distance_transform_edt(1 - padded))[1:-1, 1:-1] * res).astype('f4')[None]

def scene_labels(window, n=8, cap=12, max_time_error=0.025, static_tol=0.05):
    """Align a nine-frame, 0.5-second window and match current objects over time.

    Reject inconsistent times, nonfinite ego motion and reversing windows.
    Current input objects are the nearest cap observations within 32 metres,
    associated by track token. Static classes have zero velocity; missing
    dynamic velocities invalidate the observation. Future validity controls
    label_ok but never filters the current object input or measured v0.

    Returns current actors/mask, future/future_mask, ego xy/yaw/speed,
    measured v0, and truncation/label diagnostics. nodes is an auxiliary
    equal-distance ego-path resampling retained in the preparation schema;
    it is not used to generate diffusion candidates.
    """
    if len(window) != 9:
        return None
    p, yaw = pose(window[0])
    ts = np.array([f['timestamp'] for f in window], dtype=np.float64)
    ts = (ts - ts[0]) / 1000000.0
    time_error = np.abs(ts - np.arange(9) * 0.5).max()
    if not np.isfinite(ts).all() or time_error > max_time_error or min(np.diff(ts)) <= 0:
        return None
    poses = [pose(f) for f in window]
    xy = np.array([(pos - p) @ rot(yaw) for pos, _ in poses])
    headings = wrap(np.array([psi for _, psi in poses]) - yaw)
    dyn = np.array([f['ego_dynamic_state'][:2] for f in window])
    if not np.isfinite(xy).all() or not np.isfinite(headings).all() or (not np.isfinite(dyn).all()) or (dyn[:, 0].min() < -0.15):
        return None
    speed = np.linalg.norm(dyn, axis=-1)
    measured_v0 = np.float32(speed[0])
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=-1))]
    keep = np.r_[True, np.diff(arc) > 1e-06]
    s = np.linspace(0.0, arc[-1], n + 1)[1:]
    nodes = np.stack([np.interp(s, arc[keep], xy[keep, k]) for k in range(2)], axis=-1)
    if speed.max() <= static_tol:
        nodes[:] = 0.0
        speed[1:] = 0.0
    a0 = window[0]['anns']
    distances = np.linalg.norm(a0['gt_boxes'][:, :2], axis=1)
    ix = [i for i in np.argsort(distances, kind='stable') if distances[i] <= 32.0 and a0['gt_names'][i] != 'ego']
    ids = [a0['track_tokens'][i] for i in ix[:cap]]
    fbox = np.zeros((9, cap, 10), np.float32)
    valid = np.zeros((9, cap), bool)
    for h, frame in enumerate(window):
        ann = frame['anns']
        loc, psi = poses[h]
        counts = collections.Counter(ann['track_tokens'])
        lut = {key: i for i, key in enumerate(ann['track_tokens'])}
        for j, key in enumerate(ids):
            if not key or counts[key] != 1:
                continue
            i = lut[key]
            x, y, _, length, width, _, angle = ann['gt_boxes'][i]
            center = (np.array([x, y]) @ rot(psi).T + loc - p) @ rot(yaw)
            name = ann['gt_names'][i]
            vel = np.zeros(2) if name in STATIC_CLASSES else np.asarray(ann['gt_velocity_3d'][i, :2]) @ rot(psi - yaw).T
            row = [*center, wrap(angle + psi - yaw), length, width, *vel, float(name == 'vehicle'), CLASSES.get(name, 0), float(np.isfinite(angle))]
            if np.isfinite(row).all() and min(length, width) > 0:
                fbox[h, j] = row
                valid[h, j] = True
    reliable = bool(valid[:, :len(ids)].all() and len(ids) == len(set(ids)))
    return dict(nodes=nodes.astype('f4'), speed=speed[1:].astype('f4'), v0=measured_v0, xy=xy.astype('f4'), yaw=headings.astype('f4'), time_error_s=np.float32(time_error), actors=fbox[0].copy(), mask=valid[0].copy(), future=fbox, future_mask=valid, label_ok=reliable, roi_count=len(ix), truncated=len(ix) > cap)
