"""Build contexts for explicit, log-disjoint training/validation requests.

Requests are a JSON list of token, log and split fields. This entrypoint writes
contexts only; use scale_data for official-token selection and feature/PDM caches.
Future actor records are reserved for offline graph supervision.
"""

import argparse
import collections
import datetime
import json
import pickle
from pathlib import Path

import numpy as np

from ddv2_energy.contexts import (
    ContextError, save_context, sha256_file, validate_context,
    validate_identity, write_manifest,
)
from .preprocessing import CLASSES, pose, scene_labels, scene_sdf


def validate_requests(requests):
    if not isinstance(requests, list) or not requests:
        raise ContextError("requests must be a nonempty JSON list")
    result, tokens, log_splits = [], set(), {}
    for request in requests:
        identity = validate_identity(request)
        token, log, split = (identity[k] for k in ("token", "log", "split"))
        if token in tokens:
            raise ContextError(f"duplicate requested token: {token}")
        if log in log_splits and log_splits[log] != split:
            raise ContextError(f"train/val log overlap: {log}")
        if log.endswith(".pkl"):
            raise ContextError("request log must be the log name without .pkl")
        tokens.add(token)
        log_splits[log] = split
        result.append(identity)
    return result


def _check_annotations(frame):
    """Validate aligned actor metadata before building graph geometry."""
    required = ("token", "timestamp", "ego2global_translation", "ego2global_rotation", "ego_dynamic_state", "anns")
    missing = set(required).difference(frame)
    if missing:
        raise ContextError(f"raw frame missing fields: {sorted(missing)}")
    ann = frame["anns"]
    required_ann = {"gt_boxes", "gt_names", "track_tokens", "gt_velocity_3d"}
    if not isinstance(ann, dict) or required_ann.difference(ann):
        raise ContextError("raw frame lacks required actor annotation metadata")
    boxes = np.asarray(ann["gt_boxes"])
    names, ids = ann["gt_names"], ann["track_tokens"]
    vel = np.asarray(ann["gt_velocity_3d"])
    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise ContextError("raw gt_boxes must be [N,7]")
    n = len(boxes)
    if len(names) != n or len(ids) != n or vel.shape != (n, 3):
        raise ContextError("actor boxes/names/track tokens/velocities do not align")
    unknown = set(names).difference(CLASSES).difference({"ego"})
    if unknown:
        raise ContextError(f"unknown actor classes cannot be silently zero risk: {sorted(unknown)}")
    # A missing current position would otherwise disappear during the ROI test.
    if not np.isfinite(boxes[:, :2]).all():
        raise ContextError("nonfinite raw actor positions")


def context_from_window(request, window, sdf, actor_cap=32):
    """Construct rear-axle contexts from a complete annotated time window.

    Every current selected actor must be valid. Incomplete future tracks stay
    masked with label_ok=False, so graph fitting can explicitly reject them.
    """
    identity = validate_identity(request)
    if type(actor_cap) is not int or actor_cap < 1:
        raise ContextError("actor_cap must be a positive integer")
    if len(window) != 9:
        raise ContextError("requested token lacks eight complete future frames")
    if window[0].get("token") != identity["token"]:
        raise ContextError("requested token does not match raw current frame")
    for frame in window:
        _check_annotations(frame)
    sample = scene_labels(window, cap=actor_cap)
    if sample is None:
        raise ContextError("invalid 0.5 s timestamps, ego pose/velocity, or unsupported reverse segment")
    count = min(sample["roi_count"], actor_cap)
    if not sample["mask"][:count].all():
        raise ContextError("invalid/duplicate/missing current actor state; refusing to hide it as zero risk")
    if sdf is None:
        raise ContextError("empty drivable map for requested token")
    context = dict(identity, schema_version=1, dt=0.5, horizon=8,
                   actors=sample["actors"][:count], mask=sample["mask"][:count],
                   sdf=np.asarray(sdf, dtype=np.float32), origin=np.array([-32., -32.], dtype=np.float32),
                   resolution=0.5, v0=sample["v0"], future=sample["future"][:, :count],
                   future_mask=sample["future_mask"][:, :count],
                   label_ok=sample["label_ok"], truncated=sample["truncated"],
                   gt_poses=np.concatenate([sample["xy"][1:], sample["yaw"][1:, None]], axis=1))
    return validate_context(context)


def prepare(requests_path, logs_root, maps_root, out, actor_cap=32):
    """Fail on any requested missing/invalid record; publish no partial manifest."""
    # nuPlan/map rasterization imports are deliberately absent from module import
    # so schema tests and ContextStore need only PyTorch and NumPy.
    from nuplan.common.maps.nuplan_map.map_factory import get_maps_api

    requests_path = Path(requests_path).resolve()
    logs_root, maps_root, out = (Path(p).resolve() for p in (logs_root, maps_root, out))
    requests = validate_requests(json.loads(requests_path.read_text(encoding="utf-8")))
    if not logs_root.is_dir() or logs_root.name != "trainval":
        raise ContextError("logs-root must be the raw trainval directory, not navtest/test")
    if not maps_root.is_dir():
        raise ContextError(f"missing maps root: {maps_root}")
    if actor_cap < 1:
        raise ContextError("actor_cap must be positive")
    for source in (logs_root, maps_root):
        if out == source or source in out.parents or out in source.parents:
            raise ContextError("context output must not overlap raw logs or maps")
    out.mkdir(parents=True, exist_ok=False)
    groups = collections.defaultdict(list)
    for request in requests:
        groups[request["log"]].append(request)
    records, sources, map_apis = [], [], {}
    incomplete, truncated = 0, 0
    for log, group in groups.items():
        path = (logs_root / (log + ".pkl")).resolve()
        if logs_root not in path.parents or not path.is_file():
            raise ContextError(f"missing/escaping raw log: {log}")
        # NAVSIM raw logs are trusted user-owned pickle datasets, not downloads
        # supplied by arbitrary third parties.
        with path.open("rb") as stream:
            frames = pickle.load(stream)
        if not isinstance(frames, list) or not frames:
            raise ContextError(f"raw log must contain a nonempty frame list: {log}")
        index = {}
        for i, frame in enumerate(frames):
            if not isinstance(frame, dict) or not isinstance(frame.get("token"), str):
                raise ContextError(f"missing raw token metadata in {log}, frame {i}")
            token = frame["token"]
            if token in index:
                raise ContextError(f"ambiguous duplicate raw token {token} in {log}")
            index[token] = i
        sources.append(dict(log=log, path=str(path), sha256=sha256_file(path)))
        for request in group:
            token = request["token"]
            if token not in index:
                raise ContextError(f"requested token missing from declared log: {log}/{token}")
            window = frames[index[token]:index[token]+9]
            if len(window) != 9:
                raise ContextError(f"insufficient future frames: {log}/{token}")
            locations = [frame.get("map_location") for frame in window]
            location = locations[0]
            if not isinstance(location, str) or not location or any(x != location for x in locations):
                raise ContextError(f"missing/inconsistent map_location: {log}/{token}")
            for frame in window:
                if "log_name" in frame and frame["log_name"] != log:
                    raise ContextError(f"raw log_name disagrees with request: {log}/{token}")
                _check_annotations(frame)
            if location not in map_apis:
                map_apis[location] = get_maps_api(str(maps_root), "nuplan-maps-v1.0", location)
            sdf = scene_sdf(map_apis[location], *pose(window[0]))
            context = context_from_window(request, window, sdf, actor_cap)
            records.append(save_context(out, context))
            incomplete += not context["label_ok"]
            truncated += context["truncated"]
            print(f"context {len(records)}/{len(requests)} {request['split']} {log}/{token} "
                  f"actors={len(context['actors'])} label_ok={context['label_ok']}", flush=True)
    provenance = dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        requests_path=str(requests_path), requests_sha256=sha256_file(requests_path),
        split_provenance="caller-declared train/val requests; log-disjoint; raw trainval only; not independently official-split-certified",
        logs_root=str(logs_root), raw_logs=sources, maps_root=str(maps_root),
        map_version="nuplan-maps-v1.0", map_locations=sorted(map_apis),
        geometry_source="ddv2_energy.preprocessing.scene_labels/scene_sdf/pose",
        coordinate_frame="t0 rear axle; x forward, y left; metres/radians",
        actor_selection=f"nearest {actor_cap} current tracked objects within 32 m; no future filtering",
        actor_cap=actor_cap, incomplete_future_contexts=incomplete, truncated_contexts=truncated,
        current_annotations="privileged training reward context, never policy inference",
        future_annotations="isolated graph-supervision labels only",
        nominal_dt_s=0.5, horizon_s=4.0, maximum_timestamp_error_s=0.025,
    )
    manifest = write_manifest(out, records, provenance)
    print(f"Saved {len(records)} contexts: {manifest}", flush=True)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("requests", "logs-root", "maps-root", "out"):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--actor-cap", type=int, default=32)
    args = parser.parse_args()
    prepare(args.requests, args.logs_root, args.maps_root, args.out, args.actor_cap)


if __name__ == "__main__":
    main()
