"""Strict, token-indexed contexts used only by the opt-in RL reward branch.

Schema 1 coordinates are the current rear-axle frame: x forward, y left,
metres/radians/seconds. ``future`` contains t=0 plus eight 0.5 s observations;
it is supervision, not an input to the graph or to the driving policy.

The manifest is a JSON object with ``schema_version``, ``purpose``,
``provenance`` and a ``records`` list. Each record has token/log/split/path/
sha256; the only permitted relative path is ``contexts/<token>.pt``. Read
only trusted local caches: hashes detect accidental changes, not authenticity.
"""

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

import torch


SCHEMA_VERSION = 1
DT = 0.5
HORIZON = 8
PURPOSE = "training_only_energy_context"
CURRENT_FIELDS = (
    "schema_version", "token", "log", "split", "dt", "horizon", "actors",
    "mask", "sdf", "origin", "resolution", "v0", "truncated",
)
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}\Z")


class ContextError(ValueError):
    """An absent, ambiguous or invalid context must never become zero risk."""


def safe_name(value, field="token"):
    """Reject separators, traversal, drive letters and control characters."""
    if not isinstance(value, str) or not _NAME.fullmatch(value):
        raise ContextError(f"invalid {field}: expected a safe nonempty identifier")
    if value in {".", ".."} or value.endswith("."):
        raise ContextError(f"invalid {field}: {value!r}")
    return value


def validate_identity(record):
    if not isinstance(record, dict):
        raise ContextError("context/manifest record must be a dictionary")
    for key in ("token", "log", "split"):
        if key not in record:
            raise ContextError(f"missing identity field: {key}")
    safe_name(record["token"])
    safe_name(record["log"], "log")
    if not isinstance(record["split"], str) or record["split"] not in {"train", "val"}:
        raise ContextError("only explicit train/val contexts are allowed; no navtest")
    return {key: record[key] for key in ("token", "log", "split")}


def _tensor(value, name, boolean=False, allow_integer=False):
    try:
        result = torch.as_tensor(value).detach().cpu()
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ContextError(f"invalid tensor {name}: {exc}") from exc
    if boolean:
        if result.dtype != torch.bool:
            raise ContextError(f"{name} must have boolean dtype")
        return result.clone()
    if not result.is_floating_point() and not (
            allow_integer and result.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)):
        raise ContextError(f"{name} must have real floating dtype")
    return result.to(torch.float32).clone()


def _finite(value, name):
    if not torch.isfinite(value).all().item():
        raise ContextError(f"{name} contains nonfinite visible values")


def validate_context(context, expected=None):
    """Return a canonical CPU copy, checking both geometry and label masks.

    Missing future observations are permitted only with ``label_ok=False``.
    They are zero-filled behind an explicit false mask, never zero-risk labels.
    No future annotation is used to choose/reorder the current actor set.
    """
    identity = validate_identity(context)
    if expected is not None:
        for key, value in validate_identity(expected).items():
            if identity[key] != value:
                raise ContextError(f"context {key} mismatch: {identity[key]!r} != {value!r}")
    required = set(CURRENT_FIELDS) | {"future", "future_mask", "label_ok", "gt_poses"}
    missing = required.difference(context)
    if missing:
        raise ContextError(f"missing context fields: {sorted(missing)}")
    if type(context["schema_version"]) is not int or context["schema_version"] != SCHEMA_VERSION:
        raise ContextError("unsupported context schema_version")
    if type(context["horizon"]) is not int or context["horizon"] != HORIZON:
        raise ContextError("context must cover eight future poses / four seconds")
    try:
        dt, resolution = float(context["dt"]), float(context["resolution"])
    except (TypeError, ValueError) as exc:
        raise ContextError("dt/resolution must be finite scalars") from exc
    if not math.isfinite(dt) or abs(dt - DT) > 1e-9:
        raise ContextError("context dt must be 0.5 seconds")
    if not math.isfinite(resolution) or abs(resolution - 0.5) > 1e-9:
        raise ContextError("schema 1 SDF resolution must be 0.5 metres")
    for key in ("label_ok", "truncated"):
        if type(context[key]) is not bool:
            raise ContextError(f"{key} must be an explicit boolean")

    out = dict(identity, schema_version=SCHEMA_VERSION, horizon=HORIZON, dt=DT,
               resolution=resolution, label_ok=context["label_ok"],
               truncated=context["truncated"])
    for key in ("actors", "sdf", "origin", "v0", "future", "gt_poses"):
        out[key] = _tensor(context[key], key, allow_integer=key in {"origin", "v0"})
    for key in ("mask", "future_mask"):
        out[key] = _tensor(context[key], key, boolean=True)

    actors, mask, future, future_mask = (out[k] for k in ("actors", "mask", "future", "future_mask"))
    if actors.ndim != 2 or actors.shape[-1] != 10:
        raise ContextError("actors must be [A,10]")
    count = actors.shape[0]
    if mask.shape != (count,):
        raise ContextError("mask must be [A]")
    if future.shape != (9, count, 10) or future_mask.shape != (9, count):
        raise ContextError("future/future_mask must be [9,A,10]/[9,A]")
    if not torch.equal(future_mask[0], mask):
        raise ContextError("future_mask[0] must equal current mask")
    if (future_mask & ~mask[None]).any().item():
        raise ContextError("future may not introduce actors absent from the current actor set")
    _finite(actors[mask], "actors")
    _finite(future[future_mask], "future")
    for rows, name in ((actors[mask], "actors"), (future[future_mask], "future")):
        if rows.numel() and (rows[:, 3:5] <= 0).any().item():
            raise ContextError(f"visible {name} dimensions must be positive")
    if not torch.allclose(future[0, mask], actors[mask], atol=1e-6, rtol=0):
        raise ContextError("future[0] must match current actors in the same coordinates")
    if out["label_ok"] and not future_mask[:, mask].all().item():
        raise ContextError("label_ok=True requires all current actors at every label timestep")
    # Explicit masks are retained; zero-fill only invisible storage for safe batching.
    out["actors"] = torch.where(mask[:, None], actors, torch.zeros_like(actors))
    out["future"] = torch.where(future_mask[:, :, None], future, torch.zeros_like(future))
    sdf = out["sdf"]
    if sdf.ndim != 3 or sdf.shape[0] != 1 or min(sdf.shape[1:]) < 2:
        raise ContextError("sdf must be [1,Ny,Nx] with a nonempty 2D grid")
    _finite(sdf, "sdf")
    if not (sdf > 0).any().item():
        raise ContextError("empty drivable map cannot be used as a context")
    if out["origin"].shape != (2,) or not torch.equal(out["origin"], torch.tensor([-32., -32.])):
        raise ContextError("schema 1 SDF origin must be [-32,-32] metres")
    if out["v0"].numel() != 1:
        raise ContextError("v0 must be a scalar observed speed")
    out["v0"] = out["v0"].reshape(())
    _finite(out["v0"], "v0")
    if out["v0"].item() < 0:
        raise ContextError("v0 must be nonnegative")
    if out["gt_poses"].shape != (HORIZON, 3):
        raise ContextError("gt_poses must be [8,3] at t=0.5,...,4.0 seconds")
    _finite(out["gt_poses"], "gt_poses")
    return out


def current_only(context):
    """Whitelist scoring inputs; do not pass future labels through **kwargs.

    Call on contexts already validated by ContextStore/validate_context. Tensor
    values are cloned so a scoring consumer cannot mutate cached supervision.
    """
    try:
        return {key: context[key].clone() if torch.is_tensor(context[key]) else context[key]
                for key in CURRENT_FIELDS}
    except KeyError as exc:
        raise ContextError(f"missing current-only field: {exc}") from exc


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_records(records):
    if not isinstance(records, list) or not records:
        raise ContextError("manifest records must be a nonempty list")
    indexed, log_splits = {}, {}
    for record in records:
        identity = validate_identity(record)
        token, log, split = (identity[k] for k in ("token", "log", "split"))
        if token in indexed:
            raise ContextError(f"duplicate manifest token: {token}")
        if log in log_splits and log_splits[log] != split:
            raise ContextError(f"train/val log overlap: {log}")
        log_splits[log] = split
        if record.get("path") != f"contexts/{token}.pt":
            raise ContextError(f"invalid context relative path for {token}")
        if not isinstance(record.get("sha256"), str) or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
            raise ContextError(f"missing/invalid sha256 for {token}")
        indexed[token] = dict(record)
    return indexed


class ContextStore:
    """Load exact manifest tokens from a trusted local, immutable context set."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        try:
            manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ContextError(f"cannot read context manifest at {self.root}: {exc}") from exc
        if (not isinstance(manifest, dict) or type(manifest.get("schema_version")) is not int
                or manifest.get("schema_version") != SCHEMA_VERSION):
            raise ContextError("unsupported context manifest schema")
        if (manifest.get("purpose") != PURPOSE
                or not isinstance(manifest.get("provenance"), dict) or not manifest["provenance"]):
            raise ContextError("manifest must declare training-only purpose and source provenance")
        self.records = _validate_records(manifest.get("records"))
        self.manifest = manifest

    def load(self, token):
        token = safe_name(token)
        if token not in self.records:
            raise ContextError(f"missing context token: {token}")
        record = self.records[token]
        path = (self.root / record["path"]).resolve()
        if self.root not in path.parents:
            raise ContextError("context symlink escapes manifest root")
        if not path.is_file():
            raise ContextError(f"missing context file: {path}")
        if sha256_file(path) != record["sha256"]:
            raise ContextError(f"context sha256 mismatch: {token}")
        try:
            # This schema consists solely of tensors and primitive values. Do not
            # load arbitrary model/pickle objects from a purported context cache.
            context = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ContextError(f"cannot load context {token}: {exc}") from exc
        return validate_context(context, record)


def save_context(root, context):
    """Atomically publish one new context, never overwriting an existing file."""
    context = validate_context(context)
    root = Path(root).resolve()
    folder = root / "contexts"
    folder.mkdir(parents=True, exist_ok=True)
    if folder.resolve().parent != root:
        raise ContextError("context directory symlink escapes output root")
    path = folder / f"{context['token']}.pt"
    if path.exists():
        raise FileExistsError(f"refusing to overwrite context: {path}")
    fd, temporary = tempfile.mkstemp(prefix=".context-", suffix=".partial", dir=folder)
    os.close(fd)
    try:
        torch.save(context, temporary)
        # Same-filesystem hard link is an atomic no-replace publish (unlike
        # replace(), which could overwrite an independently created context).
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return dict(validate_identity(context), path=f"contexts/{context['token']}.pt", sha256=sha256_file(path))


def write_manifest(root, records, provenance):
    """Commit a complete cache manifest after all requested contexts succeeded."""
    _validate_records(records)
    if not isinstance(provenance, dict) or not provenance:
        raise ContextError("nonempty source provenance is required")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / "manifest.json"
    if target.exists():
        raise FileExistsError(f"refusing to replace context manifest: {target}")
    payload = dict(schema_version=SCHEMA_VERSION, purpose=PURPOSE, records=records, provenance=provenance)
    fd, temporary = tempfile.mkstemp(prefix=".manifest-", suffix=".partial", dir=root)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target
