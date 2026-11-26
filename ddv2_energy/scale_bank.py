"""Resumable candidate collection with per-token random seeds.

Each token receives a SHA256-derived seed, independent of collection order
and previously committed shards. Only a complete bank is published at
--out; partial records remain in --out.shards.
"""
import argparse
import hashlib
import json
import os
import random
import tempfile
import uuid
from pathlib import Path

import numpy as np
import torch

from .contexts import ContextStore, safe_name
from .runtime import (capture_rollout, certify_navtrain, check_context_alignment,
                      check_gpu_temperature, load_manifest, load_sample, make_agent,
                      sha256, train_head_only)


PROTOCOL = "ddv2_scale_token_sha256_rng_v1"


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def atomic_json(value, path, *, exclusive=False):
    """Write a complete JSON commit; exclusive publication never replaces data."""
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=".json-", suffix=".partial", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(name, path)
        else:
            os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def atomic_torch(value, path, *, exclusive=False):
    path = Path(path)
    fd, name = tempfile.mkstemp(prefix=".tensor-", suffix=".partial", dir=path.parent)
    os.close(fd)
    try:
        with open(name, "wb") as stream:
            torch.save(value, stream)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(name, path)
        else:
            os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def source_hashes(kind):
    """Pin executed mathematical helpers, not unrelated job/report files."""
    root = Path(__file__).resolve().parent.parent
    names = ["ddv2_energy/scale_bank.py", "ddv2_energy/runtime.py",
             "ddv2_energy/contexts.py"]
    if kind == "graph":
        names += ["ddv2_energy/scale_graph.py", "ddv2_energy/graph_fit.py",
                  "ddv2_energy/interaction.py", "ddv2_energy/trajectory.py"]
    else:
        # Fingerprint the installed V2 agent and local wrappers together.
        navsim = Path(os.environ.get("NAVSIM_DEVKIT_ROOT", str(root))) / "navsim"
        if not navsim.is_dir():
            raise FileNotFoundError("NAVSIM_DEVKIT_ROOT must name the pinned V2 repository")
        sources = {"navsim/" + str(p.relative_to(navsim)).replace("\\", "/"): sha256(p)
                   for p in sorted(navsim.rglob("*.py"))}
        sources.update({n: sha256(root / n) for n in names})
        return sources
    return {n: sha256(root / n) for n in names}


def token_seed(seed, token):
    safe_name(token)
    digest = hashlib.sha256(f"{PROTOCOL}\0{int(seed)}\0{token}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2 ** 63 - 1)


def seed_token(seed, token):
    value = token_seed(seed, token)
    random.seed(value)
    np.random.seed(value % (2 ** 32))
    torch.manual_seed(value)  # Also seeds CUDA generators; no order dependence.
    return value


def validate_records(records, tokens, candidates=80):
    if (not isinstance(records, list) or not all(isinstance(r, dict) for r in records)
            or [r.get("token") for r in records] != list(tokens)):
        raise ValueError("shard token coverage/order mismatch")
    if len(set(tokens)) != len(tokens):
        raise ValueError("duplicate shard tokens")
    for record in records:
        poses = record.get("poses")
        if (not isinstance(poses, torch.Tensor) or not poses.is_floating_point()
                or tuple(poses.shape) != (candidates, 8, 3)
                or not bool(torch.isfinite(poses).all())):
            raise ValueError("bank requires finite [80,8,3] anchor/group candidates")


def committed_shard(folder, index, spec_hash, tokens):
    """Only a hash-verified JSON commit makes a shard reusable after a crash."""
    folder = Path(folder)
    commit = folder / f"shard-{index:06d}.json"
    if not commit.exists():
        return None
    meta = json.loads(commit.read_text(encoding="utf-8"))
    name = meta.get("file", "")
    if (Path(name).name != name or not name.startswith("data-") or not name.endswith(".pt")
            or meta.get("spec_hash") != spec_hash or meta.get("index") != index):
        raise ValueError("incompatible shard commit")
    path = folder / name
    if path.is_symlink() or path.resolve().parent != folder.resolve() or sha256(path) != meta.get("sha256"):
        raise ValueError("shard content/hash mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("spec_hash") != spec_hash or payload.get("index") != index:
        raise ValueError("shard identity mismatch")
    validate_records(payload.get("records"), tokens)
    return payload["records"]


def publish_shard(folder, index, spec_hash, records):
    """Orphan payloads from an interrupted commit are preserved but never reused."""
    folder = Path(folder)
    name = f"data-{index:06d}-{uuid.uuid4().hex}.pt"
    path = folder / name
    atomic_torch({"spec_hash": spec_hash, "index": index, "records": records}, path, exclusive=True)
    atomic_json({"spec_hash": spec_hash, "index": index, "file": name, "sha256": sha256(path)},
                folder / f"shard-{index:06d}.json", exclusive=True)


def merge_shards(folder, spec, output):
    """Publish v1 compatibility only after exact ordered coverage is proven."""
    tokens, size = spec["tokens"], spec["shard_size"]
    records = []
    identity = canonical_hash(spec)
    for index, lo in enumerate(range(0, len(tokens), size)):
        shard = committed_shard(folder, index, identity, tokens[lo:lo + size])
        if shard is None:
            raise ValueError("cannot publish incomplete bank")
        records.extend(shard)
    validate_records(records, tokens)
    payload = {"format": "ddv2_candidate_bank_v1", "source": "DiffusionDriveV2",
               "checkpoint_sha256": spec["checkpoint_sha256"], "seed": spec["seed"],
               "manifest_sha256": spec["manifest_sha256"],
               "split_provenance": spec["split_provenance"], "records": records,
               "collection_protocol": PROTOCOL, "collection_spec": spec,
               "collection_spec_sha256": identity, "coverage_complete": True,
               "pilot_bitwise_equivalent": False}
    atomic_torch(payload, Path(output), exclusive=True)
    return len(records)


def verify_published_bank(folder, spec, output):
    """Recover a crash after bank publication but before pipeline ledger commit.

    This performs the work's verification again; it does not edit a failed
    stage into success or trust mere file existence. Every final tensor must
    exactly equal its hash-verified immutable completed shard.
    """
    payload = torch.load(output, map_location="cpu", weights_only=False)
    identity = canonical_hash(spec)
    if (payload.get("format") != "ddv2_candidate_bank_v1"
            or payload.get("source") != "DiffusionDriveV2"
            or payload.get("collection_protocol") != PROTOCOL
            or payload.get("coverage_complete") is not True
            or payload.get("collection_spec") != spec
            or payload.get("collection_spec_sha256") != identity
            or payload.get("checkpoint_sha256") != spec["checkpoint_sha256"]
            or payload.get("manifest_sha256") != spec["manifest_sha256"]
            or payload.get("seed") != spec["seed"]
            or payload.get("split_provenance") != spec["split_provenance"]
            or payload.get("pilot_bitwise_equivalent") is not False):
        raise ValueError("published bank does not match frozen collection spec")
    records, tokens, size = payload.get("records"), spec["tokens"], spec["shard_size"]
    validate_records(records, tokens)
    for index, lo in enumerate(range(0, len(tokens), size)):
        shard = committed_shard(folder, index, identity, tokens[lo:lo + size])
        if shard is None or any(not torch.equal(a["poses"], b["poses"])
                                for a, b in zip(shard, records[lo:lo + size])):
            raise ValueError("published bank differs from committed shards")
    progress_path = Path(folder) / "progress.json"
    if progress_path.exists():
        previous = json.loads(progress_path.read_text(encoding="utf-8"))
        if previous.get("bank_sha256") and previous["bank_sha256"] != sha256(output):
            raise ValueError("previously committed final bank checksum changed")
    return len(records)


def collect_bank(manifest, contexts, checkpoint, out, *, device="cuda:0", seed=0,
                 resume=False, shard_size=128):
    if not isinstance(shard_size, int) or shard_size < 1:
        raise ValueError("shard_size must be positive")
    output = Path(out)
    if output.exists() and not resume:
        raise FileExistsError("completed bank already exists: " + str(output))
    rows = load_manifest(manifest, require_metric=False)
    split = certify_navtrain(rows)
    store = ContextStore(contexts)
    inputs = []
    # Hash actual frozen feature/target bytes once before collection. The context
    # loader independently validates each context against its manifest checksum.
    for row in rows:
        ctx = store.load(row["token"])
        if (ctx["split"] != row["split"] or ctx["log"] != row["log"]
                or not ctx["label_ok"] or ctx["truncated"]):
            raise ValueError("unmatched/incomplete bank context: " + row["token"])
        inputs.append({"token": row["token"], "feature": sha256(row["feature_path"]),
                       "target": sha256(row["target_path"])})
    spec = {"protocol": PROTOCOL, "seed": seed, "shard_size": shard_size,
            "candidate_shape": [80, 8, 3], "tokens": [r["token"] for r in rows],
            "buffer_policy": "restore_initial_registered_buffers_before_each_token",
            "checkpoint_sha256": sha256(checkpoint), "manifest_sha256": sha256(manifest),
            "contexts_manifest_sha256": sha256(Path(contexts) / "manifest.json"),
            "input_sha256": canonical_hash(inputs), "sources": source_hashes("bank"),
            "split_provenance": split, "torch_version": str(torch.__version__),
            "cuda_version": torch.version.cuda, "device_type": torch.device(device).type}
    identity = canonical_hash(spec)
    folder = Path(str(output) + ".shards")
    if folder.exists():
        if not resume:
            raise FileExistsError("partial bank exists; use explicit --resume")
        if json.loads((folder / "spec.json").read_text(encoding="utf-8")) != spec:
            raise ValueError("bank resume source/input/settings mismatch")
    else:
        if resume:
            raise FileNotFoundError("no partial bank to resume")
        folder.mkdir(parents=True, exist_ok=False)
        atomic_json(spec, folder / "spec.json", exclusive=True)
    # Advisory process lock prevents two collectors publishing the same shards;
    # OS releases it on death, without unsafe stale-lock deletion/reassignment.
    import fcntl  # Server is Linux; pure shard helpers remain portable for tests.
    with (folder / "collector.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists():
            count = verify_published_bank(folder, spec, output)
            progress = {"stage": "complete", "completed_tokens": count,
                        "total_tokens": len(rows), "spec_sha256": identity,
                        "bank_sha256": sha256(output), "verified_published_recovery": True}
            atomic_json(progress, folder / "progress.json")
            print(json.dumps(progress), flush=True)
            return
        agent, buffers, completed = None, None, 0
        try:
            for index, lo in enumerate(range(0, len(rows), shard_size)):
                selected = rows[lo:lo + shard_size]
                tokens = [r["token"] for r in selected]
                existing = committed_shard(folder, index, identity, tokens)
                if existing is None:
                    if agent is None:
                        if torch.device(device).type == "cuda":
                            check_gpu_temperature()
                        agent = make_agent(checkpoint).to(device)
                        train_head_only(agent)
                        # Keep per-token independence even if a future pinned
                        # head contains mutable running-statistic buffers. This
                        # changes no weights and is explicit in the protocol.
                        buffers = {name: value.detach().clone()
                                   for name, value in agent.named_buffers()}
                    records = []
                    for offset, row in enumerate(selected):
                        if torch.device(device).type == "cuda":
                            check_gpu_temperature()
                        expected = inputs[lo + offset]
                        if (sha256(row["feature_path"]) != expected["feature"]
                                or sha256(row["target_path"]) != expected["target"]):
                            raise ValueError("frozen input changed during collection")
                        features, targets = load_sample(row, device)
                        check_context_alignment(store.load(row["token"]), features, targets)
                        with torch.no_grad():
                            for name, value in agent.named_buffers():
                                value.copy_(buffers[name])
                        seed_token(seed, row["token"])
                        poses = capture_rollout(agent, features, targets)[0].detach().cpu()
                        records.append({"token": row["token"], "poses": poses})
                    validate_records(records, tokens)
                    publish_shard(folder, index, identity, records)
                completed += len(selected)
                progress = {"stage": "collecting", "completed_tokens": completed,
                            "total_tokens": len(rows), "committed_shards": index + 1,
                            "spec_sha256": identity}
                atomic_json(progress, folder / "progress.json")
                print(json.dumps(progress), flush=True)
            if (sha256(manifest) != spec["manifest_sha256"]
                    or sha256(Path(contexts) / "manifest.json") != spec["contexts_manifest_sha256"]):
                raise ValueError("manifest changed during bank collection")
            merge_shards(folder, spec, output)
            atomic_json(dict(progress, stage="complete", bank_sha256=sha256(output)), folder / "progress.json")
        finally:
            if agent is not None:
                agent._transfuser_model._trajectory_head._pdm_pool.shutdown(wait=True, cancel_futures=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "contexts", "checkpoint", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard-size", type=int, default=128)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    collect_bank(**vars(args))


if __name__ == "__main__":
    main()
