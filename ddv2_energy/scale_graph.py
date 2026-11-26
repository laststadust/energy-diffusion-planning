"""Candidate-supervised graph training with microbatch accumulation.

An optimizer step uses the sample-weighted effective batch loss. Validation
selects a graph checkpoint using held-out-log Huber loss. Future boxes are
used only in detached supervision targets, not in the graph forward pass.
"""
import argparse
import copy
import json
import math
import os
import random
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .interaction import InteractionConfig, InteractionEnergy
from .graph_fit import evaluate, load_bank, make_batch
from .runtime import check_gpu_temperature, sha256
from .scale_bank import atomic_json, atomic_torch, canonical_hash, source_hashes


RESUME_FORMAT = "ddv2_scale_graph_epoch_resume_v1"


def capture_rng(device):
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state().clone(),
            "torch_cuda": [s.cpu().clone() for s in torch.cuda.get_rng_state_all()]
            if torch.device(device).type == "cuda" else None}


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.device(device).type == "cuda":
        if state["torch_cuda"] is None or len(state["torch_cuda"]) != torch.cuda.device_count():
            raise ValueError("resume CUDA generator topology mismatch")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    elif state["torch_cuda"] is not None:
        raise ValueError("cannot silently resume a CUDA experiment on CPU")


def experiment_spec(bank, contexts, epochs, batch_size, seed, lr, config, device, microbatch_size=16):
    return {"format": RESUME_FORMAT, "bank_sha256": sha256(bank),
            "contexts_manifest_sha256": sha256(Path(contexts) / "manifest.json"),
            "sources": source_hashes("graph"), "config": asdict(config),
            "settings": {"epochs": epochs, "batch_size": batch_size, "seed": seed,
                         "microbatch_size": microbatch_size,
                         "accumulation": "sample_weighted_mean_one_step_per_effective_batch",
                         "lr": lr, "optimizer": "AdamW", "weight_decay": .01,
                         "clip_grad_norm": 10., "loss": "smooth_l1"},
            "runtime": {"torch": str(torch.__version__), "numpy": np.__version__,
                        "cuda": torch.version.cuda, "device_type": torch.device(device).type,
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "cudnn_benchmark": torch.backends.cudnn.benchmark,
                        "cudnn_deterministic": torch.backends.cudnn.deterministic,
                        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32}}


def validate_scale_bank(bank):
    """Validate complete shard metadata when the bank provides a collection protocol."""
    raw = torch.load(bank, map_location="cpu", weights_only=False)
    if "collection_protocol" in raw:
        if (raw.get("coverage_complete") is not True
                or canonical_hash(raw.get("collection_spec")) != raw.get("collection_spec_sha256")
                or [r["token"] for r in raw["records"]] != raw["collection_spec"]["tokens"]):
            raise ValueError("partial or inconsistent scale bank cannot train a graph")


def _alias_checkpoint(source, target):
    """Atomic hard-link alias, keeping immutable per-epoch checkpoints intact."""
    temp = target.parent / (".alias-" + uuid.uuid4().hex)
    try:
        os.link(source, temp)
        os.replace(temp, target)
    finally:
        temp.unlink(missing_ok=True)


def commit_checkpoint(checkpoint, output):
    epoch = checkpoint["completed_epoch"]
    path = output / f"epoch-{epoch:04d}.pt"
    atomic_torch(checkpoint, path, exclusive=True)
    checksum = sha256(path)
    meta = {"sha256": checksum, "completed_epoch": epoch,
            "spec_sha256": canonical_hash(checkpoint["resume_spec"])}
    atomic_json(meta, Path(str(path) + ".sha256.json"), exclusive=True)
    _alias_checkpoint(path, output / "last.pt")
    atomic_json(meta, output / "last.pt.sha256.json")
    return path


def read_resume(path, spec):
    path = Path(path)
    meta = json.loads(Path(str(path) + ".sha256.json").read_text(encoding="utf-8"))
    if meta.get("sha256") != sha256(path) or meta.get("spec_sha256") != canonical_hash(spec):
        raise ValueError("resume checkpoint/source/data/settings hash mismatch")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (state.get("resume_format") != RESUME_FORMAT or state.get("resume_spec") != spec
            or state.get("epoch_boundary") is not True):
        raise ValueError("checkpoint is not a matching full epoch boundary")
    epoch, history = state.get("completed_epoch"), state.get("history")
    if (type(epoch) is not int or not 1 <= epoch <= spec["settings"]["epochs"]
            or meta.get("completed_epoch") != epoch or not isinstance(history, list)
            or [m.get("epoch") for m in history] != list(range(1, epoch + 1))
            or state.get("metrics") != history[-1]):
        raise ValueError("inconsistent epoch/history in graph checkpoint")
    best = min(history, key=lambda m: m["validation"]["huber"])
    if (state.get("best_validation_huber") != best["validation"]["huber"]
            or state.get("best_snapshot", {}).get("metrics") != best):
        raise ValueError("best checkpoint selection/history mismatch")
    if "optimizer" not in state or "rng" not in state:
        raise ValueError("resume requires optimizer and all RNG states")
    return state


def accumulate_graph_gradients(model, samples, device, config, microbatch_size, on_phase=None):
    """Accumulate exactly one effective-batch mean loss without changing updates.

    Caller zeros gradients once before this function, and clips/steps once
    after it. Each independent candidate's loss has weight 1/len(samples),
    including a short last group or last microbatch. InteractionEnergy has no
    batch normalization/dropout; its LayerNorm and graph edges are sample-local.
    Future labels remain confined to make_batch, not the forward signature.
    """
    if type(microbatch_size) is not int or microbatch_size < 1 or not samples:
        raise ValueError("positive microbatch_size and nonempty effective batch required")
    total = 0.
    for number, lo in enumerate(range(0, len(samples), microbatch_size), 1):
        chunk = samples[lo:lo + microbatch_size]
        if torch.device(device).type == "cuda":
            check_gpu_temperature()
        if on_phase:
            on_phase("make_batch", number, len(chunk))
        traj, actors, mask, target = make_batch(chunk, device, config)
        if target.requires_grad or mask.dtype != torch.bool:
            raise ValueError("graph labels must be detached and actor mask boolean")
        if on_phase:
            on_phase("forward", number, len(chunk))
        prediction = model(traj, actors, mask, dt=.5)
        loss = F.smooth_l1_loss(prediction, target)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("nonfinite supervised graph microbatch loss")
        if on_phase:
            on_phase("backward", number, len(chunk))
        # Weight by actual sizes, not ceil(batch/microbatch): this preserves a
        # true candidate mean even for the last partial effective batch.
        (loss * (len(chunk) / len(samples))).backward()
        total += float(loss.detach()) * len(chunk)
        if on_phase:
            on_phase("microbatch_complete", number, len(chunk))
        # Release the completed graph before allocating the next microbatch.
        del prediction, loss, target, traj, actors, mask
    return total / len(samples)


def fit_graph(bank, contexts, out, *, epochs=10, device="cuda:0", batch_size=128,
              seed=0, lr=.001, config=None, resume=None, microbatch_size=16):
    """Create a fresh output even when resuming; never mutate a failed attempt."""
    if (type(epochs) is not int or epochs < 1 or type(batch_size) is not int
            or batch_size < 1 or type(microbatch_size) is not int or microbatch_size < 1
            or not math.isfinite(lr) or lr <= 0):
        raise ValueError("positive epochs/batch/microbatch sizes and finite positive lr required")
    output = Path(out)
    if output.exists():
        raise FileExistsError("graph output must be a NEW directory: " + str(output))
    config = config or InteractionConfig()
    spec = experiment_spec(bank, contexts, epochs, batch_size, seed, lr, config, device, microbatch_size)
    recovered = read_resume(resume, spec) if resume else None
    validate_scale_bank(bank)
    records, provenance = load_bank(bank, contexts)
    if recovered and recovered["provenance"] != provenance:
        raise ValueError("resume bank/context provenance mismatch")
    if torch.device(device).type == "cuda":
        check_gpu_temperature()
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    model = InteractionEnergy(config).to(device).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train = [r for r in records if r.split == "train"]
    val = [r for r in records if r.split == "val"]
    samples = [(r, k) for r in train for k in range(len(r.poses))]
    batches = math.ceil(len(samples) / batch_size)
    start, steps, history, best, snapshot = 0, 0, [], float("inf"), None
    if recovered:
        start, steps = recovered["completed_epoch"], recovered["trained_steps"]
        if steps != start * batches:
            raise ValueError("resume optimizer-step count does not match completed epochs")
        model.load_state_dict(recovered["state_dict"], strict=True)
        optimizer.load_state_dict(recovered["optimizer"])
        history = copy.deepcopy(recovered["history"])
        best = recovered["best_validation_huber"]
        snapshot = copy.deepcopy(recovered["best_snapshot"])
        # Initialization above consumes RNG. Restore only after constructing and
        # loading everything so next randperm is exactly the next epoch's draw.
        restore_rng(recovered["rng"], device)
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(spec, output / "spec.json", exclusive=True)
    if recovered:
        atomic_torch(snapshot, output / "best.pt", exclusive=True)
        commit_checkpoint(recovered, output)
    started = time.monotonic()
    try:
        for epoch in range(start, epochs):
            model.train()
            permutation = torch.randperm(len(samples)).tolist()
            total, count, max_grad = 0., 0, 0.
            for batch_number, lo in enumerate(range(0, len(samples), batch_size), 1):
                if torch.device(device).type == "cuda":
                    check_gpu_temperature()
                batch = [samples[k] for k in permutation[lo:lo + batch_size]]
                optimizer.zero_grad(set_to_none=True)
                batch_loss = accumulate_graph_gradients(model, batch, device, config, microbatch_size)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10., error_if_nonfinite=True)
                optimizer.step()
                steps += 1
                count += len(batch)
                total += batch_loss * len(batch)
                max_grad = max(max_grad, float(norm))
                if batch_number == 1 or batch_number % 100 == 0 or batch_number == batches:
                    atomic_json({"stage": "training", "epoch": epoch + 1, "epochs": epochs,
                                 "batch": batch_number, "batches_per_epoch": batches,
                                 "effective_batch_size": batch_size, "microbatch_size": microbatch_size,
                                 "trained_steps": steps, "safe_completed_epoch": epoch,
                                 "elapsed_seconds": time.monotonic() - started}, output / "progress.json")
            atomic_json({"stage": "validation", "epoch": epoch + 1,
                         "safe_completed_epoch": epoch, "trained_steps": steps}, output / "progress.json")
            metrics = {"epoch": epoch + 1, "train_huber": total / count,
                       "max_preclip_gradient_norm": max_grad,
                       "validation": evaluate(model, val, device, microbatch_size)}
            history.append(metrics)
            core = {"format": "ddv2_dual_graph_v1", "config": asdict(config),
                    "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "provenance": provenance, "trained_steps": steps, "metrics": metrics,
                    "seed": seed, "training_settings": {"epochs": epochs, "batch_size": batch_size,
                                                          "microbatch_size": microbatch_size, "lr": lr}}
            if metrics["validation"]["huber"] < best:
                best = metrics["validation"]["huber"]
                snapshot = copy.deepcopy(core)
            checkpoint = dict(core, optimizer=optimizer.state_dict(), rng=capture_rng(device),
                              resume_format=RESUME_FORMAT, resume_spec=spec, epoch_boundary=True,
                              completed_epoch=epoch + 1, history=copy.deepcopy(history),
                              best_validation_huber=best, best_snapshot=snapshot)
            commit_checkpoint(checkpoint, output)
            atomic_torch(snapshot, output / "best.pt")
            atomic_json({"stage": "epoch_complete", "completed_epoch": epoch + 1,
                         "epochs": epochs, "trained_steps": steps,
                         "best_validation_huber": best, "metrics": metrics}, output / "progress.json")
            print(json.dumps(metrics, allow_nan=False), flush=True)
        report = {"format": "ddv2_graph_training_report_v1", "provenance": provenance,
                  "trained_steps": steps, "history": history, "best_validation_huber": best,
                  "checkpoint_selection": "minimum held-out-log validation Huber; not navtest",
                  "future_labels_are_model_inputs": False, "policy_parameters_updated": False,
                  "completion": "complete", "completed_epochs": epochs,
                  "effective_batch_size": batch_size, "microbatch_size": microbatch_size,
                  "optimizer_steps_per_epoch": batches,
                  "resume_spec_sha256": canonical_hash(spec),
                  "resumed_from": str(Path(resume).resolve()) if resume else None,
                  "best_checkpoint_sha256": sha256(output / "best.pt"),
                  "last_checkpoint_sha256": sha256(output / "last.pt")}
        atomic_json(report, output / "report.json", exclusive=True)
        atomic_json({"stage": "complete", "completed_epoch": epochs, "trained_steps": steps,
                     "report_sha256": sha256(output / "report.json")}, output / "progress.json")
        return report
    except BaseException as exc:
        # Preserve error truth and previous committed epoch. Never rewrite a
        # thermal/device abort as success or launch a replacement automatically.
        atomic_json({"stage": "failed", "error_type": type(exc).__name__, "error": str(exc),
                     "safe_completed_epoch": len(history) - int(len(history) > 0 and
                         not (output / f"epoch-{len(history):04d}.pt.sha256.json").exists()),
                     "trained_steps_in_memory": steps}, output / "progress.json")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "contexts", "out"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--microbatch-size", type=int, default=16,
                        help="fixed live forward batch; gradient accumulation preserves effective batch-size")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--resume", help="verified last.pt or epoch-XXXX.pt; --out must be NEW")
    fit_graph(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
