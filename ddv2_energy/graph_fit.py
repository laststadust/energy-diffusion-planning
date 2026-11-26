"""Candidate-level supervision for contour and vehicle-safety graphs.

Candidate coordinates come from V2. Current observations enter the network;
recorded future boxes are used only in label_cost. The optimizer updates
InteractionEnergy without modifying the planning head.

Only load candidate banks created locally or obtained from trusted sources.
Checkpoints use pickle. Place banks and outputs on a disk with sufficient space.
"""
import argparse
import json
import math
import os
import random
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from .interaction import InteractionConfig, InteractionEnergy, label_cost
from .trajectory import pose_traj
from .contexts import ContextStore
from .runtime import check_gpu_temperature


@dataclass
class CandidateRecord:
    token: str
    log: str
    split: str
    poses: torch.Tensor
    context: dict


def load_bank(bank_path, context_root):
    """Validate V2 provenance, complete labels and log-isolated train/val splits."""
    bank = torch.load(bank_path, map_location="cpu", weights_only=False)
    if not isinstance(bank, dict) or bank.get("format") != "ddv2_candidate_bank_v1":
        raise ValueError("expected ddv2_candidate_bank_v1")
    if bank.get("source") != "DiffusionDriveV2":
        raise ValueError("graph supervision requires actual DiffusionDriveV2 candidates")
    checksum = bank.get("checkpoint_sha256", "")
    if not isinstance(checksum, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", checksum):
        raise ValueError("candidate bank requires a valid source checkpoint SHA256")
    rows = bank.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("candidate bank records must be a nonempty list")
    store = ContextStore(context_root)
    records, seen = [], set()
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("token"), str):
            raise ValueError("each candidate record requires a token")
        token = row["token"]
        if token in seen:
            raise ValueError("duplicate candidate token: " + token)
        seen.add(token)
        poses = row.get("poses")
        if (not isinstance(poses, torch.Tensor) or not poses.is_floating_point()
                or poses.ndim != 3 or tuple(poses.shape[1:]) != (8, 3)
                or poses.shape[0] < 1 or not bool(torch.isfinite(poses).all())):
            raise ValueError("each V2 candidate bank row requires finite floating [K,8,3] poses")
        ctx = store.load(token)
        if ctx.get("token") != token or ctx.get("split") not in ("train", "val"):
            raise ValueError("candidate/context token or split mismatch")
        if ctx.get("horizon") != 8 or not math.isclose(float(ctx.get("dt", 0)), .5):
            raise ValueError("candidate supervision requires eight 0.5-second future poses")
        if not bool(ctx.get("label_ok")) or bool(ctx.get("truncated")):
            raise ValueError("incomplete or actor-truncated supervision: " + token)
        mask, fm = ctx["mask"].bool(), ctx["future_mask"].bool()
        future = ctx["future"]
        if (future.ndim != 3 or tuple(future.shape) != (9, mask.numel(), 10)
                or tuple(fm.shape) != (9, mask.numel())
                or not bool(torch.isfinite(future).all())
                or not bool(fm[:, mask].all())
                or bool(fm[:, ~mask].any())):
            raise ValueError("future labels must match every currently observed actor: " + token)
        log = ctx.get("log")
        if not isinstance(log, str) or not log:
            raise ValueError("every supervised scene needs a log identifier")
        records.append(CandidateRecord(token, log, ctx["split"],
                                       poses.detach().float().cpu().clone(), ctx))
    by_split = {s: [r for r in records if r.split == s] for s in ("train", "val")}
    if not all(by_split.values()):
        raise ValueError("both train and val candidate records are required")
    logs = {s: sorted({r.log for r in by_split[s]}) for s in by_split}
    if set(logs["train"]) & set(logs["val"]):
        raise ValueError("train/val log overlap is forbidden")
    if not any(bool(r.context["mask"].any()) for r in by_split["train"]):
        raise ValueError("all training scenes lack actors; graph cannot learn interaction risk")
    synthetic = bank.get("synthetic", False)
    if not isinstance(synthetic, bool):
        raise ValueError("synthetic provenance must be boolean")
    provenance = {"source": "DiffusionDriveV2", "checkpoint_sha256": checksum.lower(),
                  "synthetic": synthetic,
                  "split_provenance": bank.get("split_provenance", {}),
                  "train_tokens": [r.token for r in by_split["train"]],
                  "val_tokens": [r.token for r in by_split["val"]],
                  "train_logs": logs["train"], "val_logs": logs["val"]}
    return records, provenance


def make_batch(samples, device, config):
    """Prepare candidates/current inputs and independently computed label targets.

    samples is a list of (CandidateRecord, candidate_index).  Padding never turns
    a missing future annotation into a zero-risk label: load_bank rejects that.
    """
    n = len(samples)
    if not n:
        raise ValueError("cannot construct an empty graph training batch")
    device = torch.device(device)
    amax = max(r.context["actors"].shape[0] for r, _ in samples)
    actors = torch.zeros((n, amax, 10), dtype=torch.float32, device=device)
    mask = torch.zeros((n, amax), dtype=torch.bool, device=device)
    future = torch.zeros((n, 9, amax, 10), dtype=torch.float32, device=device)
    fm = torch.zeros((n, 9, amax), dtype=torch.bool, device=device)
    trajectories = []
    for i, (record, index) in enumerate(samples):
        ctx = record.context
        a = ctx["actors"].shape[0]
        actors[i, :a] = ctx["actors"].to(device=device, dtype=torch.float32)
        mask[i, :a] = ctx["mask"].to(device=device, dtype=torch.bool)
        future[i, :, :a] = ctx["future"].to(device=device, dtype=torch.float32)
        fm[i, :, :a] = ctx["future_mask"].to(device=device, dtype=torch.bool)
        poses = record.poses[index:index + 1].to(device=device, dtype=torch.float32)
        trajectories.append(pose_traj(poses, ctx["v0"], dt=.5))
    traj = {key: torch.cat([t[key] for t in trajectories], 0) for key in trajectories[0]}
    # Future observations are consumed here, never passed to InteractionEnergy.
    with torch.no_grad():
        target = label_cost(traj, future, fm, config).detach()
    if not bool(torch.isfinite(target).all()) or bool((target < 0).any()):
        raise ValueError("candidate supervision produced invalid risk targets")
    return traj, actors, mask, target


def evaluate(model, records, device, batch_size):
    """Per-candidate errors and within-scene risk-ordering diagnostics.

    Tied target pairs are omitted; tied predictions receive half credit.  Recall
    checks whether the predicted maximum-risk candidate belongs to the actual
    maximum-risk set, excluding scenes whose labels are all identical.
    """
    model.eval()
    sums = {"huber": 0., "mae": 0., "target_sum": 0., "prediction_sum": 0.}
    count = positive = pair_count = ranked_scenes = max_hits = 0
    pair_credit = 0.
    with torch.no_grad():
        for record in records:
            predictions, targets = [], []
            for lo in range(0, len(record.poses), batch_size):
                if torch.device(device).type == "cuda":
                    check_gpu_temperature()
                samples = [(record, k) for k in range(lo, min(lo + batch_size, len(record.poses)))]
                traj, actors, mask, y = make_batch(samples, device, model.cfg)
                pred = model(traj, actors, mask, dt=.5)
                if not bool(torch.isfinite(pred).all()):
                    raise FloatingPointError("nonfinite dual-graph validation output")
                n = y.numel()
                count += n
                positive += int((y > 1e-6).sum().item())
                sums["huber"] += F.smooth_l1_loss(pred, y, reduction="sum").item()
                sums["mae"] += (pred - y).abs().sum().item()
                sums["target_sum"] += y.sum().item()
                sums["prediction_sum"] += pred.sum().item()
                predictions.append(pred.cpu())
                targets.append(y.cpu())
            p, y = torch.cat(predictions), torch.cat(targets)
            i, j = torch.triu_indices(len(y), len(y), offset=1)
            dy, dp = y[i] - y[j], p[i] - p[j]
            distinct = dy.abs() > 1e-6
            pair_count += int(distinct.sum().item())
            pair_credit += float(((dy * dp > 0).float() + .5 * (dp == 0).float())[distinct].sum())
            if float(y.max() - y.min()) > 1e-6:
                ranked_scenes += 1
                max_hits += int(bool(y[p.argmax()] >= y.max() - 1e-6))
    if not count:
        raise ValueError("evaluation requires candidate records")
    return {"candidates": count, "huber": sums["huber"] / count,
            "mae": sums["mae"] / count, "target_mean": sums["target_sum"] / count,
            "prediction_mean": sums["prediction_sum"] / count,
            "positive_label_fraction": positive / count,
            "ranking_pair_count": pair_count,
            "pairwise_ranking_accuracy": pair_credit / pair_count if pair_count else None,
            "ranked_scene_count": ranked_scenes,
            "max_risk_recall": max_hits / ranked_scenes if ranked_scenes else None}


def _save_atomic(payload, path):
    # Only files inside this invocation's newly created output directory change.
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    os.close(fd)
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def fit_graph(bank, contexts, out, *, epochs=10, device="cuda:0", batch_size=16,
              seed=0, lr=1e-3, config=None):
    """Train a new dual graph; no V2 parameters or existing graph weights change."""
    if (not isinstance(epochs, int) or epochs < 1 or not isinstance(batch_size, int)
            or batch_size < 1 or not math.isfinite(lr) or lr <= 0):
        raise ValueError("epochs/batch_size must be positive integers and lr positive finite")
    records, provenance = load_bank(bank, contexts)
    output = Path(out)
    if output.exists():
        raise FileExistsError("graph output must be a new directory: " + str(output))
    config = config or InteractionConfig()
    if torch.device(device).type == "cuda":
        check_gpu_temperature()
    random.seed(seed)
    torch.manual_seed(seed)
    model = InteractionEnergy(config).to(torch.device(device)).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    train_records = [r for r in records if r.split == "train"]
    val_records = [r for r in records if r.split == "val"]
    samples = [(r, k) for r in train_records for k in range(len(r.poses))]
    output.mkdir(parents=True, exist_ok=False)
    history, trained_steps, best = [], 0, float("inf")
    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(len(samples)).tolist()
        train_sum, count, max_grad = 0., 0, 0.
        for lo in range(0, len(samples), batch_size):
            if torch.device(device).type == "cuda":
                check_gpu_temperature()
            batch = [samples[k] for k in permutation[lo:lo + batch_size]]
            traj, actors, mask, target = make_batch(batch, device, config)
            optimizer.zero_grad(set_to_none=True)
            # Future labels do not enter this call; only this graph is optimized.
            prediction = model(traj, actors, mask, dt=.5)
            loss = F.smooth_l1_loss(prediction, target)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite supervised graph loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10., error_if_nonfinite=True)
            optimizer.step()
            max_grad = max(max_grad, float(norm))
            trained_steps += 1
            count += len(batch)
            train_sum += float(loss.detach()) * len(batch)
        validation = evaluate(model, val_records, device, batch_size)
        metrics = {"epoch": epoch + 1, "train_huber": train_sum / count,
                   "max_preclip_gradient_norm": max_grad, "validation": validation}
        history.append(metrics)
        checkpoint = {"format": "ddv2_dual_graph_v1", "config": asdict(config),
                      "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                      "provenance": provenance, "trained_steps": trained_steps, "metrics": metrics,
                      "optimizer": optimizer.state_dict(), "seed": seed,
                      "training_settings": {"epochs": epochs, "batch_size": batch_size, "lr": lr}}
        _save_atomic(checkpoint, output / "last.pt")
        if validation["huber"] < best:
            best = validation["huber"]
            _save_atomic(checkpoint, output / "best.pt")
        print(json.dumps(metrics, ensure_ascii=False, allow_nan=False), flush=True)
    report = {"format": "ddv2_graph_training_report_v1", "provenance": provenance,
              "trained_steps": trained_steps, "history": history,
              "best_validation_huber": best,
              "checkpoint_selection": "minimum held-out-log validation Huber; not navtest",
              "future_labels_are_model_inputs": False, "policy_parameters_updated": False}
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True)
    parser.add_argument("--contexts", required=True)
    parser.add_argument("--out", required=True, help="new output directory")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()
    fit_graph(args.bank, args.contexts, args.out, epochs=args.epochs, device=args.device,
              batch_size=args.batch_size, seed=args.seed, lr=args.lr)


if __name__ == "__main__":
    main()
