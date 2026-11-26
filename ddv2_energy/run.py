"""RL fine-tuning from explicit, log-disjoint NAVSIM train/validation manifests.

This entrypoint consumes prepared caches and a supervised graph checkpoint.
Candidate sampling and the selector remain in the pinned upstream model.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from .runtime import (make_agent, train_head_only, load_manifest, load_sample, sha256,
                      check_gpu_temperature, check_context_alignment, certify_navtrain)


class ManifestDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        features, targets = load_sample(row, "cpu")
        return ({k: v[0] for k, v in features.items()}, {k: v[0] for k, v in targets.items()},
                row["metric_cache_path"], row["token"])


class EpochSeededSampler(Sampler):
    """Order is a function of seed/epoch only, independent of graph/RNG activity."""
    def __init__(self, data_source, seed=0):
        self.size, self.seed, self.epoch = len(data_source), int(seed), 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return self.size

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        return iter(torch.randperm(self.size, generator=generator).tolist())


def atomic_json(path, value):
    """Only update files owned by this newly created experiment directory."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def state_fingerprints(agent):
    result = {}
    for name, tensor in agent.state_dict().items():
        raw = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        result[name] = hashlib.sha256(raw).hexdigest()
    return result


def audit_weights(initial, current, *, require_change=False):
    if set(initial) != set(current):
        raise RuntimeError("Policy state keys changed during training")
    changed = sorted(key for key in initial if initial[key] != current[key])
    prefix = "_transfuser_model._trajectory_head."
    unexpected = [key for key in changed if not key.startswith(prefix) or key == prefix + "plan_anchor"]
    if unexpected:
        raise RuntimeError("Frozen/anchor tensors changed: " + repr(unexpected))
    denoiser = [key for key in changed if key.startswith(prefix + "diff_decoder.")]
    if require_change and not denoiser:
        raise RuntimeError("Optimizer gate failed: no changed diff_decoder tensor")
    return {"changed_head_tensors": len(changed), "changed_denoiser_tensors": len(denoiser),
            "frozen_tensors_unchanged": True, "anchor_unchanged": True,
            "state_keys_unchanged": True}


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def finite_scalar_metrics(metrics):
    result = {}
    for key, value in metrics.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            value = float(value.detach().cpu())
        if isinstance(value, (float, int)):
            if not math.isfinite(value):
                raise FloatingPointError("Nonfinite logged metric: " + str(key))
            result[str(key)] = value
    return result


def validate_gate_args(args):
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("Positive epochs/batch size and nonnegative workers required")
    if not math.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("lr must be finite and positive")
    for value in (args.limit_train_batches, args.limit_val_batches):
        if value is not None and value < 1:
            raise ValueError("Batch limits must be positive integers")
    if args.max_steps != -1 and args.max_steps < 1:
        raise ValueError("max_steps must be -1 or positive")
    if args.gate:
        if (args.limit_train_batches is None or args.limit_train_batches > 2
                or args.limit_val_batches is None or args.limit_val_batches > 2
                or args.max_steps < 1 or args.max_steps > 2 or args.epochs != 1):
            raise ValueError("Gate requires epochs=1, limits<=2 and max_steps=1 or2")
    elif args.limit_train_batches is not None or args.limit_val_batches is not None or args.max_steps != -1:
        raise ValueError("Bounded batch/step limits are only allowed with --gate")


RESUME_FIELDS = ("mode", "checkpoint_sha256", "manifest_sha256", "graph_sha256", "context_manifest_sha256",
                 "strength", "epochs", "batch_size", "workers", "seed", "lr", "gate",
                 "source_hashes", "precision", "motion_energy", "sampler")


def verify_resume(checkpoint_path, metadata):
    """Reject method/data/config changes and non-epoch-boundary restoration."""
    path = Path(checkpoint_path)
    previous_path = path.parent.parent / "run.json"
    if not previous_path.is_file():
        raise ValueError("Resume checkpoint must have its parent run.json")
    previous = json.loads(previous_path.read_text())
    changed = [name for name in RESUME_FIELDS if previous.get(name) != metadata.get(name)]
    if changed:
        raise ValueError("Resume would change experiment settings: " + repr(changed))
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    control = checkpoint.get("ddv2_run_control", {})
    if control.get("gate") or not control.get("epoch_boundary") or "rng" not in control:
        raise ValueError("Only non-gate epoch-boundary checkpoints with RNG state may resume")
    if control.get("source_hashes") != metadata["source_hashes"]:
        raise ValueError("Resume source code hashes differ")
    return control


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True); p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--mode", choices=("original", "dual_graph"), default="original")
    p.add_argument("--contexts", default=""); p.add_argument("--graph", default="")
    p.add_argument("--strength", type=float, default=.1)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--resume", default=None)
    p.add_argument("--gate", action="store_true", help="Bounded real-PDM optimizer check; never an experimental initialization")
    p.add_argument("--limit-train-batches", type=int, default=None)
    p.add_argument("--limit-val-batches", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=-1)
    args = p.parse_args()
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, Callback
    from pytorch_lightning.loggers import CSVLogger
    from navsim.planning.training.agent_lightning_module import AgentLightningModule
    validate_gate_args(args)
    out = Path(args.out)
    if out.exists():
        raise FileExistsError("Use a NEW output directory, including for a resumed run: " + str(out))
    rows = load_manifest(args.manifest)
    split_provenance = certify_navtrain(rows)
    train, val = ([r for r in rows if r["split"] == split] for split in ("train", "val"))
    if not train or not val:
        raise ValueError("Need nonempty, log-disjoint training and validation samples")
    initial_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if initial_payload.get("ddv2_run_control", {}).get("gate", False):
        raise ValueError("Gate weights must never initialize formal or subsequent runs")
    del initial_payload
    if args.mode == "dual_graph" and args.strength > 0:
        from .contexts import ContextStore
        from .config import EnergyRLConfig
        from .engine import FrozenEnergyEngine
        store = ContextStore(args.contexts)
        for row in rows:
            ctx = store.load(row["token"])
            if (ctx["log"], ctx["split"]) != (row["log"], row["split"]) or ctx["truncated"]:
                raise ValueError("Invalid/truncated reward context: " + row["token"])
            features, targets = load_sample(row, "cpu")
            check_context_alignment(ctx, features, targets)
        # Fail before a long training launch when graph weights are missing/test-only.
        engine = FrozenEnergyEngine(EnergyRLConfig(mode="dual_graph", context_root=args.contexts,
                                          graph_checkpoint=args.graph, strength=args.strength), "cpu")
        if set(engine.provenance["train_logs"]) & {r["log"] for r in val}:
            raise ValueError("Graph training logs overlap the RL validation logs")
        certify_navtrain([{"log": log} for log in engine.provenance["train_logs"] + engine.provenance["val_logs"]])
        del engine
    initial_temperatures = check_gpu_temperature()
    pl.seed_everything(args.seed, workers=True)
    agent = make_agent(args.checkpoint, mode=args.mode, contexts=args.contexts,
                       graph=args.graph, strength=args.strength, lr=args.lr)
    initial_weights = state_fingerprints(agent)
    train_sampler = EpochSeededSampler(train, args.seed)
    pending_rng = None
    class HeadModule(AgentLightningModule):
        def on_train_start(self):
            if pending_rng is not None:
                restore_rng(pending_rng)
        def on_train_epoch_start(self):
            train_head_only(self.agent)
            train_sampler.set_epoch(self.current_epoch)
        def on_before_optimizer_step(self, optimizer):
            for name, param in self.agent.named_parameters():
                if param.grad is not None and not bool(torch.isfinite(param.grad).all()):
                    raise FloatingPointError("Nonfinite policy gradient: " + name)
            engine = getattr(self.agent._transfuser_model._trajectory_head, "_energy_reward_engine", None)
            if engine is not None and any(p.grad is not None for p in engine.graph.parameters()):
                raise RuntimeError("Frozen reward graph received an RL gradient")
        def on_save_checkpoint(self, checkpoint):
            checkpoint["ddv2_run_control"] = dict(
                gate=args.gate, source_hashes=metadata["source_hashes"], rng=capture_rng(),
                epoch_boundary=(not args.gate and progress.epoch_train_batches == len(loaders[0])),
                last_completed_epoch=progress.last_completed_epoch)
        def configure_optimizers(self):
            settings = super().configure_optimizers()
            scheduler = settings["lr_scheduler"]
            if isinstance(scheduler, dict):
                scheduler = scheduler["scheduler"]
            # Same schedule family in both arms; fit its period to this run instead
            # of allowing upstream's hard-coded 10-epoch cosine to rise again.
            scheduler.epochs = max(args.epochs, scheduler.warmup_epochs + 1)
            return settings
    class ProgressGuard(Callback):
        def __init__(self):
            self.started = time.monotonic()
            self.epoch_started = self.started
            self.last_completed_epoch = 0
            self.epoch_train_batches = 0
            self.validation_batches = 0
            self.history = []
            self.temperatures = list(initial_temperatures)
            self.max_temperature = max(initial_temperatures)
            self.last_audit = None
            self.expected_epoch_tokens = []
            self.seen_epoch_tokens = []

        def status(self, trainer, state, error=None):
            metrics = finite_scalar_metrics(trainer.callback_metrics) if state != "failed" else {}
            value = dict(state=state, gate=args.gate, epoch=min(int(trainer.current_epoch) + 1, args.epochs),
                         last_completed_epoch=self.last_completed_epoch, global_step=int(trainer.global_step),
                         epochs=args.epochs, elapsed_s=time.monotonic()-self.started,
                         epoch_train_batches=self.epoch_train_batches,
                         validation_batches_observed=self.validation_batches, metrics=metrics,
                         temperature=self.temperatures, max_temperature=self.max_temperature,
                         history=self.history, weight_audit=self.last_audit,
                         learning_rates=[float(g["lr"]) for opt in trainer.optimizers for g in opt.param_groups],
                         peak_gpu_memory_bytes=int(torch.cuda.max_memory_allocated()),
                         cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), error=error)
            atomic_json(out / "progress.json", value)
            return value

        def thermal(self, phase, trainer, batch_idx):
            self.temperatures = check_gpu_temperature()
            self.max_temperature = max(self.max_temperature, max(self.temperatures))
            with (out / "thermals.csv").open("a", encoding="utf-8") as stream:
                stream.write(f"{time.time():.3f},{phase},{trainer.current_epoch},{batch_idx}," +
                             ";".join(str(x) for x in self.temperatures) + "\n")

        def on_train_epoch_start(self, trainer, pl_module):
            self.epoch_started = time.monotonic()
            self.epoch_train_batches = 0
            train_sampler.set_epoch(trainer.current_epoch)
            self.expected_epoch_tokens = [train[i]["token"] for i in train_sampler]
            self.seen_epoch_tokens = []
            self.status(trainer, "training")

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            self.thermal("train", trainer, batch_idx)
            tokens = list(batch[-1])
            offset = len(self.seen_epoch_tokens)
            if tokens != self.expected_epoch_tokens[offset:offset+len(tokens)]:
                raise RuntimeError("Actual train token order differs from fixed seed/epoch sampler")
            self.seen_epoch_tokens.extend(tokens)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            loss = outputs.get("loss") if isinstance(outputs, dict) else outputs
            if isinstance(loss, torch.Tensor) and not bool(torch.isfinite(loss).all()):
                raise FloatingPointError("Nonfinite training loss")
            self.epoch_train_batches += 1
            self.status(trainer, "training")

        def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
            self.thermal("validation", trainer, batch_idx)

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            self.validation_batches += 1
            self.status(trainer, "validating")

        def on_train_epoch_end(self, trainer, pl_module):
            self.last_audit = audit_weights(initial_weights, state_fingerprints(agent), require_change=True)
            full_epoch = self.epoch_train_batches == len(loaders[0])
            if full_epoch:
                self.last_completed_epoch = int(trainer.current_epoch) + 1
            self.history.append(dict(epoch=int(trainer.current_epoch)+1, full_epoch=full_epoch,
                                     train_batches=self.epoch_train_batches,
                                     seen_samples=len(self.seen_epoch_tokens),
                                     sample_order_sha256=hashlib.sha256("\n".join(self.seen_epoch_tokens).encode()).hexdigest(),
                                     elapsed_s=time.monotonic()-self.epoch_started,
                                     metrics=finite_scalar_metrics(trainer.callback_metrics)))
            self.status(trainer, "epoch_finished")

        def on_exception(self, trainer, pl_module, exception):
            self.status(trainer, "failed", str(exception))
    module = HeadModule(agent)
    out.mkdir(parents=True, exist_ok=False)
    metadata = dict(vars(args), checkpoint_sha256=sha256(args.checkpoint),
                    manifest_sha256=sha256(args.manifest), motion_energy=False,
                    graph_sha256=sha256(args.graph) if args.graph else None,
                    train_count=len(train), val_count=len(val), precision="32-true",
                    note="Baseline and energy modes share this runner and upstream policy objective")
    metadata["scheduler_epochs"] = max(args.epochs, 2)
    metadata["split_provenance"] = split_provenance
    metadata["sampler"] = "randperm(seed+epoch); separate loader generators"
    metadata["context_manifest_sha256"] = sha256(Path(args.contexts) / "manifest.json") if args.contexts else None
    repo = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
    source_files = [Path(__file__), Path(__file__).with_name("runtime.py"),
                    Path(__file__).with_name("engine.py"), Path(__file__).with_name("rewards.py"),
                    Path(__file__).with_name("config.py"),
                    Path(__file__).with_name("contexts.py"),
                    Path(__file__).with_name("geometry.py"),
                    Path(__file__).with_name("interaction.py"),
                    Path(__file__).with_name("trajectory.py"),
                    repo / "navsim/agents/diffusiondrivev2/diffusiondrivev2_model_rl.py",
                    repo / "navsim/agents/diffusiondrivev2/diffusiondrivev2_rl_config.py"]
    metadata["source_hashes"] = {p.name: sha256(p) for p in source_files}
    if args.resume:
        control = verify_resume(args.resume, metadata)
        pending_rng = control["rng"]
    atomic_json(out / "run.json", metadata)
    loaders = [DataLoader(ManifestDataset(r), batch_size=args.batch_size,
                          sampler=train_sampler if i == 0 else None, shuffle=False,
                          generator=torch.Generator().manual_seed(args.seed + 10000 + i),
                          num_workers=args.workers) for i, r in enumerate((train, val))]
    progress = ProgressGuard()
    if args.resume:
        progress.last_completed_epoch = int(control["last_completed_epoch"])
    (out / "thermals.csv").write_text("unix_time,phase,epoch,batch,gpu_temperatures_c\n")
    trainer = pl.Trainer(default_root_dir=str(out), accelerator="gpu", devices=1,
                         max_epochs=args.epochs, precision="32-true", num_sanity_val_steps=0,
                         callbacks=[progress, ModelCheckpoint(dirpath=str(out / "checkpoints"),
                             filename="epoch={epoch}-step={step}", auto_insert_metric_name=False,
                             save_top_k=-1, save_last=True, every_n_epochs=1, save_on_train_epoch_end=True)],
                         logger=CSVLogger(str(out), name="csv_logs"),
                         limit_train_batches=args.limit_train_batches if args.gate else 1.0,
                         limit_val_batches=args.limit_val_batches if args.gate else 1.0,
                         max_steps=args.max_steps, enable_progress_bar=False, log_every_n_steps=1)
    try:
        trainer.fit(module, loaders[0], loaders[1], ckpt_path=args.resume)
        progress.last_audit = audit_weights(initial_weights, state_fingerprints(agent), require_change=True)
        if args.gate:
            if trainer.global_step < 1 or trainer.global_step > 2:
                raise RuntimeError("Gate did not execute one or two optimizer steps")
            result = progress.status(trainer, "gate_passed")
            result.update(passed=True, purpose="real_PDM_optimizer_gate_only_not_experimental_weights")
            atomic_json(out / "gate_result.json", result)
        else:
            if progress.last_completed_epoch != args.epochs:
                raise RuntimeError("Training ended before the requested final epoch")
            result = progress.status(trainer, "complete")
            checkpoint_path = out / "checkpoints/last.ckpt"
            if not checkpoint_path.is_file():
                raise FileNotFoundError("Completed training has no final checkpoint")
            result.update(checkpoint=str(checkpoint_path), checkpoint_sha256=sha256(checkpoint_path))
            atomic_json(out / "training_result.json", result)
    except Exception as exc:
        progress.status(trainer, "failed", str(exc))
        raise
    finally:
        agent._transfuser_model._trajectory_head._pdm_pool.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
