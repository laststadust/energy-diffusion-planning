"""Collect V2 anchor-diffusion proposals for candidate-level graph supervision."""
import argparse
import json
from pathlib import Path
import torch
from .contexts import ContextStore
from .runtime import (make_agent, train_head_only, load_manifest, load_sample,
                      capture_rollout, sha256, check_gpu_temperature, check_context_alignment, certify_navtrain)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True); p.add_argument("--contexts", required=True)
    p.add_argument("--checkpoint", required=True); p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0"); p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    rows = load_manifest(args.manifest, require_metric=False)
    split_provenance = certify_navtrain(rows)
    store = ContextStore(args.contexts)
    # Preflight all token/split associations before loading a GPU model.
    for row in rows:
        ctx = store.load(row["token"])
        if ctx["split"] != row["split"] or ctx["log"] != row["log"] or not ctx["label_ok"] or ctx["truncated"]:
            raise ValueError("Unmatched/incomplete graph-supervision context: " + row["token"])
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        check_gpu_temperature()
    agent = make_agent(args.checkpoint).to(args.device)
    train_head_only(agent)
    records = []
    try:
        for row in rows:
            if args.device.startswith("cuda"):
                check_gpu_temperature()
            features, targets = load_sample(row, args.device)
            ctx = store.load(row["token"])
            check_context_alignment(ctx, features, targets)
            poses = capture_rollout(agent, features, targets)[0].cpu()
            records.append({"token": row["token"], "poses": poses})
            print(json.dumps({"bank_samples": len(records), "token": row["token"]}), flush=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation: never overwrite an existing experimental bank.
        with out.open("xb") as f:
            torch.save(dict(format="ddv2_candidate_bank_v1", source="DiffusionDriveV2",
                            checkpoint_sha256=sha256(args.checkpoint), seed=args.seed,
                            manifest_sha256=sha256(args.manifest), split_provenance=split_provenance,
                            records=records), f)
    finally:
        agent._transfuser_model._trajectory_head._pdm_pool.shutdown(wait=True, cancel_futures=True)


if __name__ == "__main__":
    main()
