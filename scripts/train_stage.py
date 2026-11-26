"""Run one training stage using the shared recipe and explicit local paths."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def build_command(args, recipe):
    allowed = {"graph", "policy", "energy", "seeds"}
    if set(recipe) != allowed:
        raise ValueError("Recipe requires exactly graph, policy, energy and seeds")
    keys = {"graph": {"epochs", "batch_size", "microbatch_size", "lr"},
            "policy": {"epochs", "batch_size", "workers", "lr"}, "energy": {"strength"}}
    for section, fields in keys.items():
        if set(recipe[section]) != fields:
            raise ValueError("Unknown or missing recipe fields in " + section)
    command = [sys.executable, "-m", "ddv2_energy.scale_graph" if args.stage == "graph" else "ddv2_energy.run"]
    command += ["--out", args.out, "--seed", str(args.seed)]
    for key, value in recipe[args.stage].items():
        command += ["--" + key.replace("_", "-"), str(value)]
    if args.stage == "graph":
        if not args.bank or not args.contexts:
            raise ValueError("Graph training requires --bank and --contexts")
        command += ["--bank", args.bank, "--contexts", args.contexts, "--device", args.device]
    else:
        if not args.manifest or not args.checkpoint:
            raise ValueError("Policy training requires --manifest and --checkpoint")
        command += ["--manifest", args.manifest, "--checkpoint", args.checkpoint,
                    "--mode", args.mode, "--strength", str(recipe["energy"]["strength"])]
        if args.mode == "dual_graph":
            if not args.contexts or not args.graph:
                raise ValueError("Energy policy training requires --contexts and --graph")
            command += ["--contexts", args.contexts, "--graph", args.graph]
    if args.resume:
        command += ["--resume", args.resume]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("graph", "policy"))
    parser.add_argument("--config", default=str(ROOT / "configs/training.json"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bank")
    parser.add_argument("--contexts")
    parser.add_argument("--manifest")
    parser.add_argument("--checkpoint")
    parser.add_argument("--graph")
    parser.add_argument("--mode", choices=("original", "dual_graph"), default="dual_graph")
    parser.add_argument("--device", default="cuda:0", help="Graph device; policy uses the visible CUDA device")
    parser.add_argument("--resume")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    recipe = json.loads(Path(args.config).read_text(encoding="utf-8"))
    command = build_command(args, recipe)
    print(shlex.join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
