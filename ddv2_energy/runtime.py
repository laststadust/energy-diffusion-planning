"""Shared training/bank helpers. No simulator, checkpoint or feature rewriting."""
import hashlib
import json
from pathlib import Path
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_agent(checkpoint, *, mode="original", contexts="", graph="", strength=.1, lr=1e-5):
    from navsim.agents.diffusiondrivev2.diffusiondrivev2_rl_agent import Diffusiondrivev2_Rl_Agent
    from navsim.agents.diffusiondrivev2.diffusiondrivev2_rl_config import TransfuserConfig
    cfg = TransfuserConfig(energy_mode=mode, energy_context_root=str(contexts),
                           energy_graph_checkpoint=str(graph), energy_strength=strength)
    agent = Diffusiondrivev2_Rl_Agent(cfg, lr, str(checkpoint))
    # Official strict=False is retained in the original source, but our entry
    # refuses missing/uninitialized RL tensors. Selector-only extra keys are fine.
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    state = {k.removeprefix("agent."): v for k, v in state.items()}
    actual = agent.state_dict()
    missing = [k for k in actual if k not in state]
    unequal = [k for k in actual if k in state and not torch.equal(actual[k], state[k])]
    if missing or unequal:
        agent._transfuser_model._trajectory_head._pdm_pool.shutdown(wait=True)
        raise ValueError("RL checkpoint mismatch: " + repr(dict(missing=missing, unequal=unequal)))
    return agent


def train_head_only(agent):
    """Freeze feature extractor statistics as well as its original frozen weights.

    Both baseline and energy training use this mode switch. Calling the
    parent .train() alone would re-enable non-head batch-normalization updates.
    """
    agent.eval()
    agent._transfuser_model.train(False)
    agent._transfuser_model._trajectory_head.train(True)


def load_manifest(path, *, require_metric=True):
    rows = json.loads(Path(path).read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest must be a nonempty JSON list")
    tokens, logs = set(), {"train": set(), "val": set()}
    for row in rows:
        if row["token"] in tokens or row["split"] not in logs:
            raise ValueError("Duplicate token or invalid train/val split")
        tokens.add(row["token"]); logs[row["split"]].add(row["log"])
        for field in ("feature_path", "target_path") + (("metric_cache_path",) if require_metric else ()):
            if not Path(row[field]).is_file():
                raise FileNotFoundError(row[field])
        # A filename/path typo must never silently mix sample contexts and features.
        checked_fields = ("feature_path", "target_path") + (("metric_cache_path",) if require_metric else ())
        if any(row["token"] not in Path(row[k]).parts or row["log"] not in Path(row[k]).parts
               for k in checked_fields):
            raise ValueError("Feature/target/metric cache path does not match the requested token AND log")
    if logs["train"] & logs["val"]:
        raise ValueError("Train/validation logs overlap")
    return rows


def certify_navtrain(rows):
    """Check the pinned repo's official log lists, not just directory names."""
    import os
    from omegaconf import OmegaConf
    repo = Path(os.environ["NAVSIM_DEVKIT_ROOT"])
    directory = repo / "navsim/planning/script/config/common/train_test_split/scene_filter"
    train = set(OmegaConf.load(directory / "navtrain.yaml")["log_names"])
    test = set(OmegaConf.load(directory / "navtest.yaml")["log_names"])
    requested = {r["log"] for r in rows}
    if requested & test or not requested <= train:
        raise ValueError("Training requests contain non-navtrain/navtest logs: " + repr(sorted((requested - train) | (requested & test))))
    return {"navtrain_log_list_sha256": sha256(directory / "navtrain.yaml"),
            "navtest_log_list_sha256": sha256(directory / "navtest.yaml")}


def load_sample(row, device):
    from navsim.planning.training.dataset import load_feature_target_from_pickle
    features = load_feature_target_from_pickle(Path(row["feature_path"]))
    targets = load_feature_target_from_pickle(Path(row["target_path"]))
    return ({k: v[None].to(device) for k, v in features.items()},
            {k: v[None].to(device) for k, v in targets.items()})


def check_context_alignment(context, features, targets):
    """Fail on stale feature caches or a different planning instant/frame."""
    target = targets["trajectory"][0].detach().cpu().float()
    expected = torch.as_tensor(context["gt_poses"]).float()
    if target.shape != expected.shape or not torch.allclose(target, expected, atol=.02, rtol=0):
        raise ValueError("Cached GT/context mismatch: " + context["token"])
    # Official feature builder concatenates command[4], velocity[2], acceleration[2].
    measured = torch.linalg.vector_norm(features["status_feature"][0, -4:-2].detach().cpu().float())
    if not torch.isfinite(measured) or abs(float(measured) - float(context["v0"])) > .02:
        raise ValueError("Cached current speed/context mismatch: " + context["token"])


@torch.no_grad()
def capture_rollout(agent, features, targets):
    """Capture ONE real old-policy rollout; don't recompute loss or run PDM.

    A temporary instance wrapper reuses the complete upstream feature path.
    Both methods and the capture flag are restored, including on failure.
    """
    from types import MethodType
    model = agent._transfuser_model
    head = model._trajectory_head
    original = head.get_rlloss
    previous = head._config.energy_capture_candidates
    def return_old(self, *args, **kwargs):
        return kwargs.get("old_pred", args[-1] if args else None)
    head.get_rlloss = MethodType(return_old, head)
    head._config.energy_capture_candidates = True
    try:
        return model(features, targets, eta=1.0, cal_pdm=False)["energy_candidates"]
    finally:
        head.get_rlloss = original
        head._config.energy_capture_candidates = previous


def check_gpu_temperature(limit=80):
    """Check only GPU(s) explicitly assigned to this process; no device changes."""
    import os
    import subprocess
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES explicitly before a GPU run")
    result = subprocess.run(["nvidia-smi", "-i", visible,
                             "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                            text=True, capture_output=True, timeout=10, check=True)
    temperatures = [int(s.strip()) for s in result.stdout.splitlines()]
    if not temperatures or max(temperatures) >= limit:
        raise RuntimeError("THERMAL_ABORT: " + repr(temperatures))
    return temperatures
