"""Merge a trained RL policy into a complete SEL checkpoint.

Only load local checkpoints from trusted sources: torch.load uses pickle.
The exported file retains the base selector and is intended for evaluation,
not for resuming training.
"""
import argparse
from collections import OrderedDict
from collections.abc import Mapping
import copy
import datetime
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch


HEAD = "_transfuser_model._trajectory_head."
POLICY_ROOTS = {"diff_decoder", "plan_anchor_encoder", "time_mlp"}
SELECTOR_ROOTS = {
    "plan_anchor_scorer_encoder", "scorer_decoder", "fine_scorer_decoder",
    "NC_head", "EP_head", "DAC_head", "TTC_head", "C_head",
    "fine_NC_head", "fine_EP_head", "fine_DAC_head", "fine_TTC_head", "fine_C_head",
}


def _selector_key(key):
    root = key[len(HEAD):].split(".", 1)[0] if key.startswith(HEAD) else ""
    return root in SELECTOR_ROOTS or "scorer" in key.lower() or "selector" in key.lower()


def _index_state(state, name):
    if not isinstance(state, Mapping) or not state:
        raise ValueError(name + " must be a nonempty state_dict mapping")
    result = {}
    for original_key, value in state.items():
        if not isinstance(original_key, str):
            raise TypeError(name + " state keys must be strings")
        key = original_key.removeprefix("agent.")
        if key in result:
            raise ValueError(name + " has duplicate normalized key: " + key)
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise TypeError(name + " must contain dense tensors: " + key)
        if not torch.isfinite(value).all():
            raise ValueError(name + " contains nonfinite tensor: " + key)
        result[key] = (original_key, value)
    return result


def _bitwise_equal(a, b):
    # Byte comparison distinguishes signed zero when checking frozen tensors.
    a_bytes = a.detach().contiguous().reshape(-1).view(torch.uint8)
    b_bytes = b.detach().contiguous().reshape(-1).view(torch.uint8)
    if a_bytes.device != b_bytes.device:
        a_bytes, b_bytes = a_bytes.cpu(), b_bytes.cpu()
    return torch.equal(a_bytes, b_bytes)


def merge_state(base_state, trained_state):
    """Merge denoising tensors while retaining base-state keys and frozen values.

    Strip the leading agent. prefix only for comparison. The trained state must
    be an RL state without selector tensors. At least one diff_decoder tensor
    must differ from the base so that a no-op export cannot look like training.
    """
    base = _index_state(base_state, "base")
    trained = _index_state(trained_state, "trained")
    updates, changed = [], []
    for key, (_, value) in trained.items():
        if key not in base:
            raise ValueError("trained key is absent from official SEL base: " + key)
        reference = base[key][1]
        if value.shape != reference.shape or value.dtype != reference.dtype:
            raise ValueError("shape/dtype mismatch for trained tensor: " + key)
        if _selector_key(key):
            raise ValueError("trained RL checkpoint must not contain selector tensors: " + key)
        equal = _bitwise_equal(value, reference)
        if key.startswith(HEAD):
            root = key[len(HEAD):].split(".", 1)[0]
            if root == "plan_anchor":
                if not equal:
                    raise ValueError("frozen plan_anchor changed: " + key)
                continue
            if root in POLICY_ROOTS:
                updates.append(key)
                if not equal:
                    changed.append(key)
                continue
            # Unrecognized head buffers must remain unchanged.
        if not equal:
            raise ValueError("non-denoising/frozen tensor changed: " + key)
    if not any(key.startswith(HEAD + "diff_decoder.") for key in changed):
        raise ValueError("no changed diff_decoder tensor: refusing an untrained/no-op export")

    result = OrderedDict((key, value.detach().clone()) for key, value in base_state.items())
    if hasattr(base_state, "_metadata"):
        result._metadata = copy.deepcopy(base_state._metadata)
    for key in updates:
        result[base[key][0]] = trained[key][1].detach().clone()
    selector_keys = sorted(key for key in base if _selector_key(key))
    report = {
        "base_tensor_count": len(base), "trained_tensor_count": len(trained),
        "updated_policy_keys": sorted(updates), "changed_policy_keys": sorted(changed),
        "preserved_selector_keys": selector_keys,
        "preserved_base_only_count": len(set(base) - set(trained)),
        "plan_anchor_unchanged": True,
        "frozen_tensors_bitwise_verified": True,
    }
    return result, report


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Trusted FULL official V2 SEL checkpoint")
    parser.add_argument("--trained", required=True, help="Trusted trained RL Lightning checkpoint")
    parser.add_argument("--out", required=True, help="NEW evaluation checkpoint path")
    args = parser.parse_args()
    base_path, trained_path, out = (Path(p).resolve() for p in (args.base, args.trained, args.out))
    if out.exists():
        raise FileExistsError("Refusing to overwrite: " + str(out))
    for path in (base_path, trained_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    trained = torch.load(trained_path, map_location="cpu", weights_only=False)
    if not isinstance(base, Mapping) or not isinstance(trained, Mapping):
        raise ValueError("Both inputs must be Lightning checkpoint dictionaries")
    if "state_dict" not in base or "state_dict" not in trained:
        raise ValueError("Both checkpoints must contain state_dict")
    normalized = _index_state(base["state_dict"], "base")
    available_roots = {key[len(HEAD):].split(".", 1)[0] for key in normalized if key.startswith(HEAD)}
    missing_selectors = SELECTOR_ROOTS - available_roots
    if missing_selectors:
        raise ValueError("Base is not a full SEL checkpoint; missing selectors: " + repr(sorted(missing_selectors)))
    state, report = merge_state(base["state_dict"], trained["state_dict"])
    provenance = dict(
        format="ddv2_energy_policy_export_v1", purpose="evaluation_only_not_training_resume",
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        base_path=str(base_path), base_sha256=_sha256(base_path),
        trained_path=str(trained_path), trained_sha256=_sha256(trained_path),
        merge_report=report,
        note="Trained anchor-diffusion policy; unchanged official coarse/fine selector. No optimizer state.",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".policy-export-", suffix=".partial", dir=out.parent)
    os.close(fd)
    try:
        # Inference exports exclude optimizer, scheduler and training-resume state.
        torch.save({"state_dict": state, "energy_rl_export": provenance}, temporary)
        # Publish atomically on the same filesystem without overwriting a target.
        os.link(temporary, out)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(json.dumps({"out": str(out), "changed_policy_tensors": len(report["changed_policy_keys"]),
                      "preserved_selector_tensors": len(report["preserved_selector_keys"]),
                      "sha256": _sha256(out)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
