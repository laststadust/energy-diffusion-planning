"""Apply the energy integration to a verified DiffusionDriveV2 checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def normalized(data):
    return data.decode("utf-8").replace("\r\n", "\n").rstrip("\n") + "\n"


def source_hash(path):
    return hashlib.sha256(normalized(Path(path).read_bytes()).encode()).hexdigest()


def apply_integration(repo, lock, patch, *, check_only=False):
    repo, patch = Path(repo).resolve(), Path(patch).resolve()
    if not (repo / ".git").exists():
        raise ValueError("--repo must be a Git checkout")
    head = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if head != lock["commit"]:
        raise ValueError("Upstream commit differs from upstream.lock")
    if hashlib.sha256(normalized(patch.read_bytes()).encode()).hexdigest() != lock["patch_sha256"]:
        raise ValueError("Integration patch checksum mismatch")
    states = []
    for entry in lock["files"]:
        target = (repo / entry["path"]).resolve()
        if repo not in target.parents:
            raise ValueError("Patch target escapes repository")
        value = source_hash(target)
        if value == entry["before_sha256"]:
            states.append("before")
        elif value == entry["after_sha256"]:
            states.append("after")
        else:
            raise ValueError("Unexpected local modification: " + entry["path"])
    if states and all(state == "after" for state in states):
        return "already applied"
    if not states or not all(state == "before" for state in states):
        raise ValueError("Partial integration detected; restore the two targets from a clean checkout")
    command = ["git", "-C", str(repo), "apply"]
    subprocess.run(command + ["--check", str(patch)], check=True)
    if check_only:
        return "compatible"
    subprocess.run(command + [str(patch)], check=True)
    if any(source_hash(repo / e["path"]) != e["after_sha256"] for e in lock["files"]):
        raise RuntimeError("Post-application source verification failed")
    return "applied"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--check", action="store_true", help="Validate without changing upstream files")
    args = parser.parse_args()
    lock = json.loads((ROOT / "upstream.lock").read_text(encoding="utf-8"))
    print(apply_integration(args.repo, lock, ROOT / lock["patch"], check_only=args.check))


if __name__ == "__main__":
    main()
