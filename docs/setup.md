# Setup

## Environment

The training target is Linux with an NVIDIA GPU and the pinned DiffusionDriveV2
checkout. Python 3.9, PyTorch 2.0.1 and Lightning 2.2.1 follow that checkout's
requirements. Data preparation and candidate collection use Linux file locks.
The small core package can also be imported without NAVSIM for synthetic tests.
Core tests have been run with Python 3.12, CPU PyTorch 2.7.1 and NumPy 2.3.5.
That test environment is not the full NAVSIM training environment below.

The dependency files have separate roles:

- `requirements-core.txt`: direct dependencies of geometry, graphs and rewards.
- Upstream `requirements.txt`: the NAVSIM model, maps, caching and scorer stack.
- `requirements-navsim.txt`: direct V2 extensions absent from that upstream list
  or left unpinned there, including diffusers, einops, timm and cloudpickle.

The extra package pins define a proposed installation baseline. A clean Linux
CUDA installation remains to be validated. Use a separate Conda environment
for setup and testing.

From the release directory, after cloning and patching upstream as in README:

```bash
conda env create -f environment.yml
conda activate energy-navsim
export RELEASE_ROOT="$(pwd)"
export NAVSIM_DEVKIT_ROOT="$RELEASE_ROOT/vendor/DiffusionDriveV2"

# Select the wheel family supported by the target machine's driver.
python -m pip install torch==2.0.1 torchvision==0.15.2 \
  --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r "$NAVSIM_DEVKIT_ROOT/requirements.txt" \
  -r "$RELEASE_ROOT/requirements-navsim.txt"
python -m pip install --no-deps -e "$NAVSIM_DEVKIT_ROOT"
python -m pip install --no-deps -e "$RELEASE_ROOT"
python -m pip check
```

The upstream requirements still include tools used indirectly by NAVSIM and
nuPlan. They are external framework dependencies, not imports of any earlier
energy-planning experiment package. This release does not require a copy of
the private research workspace.

## External assets

Follow the pinned [upstream asset instructions](https://github.com/hustvl/DiffusionDriveV2/blob/1cd12a1e155c34dcc471261835444c8d5587580b/docs/train_eval.md).
For this workflow, provide the full official SEL initialization checkpoint and
the ResNet-34 backbone file. The agent constructor loads the backbone file
before loading the full checkpoint, so the latter alone is insufficient.

Expected paths relative to the upstream checkout:

```text
ckpts/resnet34.a1_in1k/pytorch_model.bin
ckpts/diffusiondrivev2_sel.ckpt
kmeans_navsim_traj_20.npy
```

The anchor file is provided by upstream. The official initialization assets
remain external; this repository does not redistribute them. GTRS vocabulary
files are for upstream selector training, which is not part of this workflow.
Only load checkpoints, candidate banks and cached pickles from trusted sources.
Their Python serialization can execute code during loading.

## Paths

Edit `configs/paths.example.sh` or set the same variables in your shell. Use
absolute paths for the dataset, working directory and initialization checkpoint.
Large generated artifacts stay under `WORK_ROOT`; it can reside on any disk.
There is no required home directory or mount name.

Set `CUDA_VISIBLE_DEVICES` to one available GPU. The runners use `cuda:0` within
that process. They read `nvidia-smi` and stop if an assigned device reaches
80 degrees Celsius or the query fails; they do not alter clocks, power limits
or other users' jobs. Independent seed jobs may use different GPUs.

Run NAVSIM stages from `NAVSIM_DEVKIT_ROOT`, because upstream anchors and
backbone paths are relative to that directory. The release package remains
importable through its editable installation.

## Checks

```bash
python "$RELEASE_ROOT/scripts/apply_upstream.py" --repo "$NAVSIM_DEVKIT_ROOT" --check
python -c 'import torch, navsim, ddv2_energy, diffusers, cloudpickle; print(torch.cuda.is_available())'
cd "$RELEASE_ROOT"
python -B -m unittest discover -s tests -t . -v
```

These checks cover package imports, CUDA availability and synthetic tests.
For NAVSIM scoring, follow [training and evaluation](training.md#evaluation).
