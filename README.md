# Energy-based diffusion planning

Candidate-energy rewards for anchor-based trajectory planning on NAVSIM.
Road clearance and a dual-graph interaction evaluator shape the reward used to
optimize the diffusion policy. The graph is trained on candidate-level risk
labels and frozen during policy optimization. Inference uses the optimized
policy and the existing trajectory selector.

The implementation builds on DiffusionDriveV2. This repository contains the
method modules, data preparation, training utilities and a small integration
patch for a fixed upstream revision.

## Evaluation scripts

The current evaluation entrypoint is [scripts/evaluate.sh](scripts/evaluate.sh),
which runs the NAVSIM v1 `navtest` benchmark with a 4-second planning horizon.

Model weights and detailed dataset configurations will be released in a future
update.

## Setup

Use Linux for data preparation and training. See [setup](docs/setup.md) for the
environment and required upstream assets. From this repository:

```bash
git clone https://github.com/hustvl/DiffusionDriveV2.git vendor/DiffusionDriveV2
git -C vendor/DiffusionDriveV2 checkout 1cd12a1e155c34dcc471261835444c8d5587580b
python scripts/apply_upstream.py --repo vendor/DiffusionDriveV2 --check
python scripts/apply_upstream.py --repo vendor/DiffusionDriveV2
```

The installer verifies the upstream revision and applies the integration patch.

Configure storage and GPU paths using [configs/paths.example.sh](configs/paths.example.sh).
The file includes optional cache and terminal examples.

## Training

The workflow is:

1. Prepare log-disjoint training and validation samples, features, PDM caches
   and energy contexts.
2. Collect final trajectories from the initialized diffusion policy.
3. Train the dual graph with candidate-level risk supervision.
4. Freeze the graph and optimize the diffusion policy using the energy reward.
5. Merge the trained policy into the full selector checkpoint for evaluation.

See [data preparation](docs/data.md) and [training](docs/training.md) for commands.
[configs/training.json](configs/training.json) is consumed by
`scripts/train_stage.py`; it sets ten graph epochs and ten policy epochs.
Each invocation runs one seed on one visible GPU. Multiple seeds can be launched
as independent jobs.

Use `--mode original` for the matched baseline and `--mode dual_graph` for the
energy branch. [Method details](docs/method.md) describe the reward, advantage
rules and frozen parameters.

## Evaluation

Evaluation uses the exported SEL checkpoint and the pinned upstream PDM scorer.
The script takes a checkpoint path and an output directory. See
[training and evaluation](docs/training.md#evaluation) for the command and cache
configuration.

## Code layout

```text
ddv2_energy/       Geometry, graphs, rewards and training/data entrypoints
configs/          Training recipe and external path examples
patches/          Minimal upstream integration
scripts/          Checked patch installer and stage launchers
tests/            Synthetic unit and integration tests
docs/             Setup, data, method and training instructions
```

## Tests

With the core dependencies installed:

```bash
python -B -m unittest discover -s tests -t . -v
```

The synthetic tests cover graph geometry, reward rules, input alignment and
checkpoint handling. See the setup guide for the NAVSIM training and evaluation
environment.

## Acknowledgments

This project builds on DiffusionDriveV2 and NAVSIM. See
[third-party notices](THIRD_PARTY_NOTICES.md) and [license information](LICENSE.md)
for attribution and licensing details.

## Citation and license

Version [v0.1.0](https://github.com/laststadust/energy-diffusion-planning/releases/tag/v0.1.0)
is documented in the [changelog](CHANGELOG.md). Use [CITATION.cff](CITATION.cff)
to cite this software release.

Project-specific code is distributed under the [MIT License](LICENSE.md).
Third-party code retains its original license.
