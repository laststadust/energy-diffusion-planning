# Data preparation

Preparation and training use NAVSIM v1 `navtrain` with the scene filters from
the pinned DiffusionDriveV2 checkout. `navtest` is reserved for evaluation.
Obtain OpenScene records, sensor files and nuPlan maps through the official
[NAVSIM installation instructions](https://github.com/autonomousvision/navsim/blob/v1.1/docs/install.md)
under their respective licenses. This repository does not distribute these
datasets, generated caches or sample manifests.

## Paths

Use Linux for NAVSIM preparation and training. The preparers use POSIX file
locks and spawn worker processes. Data, generated files and the upstream
checkout are separate directories:

```text
DATA_ROOT/
  maps/
  navsim_logs/trainval/
  sensor_blobs/trainval/
WORK_ROOT/
UPSTREAM_ROOT/
```

Set paths in the current shell, using absolute directories on a volume with
enough free space:

```bash
export DATA_ROOT=/path/to/navsim-data
export WORK_ROOT=/path/to/generated-files
export NAVSIM_DEVKIT_ROOT=/path/to/DiffusionDriveV2
export OPENSCENE_DATA_ROOT="$DATA_ROOT"
export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps"
export NUPLAN_MAP_VERSION=nuplan-maps-v1.0
```

The code does not assume a particular disk mount or user name. The output must
not overlap the raw dataset or a reused feature cache. An existing output can
be resumed only when its preparation specification and committed artifacts
match. Arbitrary existing directories are rejected.

## Full training preparation

Run from the release environment after installing the pinned upstream code
and the integration patch:

```bash
python -m ddv2_energy.scale_data \
  --repo "$NAVSIM_DEVKIT_ROOT" \
  --data-root "$DATA_ROOT" \
  --out "$WORK_ROOT/prepared" \
  --workers 4 --seed 0 --actor-cap 128 --min-free-gib 300
```

The command creates feature, target, context and PDM metric caches. It requires
raw sensor files when feature caches need to be generated. It uses CPUs for
this stage and does not train a model.

`--feature-cache /path/to/existing-cache` optionally reuses existing compatible
features and targets from the same pinned V2 preprocessing. Shape checks do
not establish preprocessing equivalence and are not a V1-to-V2 conversion.
Reuse requires the expected tensor shapes, finite values,
ground-truth trajectory alignment and current-speed alignment. Invalid entries
are rebuilt under the new output directory with the upstream feature builder;
the supplied cache is never modified. Without this option, all required
features are generated in the output directory. The dataset and reused pickle
files must come from trusted sources.

`--min-free-gib` controls the free-space guard on the output filesystem. Its
default is 300 GiB, which is a guard threshold, not a measured dataset size or
a guarantee that preparation will fit. Available storage is checked during
the build. Data volume and cache reuse determine the actual requirement.

### Selection

The preparation procedure:

1. Reads the official `navtrain` log and token allowlists and rejects overlap
   with `navtest` logs.
2. Shuffles sorted training logs with split seed 0 and assigns every fifth log
   to internal validation. Training and validation are disjoint by log.
3. Inspects every official token in the available raw logs, with no per-log
   sample quota and a frame gap of one.
4. Requires four history frames including the current frame and ten future
   frames for the upstream PDM cache. The trajectory and energy supervision
   horizon is eight future points at 0.5-second intervals.
5. Excludes windows with missing route information, invalid timing or poses,
   unsupported reverse motion, invalid current actors, incomplete matched
   future actor labels or more than 128 current actors in the configured ROI.
6. Freezes selection before computing model predictions, rewards or PDM scores.

The fixed full-preparation protocol checks the upstream allowlist counts
(1,192 logs and 103,288 tokens), split seed 0 and actor cap 128. These are
protocol checks, not counts of the eligible training subset. Exclusions are
recorded separately. Changing the upstream version or scientific selection
rules requires a separately defined preparation protocol and output directory.

Current map and actor annotations supply privileged training reward context.
Future actor annotations are stored for graph supervision only. They are not
inputs to the policy or to the graph's current-scene forward pass.

### Outputs

```text
prepared/
  manifest.json
  engineering_manifest.json
  contexts/manifest.json
  contexts/contexts/*.pt
  feature_cache/...
  metric_cache/...
  preparation_spec.json
  preparation_report.json
```

`manifest.json` contains one row per eligible sample, with `token`, `log`,
`split`, `feature_path`, `target_path` and `metric_cache_path`. Reused feature
paths can point outside `prepared/`. The context manifest records checksums
and sample identities. `engineering_manifest.json` contains the first 32
training and 16 validation records for a bounded integration check, not a
separate benchmark.

Continue to candidate collection only after `preparation_report.json` reports
`state: COMPLETE`. An interrupted build may have partial caches but no valid
consumer manifest. Re-running the same command verifies and resumes matching
artifacts; changes to raw data, code or the frozen specification are rejected.

## Small integration subset

For installation and integration checks before full preparation:

```bash
python -m ddv2_energy.prepare_experiment \
  --repo "$NAVSIM_DEVKIT_ROOT" \
  --data-root "$DATA_ROOT" \
  --out "$WORK_ROOT/subset" \
  --train-count 32 --val-count 16 --max-per-log 8 \
  --frame-gap 8 --actor-cap 128 --workers 2 --seed 0
```

This entrypoint uses explicit sample quotas and temporal spacing. It shares
the annotation checks and cache builders with full preparation but does not
reproduce the full training selection. Keep subset outputs separate.

## Context-only preparation

`ddv2_energy.prepare_contexts` accepts an explicit JSON list containing `token`,
`log` and `split` fields. It builds contexts but no feature or PDM caches:

```bash
python -m ddv2_energy.prepare_contexts \
  --requests /path/to/requests.json \
  --logs-root "$DATA_ROOT/navsim_logs/trainval" \
  --maps-root "$DATA_ROOT/maps" \
  --out "$WORK_ROOT/requested-contexts" --actor-cap 128
```

Caller-provided requests are checked for duplicate tokens and log overlap.
This lower-level entrypoint does not independently certify official split
membership. Use full preparation for the documented training protocol.
