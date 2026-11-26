# Training

Set the paths in [setup](setup.md) and complete [data preparation](data.md).
The examples below use one seed. Repeat with the other configured seeds and
separate output directories for independent runs.

```bash
cd "$NAVSIM_DEVKIT_ROOT"
export PREPARED="$WORK_ROOT/prepared"
export RUN_ROOT="$WORK_ROOT/seed0"
mkdir -p "$RUN_ROOT"
```

## Candidate collection

Collect terminal training-sampler trajectories from the same initialized policy
used by both policy-training arms. The bank uses 20 anchors with four samples
per anchor, yielding 80 candidates with eight future poses each. It does not
add the demonstrated ego trajectory as a candidate.

```bash
python -m ddv2_energy.scale_bank \
  --manifest "$PREPARED/manifest.json" \
  --contexts "$PREPARED/contexts" \
  --checkpoint "$BASE_CHECKPOINT" \
  --out "$WORK_ROOT/candidates.pt" \
  --device cuda:0 --seed 0 --shard-size 128
```

The collector fingerprints its input files and uses a per-token random stream.
Use `--resume` only for an interrupted collection with the same inputs and
settings. Preparation filters incomplete future labels and actor truncation;
the bank loader independently rejects them.

## Graph training

The graph receives candidate ego motion and current surrounding-object states.
Future recorded objects provide supervision only. The input uses constant-
velocity extrapolation and does not read those future labels during prediction.

```bash
python "$RELEASE_ROOT/scripts/train_stage.py" graph \
  --bank "$WORK_ROOT/candidates.pt" \
  --contexts "$PREPARED/contexts" \
  --out "$RUN_ROOT/graph" --seed 0
```

`configs/training.json` sets ten epochs, effective batch size 128, microbatch
size 16 and learning rate 0.001. Gradient accumulation preserves the effective
batch while bounding the live graph size. The trainer selects `best.pt` by
held-out validation Huber loss and also saves full epoch checkpoints.

## Policy optimization

Start both arms from the same full official initialization checkpoint. Do not
initialize one arm from the other arm's trained output.

```bash
# Matched baseline continuation.
python "$RELEASE_ROOT/scripts/train_stage.py" policy \
  --manifest "$PREPARED/manifest.json" \
  --checkpoint "$BASE_CHECKPOINT" \
  --mode original --out "$RUN_ROOT/policy_original" --seed 0

# Road and interaction energy reward.
python "$RELEASE_ROOT/scripts/train_stage.py" policy \
  --manifest "$PREPARED/manifest.json" \
  --checkpoint "$BASE_CHECKPOINT" \
  --mode dual_graph --contexts "$PREPARED/contexts" \
  --graph "$RUN_ROOT/graph/best.pt" \
  --out "$RUN_ROOT/policy_energy" --seed 0
```

The recipe sets ten epochs, batch size four, learning rate 1e-5 and energy
strength 0.1. Both arms use the same ordered samples, epoch-seeded sampler and
training schedule. The image backbone, anchors, graph and selector are frozen;
the denoising policy is updated. See [method](method.md) for the advantage rules.

Use `--dry-run` on either stage launcher to print its command without starting
training. `seeds` in the recipe lists the intended independent runs; each
invocation takes an explicit `--seed` and does not launch them automatically.

The policy runner writes full checkpoints under `checkpoints/`, with optimizer,
scheduler, RNG and configuration metadata. Resume requires the same source,
data and budget and a **new output directory**. Increasing the original epoch
budget is not supported by the strict resume identity in this release.

For a bounded GPU integration check before a full run, invoke the policy
module directly using the prepared engineering manifest:

```bash
python -m ddv2_energy.run \
  --manifest "$PREPARED/engineering_manifest.json" \
  --checkpoint "$BASE_CHECKPOINT" --mode original \
  --out "$WORK_ROOT/integration_original" \
  --epochs 1 --batch-size 1 --workers 0 --seed 0 \
  --gate --limit-train-batches 2 --limit-val-batches 2 --max-steps 2
```

This check changes model parameters only in its new local output directory.
Its checkpoints are marked as integration-test outputs and cannot initialize
formal training or resume. It does not replace validation of the energy arm
with a genuinely trained graph.

## Weight export

The RL checkpoint is not a complete inference checkpoint. Export the changed
policy tensors into the official full SEL checkpoint; retain the original
selector and frozen tensors.

```bash
python -m ddv2_energy.export_policy \
  --base "$BASE_CHECKPOINT" \
  --trained "$RUN_ROOT/policy_energy/checkpoints/last.ckpt" \
  --out "$RUN_ROOT/policy_energy_sel.ckpt"
```

The exporter checks tensor names, shapes, dtypes and frozen tensor values. It
refuses missing policy updates and selector changes. The output is for
evaluation, not for resuming optimizer state. This command operates on your
locally trained files; no method checkpoint is included in this release.

## Evaluation

Prepare the test caches using the pinned upstream scripts, separately from the
training contexts:

```bash
export TEST_FEATURE_CACHE="$WORK_ROOT/navtest_features"
export TEST_METRIC_CACHE="$WORK_ROOT/navtest_metrics"
cd "$NAVSIM_DEVKIT_ROOT"
python navsim/planning/script/run_dataset_caching.py \
  agent=diffusiondrivev2_rl_agent experiment_name=navtest_features \
  train_test_split=navtest "cache_path=$TEST_FEATURE_CACHE"
python navsim/planning/script/run_metric_caching.py \
  train_test_split=navtest "cache.cache_path=$TEST_METRIC_CACHE"
bash "$RELEASE_ROOT/scripts/evaluate.sh" \
  "$RUN_ROOT/policy_energy_sel.ckpt" "$RUN_ROOT/navtest_energy"
```

This wrapper invokes the upstream fast SEL evaluation path, with its batch size
of 64, one visible GPU and mixed precision. Use the same cache, scorer version
and settings for all compared checkpoints. It does not load the training graph
or apply an energy-gradient correction to the inference trajectory.

The wrapper exposes the upstream aggregate evaluation output. It does not
include private experiment ledgers or the previous server's reporting package.
No published score is claimed by these examples.
