#!/usr/bin/env bash
# Uses the pinned upstream SEL evaluator; no energy model is loaded at inference.
set -Eeuo pipefail
: "${NAVSIM_DEVKIT_ROOT:?Set NAVSIM_DEVKIT_ROOT}"
: "${CUDA_VISIBLE_DEVICES:?Select one GPU explicitly}"
: "${TEST_FEATURE_CACHE:?Set TEST_FEATURE_CACHE}"
: "${TEST_METRIC_CACHE:?Set TEST_METRIC_CACHE}"
if [ "$#" -ne 2 ]; then
  echo "Usage: bash scripts/evaluate.sh CHECKPOINT NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi
CHECKPOINT=$(realpath "$1")
OUTPUT=$(realpath -m "$2")
test -f "$CHECKPOINT"
test -d "$TEST_FEATURE_CACHE"
test -d "$TEST_METRIC_CACHE"
TEST_FEATURE_CACHE=$(realpath "$TEST_FEATURE_CACHE")
TEST_METRIC_CACHE=$(realpath "$TEST_METRIC_CACHE")
if [ -e "$OUTPUT" ]; then
  echo "Refusing to overwrite an existing evaluation directory: $OUTPUT" >&2
  exit 2
fi
cd "$NAVSIM_DEVKIT_ROOT"
python navsim/planning/script/run_pdm_score_fast.py \
  agent=diffusiondrivev2_sel_agent \
  experiment_name=energy_planning_eval \
  train_test_split=navtest \
  "agent.checkpoint_path=$CHECKPOINT" \
  "+metric_cache_path=$TEST_METRIC_CACHE" \
  "+test_cache_path=$TEST_FEATURE_CACHE" \
  trainer.params.strategy=auto \
  +trainer.params.devices=1 \
  trainer.params.precision=16-mixed \
  dataloader.params.num_workers=4 \
  "output_dir=$OUTPUT" \
  seed=0
