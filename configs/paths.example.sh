#!/usr/bin/env bash
# Set paths on the machine where NAVSIM runs, then source this file.
export RELEASE_ROOT="/path/to/release_draft"
export NAVSIM_DEVKIT_ROOT="/path/to/DiffusionDriveV2"
export DATA_ROOT="/path/to/navsim_dataset"
export WORK_ROOT="/path/to/large_storage/energy_planning"
export NAVSIM_EXP_ROOT="$WORK_ROOT/navsim"
export OPENSCENE_DATA_ROOT="$DATA_ROOT"
export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps"
export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export BASE_CHECKPOINT="$NAVSIM_DEVKIT_ROOT/ckpts/diffusiondrivev2_sel.ckpt"
export CUDA_VISIBLE_DEVICES=0

# Optional cache relocation. Create these directories before enabling them.
# export HF_HOME="$WORK_ROOT/cache/huggingface"
# export TORCH_HOME="$WORK_ROOT/cache/torch"
# export TMPDIR="$WORK_ROOT/tmp"
# export PIP_CACHE_DIR="$WORK_ROOT/cache/pip"

# Host-specific terminal/session startup is intentionally not executed here.
# screen -S planning
