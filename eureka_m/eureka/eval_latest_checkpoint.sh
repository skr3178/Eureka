#!/bin/bash
# Script to evaluate the latest checkpoint from a Eureka run

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate eureka3

# Default to the most recent run
RUN_DIR="${1:-$(ls -td /home/skr/eureka_p/eureka_m/eureka/outputs/eureka_mujoco/*/ | head -1)}"

echo "Using run directory: $RUN_DIR"

# Find the best checkpoint (usually the one without "last_" prefix)
CHECKPOINT_DIR="$RUN_DIR/runs/RoboPianistMujoco_SAC/nn"
if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

# Find the best checkpoint (prefer the specific epoch 600 checkpoint)
BEST_CHECKPOINT=$(find "$CHECKPOINT_DIR" -name "last_RoboPianistMujoco_SAC_ep_600_rew_25001352.0.pth" | head -1)
if [ -z "$BEST_CHECKPOINT" ]; then
    # Fallback to the best checkpoint (without "last_" prefix)
    BEST_CHECKPOINT=$(find "$CHECKPOINT_DIR" -name "RoboPianistMujoco_SAC.pth" | head -1)
fi
if [ -z "$BEST_CHECKPOINT" ]; then
    # Final fallback to latest checkpoint
    BEST_CHECKPOINT=$(ls -t "$CHECKPOINT_DIR"/*.pth | head -1)
fi

if [ -z "$BEST_CHECKPOINT" ]; then
    echo "Error: No checkpoint found in $CHECKPOINT_DIR"
    exit 1
fi

echo "Found checkpoint: $BEST_CHECKPOINT"

# Find the environment file (prefer the final one, or latest iteration)
# Exclude _rewardonly files as they only contain reward functions, not the full environment
ENV_FILE=$(find "$RUN_DIR" -name "env_iter*_response*.py" ! -name "*_rewardonly.py" | sort -V | tail -1)
if [ -z "$ENV_FILE" ]; then
    ENV_FILE="$RUN_DIR/env_init_obs.py"
fi

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: Environment file not found in $RUN_DIR"
    exit 1
fi

echo "Using environment file: $ENV_FILE"

# Config file
CONFIG_FILE="/home/skr/eureka_p/eureka_m/eureka/cfg/train/robopianist_sac.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    exit 1
fi

# Change to the run directory for relative paths
cd "$RUN_DIR"

# Run evaluation with GUI
python /home/skr/eureka_p/eureka_m/eureka/eval_checkpoint_mujoco.py \
    --checkpoint "$BEST_CHECKPOINT" \
    --env_file "$ENV_FILE" \
    --config "$CONFIG_FILE" \
    --num_episodes 1 \
    --seed 42 \
    --use_viewer

