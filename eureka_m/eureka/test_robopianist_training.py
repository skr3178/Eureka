#!/usr/bin/env python3
"""
Test script to run training with generated RoboPianist environment files
without going through the full Eureka LLM loop.
"""

import os
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Path to generated environment file
GENERATED_ENV_FILE = "/home/skr/eureka_p/eureka_m/eureka/outputs/eureka_mujoco/2025-11-16_16-14-11/env_iter0_response0.py"
CONFIG_FILE = "/home/skr/eureka_p/eureka_m/eureka/cfg/train/robopianist_ppo.yaml"
TRAIN_SCRIPT = "/home/skr/eureka_p/eureka_m/eureka/train_mujoco.py"

def main():
    # Check if files exist
    if not os.path.exists(GENERATED_ENV_FILE):
        logging.error(f"Generated environment file not found: {GENERATED_ENV_FILE}")
        return 1
    
    if not os.path.exists(CONFIG_FILE):
        logging.error(f"Config file not found: {CONFIG_FILE}")
        return 1
    
    if not os.path.exists(TRAIN_SCRIPT):
        logging.error(f"Training script not found: {TRAIN_SCRIPT}")
        return 1
    
    logging.info(f"Testing with generated environment: {GENERATED_ENV_FILE}")
    logging.info(f"Using config: {CONFIG_FILE}")
    
    # Change to the eureka directory
    eureka_dir = "/home/skr/eureka_p/eureka_m/eureka"
    os.chdir(eureka_dir)
    
    # Run training with minimal iterations for testing
    import subprocess
    cmd = [
        "python", "-u", TRAIN_SCRIPT,
        f"--env_file={GENERATED_ENV_FILE}",
        f"--config={CONFIG_FILE}",
        "--num_envs=1",
        "--seed=42",
        "--wandb_activate=False",
        "--max_iterations=10",  # Just 10 iterations for testing
        f"--output_dir={os.path.dirname(GENERATED_ENV_FILE)}"
    ]
    
    logging.info(f"Running command: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=False, capture_output=False)
        return result.returncode
    except Exception as e:
        logging.error(f"Error running training: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())

