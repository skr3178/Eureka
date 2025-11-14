#!/usr/bin/env python3
"""
Training script for MuJoCo environments using rl_games.
Compatible with Eureka's LLM-based reward generation workflow.
"""

import os
import sys
import argparse
import logging
import datetime
from pathlib import Path
import shutil
import importlib.util

import numpy as np
import torch
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv
from omegaconf import OmegaConf

# Add paths for imports
EUREKA_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, EUREKA_ROOT_DIR)
sys.path.insert(0, os.path.join(EUREKA_ROOT_DIR, 'gymnasium'))

from rl_games.torch_runner import Runner
from rl_games.common import env_configurations, vecenv
from rl_games.common.ivecenv import IVecEnv
from rl_games.common.algo_observer import AlgoObserver
from rl_games.algos_torch import torch_ext
# Note: gymnasium is imported as 'gym' above, don't import old gym here


class MujocoAlgoObserver(AlgoObserver):
    """Custom algo observer for MuJoCo environments to log Eureka metrics."""
    
    def __init__(self):
        super().__init__()
        self.writer = None
        self.last_mean_rewards = -1000000000
        self.last_mean_gt_reward = -1000000000
        self.last_mean_gpt_reward = -1000000000
        self.last_consecutive_successes = 0
        
    def after_init(self, algo):
        """Initialize TensorBoard writer after algorithm initialization."""
        from torch.utils.tensorboard import SummaryWriter
        if hasattr(algo, 'writer') and algo.writer is not None:
            self.writer = algo.writer
            # Extract log directory from writer if possible
            if hasattr(algo.writer, 'log_dir'):
                log_dir = Path(algo.writer.log_dir)
                print(f"Tensorboard Directory: {log_dir}")
            elif hasattr(algo, 'experiment_name'):
                # Try to construct path from experiment name
                log_dir = Path.cwd() / "runs" / algo.experiment_name / "summaries"
                print(f"Tensorboard Directory: {log_dir}")
        else:
            # Create our own writer if algo doesn't have one
            log_dir = Path.cwd() / "runs" / datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            log_dir.mkdir(parents=True, exist_ok=True)
            summaries_dir = log_dir / "summaries"
            summaries_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(str(summaries_dir))
            print(f"Tensorboard Directory: {summaries_dir}")
    
    def process_infos(self, infos, done_indices):
        """Extract Eureka metrics from infos."""
        if not infos:
            return
        
        # Handle dict format (from vectorized environments)
        if isinstance(infos, dict):
            if 'gt_reward' in infos:
                values = infos['gt_reward']
                if isinstance(values, (np.ndarray, list)):
                    self.last_mean_gt_reward = np.mean(values)
                else:
                    self.last_mean_gt_reward = values
            if 'gpt_reward' in infos:
                values = infos['gpt_reward']
                if isinstance(values, (np.ndarray, list)):
                    self.last_mean_gpt_reward = np.mean(values)
                else:
                    self.last_mean_gpt_reward = values
            if 'consecutive_successes' in infos:
                values = infos['consecutive_successes']
                if isinstance(values, (np.ndarray, list)):
                    self.last_consecutive_successes = np.mean(values)
                else:
                    self.last_consecutive_successes = values
        # Handle list format
        elif isinstance(infos, list):
            gt_rewards = []
            gpt_rewards = []
            consecutive_successes = []
            
            for info in infos:
                if isinstance(info, dict):
                    if 'gt_reward' in info:
                        gt_rewards.append(info['gt_reward'])
                    if 'gpt_reward' in info:
                        gpt_rewards.append(info['gpt_reward'])
                    if 'consecutive_successes' in info:
                        consecutive_successes.append(info['consecutive_successes'])
            
            if gt_rewards:
                self.last_mean_gt_reward = np.mean(gt_rewards)
            if gpt_rewards:
                self.last_mean_gpt_reward = np.mean(gpt_rewards)
            if consecutive_successes:
                self.last_consecutive_successes = np.mean(consecutive_successes)
    
    def after_print_stats(self, frame, epoch_num, total_time):
        """Log metrics to TensorBoard."""
        if self.writer is None:
            return
            
        if self.last_mean_gt_reward > -1000000000:
            self.writer.add_scalar('gt_reward', self.last_mean_gt_reward, frame)
        if self.last_mean_gpt_reward > -1000000000:
            self.writer.add_scalar('gpt_reward', self.last_mean_gpt_reward, frame)
        if self.last_consecutive_successes > 0:
            self.writer.add_scalar('consecutive_successes', self.last_consecutive_successes, frame)


def load_generated_env(env_file_path):
    """Load the generated environment module from file."""
    spec = importlib.util.spec_from_file_location("generated_env", env_file_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load environment from {env_file_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Find the environment class in the module
    # It could be AntEnv, AntEnvGPT, or any class ending with Env
    env_class = None
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if (isinstance(attr, type) and 
            (attr_name == 'AntEnv' or attr_name.endswith('Env') or 'AntEnv' in attr_name)):
            env_class = attr
            break
    
    if env_class is None:
        # List all classes in the module for debugging
        classes = [name for name in dir(module) 
                  if isinstance(getattr(module, name, None), type) 
                  and not name.startswith('_')]
        raise ValueError(f"Environment class not found in {env_file_path}. "
                        f"Available classes: {classes}")
    
    return env_class


class MujocoVecEnv(IVecEnv):
    """Wrapper for gymnasium SyncVectorEnv to work with rl_games."""
    
    def __init__(self, env):
        self.env = env
        self.num_actors = env.num_envs if hasattr(env, 'num_envs') else 1
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
    def step(self, actions):
        obs, rewards, dones, truncs, infos = self.env.step(actions)
        # Combine terminated and truncated into single done signal
        dones = dones | truncs
        return obs, rewards, dones, infos
    
    def reset(self):
        obs, infos = self.env.reset()
        return obs
    
    def get_env_info(self):
        # For vectorized environments, use single_observation_space (not batched)
        if hasattr(self.env, 'single_observation_space'):
            obs_space = self.env.single_observation_space
        else:
            obs_space = self.observation_space
        
        # For action space, also use single if available
        if hasattr(self.env, 'single_action_space'):
            action_space = self.env.single_action_space
        else:
            action_space = self.action_space
        
        info = {
            'observation_space': obs_space,
            'action_space': action_space,
            'agents': 1,
            'value_size': 1,
            'use_global_observations': False,
        }
        return info
    
    def seed(self, seed):
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)
    
    def has_action_masks(self):
        return False


def create_mujoco_env(env_class, num_envs=1, **kwargs):
    """Create vectorized MuJoCo environment."""
    def make_env():
        env = env_class(**kwargs)
        return env
    
    if num_envs == 1:
        # For single env, create a single-element vector env
        vec_env = SyncVectorEnv([make_env])
        return MujocoVecEnv(vec_env)
    else:
        envs = [make_env for _ in range(num_envs)]
        vec_env = SyncVectorEnv(envs)
        return MujocoVecEnv(vec_env)


def register_mujoco_env(env_name, env_class, **env_kwargs):
    """Register MuJoCo environment with rl_games."""
    def env_creator(**kwargs):
        num_envs = kwargs.pop('num_actors', 1)
        combined_kwargs = {**env_kwargs, **kwargs}
        return create_mujoco_env(env_class, num_envs=num_envs, **combined_kwargs)
    
    env_configurations.register(env_name, {
        'vecenv_type': 'MUJOCO',
        'env_creator': env_creator,
    })
    
    # Register the vecenv type
    def create_mujoco_vecenv(config_name, num_actors, **kwargs):
        return env_creator(num_actors=num_actors, **kwargs)
    
    vecenv.register('MUJOCO', create_mujoco_vecenv)


def main():
    parser = argparse.ArgumentParser(description='Train MuJoCo environment with rl_games')
    parser.add_argument('--env_file', type=str, required=True,
                        help='Path to generated environment file')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to rl_games config YAML file')
    parser.add_argument('--num_envs', type=int, default=1,
                        help='Number of parallel environments')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--wandb_activate', type=bool, default=False,
                        help='Activate wandb logging')
    parser.add_argument('--wandb_entity', type=str, default='',
                        help='Wandb entity')
    parser.add_argument('--wandb_project', type=str, default='',
                        help='Wandb project')
    parser.add_argument('--max_iterations', type=int, default=1000,
                        help='Maximum training iterations')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load generated environment
    logging.info(f"Loading environment from {args.env_file}")
    EnvClass = load_generated_env(args.env_file)
    
    # Register environment with rl_games
    env_name = 'mujoco_ant'
    register_mujoco_env(env_name, EnvClass)
    
    # Load rl_games config
    logging.info(f"Loading config from {args.config}")
    config_dict = OmegaConf.load(args.config)
    
    # Convert OmegaConf to regular Python dict to avoid issues with rl_games
    # rl_games modifies the config (e.g., creates DefaultRewardsShaper objects)
    # which OmegaConf doesn't support
    config_dict = OmegaConf.to_container(config_dict, resolve=True)
    
    # Update config with command line arguments
    if 'params' not in config_dict:
        config_dict['params'] = {}
    if 'config' not in config_dict['params']:
        config_dict['params']['config'] = {}
    
    config_dict['params']['config']['env_name'] = env_name
    config_dict['params']['config']['num_actors'] = args.num_envs
    config_dict['params']['seed'] = args.seed
    if 'max_epochs' in config_dict['params']['config']:
        config_dict['params']['config']['max_epochs'] = args.max_iterations
    
    # Create algo observer
    observer = MujocoAlgoObserver()
    
    # Create runner
    runner = Runner(observer)
    runner.load(config_dict)
    
    # Patch experience buffer to handle gymnasium spaces
    # rl_games uses 'type(space) is gym.spaces.Box' which fails for gymnasium
    # We need to patch both __init__ (for action space detection) and _create_tensor_from_space
    from rl_games.common import experience
    import gymnasium.spaces as gymnasium_spaces
    
    # Store original methods
    original_init = experience.ExperienceBuffer.__init__
    original_create_tensor = experience.ExperienceBuffer._create_tensor_from_space
    
    def patched_init(self, env_info, algo_info, device, aux_tensor_dict=None):
        """Patched __init__ that handles gymnasium action spaces."""
        # Call original init but patch action space detection
        self.env_info = env_info
        self.algo_info = algo_info
        self.device = device

        self.num_agents = env_info.get('agents', 1)
        self.action_space = env_info['action_space']
        
        self.num_actors = algo_info['num_actors']
        self.horizon_length = algo_info['horizon_length']
        self.has_central_value = algo_info['has_central_value']
        self.use_action_masks = algo_info.get('use_action_masks', False)
        batch_size = self.num_actors * self.num_agents
        self.is_discrete = False
        self.is_multi_discrete = False
        self.is_continuous = False
        self.obs_base_shape = (self.horizon_length, self.num_agents * self.num_actors)
        self.state_base_shape = (self.horizon_length, self.num_actors)
        
        # Patch action space type checking to handle gymnasium
        if isinstance(self.action_space, gymnasium_spaces.Discrete) or (hasattr(gym, 'spaces') and type(self.action_space) is gym.spaces.Discrete):
            self.actions_shape = ()
            self.actions_num = self.action_space.n
            self.is_discrete = True
        if isinstance(self.action_space, gymnasium_spaces.Tuple) or (hasattr(gym, 'spaces') and type(self.action_space) is gym.spaces.Tuple):
            self.actions_shape = (len(self.action_space),) 
            self.actions_num = [action.n for action in self.action_space]
            self.is_multi_discrete = True
        if isinstance(self.action_space, gymnasium_spaces.Box) or (hasattr(gym, 'spaces') and type(self.action_space) is gym.spaces.Box):
            self.actions_shape = (self.action_space.shape[0],) 
            self.actions_num = self.action_space.shape[0]
            self.is_continuous = True
        
        self.tensor_dict = {}
        self._init_from_env_info(env_info)

        self.aux_tensor_dict = aux_tensor_dict
        if aux_tensor_dict is not None:
            self._init_from_aux_dict(aux_tensor_dict)
    
    def patched_create_tensor(self, space, base_shape):
        """Patched version that handles both gym and gymnasium spaces."""
        # Check for gymnasium spaces first (since we're using gymnasium)
        if isinstance(space, gymnasium_spaces.Box):
            dtype = experience.numpy_to_torch_dtype_dict[space.dtype]
            return torch.zeros(base_shape + space.shape, dtype=dtype, device=self.device)
        if isinstance(space, gymnasium_spaces.Discrete):
            dtype = experience.numpy_to_torch_dtype_dict[space.dtype]
            return torch.zeros(base_shape, dtype=dtype, device=self.device)
        if isinstance(space, gymnasium_spaces.Tuple):
            dtype = experience.numpy_to_torch_dtype_dict[space.dtype]
            tuple_len = len(space)
            return torch.zeros(base_shape + (tuple_len,), dtype=dtype, device=self.device)
        if isinstance(space, gymnasium_spaces.Dict):
            t_dict = {}
            for k, v in space.spaces.items():
                t_dict[k] = patched_create_tensor(self, v, base_shape)
            return t_dict
        
        # Fall back to original method for gym spaces
        return original_create_tensor(self, space, base_shape)
    
    experience.ExperienceBuffer.__init__ = patched_init
    experience.ExperienceBuffer._create_tensor_from_space = patched_create_tensor
    
    runner.reset()
    
    # Run training
    logging.info("Starting training...")
    statistics = runner.run({
        'train': True,
        'play': False,
    })
    
    logging.info("Training completed!")
    logging.info(f"Statistics: {statistics}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

