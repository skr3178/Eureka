#!/usr/bin/env python3
"""
Evaluate a trained checkpoint on MuJoCo environment with GUI rendering.
"""

import os
import sys
import argparse
import logging
import importlib.util
from pathlib import Path
import numpy as np
import torch

# Calculate EUREKA_ROOT_DIR
EUREKA_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if EUREKA_ROOT_DIR not in sys.path:
    sys.path.insert(0, EUREKA_ROOT_DIR)

from omegaconf import OmegaConf
from rl_games.torch_runner import Runner
from rl_games.common import env_configurations, vecenv
from rl_games.common.ivecenv import IVecEnv
from gymnasium.vector import SyncVectorEnv
from dm_control import viewer


class MujocoVecEnv(IVecEnv):
    """Wrapper for gymnasium SyncVectorEnv to work with rl_games."""
    
    def __init__(self, env):
        self.env = env
        self.num_actors = env.num_envs if hasattr(env, 'num_envs') else 1
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        
    def step(self, actions):
        obs, rewards, dones, truncs, infos = self.env.step(actions)
        dones = dones | truncs
        return obs, rewards, dones, infos
    
    def reset(self):
        obs, infos = self.env.reset()
        return obs
    
    def get_env_info(self):
        from gymnasium.spaces import Box
        import numpy as np
        
        # Try to get single observation space from SyncVectorEnv
        if hasattr(self.env, 'single_observation_space'):
            obs_space = self.env.single_observation_space
        else:
            obs_space = self.observation_space
            # If observation space has batched shape like (1, 1130), extract the single shape
            obs_shape = obs_space.shape if hasattr(obs_space, 'shape') else None
            if obs_shape and len(obs_shape) > 1 and obs_shape[0] == 1:
                # Create a new Box space with the unbatched shape
                # Handle low/high bounds - they might be arrays or scalars
                low = obs_space.low
                high = obs_space.high
                if isinstance(low, np.ndarray):
                    if low.ndim > 1:
                        low = low[0]
                    elif low.size == obs_shape[0]:
                        low = low[0] if low.size > 1 else low
                if isinstance(high, np.ndarray):
                    if high.ndim > 1:
                        high = high[0]
                    elif high.size == obs_shape[0]:
                        high = high[0] if high.size > 1 else high
                obs_space = Box(
                    low=low,
                    high=high,
                    shape=obs_shape[1:],
                    dtype=obs_space.dtype
                )
        
        # Try to get single action space from SyncVectorEnv
        if hasattr(self.env, 'single_action_space'):
            action_space = self.env.single_action_space
        else:
            action_space = self.action_space
            # If action space has batched shape, extract the single shape
            action_shape = action_space.shape if hasattr(action_space, 'shape') else None
            if action_shape and len(action_shape) > 1 and action_shape[0] == 1:
                # Handle low/high bounds - they might be arrays or scalars
                low = action_space.low
                high = action_space.high
                if isinstance(low, np.ndarray):
                    if low.ndim > 1:
                        low = low[0]
                    elif low.size == action_shape[0]:
                        low = low[0] if low.size > 1 else low
                if isinstance(high, np.ndarray):
                    if high.ndim > 1:
                        high = high[0]
                    elif high.size == action_shape[0]:
                        high = high[0] if high.size > 1 else high
                action_space = Box(
                    low=low,
                    high=high,
                    shape=action_shape[1:],
                    dtype=action_space.dtype
                )
        
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


def load_generated_env(env_file_path):
    """Load the generated environment module from file."""
    spec = importlib.util.spec_from_file_location("generated_env", env_file_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Could not load environment from {env_file_path}")
    
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    # Find the environment class
    env_class = None
    env_candidates = []
    
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type):
            if attr_name == 'Env' and hasattr(attr, '__module__') and 'gymnasium' in str(attr.__module__):
                continue
            if (attr_name.endswith('Env') or 'Env' in attr_name) and attr_name != 'Env':
                env_candidates.append((attr_name, attr))
    
    env_candidates.sort(key=lambda x: (not x[0].endswith('GPT'), -len(x[0])))
    
    if env_candidates:
        env_class = env_candidates[0][1]
        logging.info(f"Selected environment class: {env_candidates[0][0]}")
    else:
        classes = [name for name in dir(module) 
                  if isinstance(getattr(module, name, None), type) 
                  and not name.startswith('_')]
        raise ValueError(f"Environment class not found in {env_file_path}. "
                        f"Available classes: {classes}")
    
    return env_class


def create_mujoco_env(env_class, num_envs=1, **kwargs):
    """Create vectorized MuJoCo environment."""
    def make_env():
        env = env_class(**kwargs)
        return env
    
    if num_envs == 1:
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
    
    def create_mujoco_vecenv(config_name, num_actors, **kwargs):
        return env_creator(num_actors=num_actors, **kwargs)
    
    vecenv.register('MUJOCO', create_mujoco_vecenv)


def policy_step(time_step, env, agent, device=None):
    """Policy function for dm_control viewer."""
    obs = env._flatten_obs(time_step.observation) if hasattr(env, '_flatten_obs') else time_step.observation
    if isinstance(obs, dict):
        # Convert dict to array if needed
        obs = np.concatenate([np.asarray(v).flatten() for v in obs.values()])
    
    # Get action from agent
    obs_tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
    # Move to same device as model
    if device is None:
        if hasattr(agent, 'device'):
            device = agent.device
        elif hasattr(agent.model, 'device'):
            device = agent.model.device
    if device is not None:
        obs_tensor = obs_tensor.to(device)
    with torch.no_grad():
        action = agent.get_action(obs_tensor, is_deterministic=True)
    if isinstance(action, torch.Tensor):
        action = action.cpu().numpy()
    if action.ndim > 1:
        action = action[0]
    
    return action


def main():
    parser = argparse.ArgumentParser(description='Evaluate checkpoint with GUI rendering')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (.pth)')
    parser.add_argument('--env_file', type=str, required=True,
                        help='Path to generated environment file')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to rl_games config YAML file')
    parser.add_argument('--num_episodes', type=int, default=1,
                        help='Number of episodes to run')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use_viewer', action='store_true',
                        help='Use dm_control viewer for GUI (interactive)')
    parser.add_argument('--headless', action='store_true',
                        help='Run without GUI (for testing)')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load generated environment
    logging.info(f"Loading environment from {args.env_file}")
    EnvClass = load_generated_env(args.env_file)
    
    # Register environment
    env_class_name = EnvClass.__name__
    if 'RoboPianist' in env_class_name or 'robopianist' in env_class_name.lower():
        env_name = 'mujoco_robopianist'
    elif 'Ant' in env_class_name or 'ant' in env_class_name.lower():
        env_name = 'mujoco_ant'
    elif 'Humanoid' in env_class_name or 'humanoid' in env_class_name.lower():
        env_name = 'mujoco_humanoid'
    else:
        env_name = f'mujoco_{env_class_name.lower().replace("env", "")}'
    
    logging.info(f"Detected environment name: {env_name} from class {env_class_name}")
    
    # Create a single environment instance for evaluation
    env_instance = EnvClass()
    register_mujoco_env(env_name, EnvClass)
    
    # Load rl_games config
    logging.info(f"Loading config from {args.config}")
    config_dict = OmegaConf.load(args.config)
    config_dict = OmegaConf.to_container(config_dict, resolve=True)
    
    if 'params' not in config_dict:
        config_dict['params'] = {}
    if 'config' not in config_dict['params']:
        config_dict['params']['config'] = {}
    
    config_dict['params']['config']['env_name'] = env_name
    config_dict['params']['config']['num_actors'] = 1
    config_dict['params']['seed'] = args.seed
    
    # Patch env_configurations.get_env_info to handle batched observation spaces
    from rl_games.common import env_configurations
    original_get_env_info = env_configurations.get_env_info
    
    def patched_get_env_info(env):
        """Patched version that handles batched observation spaces."""
        result = original_get_env_info(env)
        
        # If observation space has batched shape, fix it
        obs_space = result.get('observation_space')
        if obs_space is not None and hasattr(obs_space, 'shape'):
            obs_shape = obs_space.shape
            if len(obs_shape) > 1 and obs_shape[0] == 1:
                from gymnasium.spaces import Box
                import numpy as np
                low = obs_space.low
                high = obs_space.high
                if isinstance(low, np.ndarray) and low.ndim > 1:
                    low = low[0]
                if isinstance(high, np.ndarray) and high.ndim > 1:
                    high = high[0]
                result['observation_space'] = Box(
                    low=low,
                    high=high,
                    shape=obs_shape[1:],
                    dtype=obs_space.dtype
                )
        
        # If action space has batched shape, fix it
        action_space = result.get('action_space')
        if action_space is not None and hasattr(action_space, 'shape'):
            action_shape = action_space.shape
            if len(action_shape) > 1 and action_shape[0] == 1:
                from gymnasium.spaces import Box
                import numpy as np
                low = action_space.low
                high = action_space.high
                if isinstance(low, np.ndarray) and low.ndim > 1:
                    low = low[0]
                if isinstance(high, np.ndarray) and high.ndim > 1:
                    high = high[0]
                result['action_space'] = Box(
                    low=low,
                    high=high,
                    shape=action_shape[1:],
                    dtype=action_space.dtype
                )
        
        return result
    
    # Apply the patch
    env_configurations.get_env_info = patched_get_env_info
    
    # Create runner and load checkpoint
    logging.info(f"Loading checkpoint from {args.checkpoint}")
    runner = Runner()
    runner.load(config_dict)
    
    # Create player (agent) from checkpoint
    player = runner.create_player()
    player.restore(args.checkpoint)
    logging.info("Checkpoint loaded successfully!")
    
    # Get the underlying dm_control environment if available
    dm_env = None
    if hasattr(env_instance, '_dm_env'):
        dm_env = env_instance._dm_env
    elif hasattr(env_instance, 'env') and hasattr(env_instance.env, '_dm_env'):
        dm_env = env_instance.env._dm_env
    
    if args.use_viewer and dm_env is not None:
        # Use dm_control viewer for interactive GUI
        logging.info("Launching dm_control viewer...")
        logging.info("Controls:")
        logging.info("  - Close window to exit")
        logging.info("  - The policy will run automatically")
        
        def policy_fn(time_step):
            # Get device from player
            device = None
            if hasattr(player, 'device'):
                device = player.device
            elif hasattr(player.model, 'device'):
                device = player.model.device
            return policy_step(time_step, env_instance, player, device)
        
        try:
            viewer.launch(dm_env, policy=policy_fn)
        except Exception as e:
            logging.error(f"Failed to launch viewer: {e}")
            logging.info("Falling back to manual evaluation loop...")
            args.use_viewer = False
    else:
        # Manual evaluation loop
        logging.info(f"Running {args.num_episodes} evaluation episode(s)...")
        
        total_rewards = []
        for episode in range(args.num_episodes):
            obs, info = env_instance.reset()
            episode_reward = 0.0
            episode_steps = 0
            done = False
            
            while not done:
                # Get action from agent
                obs_tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
                # Move to same device as model
                if hasattr(player, 'device'):
                    obs_tensor = obs_tensor.to(player.device)
                elif hasattr(player.model, 'device'):
                    obs_tensor = obs_tensor.to(player.model.device)
                with torch.no_grad():
                    action = player.get_action(obs_tensor, is_deterministic=True)
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                if action.ndim > 1:
                    action = action[0]
                
                # Step environment
                obs, reward, terminated, truncated, info = env_instance.step(action)
                done = terminated or truncated
                episode_reward += reward
                episode_steps += 1
                
                # Render if not headless
                if not args.headless and hasattr(env_instance, 'render'):
                    try:
                        frame = env_instance.render()
                        # If using MuJoCo directly, you might want to use mujoco.viewer here
                    except Exception as e:
                        # Rendering might not be available in all modes
                        pass
                
                if episode_steps > 10000:  # Safety limit
                    logging.warning(f"Episode {episode} exceeded step limit, terminating")
                    break
            
            total_rewards.append(episode_reward)
            logging.info(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {episode_steps}")
        
        logging.info(f"Evaluation complete!")
        logging.info(f"Mean reward: {np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
        logging.info(f"Rewards: {total_rewards}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()

