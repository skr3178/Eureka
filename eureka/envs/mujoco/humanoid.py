import numpy as np
from typing import Tuple, Dict

class HumanoidEnv:
    """
    MuJoCo Humanoid environment for Eureka.
    Rest of the environment definition omitted.
    """
    
    def compute_reward(self, obs, actions):
        """
        Reward function placeholder. Eureka will generate the actual implementation.
        This function should compute rewards based on observations and actions.
        
        Note: For MuJoCo environments, use numpy arrays instead of PyTorch tensors
        when performing matrix operations in the generated program.
        
        Args:
            obs: numpy array of shape (n, 376) - observations from _get_obs()
            actions: numpy array of shape (n, 17) - actions taken
        
        Returns:
            rewards: numpy array of shape (n,) - computed rewards
            rew_dict: dict with additional reward components
        """
        # Default reward function - Eureka will replace this
        # The generated function should use numpy operations, not PyTorch
        rew_dict: Dict[str, np.ndarray] = {
            "total_reward": np.zeros(obs.shape[0])
        }
        return np.zeros(obs.shape[0]), rew_dict

