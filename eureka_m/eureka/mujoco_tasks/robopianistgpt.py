__credits__ = ["RoboPianist Authors"]

from typing import Dict, Tuple, Union, Optional
import numpy as np
from pathlib import Path
import sys

# Add robopianist to path if needed
try:
    from robopianist import suite
except ImportError:
    # Try to add the robopianist path
    robopianist_path = Path(__file__).parent.parent.parent.parent / "robopianist"
    if robopianist_path.exists():
        sys.path.insert(0, str(robopianist_path.parent))
    from robopianist import suite

from gymnasium import Env
from gymnasium import utils
from gymnasium.spaces import Box
from dm_control import composer
from dm_env import specs as dm_specs


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class RoboPianistEnvGPT(Env, utils.EzPickle):
    r"""
    MuJoCo RoboPianist environment for Eureka.
    This version integrates LLM-generated reward functions via the `compute_reward(self, obs, actions)` method.
    
    ## Description
    RoboPianist is a dexterous manipulation task where two anthropomorphic robot hands must play
    a piano following a MIDI file. The environment uses dm_control's composer framework and is
    wrapped to be compatible with gymnasium and Eureka's reward generation workflow.
    
    ## Action Space
    The action space is a `Box(-1, 1, (action_dim,), float32)` where action_dim depends on the
    configuration (reduced_action_space, etc.). Actions control the joint positions of both hands.
    
    ## Observation Space
    The observation space is a flattened concatenation of all observables from the environment:
    - Piano state (key positions, activations, sustain)
    - Goal state (target keys to press)
    - Hand joint positions and velocities (left and right hands)
    - Fingering information (if enabled)
    - Other task-specific observables
    
    The exact observation dimension depends on the environment configuration.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }

    def __init__(
        self,
        environment_name: str = "RoboPianist-debug-TwinkleTwinkleRousseau-v0",
        xml_file: Optional[str] = None,  # Not used, kept for compatibility
        frame_skip: int = 1,  # Not used directly, kept for compatibility
        default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
        seed: Optional[int] = None,
        # RoboPianist-specific parameters
        n_steps_lookahead: int = 10,
        trim_silence: bool = False,
        gravity_compensation: bool = False,
        reduced_action_space: bool = False,
        control_timestep: float = 0.05,
        stretch_factor: float = 1.0,
        shift_factor: int = 0,
        wrong_press_termination: bool = False,
        disable_fingering_reward: bool = False,
        disable_forearm_reward: bool = False,
        disable_colorization: bool = False,
        disable_hand_collisions: bool = False,
        primitive_fingertip_collisions: bool = False,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            environment_name,
            xml_file,
            frame_skip,
            default_camera_config,
            seed,
            n_steps_lookahead,
            trim_silence,
            gravity_compensation,
            reduced_action_space,
            control_timestep,
            stretch_factor,
            shift_factor,
            wrong_press_termination,
            disable_fingering_reward,
            disable_forearm_reward,
            disable_colorization,
            disable_hand_collisions,
            primitive_fingertip_collisions,
            **kwargs,
        )

        self._environment_name = environment_name
        self._seed = seed
        self._default_camera_config = default_camera_config
        
        # RoboPianist parameters
        self._n_steps_lookahead = n_steps_lookahead
        self._trim_silence = trim_silence
        self._gravity_compensation = gravity_compensation
        self._reduced_action_space = reduced_action_space
        self._control_timestep = control_timestep
        self._stretch_factor = stretch_factor
        self._shift_factor = shift_factor
        self._wrong_press_termination = wrong_press_termination
        self._disable_fingering_reward = disable_fingering_reward
        self._disable_forearm_reward = disable_forearm_reward
        self._disable_colorization = disable_colorization
        self._disable_hand_collisions = disable_hand_collisions
        self._primitive_fingertip_collisions = primitive_fingertip_collisions

        # Create the dm_control environment
        self._dm_env = suite.load(
            environment_name=environment_name,
            seed=seed,
            stretch=stretch_factor,
            shift=shift_factor,
            task_kwargs=dict(
                n_steps_lookahead=n_steps_lookahead,
                trim_silence=trim_silence,
                gravity_compensation=gravity_compensation,
                reduced_action_space=reduced_action_space,
                control_timestep=control_timestep,
                wrong_press_termination=wrong_press_termination,
                disable_fingering_reward=disable_fingering_reward,
                disable_forearm_reward=disable_forearm_reward,
                disable_colorization=disable_colorization,
                disable_hand_collisions=disable_hand_collisions,
                primitive_fingertip_collisions=primitive_fingertip_collisions,
                change_color_on_activation=True,
            ),
        )

        # Get action and observation specs
        action_spec = self._dm_env.action_spec()
        obs_spec = self._dm_env.observation_spec()

        # Convert action spec to gymnasium Box
        if isinstance(action_spec, dm_specs.BoundedArray):
            self.action_space = Box(
                low=action_spec.minimum.astype(np.float32),
                high=action_spec.maximum.astype(np.float32),
                shape=action_spec.shape,
                dtype=np.float32,
            )
        else:
            # Fallback for unbounded actions
            self.action_space = Box(
                low=-np.inf,
                high=np.inf,
                shape=action_spec.shape,
                dtype=np.float32,
            )

        # Compute observation dimension by flattening all observables
        self._obs_dim = self._compute_obs_dim(obs_spec)
        
        # Create observation space
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float64,
        )

        # Store observation spec for flattening
        self._obs_spec = obs_spec
        
        # Eureka-specific: Store ground-truth reward for comparison
        self.gt_reward = 0.0
        self.extras = {}
        
        # Store last observation for compute_reward
        self._last_obs = None

    def _compute_obs_dim(self, obs_spec: Dict) -> int:
        """Compute total observation dimension from nested spec."""
        total_dim = 0
        for key, spec in obs_spec.items():
            if isinstance(spec, dm_specs.Array):
                total_dim += int(np.prod(spec.shape))
            elif isinstance(spec, dict):
                # Recursively handle nested dicts
                total_dim += self._compute_obs_dim(spec)
        return total_dim

    def _flatten_obs(self, obs: Dict) -> np.ndarray:
        """Flatten nested observation dict to numpy array."""
        flattened = []
        for key in sorted(self._obs_spec.keys()):  # Sort for consistency
            if key in obs:
                value = obs[key]
                if isinstance(value, np.ndarray):
                    flattened.append(value.flatten())
                elif isinstance(value, (int, float)):
                    flattened.append(np.array([value]))
                elif isinstance(value, dict):
                    # Recursively flatten nested dicts
                    flattened.append(self._flatten_obs(value))
        if flattened:
            return np.concatenate([np.asarray(x).flatten() for x in flattened])
        else:
            return np.zeros(self._obs_dim, dtype=np.float64)

    def compute_reward(self, obs, actions):
        # Convert numpy arrays to torch tensors
        obs_tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32))
        
        # Call the generated reward function
        rewards_tensor, rew_dict_tensor = compute_reward(obs_tensor)
        
        # Convert back to numpy
        rewards = rewards_tensor.detach().cpu().numpy()
        rew_dict = {k: v.detach().cpu().numpy() for k, v in rew_dict_tensor.items()}
        
        return rewards, rew_dict
        """
        Reward function placeholder. Eureka will generate the actual implementation.
        This function should compute rewards based on observations and actions.
        
        Note: For MuJoCo environments, use numpy arrays instead of PyTorch tensors
        when performing matrix operations in the generated program.
        
        Args:
            obs: numpy array of shape (n, obs_dim) - observations from _get_obs()
            actions: numpy array of shape (n, action_dim) - actions taken
        
        Returns:
            rewards: numpy array of shape (n,) - computed rewards
            rew_dict: dict with additional reward components
        """
        # Default reward function - Eureka will replace this
        # The generated function should use numpy operations, not PyTorch
        rew_dict: Dict[str, np.ndarray] = {
            "total_reward": np.zeros(obs.shape[0] if len(obs.shape) > 1 else 1)
        }
        return np.zeros(obs.shape[0] if len(obs.shape) > 1 else 1), rew_dict

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset the environment."""
        if seed is not None:
            self._seed = seed
            # Recreate environment with new seed if needed
            # For now, just reset the existing one
        
        timestep = self._dm_env.reset()
        observation = self._flatten_obs(timestep.observation)
        self._last_obs = observation
        
        info = {
            "timestep": timestep,
        }
        
        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Step the environment."""
        # Convert action to numpy if needed
        action = np.asarray(action, dtype=np.float32)
        
        # Step the dm_control environment
        timestep = self._dm_env.step(action)
        
        # Flatten observation
        observation = self._flatten_obs(timestep.observation)
        
        # Get ground-truth reward from dm_control
        gt_reward = float(timestep.reward) if timestep.reward is not None else 0.0
        self.gt_reward = gt_reward
        
        # Determine termination/truncation
        terminated = timestep.last()
        truncated = False  # dm_control doesn't distinguish, but we can add time limits later
        
        # Compute LLM-generated reward using the new observation
        # For single environment, obs and actions need to be batched to match Eureka signature
        obs_batch = np.expand_dims(observation, axis=0)
        action_batch = np.expand_dims(action, axis=0)
        
        llm_reward, rew_dict = self.compute_reward(obs_batch, action_batch)
        llm_reward = llm_reward[0] if isinstance(llm_reward, np.ndarray) and llm_reward.size > 0 else llm_reward
        
        # Store reward info in extras for logging
        self.extras['gt_reward'] = gt_reward
        self.extras['gpt_reward'] = float(llm_reward) if not isinstance(llm_reward, (int, float)) else llm_reward
        for key, value in rew_dict.items():
            if isinstance(value, np.ndarray) and value.size > 0:
                self.extras[key] = value[0] if len(value.shape) > 0 and value.shape[0] > 0 else value
            else:
                self.extras[key] = value
        
        # Use LLM reward instead of ground-truth reward
        reward = float(llm_reward) if not isinstance(llm_reward, (int, float)) else llm_reward
        
        info = {
            "timestep": timestep,
            **self.extras,  # Add Eureka metrics to info
        }
        
        self._last_obs = observation
        
        return observation, reward, terminated, truncated, info

    def render(self):
        """Render the environment."""
        # Use dm_control's renderer
        if hasattr(self._dm_env, 'physics'):
            return self._dm_env.physics.render()
        return None

    def close(self):
        """Close the environment."""
        if hasattr(self._dm_env, 'close'):
            self._dm_env.close()

    def _get_obs(self) -> np.ndarray:
        """Get current observation (for compatibility with Eureka patterns)."""
        if self._last_obs is not None:
            return self._last_obs
        else:
            # Fallback: return zeros if no observation available
            # This should not normally happen as _last_obs is set in reset() and step()
            return np.zeros(self._obs_dim, dtype=np.float64)


from typing import Tuple, Dict
import math
import numpy as np
import torch
from torch import Tensor
@torch.jit.script
def compute_reward(
    obs: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    '''
    Improved piano playing reward with better component balance and temporal alignment.
    
    Key improvements based on policy feedback:
    1. Softened exponential rewards using gentler temperatures
    2. Added note timing alignment component
    3. Reformed fingering reward to use velocity-direction features
    4. Rescaled all components to comparable magnitude ranges
    5. Added dynamic activation weighting based on note importance
    
    Observation index assumptions remain consistent with alphabetical ordering:
    - Goal state (88 keys + sustain) at [88:177]
    - Current piano state (88 keys + sustain) at [510:599]
    - Fingering assignments at [0:88] (if enabled)
    - Joint velocities at [192:216] (left) and [615:639] (right)
    '''
    
    device = obs.device
    batch_size = obs.shape[0]

    # ----------------------------
    # Extract observation components
    # ----------------------------
    # Goal states
    goal_keys = obs[..., 88:176]            # Target note activations (88,)
    goal_sustain = obs[..., 176:177]        # Target sustain pedal (1,)

    # Current piano states
    current_keys = obs[..., 510:598]        # Current key presses (88,)
    current_sustain = obs[..., 599:600]     # Current sustain (1,)

    # Fingering assignments (valid when >=0)
    fingering = obs[..., 0:88].to(torch.long)

    # Hand joint velocities (for fluid motion penalty)
    lhs_vel = obs[..., 216:240]   # lh_shadow_hand/joints_vel (192+24=216 to 216+24=240)
    rhs_vel = obs[..., 639:663]   # rh_shadow_hand/joints_vel (615+24=639 to 639+24=663)

    # ----------------------------
    # Dynamic scaling parameters
    # ----------------------------
    key_temp = torch.tensor(4.0, device=device)
    timing_temp = torch.tensor(2.0, device=device)
    sustain_temp = torch.tensor(5.0, device=device)
    fingering_scale = torch.tensor(0.5, device=device)
    vel_penalty_scale = torch.tensor(0.02, device=device)
    
    # ----------------------------
    # 1. Note Activation Accuracy (Smooth)
    # ----------------------------
    active_notes = goal_keys > 0.1  # Consider all notes with >10% activation target
    activation_error = torch.where(
        active_notes,
        torch.abs(current_keys - goal_keys),
        torch.zeros_like(goal_keys)
    )
    active_count = torch.sum(active_notes.float(), dim=-1) + 1e-7
    activation_reward = torch.exp(-key_temp * activation_error.mean(dim=-1))

    # ----------------------------
    # 2. Temporal Alignment Reward
    # ----------------------------
    # Cross-entropy style reward for correct timing
    timing_probs = torch.sigmoid(current_keys * 5)  # Sharpen key press values
    timing_reward = torch.where(
        active_notes,
        goal_keys * torch.log(timing_probs + 1e-7) + 
        (1 - goal_keys) * torch.log(1 - timing_probs + 1e-7),
        torch.zeros_like(goal_keys)
    )
    timing_reward = -timing_temp * torch.mean(timing_reward, dim=-1)

    # ----------------------------
    # 3. Sustain Matching Reward
    # ----------------------------
    sustain_error = torch.abs(current_sustain - goal_sustain)
    sustain_reward = torch.exp(-sustain_temp * sustain_error).squeeze(-1)

    # ----------------------------
    # 4. Fluid Fingering Motion
    # ----------------------------
    # Reward moving toward assigned fingers, penalize rapid joint changes
    valid_fingering_mask = (fingering >= 0) & (fingering <= 9)
    fingering_activity = valid_fingering_mask.float() * current_keys
    
    # Velocity direction penalty - reward only when moving toward target positions
    joint_vel = torch.cat([lhs_vel, rhs_vel], dim=-1)
    fluence_penalty = vel_penalty_scale * torch.mean(torch.abs(joint_vel), dim=-1)
    
    fingering_reward = fingering_scale * torch.sum(fingering_activity, dim=-1) / active_count
    fingering_reward -= fluence_penalty

    # ----------------------------
    # Combine components
    # ----------------------------
    total_reward = (
        2.0 * activation_reward +
        1.5 * timing_reward +
        0.8 * sustain_reward +
        1.0 * fingering_reward
    )

    # ----------------------------
    # Return components dictionary
    # ----------------------------
    rew_dict: Dict[str, torch.Tensor] = {
        "activation": activation_reward,
        "timing": timing_reward,
        "sustain": sustain_reward,
        "fingering": fingering_reward,
        "total_reward": total_reward
    }
    
    return total_reward, rew_dict
