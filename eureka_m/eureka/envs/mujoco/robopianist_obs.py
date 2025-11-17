import numpy as np

class RoboPianistEnv:
    """Rest of the environment definition omitted."""
    
    """
    ### Observation Space
    
    Observations consist of a flattened concatenation of all observables from the RoboPianist environment.
    The observation space is a `Box(-Inf, Inf, (obs_dim,), float64)` where obs_dim depends on the
    environment configuration (e.g., whether fingering is enabled, reduced action space, etc.).
    
    The observation components are concatenated in sorted key order and include:
    
    #### Piano Observables:
    - **piano/state**: Normalized state of all piano keys (88 keys by default)
      - Shape: (88,) or (n_keys,)
      - Values: 0.0 (not pressed) to 1.0 (fully pressed)
    
    - **piano/sustain_state**: Sustain pedal state
      - Shape: (1,)
      - Values: 0.0 (not pressed) to 1.0 (fully pressed)
    
    - **piano/joints_pos**: Joint positions of piano keys
      - Shape: (88,) or (n_keys,)
      - Values: Joint angles in radians
    
    - **piano/activation**: Activation state of piano keys
      - Shape: (88,) or (n_keys,)
      - Values: Binary or continuous activation values
    
    - **piano/sustain_activation**: Sustain pedal activation
      - Shape: (1,)
      - Values: Binary or continuous activation
    
    #### Goal Observables:
    - **goal**: Target state indicating which keys should be pressed
      - Shape: (88 + 1,) or (n_keys + 1,) - includes sustain pedal goal
      - Values: 0.0 (should not press) or 1.0 (should press)
      - The last element is the sustain pedal goal
    
    #### Hand Observables (Left and Right):
    For each hand (rh_shadow_hand and lh_shadow_hand):
    
    - **{hand}/joints_pos**: Joint positions of the hand
      - Shape: (24,) for Shadow Hand (24 DOF)
      - Values: Joint angles in radians
    
    - **{hand}/joints_pos_cos_sin**: Joint positions encoded as (cos, sin) pairs
      - Shape: (48,) for Shadow Hand (24 joints * 2)
      - Values: Cosine and sine of joint angles
    
    - **{hand}/joints_vel**: Joint velocities
      - Shape: (24,) for Shadow Hand
      - Values: Joint angular velocities in rad/s
    
    - **{hand}/joints_torque**: Joint torques
      - Shape: (24,) for Shadow Hand
      - Values: Torques acting on each joint axis
    
    - **{hand}/position**: Position of the hand's root body in world frame
      - Shape: (3,)
      - Values: x, y, z coordinates in meters
    
    - **{hand}/orientation**: Orientation of the hand's root body
      - Shape: (4,) if quaternion, (3,) if euler
      - Values: Quaternion (w, x, y, z) or Euler angles
    
    - **{hand}/fingertip_positions**: Positions of all fingertips
      - Shape: (15,) for Shadow Hand (5 fingers * 3 coordinates)
      - Values: x, y, z coordinates of each fingertip
    
    - **{hand}/fingertip_forces**: Forces at fingertips
      - Shape: (15,) for Shadow Hand
      - Values: Force magnitudes or 3D force vectors
    
    #### Fingering Observables (if enabled):
    - **fingering**: Fingering assignment for each key that should be pressed
      - Shape: Variable, depends on number of active keys
      - Values: Finger indices (0-4 for each hand) or -1 for no assignment
      - Only present if `disable_fingering_reward=False` and MIDI file has fingering info
    
    #### Additional Observables:
    - **action_reward_observation**: If enabled, includes previous action and reward
      - Shape: (action_dim + 1,)
      - Values: Previous action values and reward
    
    ### Observation Flattening
    
    The observations are flattened by:
    1. Sorting all observation keys alphabetically
    2. Concatenating all values in order
    3. Flattening multi-dimensional arrays
    
    The exact observation dimension depends on:
    - Number of piano keys (typically 88)
    - Whether fingering is enabled
    - Whether action_reward_observation is enabled
    - Hand configuration (Shadow Hand has 24 DOF per hand)
    
    Typical observation dimensions:
    - With fingering: ~400-500 dimensions
    - Without fingering: ~350-450 dimensions
    - With action_reward_observation: +action_dim+1 dimensions
    
    Note that the obs argument passed to the reward function is a concatenation of obs across
    all environments, so it will have shape (n, obs_dim) where n is the number of parallel environments.
    """
    
    def _get_obs(self):
        """
        Get current observation by flattening the dm_control observation dict.
        
        The observation is obtained from the underlying dm_control environment's
        timestep.observation dictionary and flattened into a single numpy array.
        
        Returns:
            numpy array of shape (obs_dim,) containing all flattened observables
        """
        # Get current timestep from dm_control environment
        if hasattr(self, '_dm_env') and hasattr(self._dm_env, '_last_time_step'):
            timestep = self._dm_env._last_time_step
            if timestep is not None:
                return self._flatten_obs(timestep.observation)
        
        # Fallback: return zeros if no observation available
        return np.zeros(self._obs_dim, dtype=np.float64)

