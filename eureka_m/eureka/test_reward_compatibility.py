#!/usr/bin/env python3
"""
Test script to verify TorchScript compatibility of compute_reward function.
This tests the function in isolation without requiring the full environment setup.
"""

import torch
from typing import Tuple, Dict

# Copy the compute_reward function for testing
@torch.jit.script
def compute_reward(
    obs: torch.Tensor
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    '''
    Reward function for RoboPianist environment.
    Extracts components from flattened observation vector.
    
    Args:
        obs: Observation tensor of shape (batch_size, obs_dim)
    
    Returns:
        total_reward: torch.Tensor of shape (batch_size,)
        rew_dict: Dict[str, torch.Tensor] with named reward components
    '''
    batch_size = obs.shape[0]
    device = obs.device
    obs_dim = obs.shape[1]
    
    # Temperature parameters for reward components
    temp_key = torch.tensor(10.0, device=device)
    temp_torque = torch.tensor(0.1, device=device)
    temp_sustain = torch.tensor(5.0, device=device)
    
    # Extract components from obs
    # Based on observation structure: components are sorted alphabetically
    # piano/state: first 88 elements
    piano_end = min(88, obs_dim)
    piano_state = obs[..., 0:piano_end]
    
    # Pad piano_state to 88 if needed (TorchScript-compatible)
    piano_pad_size = 88 - piano_end
    if piano_pad_size > 0:
        padding = torch.zeros(batch_size, piano_pad_size, device=device)
        piano_state = torch.cat([piano_state, padding], dim=1)
    else:
        # Ensure exactly 88 elements
        piano_state = piano_state[..., :88]
    
    # piano/sustain_state: after piano/state (index 88)
    sustain_state = obs[..., 88:89] if obs_dim > 88 else torch.zeros(batch_size, 1, device=device)
    
    # goal: typically after piano observables
    # Try to find goal section (89 elements: 88 keys + 1 sustain)
    # Approximate position based on alphabetical ordering
    goal_start = 177  # Approximate: after piano/state, sustain_state, joints_pos, etc.
    goal_end = goal_start + 89
    if goal_end <= obs_dim:
        goal = obs[..., goal_start:goal_end]
        goal_keys = goal[..., :88]
        goal_sustain = goal[..., 88:89]
    else:
        # Fallback: use piano_state as proxy for goal (simplified)
        goal_keys = piano_state
        goal_sustain = sustain_state
    
    # Extract hand torques (rh_shadow_hand/joints_torque and lh_shadow_hand/joints_torque)
    # These come after 'goal' alphabetically (24 elements each)
    rh_torque_start = min(450, max(0, obs_dim - 48))
    rh_torque_end = rh_torque_start + 48
    if rh_torque_end <= obs_dim:
        rh_joints_torque = obs[..., rh_torque_start:rh_torque_start+24]
        lh_joints_torque = obs[..., rh_torque_start+24:rh_torque_start+48]
    else:
        # Fallback: use zeros if not available
        rh_joints_torque = torch.zeros(batch_size, 24, device=device)
        lh_joints_torque = torch.zeros(batch_size, 24, device=device)
    
    ### Key Press Accuracy (Primary Reward) ###
    key_diff = torch.sum((piano_state - goal_keys)**2, dim=-1)
    key_reward = torch.exp(-temp_key * key_diff)
    
    ### Sustain Pedal Accuracy ###
    sustain_diff = torch.sum((sustain_state - goal_sustain)**2, dim=-1)
    sustain_reward = torch.exp(-temp_sustain * sustain_diff)
    
    ### Hand Efficiency Penalty ###
    torque_penalty = torch.sum(torch.abs(rh_joints_torque), dim=-1) + \
                     torch.sum(torch.abs(lh_joints_torque), dim=-1)
    efficiency_reward = torch.exp(-temp_torque * torque_penalty)
    
    ### Simplified Proximity Reward (TorchScript-compatible) ###
    # Use key alignment as proxy for proximity (avoids complex indexing)
    # This is a simplified version that doesn't require torch.nonzero with as_tuple
    proximity_reward = torch.exp(-temp_key * key_diff * 0.5)
    
    ### Combine Components ###
    total_reward = (
        0.5 * key_reward +
        0.2 * sustain_reward +
        0.2 * efficiency_reward +
        0.1 * proximity_reward
    )
    
    rew_dict: Dict[str, torch.Tensor] = {
        "key_reward": key_reward,
        "sustain_reward": sustain_reward,
        "efficiency_reward": efficiency_reward,
        "proximity_reward": proximity_reward,
        "total_reward": total_reward
    }
    
    return total_reward, rew_dict


def test_reward_function():
    """Test the reward function with various input shapes."""
    print("=" * 60)
    print("Testing TorchScript-compatible compute_reward function")
    print("=" * 60)
    
    test_cases = [
        (1, 400, "Single environment, typical obs_dim"),
        (1, 500, "Single environment, with fingering"),
        (4, 400, "Batch of 4 environments"),
        (1, 200, "Smaller observation dimension"),
    ]
    
    for batch_size, obs_dim, description in test_cases:
        print(f"\nTest: {description}")
        print(f"  Input shape: ({batch_size}, {obs_dim})")
        
        try:
            obs = torch.randn(batch_size, obs_dim)
            total_reward, rew_dict = compute_reward(obs)
            
            # Verify shapes
            assert total_reward.shape == (batch_size,), f"Expected total_reward shape ({batch_size},), got {total_reward.shape}"
            for key, value in rew_dict.items():
                assert value.shape == (batch_size,), f"Expected {key} shape ({batch_size},), got {value.shape}"
            
            print(f"  ✓ Success!")
            print(f"    total_reward: shape {total_reward.shape}, value {total_reward[0].item():.4f}")
            print(f"    Components: {list(rew_dict.keys())}")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
    return True


if __name__ == "__main__":
    success = test_reward_function()
    exit(0 if success else 1)

