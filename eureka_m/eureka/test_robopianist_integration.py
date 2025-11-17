#!/usr/bin/env python3
"""
Test script to verify RoboPianist integration with Eureka.
This script tests:
1. Environment loading
2. Observation space
3. Action space
4. Basic step/reset functionality
5. Reward function interface
"""

import sys
from pathlib import Path
import numpy as np

# Add paths
EUREKA_ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(EUREKA_ROOT_DIR))
sys.path.insert(0, str(EUREKA_ROOT_DIR.parent))

def test_environment_loading():
    """Test that the environment can be imported and instantiated."""
    print("Testing environment loading...")
    try:
        from envs.mujoco.robopianist import RoboPianistEnv
        
        # Try to create environment with a simple config
        env = RoboPianistEnv(
            environment_name="RoboPianist-debug-TwinkleTwinkleRousseau-v0",
            seed=42,
        )
        print("✓ Environment loaded successfully")
        return env
    except Exception as e:
        print(f"✗ Failed to load environment: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_observation_space(env):
    """Test observation space properties."""
    print("\nTesting observation space...")
    try:
        obs_space = env.observation_space
        print(f"✓ Observation space: {obs_space}")
        print(f"  Shape: {obs_space.shape}")
        print(f"  Dtype: {obs_space.dtype}")
        return True
    except Exception as e:
        print(f"✗ Failed to get observation space: {e}")
        return False

def test_action_space(env):
    """Test action space properties."""
    print("\nTesting action space...")
    try:
        action_space = env.action_space
        print(f"✓ Action space: {action_space}")
        print(f"  Shape: {action_space.shape}")
        print(f"  Dtype: {action_space.dtype}")
        print(f"  Low: {action_space.low[:5]}... (showing first 5)")
        print(f"  High: {action_space.high[:5]}... (showing first 5)")
        return True
    except Exception as e:
        print(f"✗ Failed to get action space: {e}")
        return False

def test_reset(env):
    """Test environment reset."""
    print("\nTesting reset...")
    try:
        obs, info = env.reset(seed=42)
        print(f"✓ Reset successful")
        print(f"  Observation shape: {obs.shape}")
        print(f"  Observation dtype: {obs.dtype}")
        print(f"  Observation range: [{obs.min():.3f}, {obs.max():.3f}]")
        print(f"  Info keys: {list(info.keys())}")
        return obs, info
    except Exception as e:
        print(f"✗ Failed to reset: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_step(env, obs):
    """Test environment step."""
    print("\nTesting step...")
    try:
        # Sample a random action
        action = env.action_space.sample()
        obs_new, reward, terminated, truncated, info = env.step(action)
        
        print(f"✓ Step successful")
        print(f"  New observation shape: {obs_new.shape}")
        print(f"  Reward: {reward:.4f}")
        print(f"  Terminated: {terminated}")
        print(f"  Truncated: {truncated}")
        print(f"  Info keys: {list(info.keys())}")
        if 'gt_reward' in info:
            print(f"  Ground-truth reward: {info['gt_reward']:.4f}")
        if 'gpt_reward' in info:
            print(f"  LLM reward: {info['gpt_reward']:.4f}")
        return True
    except Exception as e:
        print(f"✗ Failed to step: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_compute_reward(env):
    """Test compute_reward interface."""
    print("\nTesting compute_reward interface...")
    try:
        # Get an observation
        obs, _ = env.reset(seed=42)
        action = env.action_space.sample()
        
        # Test with batched inputs (as Eureka expects)
        obs_batch = np.expand_dims(obs, axis=0)
        action_batch = np.expand_dims(action, axis=0)
        
        reward, rew_dict = env.compute_reward(obs_batch, action_batch)
        
        print(f"✓ compute_reward successful")
        print(f"  Reward shape: {reward.shape}")
        print(f"  Reward value: {reward[0]:.4f}")
        print(f"  Reward dict keys: {list(rew_dict.keys())}")
        return True
    except Exception as e:
        print(f"✗ Failed compute_reward: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_multiple_steps(env):
    """Test multiple steps in an episode."""
    print("\nTesting multiple steps...")
    try:
        obs, _ = env.reset(seed=42)
        total_reward = 0
        steps = 0
        max_steps = 10
        
        for i in range(max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            if terminated or truncated:
                break
        
        print(f"✓ Completed {steps} steps")
        print(f"  Total reward: {total_reward:.4f}")
        print(f"  Episode ended: terminated={terminated}, truncated={truncated}")
        return True
    except Exception as e:
        print(f"✗ Failed multiple steps: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("RoboPianist Eureka Integration Test")
    print("=" * 60)
    
    # Test environment loading
    env = test_environment_loading()
    if env is None:
        print("\n✗ Cannot continue tests - environment failed to load")
        return False
    
    # Run tests
    results = []
    results.append(test_observation_space(env))
    results.append(test_action_space(env))
    obs, info = test_reset(env)
    if obs is not None:
        results.append(True)
        results.append(test_step(env, obs))
        results.append(test_compute_reward(env))
        results.append(test_multiple_steps(env))
    else:
        results.append(False)
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("✓ All tests passed!")
        return True
    else:
        print("✗ Some tests failed")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

