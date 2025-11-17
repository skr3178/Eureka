#!/usr/bin/env python3
"""Test PianoSoundVideoWrapper which might be causing the crash."""

import os
from pathlib import Path

# Set MUJOCO_GL like in run.sh
os.environ['MUJOCO_GL'] = 'egl'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MUJOCO_EGL_DEVICE_ID'] = '0'

print("Testing PianoSoundVideoWrapper...")
print("=" * 60)

try:
    from robopianist import suite
    import robopianist.wrappers as robopianist_wrappers
    
    # Create environment
    print("1. Creating base environment...")
    env = suite.load(
        environment_name="RoboPianist-debug-TwinkleTwinkleRousseau-v0",
        seed=42,
    )
    print("   ✓ Base environment created")
    
    # Try with video wrapper (like in train.py)
    print("\n2. Adding PianoSoundVideoWrapper...")
    record_dir = Path("/tmp/test_robopianist_video")
    record_dir.mkdir(parents=True, exist_ok=True)
    
    env = robopianist_wrappers.PianoSoundVideoWrapper(
        environment=env,
        record_dir=record_dir,
        record_every=1,
        camera_id="piano/back",
        height=480,
        width=640,
    )
    print("   ✓ Video wrapper added")
    
    # Try to reset
    print("\n3. Testing environment reset...")
    timestep = env.reset()
    print("   ✓ Reset successful")
    
    # Try to step
    print("\n4. Testing environment step...")
    action = env.action_spec().generate_value()
    timestep = env.step(action)
    print("   ✓ Step successful")
    
    print("\n" + "=" * 60)
    print("✓✓✓ ALL TESTS PASSED - Video wrapper works!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n✗✗✗ FAILED: {e}")
    import traceback
    traceback.print_exc()

