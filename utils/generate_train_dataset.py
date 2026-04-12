# Generate training dataset in JSON format
# Each record contains: timestamp, history trajectory, camera frame paths, future trajectory

import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation

# Configuration
NUM_HISTORY_STEPS = 16  # 1.6s at 10Hz
NUM_FUTURE_STEPS = 64   # 6.4s at 10Hz
NUM_FRAMES = 4          # 0.4s per camera at 10Hz
TIME_STEP = 0.1         # 100ms = 10Hz
SAMPLE_INTERVAL = 2.0   # 2 seconds between sampling points in seconds
SAMPLE_INTERVAL_US = int(SAMPLE_INTERVAL * 1e6)  # Convert to microseconds

# Camera order (must match the model's expected order)
CAMERAS = [
    'camera_cross_left_120fov',
    'camera_front_wide_120fov',
    'camera_cross_right_120fov',
    'camera_front_tele_30fov',
]

print("Loading trajectory mapping...")
df = pd.read_csv('pai_datasets_frames/trajectory_with_frame_mapping.csv')
print(f"Loaded {len(df)} trajectory frames")

# Calculate valid timestamp range
# Need: 1.5s history + 6.4s future
# For history, we need: target_ts + (-1.5s) >= min_ts, so target_ts >= min_ts + 1.5s
# For future, we need: target_ts + 6.4s <= max_ts, so target_ts <= max_ts - 6.4s
min_ts_us = int((NUM_HISTORY_STEPS - 1) * TIME_STEP * 1e6)  # 1500000 us
max_ts_us = int((NUM_FUTURE_STEPS + 0.5) * TIME_STEP * 1e6)  # 6400000 us

min_allowed_ts = df['timestamp'].min() + min_ts_us  # Need enough history back
max_allowed_ts = df['timestamp'].max() - max_ts_us  # Need enough future forward

print(f"\nDataset timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Valid sampling range: {min_allowed_ts} to {max_allowed_ts}")
print(f"Time range: {(max_allowed_ts - min_allowed_ts) / 1e6:.1f} seconds")

# Get sample timestamps from actual trajectory data
# Filter timestamps that are within valid range and use real data timestamps
real_timestamps = df['timestamp'].values
valid_timestamps = real_timestamps[(real_timestamps >= min_allowed_ts) & (real_timestamps <= max_allowed_ts)]

# Apply interval sampling (every 2 seconds) to reduce dataset size
sample_timestamps = []
last_selected_ts = None
for ts in valid_timestamps:
    if last_selected_ts is None or (ts - last_selected_ts) >= SAMPLE_INTERVAL_US:
        sample_timestamps.append(ts)
        last_selected_ts = ts

print(f"Total valid timestamps in range: {len(valid_timestamps)}")
print(f"Will generate {len(sample_timestamps)} training samples (using real timestamps with ~{SAMPLE_INTERVAL}s interval)")


def find_trajectory_point(df, target_ts, tolerance_us=20000):
    """Find trajectory point closest to target timestamp.
    
    Args:
        df: DataFrame with trajectory data
        target_ts: Target timestamp in microseconds
        tolerance_us: Maximum acceptable deviation from target timestamp
    
    Returns:
        DataFrame row if found within tolerance, None otherwise
    """
    diff = np.abs(df['timestamp'].values - target_ts)
    min_idx = np.argmin(diff)
    min_diff = diff[min_idx]
    
    if min_diff <= tolerance_us:
        actual_ts = df.iloc[min_idx]['timestamp']
        # If requested timestamp is in CSV, return match; otherwise use closest
        return df.iloc[min_idx]
    return None


def find_trajectory_history(df, target_ts, num_steps, time_step, tolerance_us=20000):
    """Find historical trajectory points."""
    history_offsets_us = np.arange(
        -(num_steps - 1) * time_step * 1e6,
        time_step * 1e6 / 2,
        time_step * 1e6,
    ).astype(np.int64)
    history_timestamps = target_ts + history_offsets_us
    
    history = []
    for ts in history_timestamps:
        point = find_trajectory_point(df, ts, tolerance_us)
        if point is None:
            return None
        history.append(point)
    return history


def find_frame_names(df, target_ts, num_frames, time_step, tolerance_us=20000):
    """Find frame names for camera images."""
    image_offsets_us = np.arange(
        -(num_frames - 1) * time_step * 1e6,
        time_step * 1e6 / 2,
        time_step * 1e6,
    ).astype(np.int64)
    image_timestamps = target_ts + image_offsets_us
    
    frame_names = []
    for ts in image_timestamps:
        point = find_trajectory_point(df, ts, tolerance_us)
        if point is None:
            return None
        frame_names.append(point['frame_name'])
    return frame_names


def trajectory_to_local_frame(trajectory_list):
    """Convert trajectory points to local ego frame at t0."""
    positions_world = np.array([[row['x'], row['y'], row['z']] for row in trajectory_list])
    quaternions_world = np.array([[row['qx'], row['qy'], row['qz'], row['qw']] for row in trajectory_list])
    
    # Reference frame at t0 (last history point or arbitrary point for future)
    if len(trajectory_list) > 0:
        t0_xyz = positions_world[-1].copy() if len(trajectory_list) > 1 else positions_world[0].copy()
        t0_quat = quaternions_world[-1].copy() if len(trajectory_list) > 1 else quaternions_world[0].copy()
    else:
        return None
    
    t0_rot = Rotation.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()
    
    # Transform to local frame
    positions_local = t0_rot_inv.apply(positions_world - t0_xyz)
    rotations_local = (t0_rot_inv * Rotation.from_quat(quaternions_world)).as_matrix()
    
    return {
        'positions': positions_local.tolist(),      # (N, 3)
        'rotations': rotations_local.tolist(),      # (N, 3, 3) as 3x3 matrices
        'quaternions': quaternions_world.tolist(),  # (N, 4) world frame for reference
    }


# Generate training dataset
training_data = []
failed_count = 0

for idx, target_ts in enumerate(sample_timestamps):
    if (idx + 1) % 10 == 0:
        print(f"Processing sample {idx + 1}/{len(sample_timestamps)}")
    
    # Find history trajectory (1.6s back)
    history_traj = find_trajectory_history(df, target_ts, NUM_HISTORY_STEPS, TIME_STEP)
    if history_traj is None:
        failed_count += 1
        continue
    
    # Find future trajectory (6.4s forward)
    future_offsets_us = np.arange(
        TIME_STEP * 1e6,
        (NUM_FUTURE_STEPS + 0.5) * TIME_STEP * 1e6,
        TIME_STEP * 1e6,
    ).astype(np.int64)
    future_timestamps = target_ts + future_offsets_us
    
    future_traj = []
    valid = True
    for ts in future_timestamps:
        point = find_trajectory_point(df, ts, tolerance_us=20000)
        if point is None:
            valid = False
            break
        future_traj.append(point)
    
    if not valid or len(future_traj) != NUM_FUTURE_STEPS:
        failed_count += 1
        continue
    
    # Find camera frame names (0.4s back)
    frame_names = find_frame_names(df, target_ts, NUM_FRAMES, TIME_STEP)
    if frame_names is None:
        failed_count += 1
        continue
    
    # Convert trajectories to local frame
    history_local = trajectory_to_local_frame(history_traj)
    future_local = trajectory_to_local_frame(future_traj)
    
    if history_local is None or future_local is None:
        failed_count += 1
        continue
    
    # Get reference pose at t0 from last history point
    t0_row = history_traj[-1]
    
    # Build camera frames dictionary
    camera_frames = {}
    for camera_dir in CAMERAS:
        camera_frames[camera_dir] = [
            f"{camera_dir}/{frame_name}" for frame_name in frame_names
        ]
    
    # Create training sample
    sample = {
        'timestamp': int(target_ts),
        'timestamp_seconds': target_ts / 1e6,
        'reference_pose': {
            'position': [float(t0_row['x']), float(t0_row['y']), float(t0_row['z'])],
            'quaternion': [float(t0_row['qx']), float(t0_row['qy']), float(t0_row['qz']), float(t0_row['qw'])],
            'velocity': [float(t0_row['vx']), float(t0_row['vy']), float(t0_row['vz'])],
            'acceleration': [float(t0_row['ax']), float(t0_row['ay']), float(t0_row['az'])],
        },
        'history_trajectory': {
            'positions': history_local['positions'],      # [num_steps, 3] in local frame
            'rotations': history_local['rotations'],      # [num_steps, 3, 3] 3x3 rotation matrices
            'num_steps': len(history_traj),
        },
        'future_trajectory': {
            'positions': future_local['positions'],       # [num_steps, 3] in local frame
            'rotations': future_local['rotations'],       # [num_steps, 3, 3] 3x3 rotation matrices
            'num_steps': len(future_traj),
        },
        'camera_frames': {
            camera: frame_paths for camera, frame_paths in camera_frames.items()
        },
    }
    
    training_data.append(sample)

print(f"\nGenerated {len(training_data)} training samples")
print(f"Failed: {failed_count} samples")

# Save to JSON file
output_path = 'training_dataset.json'
with open(output_path, 'w') as f:
    json.dump(training_data, f, indent=2)

print(f"\nDataset saved to: {output_path}")
print(f"File size: {Path(output_path).stat().st_size / 1024 / 1024:.2f} MB")

# Print sample structure
if training_data:
    print("\n=== Sample Record Structure ===")
    sample = training_data[0]
    print(f"Timestamp: {sample['timestamp']} us ({sample['timestamp_seconds']:.3f} s)")
    print(f"Reference pose: position={sample['reference_pose']['position']}")
    print(f"History trajectory: {sample['history_trajectory']['num_steps']} steps")
    print(f"  Positions shape: {len(sample['history_trajectory']['positions'])} x 3")
    print(f"  Rotations shape: {len(sample['history_trajectory']['rotations'])} x 3 x 3")
    print(f"Future trajectory: {sample['future_trajectory']['num_steps']} steps")
    print(f"  Positions shape: {len(sample['future_trajectory']['positions'])} x 3")
    print(f"  Rotations shape: {len(sample['future_trajectory']['rotations'])} x 3 x 3")
    print(f"Camera frames: {list(sample['camera_frames'].keys())}")
    print(f"Frames per camera: {len(sample['camera_frames'][list(sample['camera_frames'].keys())[0]])}")
