# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# End-to-end example script for the inference pipeline:
# This script loads a dataset, runs inference, and computes the minADE.
# It can be used to test the inference pipeline.

import torch
import numpy as np
import pandas as pd
import os
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation
from einops import rearrange

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1 import helper


def load_trajectory_mapping(csv_path):
    """Load trajectory mapping from CSV file.
    
    Args:
        csv_path: Path to trajectory_with_frame_mapping.csv
    
    Returns:
        pd.DataFrame: DataFrame with columns [timestamp, trajectory data, frame_name, etc.]
    """
    df = pd.read_csv(csv_path)
    return df


def find_trajectory_history(df, target_timestamp, num_history_steps=16, time_step=0.1, tolerance_us=5000):
    """Find historical trajectory points around target timestamp.
    
    Args:
        df: Trajectory mapping DataFrame
        target_timestamp: Target timestamp in microseconds (t0)
        num_history_steps: Number of history points to retrieve (default 16 for 1.6s)
        time_step: Time step in seconds (default 0.1s = 10Hz)
        tolerance_us: Tolerance for timestamp matching in microseconds
    
    Returns:
        list or None: List of trajectory rows in chronological order, or None if not enough data
    """
    # Calculate history timestamps: [t0-1.5s, t0-1.4s, ..., t0-0.1s, t0]
    history_offsets_us = np.arange(
        -(num_history_steps - 1) * time_step * 1_000_000,
        time_step * 1_000_000 / 2,
        time_step * 1_000_000,
    ).astype(np.int64)
    history_timestamps = target_timestamp + history_offsets_us
    
    trajectory_history = []
    for ts in history_timestamps:
        # Find closest timestamp within tolerance
        diff = np.abs(df['timestamp'].values - ts)
        min_idx = np.argmin(diff)
        
        if diff[min_idx] <= tolerance_us:
            trajectory_history.append(df.iloc[min_idx])
        else:
            print(f"Warning: No trajectory found for timestamp {ts}us (diff: {diff[min_idx]}us)")
            return None
    
    return trajectory_history


def load_frame_names_from_history(df, target_timestamp, num_frames=4, time_step=0.1, tolerance_us=5000):
    """Find frame names for historical camera frames.
    
    Args:
        df: Trajectory mapping DataFrame
        target_timestamp: Target timestamp in microseconds (t0)
        num_frames: Number of frames to retrieve (default 4 for 0.4s)
        time_step: Time step in seconds (default 0.1s = 10Hz)
        tolerance_us: Tolerance for timestamp matching in microseconds
    
    Returns:
        list or None: List of frame names in chronological order, or None if not enough data
    """
    # Calculate image timestamps: [t0-0.3s, t0-0.2s, t0-0.1s, t0]
    image_offsets_us = np.arange(
        -(num_frames - 1) * time_step * 1_000_000,
        time_step * 1_000_000 / 2,
        time_step * 1_000_000,
    ).astype(np.int64)
    image_timestamps = target_timestamp + image_offsets_us
    
    frame_names = []
    for ts in image_timestamps:
        diff = np.abs(df['timestamp'].values - ts)
        min_idx = np.argmin(diff)
        
        if diff[min_idx] <= tolerance_us:
            frame_names.append(df.iloc[min_idx]['frame_name'])
        else:
            print(f"Warning: No frame found for timestamp {ts}us (diff: {diff[min_idx]}us)")
            return None
    
    return frame_names


def load_camera_images_history(pai_datasets_frames_path, frame_names):
    """Load 4 camera images for a sequence of frame names.
    
    Matches the processing from load_physical_aiavdataset:
    - Load images as uint8 (H, W, 3)
    - Convert to (C, H, W) using rearrange
    - Stack cameras to get (num_cameras, num_frames, 3, H, W)
    
    Args:
        pai_datasets_frames_path: Path to pai_datasets_frames directory
        frame_names: List of frame names in chronological order
    
    Returns:
        torch.Tensor: Image tensor of shape (num_cameras, num_frames, 3, H, W)
    """
    cameras = [
        'camera_cross_left_120fov',
        'camera_front_wide_120fov',
        'camera_cross_right_120fov',
        'camera_front_tele_30fov',
    ]
    
    image_frames_list = []
    
    for camera_dir in cameras:
        frames_for_camera = []
        for frame_name in frame_names:
            image_path = os.path.join(pai_datasets_frames_path, camera_dir, frame_name)
            if os.path.exists(image_path):
                # Load image as uint8 (H, W, 3)
                img = Image.open(image_path).convert('RGB')
                img_array = np.array(img, dtype=np.uint8)  # Keep as uint8, don't normalize
                frames_for_camera.append(img_array)
            else:
                raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Stack frames for this camera: (num_frames, H, W, 3)
        camera_frames = np.stack(frames_for_camera, axis=0)
        
        # Convert to torch and rearrange: (num_frames, 3, H, W)
        frames_tensor = torch.from_numpy(camera_frames)
        frames_tensor = rearrange(frames_tensor, "t h w c -> t c h w")
        
        image_frames_list.append(frames_tensor)
        print(f"Loaded {len(frames_for_camera)} frames from {camera_dir}: shape {frames_tensor.shape}")
    
    # Stack all cameras: (num_cameras, num_frames, 3, H, W)
    image_frames = torch.stack(image_frames_list, dim=0)
    
    return image_frames


def extract_trajectory_history_arrays(trajectory_history):
    """Extract and transform trajectory arrays from history rows to local frame.
    
    Following the computation in load_physical_aiavdataset:
    1. Extract positions and quaternions in world frame
    2. Get t0 pose (last history point)
    3. Transform to local frame (ego frame at t0):
       - xyz_local = R_t0^{-1} @ (xyz_world - xyz_t0)
       - rot_local = R_t0^{-1} * R_world
    
    Args:
        trajectory_history: List of pd.Series from trajectory mapping
    
    Returns:
        tuple: (ego_history_xyz, ego_history_rot) both in local frame coordinate
               ego_history_xyz: shape (1, 1, num_history, 3)
               ego_history_rot: shape (1, 1, num_history, 3, 3)
    """
    num_history = len(trajectory_history)
    
    # Extract positions and quaternions (world frame)
    positions_world = np.array([[row['x'], row['y'], row['z']] for row in trajectory_history])
    quaternions_world = np.array([[row['qx'], row['qy'], row['qz'], row['qw']] for row in trajectory_history])
    
    # Get t0 pose (last history point)
    t0_xyz = positions_world[-1].copy()  # Position at t0
    t0_quat = quaternions_world[-1].copy()  # Orientation at t0 (quaternion)
    t0_rot = Rotation.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()  # Inverse rotation
    
    # Transform positions to local frame (ego frame at t0)
    # xyz_local = R_t0^{-1} @ (xyz_world - xyz_t0)
    ego_history_xyz_local = t0_rot_inv.apply(positions_world - t0_xyz)
    
    # Transform rotations to local frame
    # rot_local = R_t0^{-1} * R_world (composition of rotations)
    ego_history_rot_local = (t0_rot_inv * Rotation.from_quat(quaternions_world)).as_matrix()
    
    # Convert to torch tensors with batch dimensions: (B=1, n_traj_group=1, T, ...)
    ego_history_xyz_tensor = torch.from_numpy(ego_history_xyz_local).float().unsqueeze(0).unsqueeze(0)
    ego_history_rot_tensor = torch.from_numpy(ego_history_rot_local).float().unsqueeze(0).unsqueeze(0)
    
    return ego_history_xyz_tensor, ego_history_rot_tensor


# Configuration
PAI_DATASETS_FRAMES_PATH = "pai_datasets_frames"  # Path to pai_datasets_frames directory
TRAJECTORY_CSV_PATH = os.path.join(PAI_DATASETS_FRAMES_PATH, "trajectory_with_frame_mapping.csv")

# Parameters matching original format
NUM_HISTORY_STEPS = 16  # 1.6s of history at 10Hz
NUM_FRAMES = 4          # 0.4s of frames per camera at 10Hz
TIME_STEP = 0.1         # 100ms = 10Hz sampling
TARGET_TIMESTAMP = -199719  # Example timestamp from the CSV (microseconds)

print(f"Loading trajectory mapping from {TRAJECTORY_CSV_PATH}...")
df = load_trajectory_mapping(TRAJECTORY_CSV_PATH)
print(f"Loaded {len(df)} trajectory frames\n")

# Find trajectory history (1.6s of history)
print(f"Searching for trajectory history around timestamp: {TARGET_TIMESTAMP}us...")
print(f"Will load {NUM_HISTORY_STEPS} historical trajectory points (1.6s)...")
trajectory_history = find_trajectory_history(
    df, TARGET_TIMESTAMP, 
    num_history_steps=NUM_HISTORY_STEPS, 
    time_step=TIME_STEP
)

if trajectory_history is None:
    print("Failed to find sufficient trajectory history.")
    exit(1)

print(f"Successfully loaded {len(trajectory_history)} trajectory points")

# Find frame names for camera images (0.4s of frames)
print(f"\nSearching for camera frame names around timestamp: {TARGET_TIMESTAMP}us...")
print(f"Will load {NUM_FRAMES} frames per camera (0.4s)...")
frame_names = load_frame_names_from_history(
    df, TARGET_TIMESTAMP,
    num_frames=NUM_FRAMES,
    time_step=TIME_STEP
)

if frame_names is None:
    print("Failed to find sufficient camera frames.")
    exit(1)

print(f"Found frames: {frame_names}")

# Load camera images (4 cameras x 4 frames = 16 images total)
print(f"\nLoading camera images...")
image_frames = load_camera_images_history(PAI_DATASETS_FRAMES_PATH, frame_names)
print(f"Image tensor shape: {image_frames.shape}")  # Should be [4, 4, 3, H, W]

# Extract trajectory history as arrays in local frame
print(f"\nExtracting trajectory arrays...")
ego_history_xyz, ego_history_rot = extract_trajectory_history_arrays(trajectory_history)
print(f"Ego history position shape: {ego_history_xyz.shape}")  # Should be [1, 1, 16, 3]
print(f"Ego history rotation shape: {ego_history_rot.shape}")  # Should be [1, 1, 16, 3, 3]

# Prepare message for model (flatten cameras x frames into single dimension)
print(f"\nPreparing input for model...")
messages = helper.create_message(image_frames.flatten(0, 1))  # [16, 3, H, W]

# Load model
print("Loading Alpamayo-R1 model...")
model = AlpamayoR1.from_pretrained("nvidia/Alpamayo-R1-10B", dtype=torch.bfloat16).to("cuda")
processor = helper.get_processor(model.tokenizer)

# Process inputs
print("Processing chat template...")
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
    continue_final_message=True,
    return_dict=True,
    return_tensors="pt",
)

# Prepare model inputs
model_inputs = {
    "tokenized_data": inputs,
    "ego_history_xyz": ego_history_xyz,
    "ego_history_rot": ego_history_rot,
}

model_inputs = helper.to_device(model_inputs, "cuda")

# Run inference
print("\n" + "="*80)
print("Running inference...")
print("="*80)
torch.cuda.manual_seed_all(42)
with torch.autocast("cuda", dtype=torch.bfloat16):
    pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
        data=model_inputs,
        top_p=0.98,
        temperature=0.6,
        num_traj_samples=1,
        max_generation_length=256,
        return_extra=True,
    )

print("\n" + "="*80)
print("INFERENCE RESULTS")
print("="*80)
print("\nChain-of-Thought reasoning:")
print(extra["cot"][0])

print(f"\nPredicted trajectory shape: {pred_xyz.shape}")
print(f"Predicted future positions (first 5 steps in ego frame):")
print(pred_xyz.cpu().numpy()[0, 0, :5, :])

print("\n✓ Inference completed successfully!")
print(f"✓ Used {NUM_HISTORY_STEPS} historical trajectory points (1.6s)")
print(f"✓ Used {NUM_FRAMES} historical frames per camera (0.4s) from 4 cameras")