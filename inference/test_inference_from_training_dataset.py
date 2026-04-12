# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Test inference script using data from training_dataset.json
# Compares input data with test_inference_extracted.py to ensure consistency

import torch
import numpy as np
import json
import os
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation
from einops import rearrange

from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1
from alpamayo_r1 import helper


def load_training_dataset(json_path):
    """Load training dataset from JSON file.
    
    Args:
        json_path: Path to training_dataset.json
    
    Returns:
        list: List of training samples
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def load_camera_images_from_dataset(pai_datasets_frames_path, sample):
    """Load 4 camera images from a training dataset sample.
    
    Args:
        pai_datasets_frames_path: Path to pai_datasets_frames directory
        sample: Training dataset sample containing camera_frames
    
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
        if camera_dir not in sample['camera_frames']:
            raise KeyError(f"Camera {camera_dir} not in sample['camera_frames']")
        
        frame_paths = sample['camera_frames'][camera_dir]
        frames_for_camera = []
        
        for frame_path in frame_paths:
            full_path = os.path.join(pai_datasets_frames_path, frame_path)
            if os.path.exists(full_path):
                # Load image as uint8 (H, W, 3)
                img = Image.open(full_path).convert('RGB')
                img_array = np.array(img, dtype=np.uint8)
                frames_for_camera.append(img_array)
            else:
                raise FileNotFoundError(f"Image not found: {full_path}")
        
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


def extract_trajectory_arrays_from_dataset(sample):
    """Extract trajectory arrays from training dataset sample.
    
    The sample already contains history_trajectory with positions and rotations
    in local frame (ego frame at t0), so we just need to convert to torch tensors
    with proper batch dimensions.
    
    Args:
        sample: Training dataset sample
    
    Returns:
        tuple: (ego_history_xyz, ego_history_rot) both in local frame
               ego_history_xyz: shape (1, 1, num_history, 3)
               ego_history_rot: shape (1, 1, num_history, 3, 3)
    """
    history_data = sample['history_trajectory']
    
    positions_local = np.array(history_data['positions'])  # (N, 3)
    rotations_local = np.array(history_data['rotations'])  # (N, 3, 3)
    
    # Convert to torch tensors with batch dimensions: (B=1, n_traj_group=1, T, ...)
    ego_history_xyz_tensor = torch.from_numpy(positions_local).float().unsqueeze(0).unsqueeze(0)
    ego_history_rot_tensor = torch.from_numpy(rotations_local).float().unsqueeze(0).unsqueeze(0)
    
    return ego_history_xyz_tensor, ego_history_rot_tensor


# ============================================================================
# COMPARISON FUNCTIONS
# ============================================================================




# ============================================================================
# MAIN INFERENCE COMPARISON
# ============================================================================

def main():
    # Configuration
    PAI_DATASETS_FRAMES_PATH = "pai_datasets_frames"
    TRAINING_DATASET_PATH = os.path.join(PAI_DATASETS_FRAMES_PATH, "training_dataset.json")
    
    # Check if training dataset exists
    if not os.path.exists(TRAINING_DATASET_PATH):
        print(f"❌ Training dataset not found: {TRAINING_DATASET_PATH}")
        print("Please generate it using: python generate_train_dataset.py")
        return
    
    print("="*80)
    print("INFERENCE COMPARISON TEST")
    print("Using training_dataset.json vs test_inference_extracted.py")
    print("="*80)
    
    # Load training dataset
    print(f"\nLoading training dataset from {TRAINING_DATASET_PATH}...")
    training_data = load_training_dataset(TRAINING_DATASET_PATH)
    print(f"Loaded {len(training_data)} training samples")
    
    # Use first sample for comparison
    sample = training_data[36]
    timestamp = sample['timestamp']
    print(f"\nUsing sample at timestamp: {timestamp}us ({sample['timestamp_seconds']:.3f}s)")
    
    # ========== METHOD 1: Using training_dataset.json ==========
    print("\n" + "-"*80)
    print("METHOD 1: Loading from training_dataset.json")
    print("-"*80)
    
    print("\nLoading camera images from dataset...")
    images_dataset = load_camera_images_from_dataset(PAI_DATASETS_FRAMES_PATH, sample)
    print(f"Loaded image tensor shape: {images_dataset.shape}")
    
    print("\nExtracting trajectory data from dataset...")
    ego_xyz_dataset, ego_rot_dataset = extract_trajectory_arrays_from_dataset(sample)
    print(f"Ego history XYZ shape: {ego_xyz_dataset.shape}")
    print(f"Ego history rotation shape: {ego_rot_dataset.shape}")
    
    print("\nPreparing model input from dataset...")
    messages_dataset = helper.create_message(images_dataset.flatten(0, 1))  # [16, 3, H, W]
    model = AlpamayoR1.from_pretrained("./alpamayo_weights", dtype=torch.bfloat16).to("cuda")
    processor = helper.get_processor(model.tokenizer)
    
    inputs_dataset = processor.apply_chat_template(
        messages_dataset,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    
    model_inputs_dataset = {
        "tokenized_data": inputs_dataset,
        "ego_history_xyz": ego_xyz_dataset,
        "ego_history_rot": ego_rot_dataset,
    }
    
    model_inputs_dataset = helper.to_device(model_inputs_dataset, "cuda")
    print("\nMethod 1: Using training_dataset.json...")
    torch.cuda.manual_seed_all(42)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs_dataset,
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


if __name__ == "__main__":
    main()
