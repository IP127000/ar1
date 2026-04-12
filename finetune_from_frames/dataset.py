# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Dataset for training from pai_datasets_frames using training_dataset.json

import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from PIL import Image
from einops import rearrange
from torch.utils.data import Dataset
from hydra.utils import instantiate
from omegaconf import OmegaConf


class FramesDataset(Dataset):
    """Dataset for loading and processing samples from pai_datasets_frames/training_dataset.json."""

    def __init__(
        self,
        json_path: str,
        pai_datasets_frames_path: str,
        model_config: Any | None = None,
        vla_preprocess_args: dict | None = None,
        split: str = "train",
        split_ratio: float = 0.9,
        seed: int = 42,
    ):
        """Initialize dataset.

        Args:
            json_path: Path to training_dataset.json file
            pai_datasets_frames_path: Path to pai_datasets_frames directory
            model_config: Optional config (dict is converted with OmegaConf) passed as
                ``model_config`` into ``hydra.utils.instantiate`` when building the preprocessor.
            vla_preprocess_args: If set, Hydra config dict to instantiate a preprocessor; its result
                is stored as ``tokenized_data`` on each sample.
            split: Either "train" or "val"
            split_ratio: Fraction of data for training (rest goes to validation)
            seed: Random seed for dataset split
        """
        self.json_path = json_path
        self.pai_datasets_frames_path = pai_datasets_frames_path
        self.split = split
        self.split_ratio = split_ratio
        self.seed = seed

        # Load training dataset
        with open(json_path, 'r') as f:
            self.all_samples = json.load(f)

        # Split dataset
        np.random.seed(seed)
        indices = np.arange(len(self.all_samples))
        np.random.shuffle(indices)
        split_idx = int(len(indices) * split_ratio)
        
        if split == "train":
            self.indices = indices[:split_idx]
        else:
            self.indices = indices[split_idx:]

        # Setup VLA preprocessing
        self.vla_preprocess_func = None
        if model_config is not None and isinstance(model_config, dict):
            model_config = OmegaConf.create(model_config)
        if vla_preprocess_args is not None:
            self.vla_preprocess_func = instantiate(vla_preprocess_args, model_config=model_config)

    def __len__(self) -> int:
        """Return the number of samples in this split."""
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any] | None:
        """Load and process a single sample.

        Returns:
            Dictionary with processed inputs for the model, or None if loading fails
        """
        try:
            sample_idx = self.indices[idx]
            sample = self.all_samples[sample_idx]

            # Load camera images
            images = self._load_camera_images(sample)

            # Extract trajectory data
            ego_history_xyz, ego_history_rot = self._extract_trajectory(sample)

            # Prepare output dict
            output = {
                "images": images,
                "ego_history_xyz": ego_history_xyz,
                "ego_history_rot": ego_history_rot,
            }

            # Apply VLA preprocessing if configured
            if self.vla_preprocess_func is not None:
                # Flatten images from (num_cameras, num_frames, 3, H, W) to (num_cameras*num_frames, 3, H, W)
                flat_images = images.flatten(0, 1)
                processed = self.vla_preprocess_func(flat_images)
                output["tokenized_data"] = processed

            return output

        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return None

    def _load_camera_images(self, sample: dict) -> torch.Tensor:
        """Load 4 camera images from a sample.

        Args:
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
                full_path = os.path.join(self.pai_datasets_frames_path, frame_path)
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

        # Stack all cameras: (num_cameras, num_frames, 3, H, W)
        image_frames = torch.stack(image_frames_list, dim=0)

        return image_frames

    def _extract_trajectory(self, sample: dict) -> tuple:
        """Extract trajectory arrays from sample.

        Args:
            sample: Training dataset sample

        Returns:
            tuple: (ego_history_xyz, ego_history_rot) both in local frame
                   ego_history_xyz: shape (1, 1, num_history, 3)
                   ego_history_rot: shape (1, 1, num_history, 3, 3)
        """
        history_data = sample['history_trajectory']

        positions_local = np.array(history_data['positions'], dtype=np.float32)  # (N, 3)
        rotations_local = np.array(history_data['rotations'], dtype=np.float32)  # (N, 3, 3)

        # Convert to torch tensors with batch dimensions: (B=1, n_traj_group=1, T, ...)
        ego_history_xyz_tensor = torch.from_numpy(positions_local).float().unsqueeze(0).unsqueeze(0)
        ego_history_rot_tensor = torch.from_numpy(rotations_local).float().unsqueeze(0).unsqueeze(0)

        return ego_history_xyz_tensor, ego_history_rot_tensor
