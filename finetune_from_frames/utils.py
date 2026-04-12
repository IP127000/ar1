# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Utility functions for training with frames dataset

import torch
from typing import Any, Dict, List


def collate_fn_frames(
    batch: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Collate function for FramesDataset samples.

    Args:
        batch: List of samples from FramesDataset

    Returns:
        Dictionary with collated tensors ready for model input
    """
    # Filter out None samples
    batch = [sample for sample in batch if sample is not None]
    if len(batch) == 0:
        raise ValueError("All samples in batch are None")

    # Stack images and trajectories
    images_list = []
    ego_xyz_list = []
    ego_rot_list = []
    tokenized_data_list = []

    for sample in batch:
        images_list.append(sample["images"])
        ego_xyz_list.append(sample["ego_history_xyz"])
        ego_rot_list.append(sample["ego_history_rot"])
        if "tokenized_data" in sample:
            tokenized_data_list.append(sample["tokenized_data"])

    # Stack tensors
    output = {
        "images": torch.stack(images_list, dim=0),
        "ego_history_xyz": torch.cat(ego_xyz_list, dim=0),
        "ego_history_rot": torch.cat(ego_rot_list, dim=0),
    }

    # If we have tokenized data, merge them
    if tokenized_data_list:
        # Merge tokenized data dicts
        merged_tokenized = {}
        for key in tokenized_data_list[0].keys():
            values = [d[key] for d in tokenized_data_list]
            if isinstance(values[0], torch.Tensor):
                merged_tokenized[key] = torch.cat(values, dim=0)
            elif isinstance(values[0], dict):
                # Nested dict (should not happen for tokenized data)
                pass
            else:
                merged_tokenized[key] = values
        output["tokenized_data"] = merged_tokenized

    return output
