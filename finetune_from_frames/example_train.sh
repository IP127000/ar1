#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Example training script for Alpamayo SFT using frames dataset
# 
# Usage:
#   bash finetune_from_frames/example_train.sh [num_gpus]
#
# Examples:
#   bash finetune_from_frames/example_train.sh        # Defaults to 8 GPUs
#   bash finetune_from_frames/example_train.sh 4      # Use 4 GPUs

# Default to 8 GPUs if not specified
NUM_GPUS=${1:-8}

echo "=========================================="
echo "Starting Alpamayo SFT Training"
echo "Configuration: sft_stage1_frames"
echo "Number of GPUs: $NUM_GPUS"
echo "=========================================="

# Run training with torchrun
torchrun --nproc_per_node $NUM_GPUS \
    -m finetune_from_frames.train_frames \
    --config-path ./finetune_from_frames/configs \
    --config-name sft_stage1_frames

echo "=========================================="
echo "Training completed!"
echo "Output directory: output_sft_frames_stage1/"
echo "=========================================="
