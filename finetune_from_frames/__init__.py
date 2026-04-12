# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Training module for Alpamayo using frames dataset."""

from .dataset import FramesDataset
from .utils import collate_fn_frames

__all__ = ["FramesDataset", "collate_fn_frames"]
