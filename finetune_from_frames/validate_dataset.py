#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Dataset validation script
# Checks if training_dataset.json and all referenced images are accessible

import json
import os
from pathlib import Path
import argparse
from collections import defaultdict


def validate_dataset(json_path, frames_path):
    """Validate training dataset structure and file accessibility.
    
    Args:
        json_path: Path to training_dataset.json
        frames_path: Path to pai_datasets_frames directory
    
    Returns:
        bool: True if dataset is valid, False otherwise
    """
    print("=" * 80)
    print("DATASET VALIDATION")
    print("=" * 80)
    
    # Check JSON file
    print(f"\n[1/4] Checking JSON file: {json_path}")
    if not os.path.exists(json_path):
        print(f"❌ JSON file not found: {json_path}")
        return False
    print(f"✓ JSON file found")
    
    # Load JSON
    print(f"\n[2/4] Loading JSON...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        print(f"✓ Loaded {len(data)} samples")
    except Exception as e:
        print(f"❌ Error loading JSON: {e}")
        return False
    
    # Validate frames path
    print(f"\n[3/4] Checking frames directory: {frames_path}")
    if not os.path.isdir(frames_path):
        print(f"❌ Frames directory not found: {frames_path}")
        return False
    print(f"✓ Frames directory exists")
    
    # Check camera directories
    cameras = [
        'camera_cross_left_120fov',
        'camera_front_wide_120fov',
        'camera_cross_right_120fov',
        'camera_front_tele_30fov',
    ]
    
    for camera in cameras:
        camera_path = os.path.join(frames_path, camera)
        if not os.path.isdir(camera_path):
            print(f"⚠ Warning: Camera directory not found: {camera}")
        else:
            image_count = len([f for f in os.listdir(camera_path) if f.endswith('.jpg')])
            print(f"  ✓ {camera}: {image_count} images")
    
    # Sample validation
    print(f"\n[4/4] Validating samples...")
    issues = defaultdict(int)
    valid_samples = 0
    
    for idx, sample in enumerate(data[:min(10, len(data))]):  # Sample first 10
        try:
            # Check required fields
            if 'camera_frames' not in sample:
                issues['missing_camera_frames'] += 1
                continue
            
            if 'history_trajectory' not in sample:
                issues['missing_history_trajectory'] += 1
                continue
            
            # Check camera frames
            sample_cameras = sample['camera_frames']
            all_exist = True
            
            for camera in cameras:
                if camera not in sample_cameras:
                    issues[f'missing_camera_{camera}'] += 1
                    all_exist = False
                    break
                
                # Check if image files exist (sample first one)
                frame_paths = sample_cameras[camera]
                if frame_paths:
                    full_path = os.path.join(frames_path, frame_paths[0])
                    if not os.path.exists(full_path):
                        issues[f'missing_image_{camera}'] += 1
                        all_exist = False
                        break
            
            if all_exist:
                valid_samples += 1
        
        except Exception as e:
            issues[f'error_sample_{idx}'] = str(e)
    
    print(f"\nValidated {min(10, len(data))} samples:")
    print(f"  ✓ Valid: {valid_samples}")
    
    if issues:
        print(f"  ⚠ Issues found:")
        for issue, count in issues.items():
            print(f"    - {issue}: {count}")
    
    # Summary statistics
    print(f"\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total samples: {len(data)}")
    print(f"Valid samples (sampled): {valid_samples}/{min(10, len(data))}")
    
    if valid_samples == min(10, len(data)):
        print("✓ Dataset validation PASSED")
        return True
    else:
        print("❌ Dataset validation FAILED - some issues detected")
        return False


def main():
    parser = argparse.ArgumentParser(description="Validate training dataset")
    parser.add_argument(
        "--json-path",
        type=str,
        default="./pai_datasets_frames/training_dataset.json",
        help="Path to training_dataset.json",
    )
    parser.add_argument(
        "--frames-path",
        type=str,
        default="./pai_datasets_frames",
        help="Path to pai_datasets_frames directory",
    )
    
    args = parser.parse_args()
    
    # Convert to absolute paths
    json_path = os.path.abspath(args.json_path)
    frames_path = os.path.abspath(args.frames_path)
    
    print(f"Using paths:")
    print(f"  JSON: {json_path}")
    print(f"  Frames: {frames_path}\n")
    
    success = validate_dataset(json_path, frames_path)
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
