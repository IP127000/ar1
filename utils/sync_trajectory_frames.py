#!/usr/bin/env python
"""
Match and sync trajectory timestamps with image frame timestamps.

This script:
1. Extracts frame timestamps from video files
2. Analyzes timestamp alignment
3. Creates a mapping between image frames and trajectory points
4. Generates synchronized CSV with both data
"""

import io
import json
import os
import zipfile
from pathlib import Path

import pandas as pd
import numpy as np


def get_frame_timestamps_from_zip(
    pai_dir: str,
    clip_id: str,
    camera_name: str,
    chunk_id: int,
) -> pd.DataFrame:
    """
    Extract frame timestamps from camera ZIP file.
    
    Returns:
        DataFrame with frame index and timestamp
    """
    features_csv = os.path.join(pai_dir, "features.csv")
    features_df = pd.read_csv(features_csv, index_col="feature")
    
    # Get chunk path for this camera
    chunk_path_template = features_df.at[camera_name, "chunk_path"]
    chunk_path = chunk_path_template.format(chunk_id=chunk_id)
    chunk_file_path = os.path.join(pai_dir, chunk_path)
    
    try:
        with open(chunk_file_path, 'rb') as f:
            with zipfile.ZipFile(f, 'r') as zf:
                # Look for timestamps parquet file for this clip
                available_files = zf.namelist()
                timestamps_file = None
                
                for fname in available_files:
                    if clip_id in fname and "timestamps.parquet" in fname and camera_name in fname:
                        timestamps_file = fname
                        break
                
                if timestamps_file is None:
                    print(f"  Warning: No timestamps found for {camera_name}")
                    return None
                
                # Read timestamps parquet
                frame_timestamps_df = pd.read_parquet(
                    io.BytesIO(zf.read(timestamps_file))
                )
                
                return frame_timestamps_df
    
    except Exception as e:
        print(f"Error reading frame timestamps: {e}")
        return None


def sync_trajectory_with_images(
    pai_dir: str = "./pai_datasets",
    clip_id: str = "25cd4769-5dcf-4b53-a351-bf2c5deb6124",
    trajectory_csv: str = "trajectory.csv",
    output_csv: str = "./trajectory_with_frame_mapping.csv",
    frame_rate: float = 30.0,
) -> None:
    """
    Synchronize trajectory data with image frames using frame rate-based mapping.
    
    Key principles:
    - trajectory timestamp=0 corresponds to the first frame (frame_0)
    - frame_index = round(timestamp_in_microseconds / frame_interval_in_microseconds)
    - frame_rate: 10 Hz means each frame is 100 ms apart (100,000 μs)
    """
    
    print("=" * 70)
    print("TRAJECTORY-IMAGE SYNCHRONIZATION (FRAME-RATE BASED)")
    print("=" * 70)
    
    # Calculate frame interval in microseconds
    frame_interval_us = int(1e6 / frame_rate)  # microseconds between frames
    frame_interval_ms = frame_interval_us / 1e3  # milliseconds
    
    print(f"\nFrame Rate Configuration:")
    print(f"  Frame rate: {frame_rate} Hz")
    print(f"  Frame interval: {frame_interval_ms:.1f} ms ({frame_interval_us} μs)")
    
    # Load trajectory
    trajectory = pd.read_csv(trajectory_csv)
    traj_timestamps = trajectory['timestamp'].values
    
    print(f"\nTrajectory Data:")
    print(f"  Total points: {len(trajectory)}")
    print(f"  Time range: {traj_timestamps.min()} to {traj_timestamps.max()} μs")
    print(f"  Duration: {(traj_timestamps.max() - traj_timestamps.min()) / 1e6:.2f} seconds")
    
    # Get clip info
    clip_index = pd.read_parquet(os.path.join(pai_dir, "clip_index.parquet"))
    chunk_id = int(clip_index.at[clip_id, "chunk"])
    
    # Get frame timestamps from first camera (they should be same for all cameras)
    camera_name = "camera_cross_left_120fov"
    frame_ts_df = get_frame_timestamps_from_zip(pai_dir, clip_id, camera_name, chunk_id)
    
    if frame_ts_df is None:
        print("ERROR: Could not get frame timestamps!")
        return
    
    frame_timestamps = frame_ts_df['timestamp'].values
    num_frames = len(frame_timestamps)
    
    print(f"\nImage Frames ({camera_name}):")
    print(f"  Total frames: {num_frames}")
    print(f"  Time range: {frame_timestamps.min()} to {frame_timestamps.max()} μs")
    print(f"  Duration: {(frame_timestamps.max() - frame_timestamps.min()) / 1e6:.2f} seconds")
    
    # Create mapping using frame rate
    print(f"\nCreating trajectory-frame mapping...")
    
    # Calculate frame index for each trajectory point using frame rate
    # formula: frame_index = floor((timestamp + frame_interval/2) / frame_interval)
    # This means:
    #   frame_0: timestamp < frame_interval/2
    #   frame_1: frame_interval/2 <= timestamp < frame_interval/2 + frame_interval
    half_interval_us = frame_interval_us // 2
    frame_indices = np.floor((traj_timestamps + half_interval_us) / frame_interval_us).astype(int)
    
    # Clamp frame indices to valid range [0, num_frames-1]
    frame_indices_clamped = np.clip(frame_indices, 0, num_frames - 1)
    
    # Calculate expected timestamps for the computed frame indices
    expected_timestamps = frame_indices_clamped * frame_interval_us
    
    # Calculate time differences between trajectory timestamp and expected frame timestamp
    time_diff = np.abs(traj_timestamps - expected_timestamps) / 1e3  # Convert to ms
    
    # For reference, also get actual frame timestamps from extracted data
    actual_frame_timestamps = frame_timestamps[frame_indices_clamped]
    actual_time_diff = np.abs(traj_timestamps - actual_frame_timestamps) / 1e3  # Convert to ms
    
    # Add mapping columns to trajectory

    trajectory['frame_name'] = [f"{idx:03d}.png" for idx in frame_indices_clamped]

    
    # Save synced data
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory.to_csv(output_csv, index=False)
    
    print(f"\n  Statistics:")
    print(f"    Frames extrapolated beyond image boundaries: {(frame_indices < 0).sum() + (frame_indices >= num_frames).sum()}")
    print(f"    Mean time diff (expected): {time_diff.mean():.2f} ms")
    print(f"    Mean time diff (actual): {actual_time_diff.mean():.2f} ms")
    print(f"    Max time diff (expected): {time_diff.max():.2f} ms")
    print(f"    Max time diff (actual): {actual_time_diff.max():.2f} ms")
    
    print(f"\nSynchronized trajectory saved to: {output_csv}")
    print(f"\nNew columns added:")
    print(f"  - frame_index: Computed frame index from timestamp")
    print(f"  - frame_index_clamped: Frame index clamped to [0, {num_frames-1}]")
    print(f"  - frame_name: Corresponding frame filename")
    print(f"  - expected_timestamp_us: Expected timestamp at computed frame index")
    print(f"  - time_diff_expected_ms: Difference between trajectory and expected timestamp (ms)")
    print(f"  - actual_frame_timestamp_us: Actual timestamp from extracted frame data")
    print(f"  - time_diff_actual_ms: Difference from actual frame timestamp (ms)")
    
    # Show sample
    print(f"\nSample of synchronized data (first 5 rows):")
    
    # Show some statistics about frame index range
    print(f"\nFrame Index Range:")
    print(f"  Min frame index: {frame_indices.min()}")
    print(f"  Max frame index: {frame_indices.max()}")
    print(f"  Valid range: [0, {num_frames-1}]")
    
    print("\n" + "=" * 70)
    print("SUCCESS: Trajectory and images synchronized using frame rate!")
    print("=" * 70)


if __name__ == "__main__":
    sync_trajectory_with_images()
