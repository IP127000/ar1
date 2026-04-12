#!/usr/bin/env python
"""
Unified script to extract camera frames and trajectory data from PAI dataset,
with automatic synchronization between frames and trajectory timestamps.

This script combines functionality from:
- extract_frames_*.py
- extract_trajectory.py
- sync_trajectory_*.py

Usage:
    python unified_extraction.py --clip-id <CLIP_ID> --output-dir <OUTPUT_DIR>
    
    Or to auto-find first available clip:
    python unified_extraction.py --output-dir ./extraction_output
"""

import argparse
import io
import json
import os
import zipfile
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image


# ============================================================================
# CONFIGURATION
# ============================================================================

CAMERA_NAMES = [
    "camera_cross_left_120fov",
    "camera_cross_right_120fov",
    "camera_front_tele_30fov",
    "camera_front_wide_120fov",
]

FRAME_NAME_FORMAT = "frame_{:06d}.png"
TIME_THRESHOLD_MS = 50  # Maximum acceptable time difference in ms


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_clip_index(pai_dir: str) -> pd.DataFrame:
    """Load the clip index from PAI dataset."""
    clip_index_path = os.path.join(pai_dir, "clip_index.parquet")
    if not os.path.exists(clip_index_path):
        raise FileNotFoundError(f"Clip index not found at {clip_index_path}")
    return pd.read_parquet(clip_index_path)


def load_features_df(pai_dir: str) -> pd.DataFrame:
    """Load the features metadata."""
    features_csv = os.path.join(pai_dir, "features.csv")
    if not os.path.exists(features_csv):
        raise FileNotFoundError(f"Features CSV not found at {features_csv}")
    return pd.read_csv(features_csv, index_col="feature")


def get_first_available_clip(pai_dir: str) -> str:
    """Get the first available clip ID from the dataset."""
    clip_index = load_clip_index(pai_dir)
    return clip_index.index[0]


def get_chunk_id(clip_id: str, clip_index: pd.DataFrame) -> int:
    """Get the chunk number for a clip."""
    if clip_id not in clip_index.index:
        raise ValueError(f"Clip ID {clip_id} not found in index")
    return int(clip_index.at[clip_id, "chunk"])


# ============================================================================
# FRAME EXTRACTION
# ============================================================================

def extract_video_from_zip(
    zip_path: str,
    clip_id: str,
    camera_name: str,
    output_dir: Path,
    verbose: bool = True,
) -> Optional[int]:
    """
    Extract video frames from a ZIP file and save as PNG images.
    
    Returns:
        Number of frames extracted, or None on error.
    """
    camera_output_dir = output_dir / camera_name
    camera_output_dir.mkdir(parents=True, exist_ok=True)
    
    video_path_temp = None
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            available_files = zf.namelist()
            
            # Find video file for this clip and camera
            video_file = None
            for fname in available_files:
                if clip_id in fname and camera_name in fname:
                    if any(fname.endswith(ext) for ext in [".mp4", ".h264", ".mov"]):
                        video_file = fname
                        break
            
            if video_file is None:
                if verbose:
                    print(f"    ⚠️  Video file not found for {camera_name}")
                return None
            
            # Extract video data from ZIP
            video_data = zf.read(video_file)
            video_path_temp = camera_output_dir / "temp_video.mp4"
            
            with open(video_path_temp, 'wb') as f:
                f.write(video_data)
            
            if verbose:
                print(f"    Extracted video ({len(video_data) / 1024 / 1024:.1f} MB)")
        
        # Open video with OpenCV (after closing ZIP)
        cap = cv2.VideoCapture(str(video_path_temp))
        
        if not cap.isOpened():
            if verbose:
                print(f"    ✗ Failed to open video")
            return None
        
        frame_idx = 0
        last_printed = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Save frame as PNG
                frame_path = camera_output_dir / FRAME_NAME_FORMAT.format(frame_idx)
                success = cv2.imwrite(str(frame_path), frame)
                
                if not success and verbose:
                    print(f"      Warning: Failed to save frame {frame_idx}")
                
                frame_idx += 1
                
                if verbose and frame_idx - last_printed >= 100:
                    print(f"      Extracted {frame_idx} frames...")
                    last_printed = frame_idx
        finally:
            cap.release()
        
        if video_path_temp and video_path_temp.exists():
            video_path_temp.unlink()
        
        if verbose:
            print(f"    ✓ Extracted {frame_idx} frames")
        return frame_idx
            
    except KeyboardInterrupt:
        if verbose:
            print(f"    ⚠️  Interrupted by user")
        if video_path_temp and video_path_temp.exists():
            video_path_temp.unlink()
        return None
    except Exception as e:
        if verbose:
            print(f"    ✗ Error: {e}")
        if video_path_temp and video_path_temp.exists():
            video_path_temp.unlink()
        return None


def extract_frames_from_clip(
    clip_id: str,
    pai_dir: str,
    output_dir: Path,
    verbose: bool = True,
) -> Dict[str, int]:
    """
    Extract frames from all cameras for a given clip.
    
    Returns:
        Dictionary mapping camera_name -> number of frames extracted.
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"FRAME EXTRACTION")
        print(f"{'='*70}")
        print(f"Clip ID: {clip_id}")
    
    # Get metadata
    clip_index = load_clip_index(pai_dir)
    chunk_id = get_chunk_id(clip_id, clip_index)
    features_df = load_features_df(pai_dir)
    
    if verbose:
        print(f"Chunk ID: {chunk_id}")
        print(f"\nExtracting frames from {len(CAMERA_NAMES)} cameras:")
    
    extracted_frames = {}
    
    for camera_name in CAMERA_NAMES:
        if verbose:
            print(f"\n  {camera_name}:")
        
        # Build ZIP file path
        try:
            chunk_path_template = features_df.at[camera_name, "chunk_path"]
            chunk_path = chunk_path_template.format(chunk_id=chunk_id)
            chunk_file_path = os.path.join(pai_dir, chunk_path)
            
            if not os.path.exists(chunk_file_path):
                if verbose:
                    print(f"    ⚠️  File not found: {chunk_file_path}")
                continue
            
            result = extract_video_from_zip(
                chunk_file_path,
                clip_id,
                camera_name,
                output_dir,
                verbose=verbose,
            )
            
            if result is not None:
                extracted_frames[camera_name] = result
        
        except Exception as e:
            if verbose:
                print(f"    ✗ Error: {e}")
            continue
    
    if verbose:
        print(f"\n✓ Successfully extracted frames from {len(extracted_frames)} cameras")
    
    return extracted_frames


# ============================================================================
# TRAJECTORY EXTRACTION
# ============================================================================

def extract_trajectory_from_clip(
    clip_id: str,
    pai_dir: str,
    output_csv: str,
    verbose: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Extract trajectory/egomotion data from a clip and save as CSV.
    
    Returns:
        DataFrame with trajectory data, or None on error.
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"TRAJECTORY EXTRACTION")
        print(f"{'='*70}")
        print(f"Clip ID: {clip_id}")
    
    # Get metadata
    clip_index = load_clip_index(pai_dir)
    chunk_id = get_chunk_id(clip_id, clip_index)
    features_df = load_features_df(pai_dir)
    
    if verbose:
        print(f"Chunk ID: {chunk_id}")
    
    # Build path to egomotion ZIP file
    egomotion_feature = "egomotion"
    
    if egomotion_feature not in features_df.index:
        if verbose:
            print(f"✗ {egomotion_feature} feature not found in features.csv")
        return None
    
    chunk_path_template = features_df.at[egomotion_feature, "chunk_path"]
    chunk_path = chunk_path_template.format(chunk_id=chunk_id)
    chunk_file_path = os.path.join(pai_dir, chunk_path)
    
    if verbose:
        print(f"Reading from: {chunk_file_path}")
    
    if not os.path.exists(chunk_file_path):
        if verbose:
            print(f"✗ File not found: {chunk_file_path}")
        return None
    
    # Extract egomotion data from ZIP
    try:
        with open(chunk_file_path, 'rb') as f:
            with zipfile.ZipFile(f, 'r') as zf:
                available_files = zf.namelist()
                
                # Find parquet file for this clip
                clip_file = None
                for fname in available_files:
                    if clip_id in fname and fname.endswith('.parquet'):
                        clip_file = fname
                        break
                
                if clip_file is None:
                    if verbose:
                        print(f"✗ No parquet file found for clip {clip_id}")
                    return None
                
                if verbose:
                    print(f"Found trajectory file: {clip_file}")
                
                # Read parquet file
                trajectory_data = pd.read_parquet(
                    io.BytesIO(zf.read(clip_file))
                )
        
        # Save to CSV
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        trajectory_data.to_csv(output_csv, index=False)
        
        if verbose:
            print(f"\n✓ Extracted {len(trajectory_data)} trajectory points")
            print(f"  Shape: {trajectory_data.shape}")
            print(f"  Columns: {', '.join(trajectory_data.columns.tolist())}")
            print(f"  Time range: {trajectory_data['timestamp'].min()} to {trajectory_data['timestamp'].max()} μs")
            print(f"  Duration: {(trajectory_data['timestamp'].max() - trajectory_data['timestamp'].min()) / 1e6:.2f} s")
            print(f"  Saved to: {output_csv}")
        
        return trajectory_data
    
    except Exception as e:
        if verbose:
            print(f"✗ Error extracting trajectory: {e}")
            import traceback
            traceback.print_exc()
        return None


# ============================================================================
# FRAME TIMESTAMP EXTRACTION AND SYNCHRONIZATION
# ============================================================================

def get_frame_timestamps_from_zip(
    pai_dir: str,
    clip_id: str,
    camera_name: str,
    chunk_id: int,
    verbose: bool = False,
) -> Optional[np.ndarray]:
    """
    Extract frame timestamps from camera ZIP file.
    
    Returns:
        Array of frame timestamps in microseconds, or None on error.
    """
    try:
        features_df = load_features_df(pai_dir)
        
        chunk_path_template = features_df.at[camera_name, "chunk_path"]
        chunk_path = chunk_path_template.format(chunk_id=chunk_id)
        chunk_file_path = os.path.join(pai_dir, chunk_path)
        
        with open(chunk_file_path, 'rb') as f:
            with zipfile.ZipFile(f, 'r') as zf:
                available_files = zf.namelist()
                
                # Look for timestamps parquet file for this clip
                timestamps_file = None
                for fname in available_files:
                    if clip_id in fname and "timestamps.parquet" in fname:
                        timestamps_file = fname
                        break
                
                if timestamps_file is None:
                    if verbose:
                        print(f"    Warning: No timestamps found for {camera_name}")
                    return None
                
                # Read timestamps parquet
                frame_ts_df = pd.read_parquet(
                    io.BytesIO(zf.read(timestamps_file))
                )
                
                if 'timestamp' not in frame_ts_df.columns:
                    if verbose:
                        print(f"    Warning: 'timestamp' column not found")
                    return None
                
                return frame_ts_df['timestamp'].values
    
    except Exception as e:
        if verbose:
            print(f"    Error reading frame timestamps: {e}")
        return None


def sync_trajectory_with_frames(
    trajectory_df: pd.DataFrame,
    pai_dir: str,
    clip_id: str,
    chunk_id: int,
    output_csv: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Synchronize trajectory data with image frame timestamps across all cameras.
    
    For each trajectory point, find the nearest frame in each camera and add:
    - camera_name_frame_idx: Index of nearest frame
    - camera_name_frame_name: Filename of nearest frame
    - camera_name_time_diff_ms: Time difference in milliseconds
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"SYNCHRONIZATION: TRAJECTORY ↔ FRAMES")
        print(f"{'='*70}")
    
    traj_timestamps = trajectory_df['timestamp'].values
    
    if verbose:
        print(f"\nTrajectory: {len(trajectory_df)} points")
        print(f"  Time range: {traj_timestamps.min()} to {traj_timestamps.max()} μs")
        print(f"  Duration: {(traj_timestamps.max() - traj_timestamps.min()) / 1e6:.2f} s")
    
    # Get frame timestamps from each camera
    frame_info = {}
    
    if verbose:
        print(f"\nReading frame timestamps from all cameras:")
    
    for camera_name in CAMERA_NAMES:
        frame_ts = get_frame_timestamps_from_zip(
            pai_dir,
            clip_id,
            camera_name,
            chunk_id,
            verbose=verbose,
        )
        
        if frame_ts is not None:
            frame_info[camera_name] = frame_ts
            if verbose:
                print(f"  {camera_name}: {len(frame_ts)} frames")
                print(f"    Time range: {frame_ts.min()} to {frame_ts.max()} μs")
                overlap_start = max(traj_timestamps.min(), frame_ts.min())
                overlap_end = min(traj_timestamps.max(), frame_ts.max())
                if overlap_start <= overlap_end:
                    print(f"    Overlap: {(overlap_end - overlap_start) / 1e6:.2f} s")
                else:
                    print(f"    ⚠️  WARNING: No overlap with trajectory!")
        else:
            if verbose:
                print(f"  {camera_name}: ✗ Could not read timestamps")
    
    if not frame_info:
        if verbose:
            print("✗ ERROR: No frame information available!")
        return trajectory_df
    
    # Create synchronization for each camera
    if verbose:
        print(f"\nSynchronizing trajectory with frames:")
    
    for camera_name in CAMERA_NAMES:
        if camera_name not in frame_info:
            trajectory_df[f'{camera_name}_frame_idx'] = -1
            trajectory_df[f'{camera_name}_frame_name'] = "N/A"
            trajectory_df[f'{camera_name}_time_diff_ms'] = np.nan
            continue
        
        frame_ts = frame_info[camera_name]
        
        # Find nearest frame for each trajectory point
        frame_idx = np.zeros(len(traj_timestamps), dtype=int)
        for i, t_ts in enumerate(traj_timestamps):
            frame_idx[i] = np.argmin(np.abs(frame_ts - t_ts))
        
        # Calculate time differences
        matched_ts = frame_ts[frame_idx]
        time_diff_ms = np.abs(matched_ts - traj_timestamps) / 1e3  # μs to ms
        
        # Add columns
        trajectory_df[f'{camera_name}_frame_idx'] = frame_idx
        trajectory_df[f'{camera_name}_frame_name'] = [
            FRAME_NAME_FORMAT.format(idx) for idx in frame_idx
        ]
        trajectory_df[f'{camera_name}_time_diff_ms'] = time_diff_ms
        
        if verbose:
            valid_diffs = time_diff_ms[~np.isnan(time_diff_ms)]
            within_threshold = (valid_diffs <= TIME_THRESHOLD_MS).sum()
            print(f"  {camera_name}:")
            print(f"    Mean time diff: {valid_diffs.mean():.2f} ms")
            print(f"    Max time diff: {valid_diffs.max():.2f} ms")
            print(f"    Within {TIME_THRESHOLD_MS}ms: {within_threshold}/{len(valid_diffs)} ({within_threshold/len(valid_diffs)*100:.1f}%)")
    
    # Save synchronized data
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    trajectory_df.to_csv(output_csv, index=False)
    
    if verbose:
        print(f"\n✓ Synchronized data saved to: {output_csv}")
        print(f"  Columns: {len(trajectory_df.columns)}")
        
        # Show sample
        print(f"\nSample (first 3 rows):")
        cols_to_show = ['timestamp', 'x', 'y', 'z'] + [
            f'{cam}_frame_name' for cam in CAMERA_NAMES
        ]
        print(trajectory_df[cols_to_show].head(3).to_string())
    
    return trajectory_df


# ============================================================================
# MAIN EXTRACTION PIPELINE
# ============================================================================

def extract_clip_complete(
    clip_id: str,
    pai_dir: str = "./pai_datasets",
    output_dir: str = "./extraction_output",
    verbose: bool = True,
) -> None:
    """
    Complete pipeline: extract frames, trajectory, and synchronize them.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"UNIFIED EXTRACTION AND SYNCHRONIZATION PIPELINE")
    print(f"{'='*70}")
    print(f"PAI Dataset: {pai_dir}")
    print(f"Output Directory: {output_path}")
    
    # Get clip metadata
    clip_index = load_clip_index(pai_dir)
    chunk_id = get_chunk_id(clip_id, clip_index)
    
    print(f"Clip: {clip_id}")
    print(f"Chunk: {chunk_id}")
    
    # 1. Extract frames
    extracted_frames = extract_frames_from_clip(
        clip_id=clip_id,
        pai_dir=pai_dir,
        output_dir=output_path / "frames",
        verbose=verbose,
    )
    
    if not extracted_frames:
        print("\n✗ No frames extracted. Aborting.")
        return
    
    # 2. Extract trajectory
    trajectory_csv = str(output_path / "trajectory.csv")
    trajectory_df = extract_trajectory_from_clip(
        clip_id=clip_id,
        pai_dir=pai_dir,
        output_csv=trajectory_csv,
        verbose=verbose,
    )
    
    if trajectory_df is None:
        print("\n✗ Failed to extract trajectory. Aborting.")
        return
    
    # 3. Synchronize
    synced_csv = str(output_path / "trajectory_synced.csv")
    sync_trajectory_with_frames(
        trajectory_df=trajectory_df,
        pai_dir=pai_dir,
        clip_id=clip_id,
        chunk_id=chunk_id,
        output_csv=synced_csv,
        verbose=verbose,
    )
    
    # Summary
    print(f"\n{'='*70}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*70}")
    print(f"\nOutput files:")
    print(f"  Frames: {output_path / 'frames'}/")
    for camera, count in extracted_frames.items():
        print(f"    {camera}: {count} frames")
    print(f"  Trajectory (raw): {trajectory_csv}")
    print(f"  Trajectory (synced): {synced_csv}")
    print(f"\n✓ All data extracted and synchronized successfully!")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extract camera frames and trajectory from PAI dataset clip"
    )
    parser.add_argument(
        "--clip-id",
        type=str,
        default=None,
        help="Clip ID to extract (auto-finds first if not specified)",
    )
    parser.add_argument(
        "--pai-dir",
        type=str,
        default="./pai_datasets",
        help="Path to PAI dataset directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./extraction_output",
        help="Directory to save extraction results",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Verbose output",
    )
    
    args = parser.parse_args()
    
    # Auto-find clip if not specified
    if args.clip_id is None:
        print("No clip ID specified. Finding first available clip...\n")
        args.clip_id = get_first_available_clip(args.pai_dir)
        print(f"Using clip: {args.clip_id}\n")
    
    try:
        extract_clip_complete(
            clip_id=args.clip_id,
            pai_dir=args.pai_dir,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
