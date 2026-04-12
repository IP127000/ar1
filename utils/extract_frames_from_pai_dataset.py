"""
Extract camera frames from PAI dataset clip (Improved standalone version).

This script extracts video frames from the 4 main cameras of a given clip
and saves them organized by camera name in a new directory.

Usage:
    python extract_frames_v2.py --clip-id <clip_id> --output-dir <output_dir>
"""

import argparse
import io
import json
import os
import zipfile
from pathlib import Path
from typing import Optional
import time

import cv2
import numpy as np
import pandas as pd


def extract_video_from_zip_v2(
    zip_path: str,
    clip_id: str,
    camera_name: str,
    output_dir: Path,
) -> Optional[int]:
    """
    Extract video frames from a ZIP file using cv2.imwrite for better stability.
    
    Returns the number of frames extracted, or None on error.
    """
    camera_output_dir = output_dir / camera_name
    camera_output_dir.mkdir(parents=True, exist_ok=True)
    
    video_path_temp = None
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            # List available files in the ZIP
            available_files = zf.namelist()
            
            # Find the video file for this clip and camera
            video_file = None
            
            for fname in available_files:
                if clip_id in fname and camera_name in fname:
                    if "video" in fname or fname.endswith(".mp4") or fname.endswith(".h264"):
                        video_file = fname
                        break
            
            if video_file is None:
                print(f"  ℹ️  Video file not found for {camera_name}")
                return None
            
            # Extract video data from ZIP
            video_data = zf.read(video_file)
            video_path_temp = camera_output_dir / "temp_video.mp4"
            
            try:
                with open(video_path_temp, 'wb') as f:
                    f.write(video_data)
            except Exception as e:
                print(f"  ✗ Failed to write temp video: {e}")
                return None
        
        # Open video with OpenCV (after closing the ZIP file)
        cap = cv2.VideoCapture(str(video_path_temp))
        
        if not cap.isOpened():
            print(f"  ✗ Failed to open video")
            return None
        
        frame_idx = 0
        last_printed = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Save frame directly using cv2.imwrite (more stable)
                frame_path = camera_output_dir / f"{frame_idx:03d}.png"
                success = cv2.imwrite(str(frame_path), frame)
                
                if not success:
                    print(f"    Warning: Failed to save frame {frame_idx}")
                
                frame_idx += 1
                
                if frame_idx - last_printed >= 100:
                    print(f"    Extracted {frame_idx} frames...")
                    last_printed = frame_idx
        finally:
            cap.release()
        
        print(f"  ✓ Extracted {frame_idx} frames to {camera_name}/")
        return frame_idx
            
    except KeyboardInterrupt:
        print(f"\n  ⚠️  Extraction interrupted by user")
        return None
    except Exception as e:
        print(f"  ✗ Error extracting video for {camera_name}: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        # Clean up temp video file
        if video_path_temp and video_path_temp.exists():
            try:
                video_path_temp.unlink()
            except Exception as e:
                print(f"    Warning: Could not delete temp file: {e}")


def extract_frames_from_clip_v2(
    clip_id: str,
    pai_dir: str = "../pai_datasets",
    output_dir: str = "../extracted_frames",
) -> None:
    """
    Extract all frames from a clip's 4 main cameras and save to disk.
    
    Args:
        clip_id: The clip ID to extract frames from
        pai_dir: Path to the PAI dataset root directory
        output_dir: Path to save extracted frames
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}\n")
    
    # Get the chunk for this clip
    print(f"Loading clip index...")
    clip_index_path = os.path.join(pai_dir, "clip_index.parquet")
    if not os.path.exists(clip_index_path):
        print(f"✗ Clip index not found at {clip_index_path}")
        return
    
    clip_index = pd.read_parquet(clip_index_path)
    
    if clip_id not in clip_index.index:
        print(f"✗ Clip ID {clip_id} not found in index")
        return
    
    chunk_id = int(clip_index.at[clip_id, "chunk"])
    print(f"Clip {clip_id} is in chunk {chunk_id}\n")
    
    # The 4 main cameras we want to extract
    camera_names = [
        "camera_cross_left_120fov",
        "camera_cross_right_120fov", 
        "camera_front_tele_30fov",
        "camera_front_wide_120fov",
    ]
    
    print(f"Processing {len(camera_names)} cameras:\n")
    
    # Extract frames from each camera
    extracted_cameras = 0
    total_frames = 0
    
    for camera_name in camera_names:
        print(f"Processing: {camera_name}")
        
        # Find the ZIP file for this camera and chunk
        camera_zip_pattern = f"{camera_name}.chunk_{chunk_id:04d}.zip"
        # Try different path patterns
        camera_zip = os.path.join(pai_dir, "camera", camera_name, camera_zip_pattern)
        
        if not os.path.exists(camera_zip):
            # Try alternate path
            camera_zip = os.path.join(pai_dir, "camera", camera_zip_pattern)
        
        if os.path.exists(camera_zip):
            result = extract_video_from_zip_v2(camera_zip, clip_id, camera_name, output_path)
            if result is not None:
                extracted_cameras += 1
                total_frames += result
        else:
            print(f"  ⚠️  ZIP file not found: {camera_zip_pattern}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract camera frames from PAI dataset clip"
    )
    parser.add_argument(
        "--clip-id",
        type=str,
        default="25cd4769-5dcf-4b53-a351-bf2c5deb6124",
        help="Clip ID to extract (defaults to first available clip)",
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
        default="./extracted_frames",
        help="Directory to save extracted frames",
    )
    
    args = parser.parse_args()
    
    # If no clip_id provided, get the first one
    if args.clip_id is None:
        print("No clip ID provided, loading first available clip...\n")
        clip_index_path = os.path.join(args.pai_dir, "clip_index.parquet")
        if not os.path.exists(clip_index_path):
            print(f"✗ Clip index not found at {clip_index_path}")
            return
        
        clip_index = pd.read_parquet(clip_index_path)
        args.clip_id = clip_index.index[0]
        print(f"Using clip ID: {args.clip_id}\n")
    
    extract_frames_from_clip_v2(
        clip_id=args.clip_id,
        pai_dir=args.pai_dir,
        output_dir=args.output_dir,
    )
    
    print(f"\n✓ Extraction complete!")
    
    # Summary
    output_path = Path(args.output_dir)
    for camera_dir in sorted(output_path.iterdir()):
        if camera_dir.is_dir():
            frames = list(camera_dir.glob("frame_*.png"))
            print(f"  {camera_dir.name}: {len(frames)} frames")


if __name__ == "__main__":
    main()
