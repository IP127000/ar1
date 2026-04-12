#!/usr/bin/env python
"""
Extract trajectory/egomotion data from PAI dataset clip and save as CSV.

Usage:
    python extract_trajectory.py --clip-id <CLIP_ID> --output-csv trajectory.csv
"""

import argparse
import io
import os
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np


def extract_trajectory_from_clip(
    clip_id: str,
    pai_dir: str = "./pai_datasets",
    output_csv: str = "trajectory.csv",
) -> Optional[pd.DataFrame]:
    """
    Extract trajectory (egomotion) data from a PAI clip and save as CSV.
    
    Args:
        clip_id: The clip ID to extract trajectory from
        pai_dir: Path to the PAI dataset root directory
        output_csv: Path to save the trajectory CSV
        
    Returns:
        DataFrame with trajectory data, or None on error
    """
    
    print(f"Extracting trajectory for clip: {clip_id}")
    print(f"Output file: {output_csv}\n")
    
    # Get the chunk for this clip
    print("Loading clip index...")
    clip_index_path = os.path.join(pai_dir, "clip_index.parquet")
    if not os.path.exists(clip_index_path):
        print(f"Error: Clip index not found at {clip_index_path}")
        return None
    
    clip_index = pd.read_parquet(clip_index_path)
    
    if clip_id not in clip_index.index:
        print(f"Error: Clip ID {clip_id} not found in index")
        return None
    
    chunk_id = int(clip_index.at[clip_id, "chunk"])
    print(f"Clip is in chunk {chunk_id}")
    
    # Get features metadata
    features_csv = os.path.join(pai_dir, "features.csv")
    features_df = pd.read_csv(features_csv, index_col="feature")
    
    # Find the egomotion ZIP file
    egomotion_feature = "egomotion"
    
    if egomotion_feature not in features_df.index:
        print(f"Error: {egomotion_feature} feature not found")
        return None
    
    # Build path to egomotion ZIP file using chunk_path
    chunk_path_template = features_df.at[egomotion_feature, "chunk_path"]
    chunk_path = chunk_path_template.format(chunk_id=chunk_id)
    chunk_file_path = os.path.join(pai_dir, chunk_path)
    
    print(f"Loading egomotion data from: {chunk_file_path}\n")
    
    if not os.path.exists(chunk_file_path):
        print(f"Error: File not found: {chunk_file_path}")
        return None
    
    # Extract egomotion data from ZIP
    try:
        with open(chunk_file_path, 'rb') as f:
            with zipfile.ZipFile(f, 'r') as zf:
                # List files in ZIP
                available_files = zf.namelist()
                
                # Find the parquet file for this clip
                clip_file = None
                for fname in available_files:
                    if clip_id in fname and fname.endswith('.parquet'):
                        clip_file = fname
                        break
                
                if clip_file is None:
                    print(f"Error: No parquet file found for clip {clip_id}")
                    return None
                
                print(f"Found egomotion file: {clip_file}")
                
                # Read the parquet file
                egomotion_data = pd.read_parquet(
                    io.BytesIO(zf.read(clip_file))
                )
                
                print(f"Egomotion data shape: {egomotion_data.shape}")
                print(f"Columns: {egomotion_data.columns.tolist()}\n")
                
                # Save to CSV
                output_path = Path(output_csv)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                egomotion_data.to_csv(output_csv, index=False)
                
                print(f"Trajectory saved to: {output_csv}")
                print(f"\nFirst few rows:")
                print(egomotion_data.head(10))
                
                print(f"\nStatistics:")
                print(f"Total trajectory points: {len(egomotion_data)}")
                print(f"Time range: {egomotion_data['timestamp'].min()} to {egomotion_data['timestamp'].max()} (us)")
                
                return egomotion_data
    
    except Exception as e:
        print(f"Error extracting trajectory: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Extract trajectory data from PAI dataset clip"
    )
    parser.add_argument(
        "--clip-id",
        type=str,
        default="25cd4769-5dcf-4b53-a351-bf2c5deb6124",
        help="Clip ID to extract trajectory from",
    )
    parser.add_argument(
        "--pai-dir",
        type=str,
        default="./pai_datasets",
        help="Path to PAI dataset directory",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="./trajectory.csv",
        help="Output CSV file path",
    )
    
    args = parser.parse_args()
    
    trajectory_df = extract_trajectory_from_clip(
        clip_id=args.clip_id,
        pai_dir=args.pai_dir,
        output_csv=args.output_csv,
    )
    
    if trajectory_df is not None:
        print(f"\nSuccess! Trajectory data exported to {args.output_csv}")
    else:
        print("\nFailed to extract trajectory data")


if __name__ == "__main__":
    main()
