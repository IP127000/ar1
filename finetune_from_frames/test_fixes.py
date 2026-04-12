#!/usr/bin/env python
"""
Test script to verify the fixed dataset and collate function.
"""

import sys
import torch
import json
from pathlib import Path

# Test imports
try:
    from finetune_from_frames.dataset import FramesDataset
    from finetune_from_frames.utils import collate_fn_frames
    print("✓ Successfully imported FramesDataset and collate_fn_frames")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def test_dataset_loading():
    """Test if the dataset loads correctly."""
    print("\n" + "="*80)
    print("TEST 1: Dataset Loading")
    print("="*80)
    
    json_path = "./pai_datasets_frames/training_dataset.json"
    frames_path = "./pai_datasets_frames"
    
    # Check if files exist
    if not Path(json_path).exists():
        print(f"✗ JSON file not found: {json_path}")
        return False
    
    print(f"✓ JSON file found: {json_path}")
    
    try:
        dataset = FramesDataset(
            json_path=json_path,
            pai_datasets_frames_path=frames_path,
            split="train",
            split_ratio=0.9,
        )
        print(f"✓ Dataset created successfully")
        print(f"  - Total samples: {len(dataset)}")
        return True
    except Exception as e:
        print(f"✗ Error creating dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sample_structure():
    """Test if a single sample has the correct structure."""
    print("\n" + "="*80)
    print("TEST 2: Sample Structure")
    print("="*80)
    
    json_path = "./pai_datasets_frames/training_dataset.json"
    frames_path = "./pai_datasets_frames"
    
    try:
        dataset = FramesDataset(
            json_path=json_path,
            pai_datasets_frames_path=frames_path,
            split="train",
            split_ratio=0.9,
        )
        
        # Get first sample
        sample = dataset[0]
        
        if sample is None:
            print("✗ First sample is None")
            return False
        
        print("✓ Sample loaded successfully")
        
        # Check required fields
        required_fields = [
            "image_frames",
            "camera_indices",
            "relative_timestamps",
            "ego_history_xyz",
            "ego_history_rot",
        ]
        
        for field in required_fields:
            if field not in sample:
                print(f"✗ Missing field: {field}")
                return False
            print(f"  ✓ {field}: shape {sample[field].shape if hasattr(sample[field], 'shape') else 'N/A'}")
        
        # Check optional tokenized_data field
        if "tokenized_data" in sample:
            print(f"  ✓ tokenized_data (with keys: {list(sample['tokenized_data'].keys())})")
        else:
            print("  ⚠ tokenized_data not present (VLA preprocessing may not be configured)")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading sample: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_collate_function():
    """Test if the collate function works correctly."""
    print("\n" + "="*80)
    print("TEST 3: Collate Function")
    print("="*80)
    
    json_path = "./pai_datasets_frames/training_dataset.json"
    frames_path = "./pai_datasets_frames"
    
    try:
        dataset = FramesDataset(
            json_path=json_path,
            pai_datasets_frames_path=frames_path,
            split="train",
            split_ratio=0.9,
        )
        
        # Create a small batch
        batch = [dataset[i] for i in range(min(2, len(dataset)))]
        batch = [s for s in batch if s is not None]  # Filter None samples
        
        if not batch:
            print("✗ No valid samples in batch")
            return False
        
        print(f"✓ Created batch with {len(batch)} samples")
        
        # Apply collate function
        collated = collate_fn_frames(batch)
        
        print("✓ Collate function executed successfully")
        print(f"  Output fields: {list(collated.keys())}")
        
        # Check output structure
        if "ego_history_xyz" in collated:
            print(f"  ✓ ego_history_xyz: shape {collated['ego_history_xyz'].shape}")
        if "ego_history_rot" in collated:
            print(f"  ✓ ego_history_rot: shape {collated['ego_history_rot'].shape}")
        if "tokenized_data" in collated:
            print(f"  ✓ tokenized_data: keys {list(collated['tokenized_data'].keys())}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in collate function: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("DATASET AND COLLATE FUNCTION TESTS")
    print("="*80)
    
    tests = [
        ("Dataset Loading", test_dataset_loading),
        ("Sample Structure", test_sample_structure),
        ("Collate Function", test_collate_function),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Unexpected error in {test_name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! The fixes are working correctly.")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
