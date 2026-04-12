import json
import os
import sys

# Check if JSON file exists
json_path = "pai_datasets_frames/training_dataset.json"
if not os.path.exists(json_path):
    print(f"JSON file not found: {json_path}")
    sys.exit(1)

print(f"✓ JSON file found")
with open(json_path) as f:
    data = json.load(f)
print(f"✓ Loaded {len(data)} samples")
print(f"✓ Sample keys: {list(data[0].keys())}")
print(f"✓ Camera frames in first sample: {list(data[0]['camera_frames'].keys())}")

# Try importing the fixed dataset
try:
    from finetune_from_frames.dataset import FramesDataset
    print("✓ FramesDataset imported successfully")
    
    # Try creating dataset
    print("\nCreating dataset...")
    dataset = FramesDataset(
        json_path="pai_datasets_frames/training_dataset.json",
        pai_datasets_frames_path="pai_datasets_frames",
        split="train",
        split_ratio=0.9,
    )
    print(f"✓ Dataset created with {len(dataset)} samples")
    
    # Try loading a sample
    print("\nLoading first sample...")
    sample = dataset[0]
    if sample is None:
        print("✗ Sample is None")
    else:
        print("✓ Sample loaded successfully")
        print(f"  Fields in sample: {list(sample.keys())}")
        for key, val in sample.items():
            if hasattr(val, 'shape'):
                print(f"    {key}: shape {val.shape}")
            else:
                print(f"    {key}: {type(val).__name__}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
