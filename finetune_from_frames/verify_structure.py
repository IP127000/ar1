#!/usr/bin/env python
"""
Verify the finetune_from_frames project structure is complete.
"""

import os
from pathlib import Path

def verify_project_structure():
    """Verify all required files exist."""
    
    base_path = Path(__file__).parent.resolve()
    
    required_files = {
        "Root files": [
            "__init__.py",
            "train_frames.py",
            "dataset.py",
            "utils.py",
            "validate_dataset.py",
            "example_train.sh",
            "README.md",
            "QUICKSTART.md",
            "GENERATION_SUMMARY.md",
        ],
        "Config files": [
            "configs/__init__.py",
            "configs/sft_base_frames.yaml",
            "configs/sft_stage1_frames.yaml",
            "configs/vla_processor.yaml",
            "configs/models/ar1_base.yaml",
            "configs/models/ar1_expert.yaml",
            "configs/deepspeed/zero2.json",
        ]
    }
    
    print("=" * 80)
    print("PROJECT STRUCTURE VERIFICATION")
    print("=" * 80)
    print(f"\nProject root: {base_path}\n")
    
    all_ok = True
    for category, files in required_files.items():
        print(f"[{category}]")
        for file_path in files:
            full_path = base_path / file_path
            exists = full_path.exists()
            status = "✓" if exists else "✗"
            print(f"  {status} {file_path}")
            if not exists:
                all_ok = False
        print()
    
    print("=" * 80)
    if all_ok:
        print("✓ PROJECT STRUCTURE VERIFIED - ALL FILES PRESENT")
        print("\nYou can now start training with:")
        print("  bash example_train.sh")
        print("\nOr manually with:")
        print("  torchrun --nproc_per_node 8 -m finetune_from_frames.train_frames \\")
        print("      --config-path ./finetune_from_frames/configs \\")
        print("      --config-name sft_stage1_frames")
    else:
        print("✗ MISSING FILES - PROJECT STRUCTURE INCOMPLETE")
        print("Please regenerate or manually create missing files")
    print("=" * 80)
    
    return all_ok


if __name__ == "__main__":
    import sys
    success = verify_project_structure()
    sys.exit(0 if success else 1)
