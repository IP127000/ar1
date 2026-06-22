# ar1

[English](README.md) | [Simplified Chinese](README.zh-CN.md)

`ar1` is an experimental workspace around Alpamayo R1 / Alpamayo 1 for
autonomous-driving VLA research. It collects inference helpers, Physical AI AV
dataset utilities, notebook experiments, and a frame-based supervised
fine-tuning pipeline.

## What Is Included

- `alpamayo_r1/`: local Alpamayo R1 helper modules and inference utilities
- `finetune_from_frames/`: frame-dataset SFT pipeline and training notes
- `inference/`: inference scripts for extracted data and training datasets
- `scripts/`: checkpoint conversion, dataset curation, and download helpers
- `utils/`: trajectory extraction and frame synchronization utilities
- `docs/`: SFT notes and supporting figures
- `notebooks/`: dataset inspection and inference notebooks

## Quick Start

Create a Python environment with PyTorch, Transformers, and the dependencies
required by the Alpamayo scripts you want to run. The exact package set depends
on whether you are doing inference, data conversion, or fine-tuning.

Typical setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install torch transformers datasets pillow numpy
```

For dataset access, authenticate with Hugging Face and prepare the Physical AI
AV dataset and Alpamayo model weights locally.

```bash
huggingface-cli login
```

## Frame-Based Fine-Tuning

The frame-dataset training workflow is documented in
[finetune_from_frames/README.md](finetune_from_frames/README.md).

Single-GPU debugging:

```bash
python -m finetune_from_frames.train_frames \
  --config-path ./finetune_from_frames/configs \
  --config-name sft_stage1_frames
```

Multi-GPU training:

```bash
torchrun --nproc_per_node 8 -m finetune_from_frames.train_frames \
  --config-path ./finetune_from_frames/configs \
  --config-name sft_stage1_frames
```

## Notes

This repository is a research workspace. Paths, checkpoints, and dataset
locations may need to be adjusted for your local machine before scripts can run
end to end.

## License

Apache License 2.0. See [LICENSE](LICENSE).
