# ar1

[English](README.md) | [简体中文](README.zh-CN.md)

`ar1` 是围绕 Alpamayo R1 / Alpamayo 1 的自动驾驶 VLA 研究实验仓库，包含推理辅助代码、Physical AI AV 数据集工具、notebook 实验，以及基于视频帧数据的监督微调流程。

## 包含内容

- `alpamayo_r1/`：本地 Alpamayo R1 辅助模块和推理工具
- `finetune_from_frames/`：基于帧数据集的 SFT 训练流程和说明
- `inference/`：针对抽取数据和训练数据集的推理脚本
- `scripts/`：checkpoint 转换、数据集整理和下载辅助脚本
- `utils/`：轨迹提取和帧同步工具
- `docs/`：SFT 文档和辅助图
- `notebooks/`：数据集检查和推理 notebook

## 快速开始

创建 Python 环境，并根据要运行的任务安装 PyTorch、Transformers 以及对应脚本依赖。推理、数据转换和微调所需依赖可能不同。

常见环境初始化：

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -U pip
python3 -m pip install torch transformers datasets pillow numpy
```

如需访问数据集，请先登录 Hugging Face，并在本地准备 Physical AI AV 数据集和 Alpamayo 模型权重。

```bash
huggingface-cli login
```

## 基于帧数据的微调

训练流程见 [finetune_from_frames/README.md](finetune_from_frames/README.md)。

单 GPU 调试：

```bash
python -m finetune_from_frames.train_frames \
  --config-path ./finetune_from_frames/configs \
  --config-name sft_stage1_frames
```

多 GPU 训练：

```bash
torchrun --nproc_per_node 8 -m finetune_from_frames.train_frames \
  --config-path ./finetune_from_frames/configs \
  --config-name sft_stage1_frames
```

## 说明

这是一个研究实验仓库。脚本中的路径、checkpoint 和数据集位置可能需要按本地机器环境调整，才能完整运行。

## 许可证

本项目使用 Apache License 2.0，见 [LICENSE](LICENSE)。
