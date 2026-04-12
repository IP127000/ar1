# Alpamayo 训练 - 从 Frames 数据集

这是一个使用 `pai_datasets_frames/training_dataset.json` 数据集进行 Alpamayo 模型 SFT（监督微调）训练的完整解决方案。

## 项目结构

```
finetune_from_frames/
├── __init__.py                  # Python 包初始化
├── train_frames.py              # 主训练脚本
├── dataset.py                   # FramesDataset 类实现
├── utils.py                     # 数据处理和 collate 函数
├── configs/
│   ├── sft_base_frames.yaml    # 基础训练配置
│   ├── sft_stage1_frames.yaml  # Stage 1 训练配置
│   ├── vla_processor.yaml       # VLA 处理器配置
│   └── deepspeed/
│       └── zero2.json           # DeepSpeed ZeRO-2 配置
└── README.md                    # 本文件
```

## 主要改动

相比原始的 `finetune/sft` 训练方案，本方案的主要改动包括：

### 1. 数据集实现 (`dataset.py`)

**FramesDataset 类**：
- 从 `pai_datasets_frames/training_dataset.json` 加载数据
- 支持自动的训练/验证集种分（90/10）
- 加载 4 个摄像头的图像序列
- 提取轨迹历史数据（位置和旋转）
- 与 VLA 处理器集成，支持在线数据预处理

关键方法：
- `_load_camera_images()`: 从样本加载 4 个摄像头的图像
  - 摄像头顺序：左交叉、前广角、右交叉、前长焦
  - 返回张量形状：(num_cameras=4, num_frames, 3, H, W)
  
- `_extract_trajectory()`: 提取轨迹数据
  - 从 `history_trajectory` 中读取位置和旋转矩阵
  - 返回形状：(1, 1, num_history, 3) 和 (1, 1, num_history, 3, 3)

### 2. 数据处理 (`utils.py`)

**collate_fn_frames 函数**：
- 处理来自 FramesDataset 的批次数据
- 堆叠图像张量
- 连接轨迹张量
- 合并分词后的数据字典

### 3. 配置文件 (`configs/`)

**sft_base_frames.yaml**：
- 配置 FramesDataset 参数
- 设置 VLA 处理器
- DeepSpeed Zero-2 优化
- 梯度检查点开启（节省显存）

**sft_stage1_frames.yaml**：
- 继承基础配置
- 模型特定的学习率倍数（视觉编码器 0.1x）
- Stage 1 训练的参数调整

### 4. 训练脚本 (`train_frames.py`)

- 基于原始 `finetune/sft/train_hf.py`
- 集成新的 FramesDataset 和 collate 函数
- 支持分布式训练（torchrun）
- DeepSpeed 支持

## 数据准备

### 前置要求

1. 确保 `pai_datasets_frames` 目录结构完整：
```
pai_datasets_frames/
├── training_dataset.json          # 包含所有样本元数据的 JSON 文件
├── camera_cross_left_120fov/      # 4 个摄像头目录
├── camera_front_wide_120fov/
├── camera_cross_right_120fov/
└── camera_front_tele_30fov/
```

2. `training_dataset.json` 的格式要求：
```json
[
  {
    "timestamp": 1310275,
    "timestamp_seconds": 1.310275,
    "reference_pose": {
      "position": [...],
      "quaternion": [...],
      "velocity": [...],
      "acceleration": [...]
    },
    "history_trajectory": {
      "positions": [...],        // (N, 3) 数组
      "rotations": [...]         // (N, 3, 3) 数组
    },
    "camera_frames": {
      "camera_cross_left_120fov": ["path/to/frame1.jpg", ...],
      "camera_front_wide_120fov": [...],
      "camera_cross_right_120fov": [...],
      "camera_front_tele_30fov": [...]
    }
  },
  ...
]
```

3. 下载/确保模型权重可用：
```bash
# 确保以下路径存在
./alpamayo_weights/        # 预训练权重
./qwen3_vl_2B/            # Qwen VLM 权重
```

## 训练命令

### 单 GPU 训练（调试）

```bash
python -m finetune_from_frames.train_frames \
    --config-path ./finetune_from_frames/configs \
    --config-name sft_stage1_frames
```

### 多 GPU 训练（生产环境）

```bash
torchrun --nproc_per_node 8 -m finetune_from_frames.train_frames \
    --config-path ./finetune_from_frames/configs \
    --config-name sft_stage1_frames
```

### 自定义参数示例

```bash
torchrun --nproc_per_node 8 -m finetune_from_frames.train_frames \
    --config-path ./finetune_from_frames/configs \
    --config-name sft_stage1_frames \
    trainer.num_train_epochs=5 \
    trainer.learning_rate=1e-4 \
    trainer.per_device_train_batch_size=2
```

## 配置参数说明

### 数据集配置

在 `sft_base_frames.yaml` 中的 `data.{train,val}_dataset` 部分：

```yaml
data:
  train_dataset:
    json_path: ./pai_datasets_frames/training_dataset.json
    pai_datasets_frames_path: ./pai_datasets_frames
    split: train              # "train" 实现训练集
    split_ratio: 0.9          # 90% 用于训练，10% 用于验证
  val_dataset:
    split: val                # "val" 实现验证集
```

### 训练参数

关键的训练超参数（可通过命令行覆盖）：

```yaml
trainer:
  per_device_train_batch_size: 1           # 批次大小
  learning_rate: 1e-5                      # 学习率
  warmup_steps: 500                        # 预热步数
  num_train_epochs: 3                      # 训练轮数
  gradient_accumulation_steps: 4           # 梯度累积
  
  # 学习率倍数（Stage 1）
  lr_multiplier:
    vlm.model.visual: 0.1                  # 视觉编码器用 0.1x 学习率
```

## 输出

训练结果保存在：

```
output_sft_frames_stage1/
├── config.yaml                 # 保存的完整配置
├── checkpoint-100/             # 训练检查点
├── training_args.bin
├── trainer_state.json
└── ...
```

模型权重和检查点可直接用于推理。

## 与原始 SFT 训练的对比

| 方面 | 原始 SFT | Frames 版本 |
|------|----------|-----------|
| 数据源 | PAI 原始数据集 | pai_datasets_frames |
| 加载方式 | 对象存储接口 | JSON + 本地文件 |
| 数据集类 | PAIDataset | FramesDataset |
| 配置 | sft_stage1.yaml | sft_stage1_frames.yaml |
| 适用场景 | 大规模生产 | 快速原型/本地开发 |

## 故障排除

### 错误：找不到图像文件

**原因**：`camera_frames` 中的路径不正确或图像文件不存在

**解决**：
1. 检查 `pai_datasets_frames_path` 配置是否正确
2. 验证 `training_dataset.json` 中的图像路径是否相对于 `pai_datasets_frames_path`
3. 确认所有引用的图像文件实际存在

### 错误：OOM（内存不足）

**解决方案**：
1. 减小 `per_device_train_batch_size`（目前为 1）
2. 增加 `gradient_accumulation_steps`
3. 启用更多梯度检查点（已默认启用）
4. 使用 `--nproc_per_node` 分散到更多 GPU

### 训练速度慢

**优化建议**：
1. 增加 `dataloader_num_workers`（目前为 2）
2. 启用 `dataloader_pin_memory: True`（已启用）
3. 确保 GPU 被充分利用（检查 GPU 使用率）
4. 考虑增加批次大小（如果显存允许）

## 参考资源

- 原始推理示例：[test_inference_from_training_dataset.py](../inference/test_inference_from_training_dataset.py)
- VLA 处理器：[qwen_processor.py](../../alpamayo_r1/processor/qwen_processor.py)
- 原始 SFT 训练：[finetune/sft/train_hf.py](../sft/train_hf.py)

## 许可证

SPDX-License-Identifier: Apache-2.0

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
