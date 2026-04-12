# 训练代码生成总结

## 项目结构

已在 `d:\alpamayo\finetune_from_frames` 创建完整的训练工作流：

```
finetune_from_frames/
├── __init__.py                       # Python 包初始化
├── train_frames.py                   # 主训练脚本 - 入口点
├── dataset.py                        # FramesDataset 类实现 - 核心数据加载
├── utils.py                          # Collate 函数和数据处理工具
├── validate_dataset.py               # 数据集验证脚本
├── example_train.sh                  # 训练执行脚本模板
├── README.md                         # 完整使用文档
├── QUICKSTART.md                     # 快速启动指南
├── configs/
│   ├── __init__.py                   # 配置包初始化
│   ├── sft_base_frames.yaml         # 基础训练配置
│   ├── sft_stage1_frames.yaml       # Stage 1 微调配置
│   ├── vla_processor.yaml            # VLA 处理器配置
│   ├── models/
│   │   ├── ar1_base.yaml            # 基础 Alpamayo R1 模型配置
│   │   └── ar1_expert.yaml          # 专家 Alpamayo R1 模型配置
│   └── deepspeed/
│       └── zero2.json               # DeepSpeed ZeRO-2 优化配置
└── GENERATION_SUMMARY.md            # 本文件
```

## 核心文件说明

### 1. **train_frames.py** (主训练脚本)
- 配置加载和模型初始化
- 数据集和验证集实例化
- 训练循环启动
- 分布式训练支持

**用法**：
```bash
torchrun --nproc_per_node 8 -m finetune_from_frames.train_frames \
    --config-path ./finetune_from_frames/configs \
    --config-name sft_stage1_frames
```

### 2. **dataset.py** (FramesDataset 类)
关键方法：
- `__init__()` - 初始化数据集，支持配置化、训练/验证分割
- `__getitem__()` - 加载单个样本
- `_load_camera_images()` - 从 training_dataset.json 加载 4 摄像头图像
- `_extract_trajectory()` - 提取轨迹历史数据

**核心特性**：
- 从 JSON 直接加载，无需对象存储
- 自动分割训练/验证集（90/10）
- 与 VLA 处理器集成

### 3. **utils.py** (数据处理)
- `collate_fn_frames()` - 处理批次数据，堆叠和合并张量

### 4. **validate_dataset.py** (数据验证脚本)
- 检查 JSON 文件完整性
- 验证所有图像文件存在
- 采样检测数据质量

**用法**：
```bash
python finetune_from_frames/validate_dataset.py
```

### 5. 配置文件

**sft_base_frames.yaml**：
```yaml
data:
  train_dataset:
    _target_: finetune_from_frames.dataset.FramesDataset
    json_path: ./pai_datasets_frames/training_dataset.json
    pai_datasets_frames_path: ./pai_datasets_frames
    split: train
    split_ratio: 0.9
```

**sft_stage1_frames.yaml**：
- 继承 sft_base_frames.yaml
- 设置 Stage 1 特定参数（学习率、预热步数等）
- 模型特定学习率倍数

## 主要改进点

### 相比原始 SFT 训练

| 特性 | 原始版本 | 新版本 |
|------|---------|--------|
| 数据来源 | PAI 对象存储 | 本地 JSON + 图像 |
| 数据集类 | PAIDataset | FramesDataset |
| 数据加载 | 需要特殊接口 | 直接 JSON + PIL 图像 |
| 配置方式 | 固定路径 | 完全可配置 |
| 验证工具 | 无 | 提供 validate_dataset.py |
| 快速启动 | 复杂 | QUICKSTART.md 指南 |

### 代码参考来源

- **数据处理逻辑**：参考 `inference/test_inference_from_training_dataset.py`
  - 4 摄像头加载
  - 轨迹数据提取
  - 数据形状转换

- **训练框架**：基于 `finetune/sft/train_hf.py`
  - 相同的 Trainer 和 TrainingArguments
  - 兼容现有的 VLA 处理器

- **配置结构**：遵循 `finetune/sft/configs/` 设计
  - 基础配置 + 阶段配置
  - DeepSpeed 集成
  - 模型选择

## 使用流程

### 1. 前置检查
```bash
# 验证数据集完整性
python finetune_from_frames/validate_dataset.py
```

### 2. 启动训练
```bash
# 单 GPU（调试）
python -m finetune_from_frames.train_frames \
    --config-path ./finetune_from_frames/configs \
    --config-name sft_stage1_frames

# 多 GPU（生产）
torchrun --nproc_per_node 8 -m finetune_from_frames.train_frames \
    --config-path ./finetune_from_frames/configs \
    --config-name sft_stage1_frames
```

### 3. 监控训练
```bash
# 查看输出
tail -f output_sft_frames_stage1/trainer_state.json

# 或使用 TensorBoard
tensorboard --logdir output_sft_frames_stage1
```

## 关键配置参数

可通过命令行动态调整：

```bash
torchrun --nproc_per_node 8 -m finetune_from_frames.train_frames \
    --config-path ./finetune_from_frames/configs \
    --config-name sft_stage1_frames \
    trainer.num_train_epochs=5 \
    trainer.learning_rate=1e-4 \
    trainer.per_device_train_batch_size=2 \
    trainer.gradient_accumulation_steps=2 \
    trainer.warmup_steps=500 \
    trainer.dataloader_num_workers=4
```

## 输出文件结构

```
output_sft_frames_stage1/
├── config.yaml                  # 完整配置（已保存）
├── checkpoint-500/              # 检查点
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   ├── training_args.bin
│   └── optimizer.pt
├── trainer_state.json           # 训练状态
├── training_args.bin            # 参数序列化
└── runs/                        # TensorBoard 日志
```

## 常见问题解决

### Q: 如何从检查点恢复？
```bash
... trainer.resume_from_checkpoint=output_sft_frames_stage1/checkpoint-500
```

### Q: 如何启用 W&B 日志？
在 `sft_base_frames.yaml` 中取消注释 `- /wandb: default`

### Q: 如何使用不同的模型？
```bash
... ++defaults=[ar1_expert]  # 使用 ar1_expert.yaml
```

### Q: OOM 问题如何处理？
- 减小 `per_device_train_batch_size` 
- 增加 `gradient_accumulation_steps`
- 启用更多梯度检查点（已默认）

## 下一步建议

1. **立即开始**：按 QUICKSTART.md 步骤执行
2. **自定义训练**：编辑 sft_stage1_frames.yaml 调整参数
3. **扩展功能**：在 dataset.py 中添加数据增强（flip、rotate 等）
4. **监控优化**：集成 W&B 或 TensorBoard 监控
5. **评估结果**：使用训练模型进行推理

## 文件清单

✅ 已创建的文件：
- [x] finetune_from_frames/__init__.py
- [x] finetune_from_frames/train_frames.py
- [x] finetune_from_frames/dataset.py
- [x] finetune_from_frames/utils.py
- [x] finetune_from_frames/validate_dataset.py
- [x] finetune_from_frames/example_train.sh
- [x] finetune_from_frames/README.md
- [x] finetune_from_frames/QUICKSTART.md
- [x] finetune_from_frames/GENERATION_SUMMARY.md
- [x] finetune_from_frames/configs/sft_base_frames.yaml
- [x] finetune_from_frames/configs/sft_stage1_frames.yaml
- [x] finetune_from_frames/configs/vla_processor.yaml
- [x] finetune_from_frames/configs/models/ar1_base.yaml
- [x] finetune_from_frames/configs/models/ar1_expert.yaml
- [x] finetune_from_frames/configs/deepspeed/zero2.json

**总计：16 个文件**

## 许可证

SPDX-License-Identifier: Apache-2.0

Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

---

**生成时间**: 2026-04-12
**基础版本**: finetune/sft v1.0
**数据集**: pai_datasets_frames/training_dataset.json
