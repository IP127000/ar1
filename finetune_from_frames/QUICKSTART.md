# 快速启动指南

## 前置要求

1. **环境**：Python 3.10+，CUDA 12.0+
2. **数据**：`pai_datasets_frames` 目录包含 `training_dataset.json` 和所有相关图像
3. **模型权重**：`alpamayo_weights` 和 `qwen3_vl_2B` 目录

## 快速开始（3 步）

### Step 1: 验证数据集

```bash
python finetune_from_frames/validate_dataset.py
```

输出应该显示：
```
✓ Dataset validation PASSED
```

### Step 2: 检查训练配置

```bash
# 查看当前配置
python -m hydra.main \
    --config-path finetune_from_frames/configs \
    --config-name sft_stage1_frames
```

### Step 3: 启动训练

**单 GPU（调试模式）**：
```bash
python -m finetune_from_frames.train_frames \
    --config-path ./finetune_from_frames/configs \
    --config-name sft_stage1_frames \
    trainer.per_device_train_batch_size=1 \
    trainer.per_device_eval_batch_size=1
```

**多 GPU（生产环境）**：
```bash
torchrun --nproc_per_node 8 -m finetune_from_frames.train_frames \
    --config-path ./finetune_from_frames/configs \
    --config-name sft_stage1_frames
```

**或使用提供的脚本**：
```bash
bash finetune_from_frames/example_train.sh 8  # 8 个 GPU
```

## 常见配置修改

### 调整学习率
```bash
torchrun --nproc_per_node 8 -m finetune_from_frames.train_frames \
    --config-path ./finetune_from_frames/configs \
    --config-name sft_stage1_frames \
    trainer.learning_rate=5e-5
```

### 调整训练轮数
```bash
... trainer.num_train_epochs=5
```

### 启用 W&B 日志
```bash
# 在配置中取消注释 /wandb: default
# 或从命令行：
... wandb.project=alpamayo-training wandb.entity=your-entity
```

### 修改批次大小和梯度累积
```bash
... trainer.per_device_train_batch_size=2 \
    trainer.gradient_accumulation_steps=2
```

## 输出

训练完成后，结果保存在：
```
output_sft_frames_stage1/
├── config.yaml                  # 保存的配置
├── checkpoint-500/              # 训练检查点
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── ...
└── trainer_state.json           # 训练状态
```

## 恢复训练

如果训练中断，可以从最后的检查点恢复：

```bash
torchrun --nproc_per_node 8 -m finetune_from_frames.train_frames \
    --config-path ./finetune_from_frames/configs \
    --config-name sft_stage1_frames \
    trainer.resume_from_checkpoint=output_sft_frames_stage1/checkpoint-500
```

## 监控训练

### 查看日志
```bash
# 实时查看 tensorboard
tensorboard --logdir output_sft_frames_stage1

# 或查看最后的日志信息
tail -f output_sft_frames_stage1/trainer_state.json
```

### 检查 GPU 利用率
```bash
watch -n 1 nvidia-smi  # 每秒刷新一次
```

## 故障排除

### 问题：CUDA 内存不足
**解决方案**：
1. 减小 `per_device_train_batch_size`
2. 增加 `gradient_accumulation_steps`
3. 使用更多 GPU

### 问题：数据加载缓慢
**解决方案**：
```bash
# 增加数据加载工作进程
... trainer.dataloader_num_workers=4
```

### 问题：找不到配置文件
**确保**：
1. 当前工作目录是项目根目录
2. 配置路径正确：`./finetune_from_frames/configs`

## 下一步

- 查看 [README.md](README.md) 获取完整文档
- 参考 [configs/sft_base_frames.yaml](configs/sft_base_frames.yaml) 理解所有配置选项
- 使用训练的模型进行推理或继续微调

## 获取帮助

检查这些文件了解更多详情：
- `finetune_from_frames/dataset.py` - 数据加载逻辑
- `finetune_from_frames/train_frames.py` - 训练入口
- `finetune_from_frames/configs/` - 所有配置文件
