# 协作交接文档

## 项目概述

这是一个 CIFAR-10 图像分类实验项目，通过 YAML 配置 + 命令行参数 + argparse 的方式支持多种 CNN 方法的训练、验证和对比。目前已完成项目框架搭建和 SimpleCNN 基线的完整训练。

## 已完成的工作

### 项目框架

- **YAML 驱动**：所有训练参数（模型、优化器、调度器、早停）通过 YAML 配置文件管理，`configs/` 目录下有 3 个方法的配置
- **模型注册机制**：`model.py` 中的 `MODEL_REGISTRY` 字典管理所有可用模型，添加新模型只需定义类 + 注册即可
- **命令行覆盖**：`train.py` 支持通过 `--epochs`、`--lr`、`--batch-size` 等参数覆盖配置文件中的值
- **数据集共享**：所有方法共用 `dataset/` 目录下的 CIFAR-10 数据
- **结果隔离**：`checkpoints/<method>/` 和 `results/<method>/` 按方法名分目录存放，互不干扰

### 已实现的模型

| 模型 | 文件位置 | 状态 |
| --- | --- | --- |
| SimpleCNN | `model.py` | 模型代码 + 10 轮训练均已完成 |
| ResNet20 | `model.py` | 模型代码已完成，**训练待跑** |
| VGG16-BN | `model.py` | 模型代码已完成，**训练待跑** |

### 已实现的功能

- 验证集划分（`val_split` 参数，默认 10%）
- 早停机制（`early_stopping` 配置段，监控 val_acc）
- k-fold 交叉验证（`--k-fold` 命令行参数）
- 自动化调参脚本（`tune.py`，支持网格搜索和随机采样）
- 学习率调度器（StepLR）
- 多优化器支持（Adam / SGD + momentum + weight_decay）
- 参数统计（每个模型训练时打印参数量）
- 数据增强：RandomCrop(32, padding=4) + RandomHorizontalFlip（所有方法统一）

### SimpleCNN 训练结果（上一批同学完成）

- 最佳 Test Accuracy：**79.05%**（epoch 9）
- 训练参数：Adam, lr=0.001, batch_size=64, 10 epochs
- 数据增强：RandomHorizontalFlip（初版）
- 结果文件：`results/simple_cnn/` 下有完整的曲线图、混淆矩阵和训练日志
- 模型文件：`checkpoints/simple_cnn/best_model.pth`

## 你需要做的事

### 优先级 1：跑 ResNet20（最快出结果）

ResNet20 参数量仅 272K，训练最快，建议优先完成。

```bash
# 步骤 1：快速调参（约 20 分钟，8 组参数，30 epochs/组）
python tune.py --method resnet20 --tune-epochs 30 --trials 8

# 步骤 2：查看调参结果，找到最佳 lr 和 weight_decay
cat results/tuning_summary.csv

# 步骤 3：用最佳参数修改 configs/resnet20.yaml 中的 lr 和 weight_decay

# 步骤 4：完整训练（150 epochs，有早停，实际可能提前停止）
python train.py --config configs/resnet20.yaml
```

### 优先级 2：跑 VGG16-BN（需要 GPU，时间较长）

VGG16-BN 有 15.2M 参数，CPU 上每轮约 5 分钟，强烈建议用 GPU。如果时间不够，可以先用 50 epochs 跑一个中间版本。

```bash
# 步骤 1：快速调参（约 1 小时，6 组参数，20 epochs/组）
python tune.py --method vgg16_bn --tune-epochs 20 --trials 6

# 步骤 2：查看调参结果，修改 configs/vgg16_bn.yaml

# 步骤 3：完整训练（150 epochs，建议 GPU）
python train.py --config configs/vgg16_bn.yaml
```

### 优先级 3：补充 k-fold 交叉验证（可选，提升报告质量）

如果时间充裕，可以对最佳参数做 k-fold 验证，获取均值和标准差：

```bash
python train.py --config configs/resnet20.yaml --k-fold 5
```

### 时间不够时的建议

如果时间非常有限：
1. ResNet20 的 `tune.py` 调参必跑（20 分钟能出结果）
2. ResNet20 完整训练必跑（150 epochs，CPU 约 1 小时，GPU 约 15 分钟）
3. VGG16-BN 调到参数后跑 50 epochs 即可（有早停，实际可能 30-40 轮就停了）
4. k-fold 可以省略

## 跑完之后如何更新

### 更新 README.md

1. **方法总览表**：找到 `## 方法介绍` 下的两个表格，填入 Best Val Acc、Best Test Acc、Best Epoch、训练时间
2. **ResNet20 小节**：搜索 `<!-- 实验结果占位：完成训练后请填充`，替换为实际数据表格
3. **VGG16-BN 小节**：同样替换实验结果占位

### 更新结果文件

结果文件会自动写入 `results/<method_name>/`，无需手动操作。训练结束后检查：

```
results/resnet20/
├── accuracy_curve.png
├── confusion_matrix.png
├── loss_curve.png
└── training_log.csv

checkpoints/resnet20/
└── best_model.pth
```

### 提交变更

确认以下文件正确后提交：

- `configs/` — 如修改了默认参数
- `results/*/training_log.csv` 和 `results/*/*.png` — 训练结果
- `checkpoints/*/best_model.pth` — 最佳模型（注意文件较大，考虑用 Git LFS）
- `README.md` — 更新后的实验报告

## 快速参考

### 常用命令

```bash
# 训练
python train.py --config configs/resnet20.yaml
python train.py --config configs/vgg16_bn.yaml

# 覆盖参数
python train.py --config configs/resnet20.yaml --epochs 50 --lr 0.01

# k-fold 交叉验证
python train.py --config configs/resnet20.yaml --k-fold 5

# 调参
python tune.py --method resnet20 --tune-epochs 30 --trials 8
python tune.py --method all --tune-epochs 20 --trials 6

# k-fold 调参（更准但更慢）
python tune.py --method resnet20 --tune-epochs 30 --trials 8 --k-fold 3
```

### 关键文件速查

| 文件 | 用途 |
| --- | --- |
| `configs/simple_cnn.yaml` | SimpleCNN 配置（lr=0.001, Adam, 10 epochs） |
| `configs/resnet20.yaml` | ResNet20 配置（lr=0.1, SGD, 150 epochs） |
| `configs/vgg16_bn.yaml` | VGG16-BN 配置（lr=0.05, SGD, 150 epochs） |
| `model.py` | 模型定义 + MODEL_REGISTRY |
| `train.py` | 训练入口 |
| `tune.py` | 调参脚本 |
| `utils.py` | 工具函数 |
| `HANDOFF.md` | 本文档 |

### 添加新方法的步骤

1. 在 `model.py` 中定义模型类
2. 在 `MODEL_REGISTRY` 中注册
3. 在 `configs/` 中新建 YAML 配置文件（参考已有文件）
4. 运行 `python train.py --config configs/新方法.yaml`
5. 在 `tune.py` 的 `SEARCH_GRIDS` 中添加搜索空间
6. 在 README 的方法介绍中添加小节

### 调参搜索空间（如需自己调整）

在 `tune.py` 的 `SEARCH_GRIDS` 字典中修改，格式：

```python
"method_name": {
    "optimizer.lr": [0.1, 0.01, 0.001],      # 点分隔路径
    "optimizer.name": ["adam", "sgd"],
    "model.params.dropout": [0.3, 0.5, 0.7],
    "optimizer.momentum": ["__dynamic__"],     # 根据 optimizer 自动设置
    "optimizer.weight_decay": ["__dynamic__"], # 同上
    "scheduler.gamma": [0.1, 0.2],
}
```

## 常见问题

**Q: 训练时一直打印 "New best val acc" 但 test acc 不高？**
A: 正常的，模型在持续改进。如果 val acc 不再提升，早停会自动触发停止训练。最终 test acc 会以最佳 val checkpoint 为准。

**Q: 为什么 VGG16 跑这么慢？**
A: 15.2M 参数，比 ResNet20 大 56 倍。CPU 上一轮约 5 分钟，150 epochs 约 12 小时。建议用 GPU。

**Q: 调参脚本跑到一半中断了怎么办？**
A: 已完成的试验结果在 `results/tuning_summary.csv` 中。重新运行 `tune.py --trials` 会再次随机采样，结果可能不同。建议记下已完成的最佳参数。

**Q: 可以同时跑多个方法吗？**
A: 可以，在不同终端窗口中分别运行不同的 `train.py` 命令，它们的结果目录（`checkpoints/` 和 `results/` 子目录）是相互隔离的。
