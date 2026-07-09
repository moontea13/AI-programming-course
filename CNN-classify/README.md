# 实验一 场景分类实践：CIFAR-10 CNN 多方法对比

本目录是深度学习课程作业的 CIFAR-10 图像分类实验。实验使用 PyTorch 和 torchvision，通过 YAML 配置 + 命令行参数的方式，支持多种 CNN 方法的快速切换与对比，具备验证集划分、早停机制、k-fold 交叉验证和自动化调参功能。

## 实验任务

搭建多种 CNN 结构的图像分类模型，在 CIFAR-10 数据集上进行训练和评估，对比不同方法在分类准确率、收敛速度等方面的表现，为后续调参和消融实验提供基准。

- 数据集：CIFAR-10（10 类，32x32 RGB，训练集 50,000 / 测试集 10,000）
- 评价指标：Training/Validation Loss、Training/Validation Accuracy、Test Accuracy、混淆矩阵
- 输出：训练日志（CSV）、loss/accuracy 曲线、混淆矩阵、最佳模型 checkpoint

## 目录结构

```
CNN-classify/
├── configs/                  # YAML 配置文件（每个方法一个）
│   ├── simple_cnn.yaml       # SimpleCNN 基线
│   ├── resnet20.yaml         # ResNet-20
│   └── vgg16_bn.yaml         # VGG16-BN
├── dataset/                  # 共享数据集目录（所有方法共用）
│   └── cifar-10-batches-py/
├── checkpoints/
│   └── <method_name>/        # 按方法名分目录存放模型
│       └── best_model.pth
├── results/
│   └── <method_name>/        # 按方法名分目录存放结果
│       ├── accuracy_curve.png
│       ├── confusion_matrix.png
│       ├── loss_curve.png
│       └── training_log.csv
├── model.py                  # 所有模型定义 + MODEL_REGISTRY 注册表
├── train.py                  # 训练入口（argparse + YAML 驱动）
├── tune.py                   # 自动化调参脚本
├── utils.py                  # 工具函数（seed、日志、绘图）
├── HANDOFF.md                # 协作交接文档
└── requirements.txt
```

## 使用方式

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行训练

通过 YAML 配置文件驱动所有参数，支持命令行覆盖：

```bash
# 各方法的默认配置训练
python train.py --config configs/simple_cnn.yaml     # SimpleCNN 基线 (10 epochs, Adam)
python train.py --config configs/resnet20.yaml       # ResNet-20 (150 epochs, SGD + StepLR)
python train.py --config configs/vgg16_bn.yaml       # VGG16-BN (150 epochs, SGD + StepLR)

# 命令行覆盖参数（快速调参，不需修改 yaml）
python train.py --config configs/resnet20.yaml --epochs 50 --lr 0.01

# k-fold 交叉验证（用于评估稳定性）
python train.py --config configs/simple_cnn.yaml --k-fold 5
```

### 自动化调参

```bash
# 快速调参（验证集模式，每种方法随机采样 8 组参数）
python tune.py --method simple_cnn --tune-epochs 20 --trials 8

# 全量网格搜索 + 5-fold 交叉验证（更准确但更慢）
python tune.py --method resnet20 --tune-epochs 30 --k-fold 5

# 对所有方法调参
python tune.py --method all --tune-epochs 30 --trials 12

# 调参结果自动保存到 results/tuning_summary.csv
```

调参时会自动为每组参数生成临时配置并运行训练，最终输出按验证集准确率排序的汇总表。

### 切换不同方法

1. 在 `configs/` 下为每个方法新建一个 YAML 配置文件，示例：

```yaml
# configs/my_new_method.yaml
method: my_new_method

model:
  name: MyNewCNN
  params:
    num_classes: 10
    dropout: 0.3

training:
  epochs: 20
  batch_size: 128
  seed: 42
  num_workers: 0
  val_split: 0.1             # 验证集比例（10%）

optimizer:
  name: sgd                   # adam 或 sgd
  lr: 0.1
  momentum: 0.9               # 仅 sgd 有效
  weight_decay: 0.0001        # 仅 sgd 有效

scheduler:
  name: step                  # none 或 step
  step_size: 40
  gamma: 0.1

early_stopping:
  patience: 20                # 连续 N 轮 val_acc 不提升则停止
  min_delta: 0.001            # 最小提升阈值

data:
  dir: dataset
```

2. 在 `model.py` 中定义新模型类，并在 `MODEL_REGISTRY` 中注册：

```python
MODEL_REGISTRY = {
    "SimpleCNN": SimpleCNN,
    "ResNet20": ResNet20,
    "VGG16BN": VGG16BN,
    "MyNewCNN": MyNewCNN,      # 添加新模型
}
```

3. 运行新方法：

```bash
python train.py --config configs/my_new_method.yaml
```

### 配置文件参数说明

| 配置段 | 参数 | 说明 |
| --- | --- | --- |
| `method` | — | 方法名，决定 checkpoints/results 子目录 |
| `model` | `name` | 模型类名，需在 MODEL_REGISTRY 中注册 |
| `model` | `params` | 模型初始化参数（如 num_classes, dropout） |
| `training` | `epochs` | 训练总轮数 |
| `training` | `batch_size` | 批次大小 |
| `training` | `val_split` | 验证集占比（0~1），0 表示不使用验证集 |
| `optimizer` | `name` | 优化器类型：adam 或 sgd |
| `optimizer` | `lr` | 学习率 |
| `optimizer` | `momentum` | SGD 动量（仅 sgd） |
| `optimizer` | `weight_decay` | 权重衰减（仅 sgd） |
| `scheduler` | `name` | 学习率调度器：none 或 step |
| `scheduler` | `step_size` | StepLR 衰减周期 |
| `scheduler` | `gamma` | StepLR 衰减系数 |
| `early_stopping` | `patience` | 早停耐心值（轮数） |
| `early_stopping` | `min_delta` | 最小提升阈值 |
| `data` | `dir` | 数据集根目录（所有方法共享） |

### 命令行参数说明

| 参数 | 说明 |
| --- | --- |
| `--config` | **必填**，YAML 配置文件路径 |
| `--epochs` | 覆盖配置文件中的训练轮数 |
| `--batch-size` | 覆盖配置文件中的 batch size |
| `--lr` | 覆盖配置文件中的学习率 |
| `--seed` | 覆盖配置文件中的随机种子 |
| `--k-fold` | k-fold 交叉验证（如 `--k-fold 5`） |

## 数据集说明

使用 `torchvision.datasets.CIFAR10` 自动下载 CIFAR-10 到 `dataset/` 目录（所有方法共享，该目录不需要提交到仓库）。

数据预处理（所有方法统一）：

- 训练集：`RandomCrop(32, padding=4)` → `RandomHorizontalFlip` → `ToTensor` → `Normalize(mean=(0.4914,0.4822,0.4465), std=(0.2470,0.2435,0.2616))`
- 验证集/测试集：`ToTensor` → 同上 Normalize

> 说明：由于默认官方源在当前网络环境中出现 SSL 握手失败，训练脚本将 CIFAR-10 下载 URL 指向同名数据包镜像 `https://dataset.bj.bcebos.com/cifar/cifar-10-python.tar.gz`。

## 方法介绍

目前共实现 3 种方法：

| 方法 | 模型 | 参数量 | 优化器 | Scheduler | 验证集 | 早停 | 备注 |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| SimpleCNN | 3 Conv + 2 FC | 620K | Adam, lr=0.001 | 无 | 10% | patience=10 | 基线（已完成） |
| ResNet20 | 20 层残差网络 | 272K | SGD, lr=0.1, momentum=0.9, wd=1e-4 | StepLR(40, 0.2) | 10% | patience=20 | GPU 完整训练已完成 |
| VGG16-BN | 13 Conv + 3 FC (含 BN), dropout=0.3 | 15.2M | SGD, lr=0.01, momentum=0.9, wd=5e-4 | StepLR(40, 0.1) | 10% | patience=20 | GPU 完整训练已完成 |

| 方法 | Best Val Acc | Best Test Acc | Best Epoch | 训练时间 | 备注 |
| --- | ---: | ---: | ---: | ---: | --- |
| SimpleCNN | — | **79.05%** | 9 | — | 初版完整训练已完成，未记录验证集 |
| ResNet20 | **90.24%** | **89.91%** | 52 | 25 分 54 秒 | GPU 完整训练，epoch 62 早停 |
| VGG16-BN | **92.46%** | **91.49%** | 86 | 50 分 23 秒 | GPU 完整训练，epoch 91 早停 |

> **结果说明**：ResNet20 和 VGG16-BN 已按 `HANDOFF.md` 的优先级完成调参和 GPU 完整训练；SimpleCNN 为初版结果，未记录验证集指标。

---

### SimpleCNN（基线） ✅ 已完成

> 以下内容来自初版实验文档，由上一批合作者撰写并完成训练。

**模型结构：**

```text
Conv2d(3 -> 32) -> ReLU -> MaxPool
Conv2d(32 -> 64) -> ReLU -> MaxPool
Conv2d(64 -> 128) -> ReLU -> MaxPool
Flatten
Linear(128*4*4 -> 256) -> ReLU -> Dropout
Linear(256 -> 10)
```

三层卷积（通道数 32→64→128）+ 两层全连接，参数量 620K，最后一层输出 10 类 logits。

**设计思路：** 作为最简单的 CNN baseline，使用逐步加深的卷积层提取特征，配合 Dropout 防止过拟合。

**问题分析：**

- 模型结构较浅，特征提取能力有限，在 CIFAR-10 上约 79% 准确率后难以继续提升。
- 缺乏 BatchNorm，深层梯度传播不够稳定。
- 原始版本数据增强仅使用了 RandomHorizontalFlip，当前版本已统一升级为 RandomCrop + RandomHorizontalFlip。

**优势：**

- 结构简单，训练快速，适合作为 baseline 对比。
- 代码清晰，指标完整，结果可复现。

**训练参数：**

- Loss：`CrossEntropyLoss`
- Optimizer：`Adam`，lr = 0.001
- Scheduler：无
- Epochs：10 | Batch size：64
- 数据增强：RandomHorizontalFlip（初版）
- Device：NVIDIA GeForce RTX 4080 Laptop GPU

**实验结果（初版，无验证集划分）：**

| Epoch | Train Loss | Train Acc | Test Loss | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.482842 | 0.45878 | 1.130726 | 0.5959 |
| 2 | 1.081769 | 0.61914 | 0.922611 | 0.6753 |
| 3 | 0.912406 | 0.68440 | 0.832532 | 0.7102 |
| 4 | 0.813898 | 0.71928 | 0.782184 | 0.7334 |
| 5 | 0.748869 | 0.74274 | 0.746464 | 0.7476 |
| 6 | 0.696443 | 0.76148 | 0.669849 | 0.7729 |
| 7 | 0.649500 | 0.77692 | 0.686632 | 0.7647 |
| 8 | 0.608766 | 0.79098 | 0.654972 | 0.7834 |
| 9 | 0.578846 | 0.80070 | 0.633623 | 0.7905 |
| 10 | 0.547311 | 0.81150 | 0.658239 | 0.7796 |

- 最佳 Test Accuracy：**0.7905（epoch 9）**
- 实验环境：PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, NVIDIA GeForce RTX 4080 Laptop GPU

> **注意**：该结果为初版实验产出（数据增强仅含 RandomHorizontalFlip），当前代码已统一增加 RandomCrop(32, padding=4)，重新运行结果可能略有不同。调参脚本 `tune.py` 也支持对 SimpleCNN 进行快速调参（约 3 分钟可完成 8 组参数搜索）。

---

### ResNet20 ✅ 已完成

**模型结构：**

ResNet-20 是 He et al. (2016) 提出的深度残差网络在 CIFAR-10 上的轻量版本。由 1 个初始卷积层 + 3 个 stage（各含 3 个 BasicBlock，共 18 层卷积）+ 1 个全连接层组成，总计 20 层。

```text
Conv2d(3 -> 16, 3x3) -> BN -> ReLU
Stage1: [BasicBlock(16, 16)] x3   (32x32 feature map)
Stage2: [BasicBlock(16, 32)] x3   (16x16, 首块 stride=2)
Stage3: [BasicBlock(32, 64)] x3   (8x8,  首块 stride=2)
AdaptiveAvgPool -> FC(64 -> 10)
```

每个 BasicBlock 由两层 3x3 Conv + BN + ReLU 组成，并包含 identity shortcut 连接。当通道数变化或 stride ≠ 1 时，shortcut 使用 1x1 Conv 对齐维度。

**改进点 / 设计思路：**

- **残差连接**：通过 shortcut 将输入直接加到输出，解决深层网络中的梯度消失问题，使 20 层网络能够有效训练。
- **Batch Normalization**：每层卷积后加入 BN，稳定训练过程中的激活分布，加速收敛。
- **CIFAR-10 适配**：将标准 ResNet 的 7x7 初始卷积替换为 3x3，移除初始 MaxPool，保留更多空间信息以适配 32x32 的小尺寸输入。

**问题分析：**

- 对 CIFAR-10 而言，ResNet-20 的参数量（272K）偏少，表达能力受限于模型容量而非深度，可能需要 ResNet-44 或 ResNet-56 才能充分发挥残差学习的优势。
- SGD 的学习率调度（step decay）需要合理设置 milestones，否则容易欠拟合或震荡。

**优势：**

- 相比 SimpleCNN，在相同训练条件下预期有更好的收敛性和泛化能力。
- 参数量仅 272K（不到 SimpleCNN 的一半），训练和推理效率高，适合快速实验。
- 是 CIFAR-10 上的经典 baseline，便于与其他工作的结果对比。

**训练参数：**

- Loss：`CrossEntropyLoss`
- Optimizer：`SGD`，lr = 0.1，momentum = 0.9，weight_decay = 1e-4
- Scheduler：`StepLR(step_size=40, gamma=0.2)` —— 每 40 个 epoch 将 lr 衰减为原来的 0.2
- Epochs：150 | Batch size：128
- 验证集：10% | 早停：patience=20

**实验结果：**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.625468 | 0.391333 | 2.001701 | 0.4012 |
| 10 | 0.5098 | 0.8242 | 0.5789 | 0.8086 |
| 23 | 0.3577 | 0.8764 | 0.4821 | 0.8450 |
| 40 | 0.2864 | 0.8994 | 0.4339 | 0.8590 |
| 41 | 0.1917 | 0.9342 | 0.3028 | 0.8958 |
| 52 | 0.116390 | 0.959244 | 0.343253 | 0.9024 |
| 62 | 0.1009 | 0.9641 | 0.3633 | 0.9012 |

- 调参最佳组合：lr = 0.1，weight_decay = 1e-4，gamma = 0.2（30 epochs trial 最佳 Val Acc = 0.8490）
- 最佳 Val Accuracy：**0.9024（epoch 52）**
- 对应 Test Accuracy：**0.8991**
- 训练时间：**25 分 54 秒**（GPU：NVIDIA GeForce RTX 5070 Laptop GPU）
- 早停：epoch 62 触发，结果文件见 `results/resnet20/`，最佳模型见 `checkpoints/resnet20/best_model.pth`

> **结果说明**：完整训练使用调参后的 `configs/resnet20.yaml`，训练日志显示 `Using device: cuda`，最佳 checkpoint 元数据为 epoch 52 / best_val_acc 0.9024。

---

### VGG16-BN ✅ 已完成

**模型结构：**

VGG16-BN 是 Simonyan & Zisserman (2015) 提出的经典深度卷积网络，在每层卷积后加入 Batch Normalization。对于 CIFAR-10（32x32 输入），经过 5 次 MaxPool 后特征图缩为 1x1。

```text
Conv2d(3 -> 64, 3x3) -> BN -> ReLU
Conv2d(64 -> 64, 3x3) -> BN -> ReLU -> MaxPool
Conv2d(64 -> 128, 3x3) -> BN -> ReLU
Conv2d(128 -> 128, 3x3) -> BN -> ReLU -> MaxPool
Conv2d(128 -> 256, 3x3) -> BN -> ReLU
Conv2d(256 -> 256, 3x3) -> BN -> ReLU
Conv2d(256 -> 256, 3x3) -> BN -> ReLU -> MaxPool
Conv2d(256 -> 512, 3x3) -> BN -> ReLU
Conv2d(512 -> 512, 3x3) -> BN -> ReLU
Conv2d(512 -> 512, 3x3) -> BN -> ReLU -> MaxPool
Conv2d(512 -> 512, 3x3) -> BN -> ReLU
Conv2d(512 -> 512, 3x3) -> BN -> ReLU
Conv2d(512 -> 512, 3x3) -> BN -> ReLU -> MaxPool
Flatten
FC(512 -> 512) -> ReLU -> Dropout
FC(512 -> 512) -> ReLU -> Dropout
FC(512 -> 10)
```

共 13 层卷积（5 个 stage，通道数 64→128→256→512→512）+ 3 层全连接，每层卷积后含 BatchNorm + ReLU，总参数量约 15.2M。

**改进点 / 设计思路：**

- **深层小卷积核**：全部使用 3x3 卷积核，堆叠多层小卷积核获得与 7x7 卷积核相同的感受野，但参数量更少、非线性更强。
- **Batch Normalization**：在每层卷积后加入 BN，使训练更稳定，允许使用更大的学习率，并起到一定正则化作用。
- **渐进式通道数加倍**：随着空间尺寸减半（MaxPool），通道数翻倍，保持计算量在各 stage 间相对均衡。

**问题分析：**

- 参数量极大（15.2M），是 ResNet-20 的 56 倍，训练和推理速度慢，显存占用高。在 CPU 上单轮训练约需 5 分钟以上，建议使用 GPU。
- 全连接层参数量大（512×512×2 ≈ 0.5M），容易过拟合，Dropout 是必须的。
- 对 CIFAR-10 来说模型容量过剩，可能在训练集上快速达到很高准确率，但泛化能力受限于正则化强度。
- 5 次 MaxPool 后将 32x32 缩到 1x1，刚好匹配，但损失了大量空间信息。

**优势：**

- 结构规整、实现简单，是深度卷积网络设计思路的经典代表。
- BN 的加入显著改善了深层网络的训练稳定性。
- 预期在 CIFAR-10 上可以达到 93%+ 的测试准确率。
- 在深度学习教学中是必学的经典架构，理解它的设计思路对后续学习很有帮助。

**训练参数：**

- Loss：`CrossEntropyLoss`
- Optimizer：`SGD`，lr = 0.01，momentum = 0.9，weight_decay = 5e-4
- Scheduler：`StepLR(step_size=40, gamma=0.1)` —— 每 40 个 epoch 将 lr 衰减为原来的 0.1
- Epochs：150 | Batch size：128
- 验证集：10% | 早停：patience=20 | Dropout：0.3

**实验结果：**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1.956450 | 0.230444 | 1.698643 | 0.3258 |
| 23 | 0.251017 | 0.914200 | 0.370810 | 0.8748 |
| 38 | 0.149480 | 0.949689 | 0.353216 | 0.8882 |
| 41 | 0.076320 | 0.975800 | 0.287851 | 0.9180 |
| 71 | 0.010755 | 0.996489 | 0.373651 | 0.9238 |
| 86 | 0.005670 | 0.998422 | 0.382712 | 0.9246 |
| 91 | 0.0057 | 0.9981 | 0.3877 | 0.9228 |

- 调参最佳组合：lr = 0.01，weight_decay = 5e-4，dropout = 0.3，gamma = 0.1（20 epochs trial 最佳 Val Acc = 0.8636）
- 最佳 Val Accuracy：**0.9246（epoch 86）**
- 对应 Test Accuracy：**0.9149**
- 训练时间：**50 分 23 秒**（GPU：NVIDIA GeForce RTX 5070 Laptop GPU）
- 早停：epoch 91 触发，结果文件见 `results/vgg16_bn/`，最佳模型见 `checkpoints/vgg16_bn/best_model.pth`

> **结果说明**：完整训练使用调参后的 `configs/vgg16_bn.yaml`，训练日志显示 `Using device: cuda`，最佳 checkpoint 元数据为 epoch 86 / best_val_acc 0.9246。

---

## 输出文件说明

每个方法的输出独立存放在 `results/<method_name>/` 和 `checkpoints/<method_name>/` 下：

- `training_log.csv`：每轮训练和验证指标（epoch, train_loss, train_acc, val_loss, val_acc）
- `loss_curve.png`：训练/验证 loss 曲线
- `accuracy_curve.png`：训练/验证 accuracy 曲线
- `confusion_matrix.png`：测试集混淆矩阵（基于最佳验证集 checkpoint）
- `best_model.pth`：验证集 accuracy 最好的模型参数

调参脚本额外输出：

- `results/tuning_summary.csv`：所有调参试验的汇总结果

## 参考链接

- https://www.cnblogs.com/Jerry-Dong/p/8109938.html
- https://zh.d2l.ai/chapter_computer-vision/kaggle-cifar10.html
- https://docs.ultralytics.com/zh/datasets/classify/cifar10/
- https://arxiv.org/abs/1512.03385 (ResNet)
- https://arxiv.org/abs/1409.1556 (VGG)
