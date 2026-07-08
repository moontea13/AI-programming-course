# 实验一 场景分类实践：CIFAR-10 CNN Baseline

本目录是深度学习课程作业的 CIFAR-10 图像分类 baseline 实验。实验使用 PyTorch 和 torchvision 搭建一个简单 CNN，完成数据下载、模型训练、测试集评估、曲线绘制、混淆矩阵生成和最佳模型保存。

## 实验目的

搭建一个结构简单、指标完整、结果可复现的 CNN baseline，明确 CIFAR-10 数据划分、训练参数、评价指标和输出文件，为后续调参、消融实验和多方法对比提供基准结果。

## 安装依赖

```powershell
pip install -r requirements.txt
```

如果本机已经安装 PyTorch 和 torchvision，可以直接运行训练脚本。

## 运行训练

Smoke test：

```powershell
python train.py --epochs 3
```

正式 baseline：

```powershell
python train.py --epochs 10
```

脚本会自动选择设备：CUDA 可用时使用 GPU，否则使用 CPU。

## 数据集说明

使用 `torchvision.datasets.CIFAR10` 自动下载 CIFAR-10：

- `train=True`：训练集，50,000 张图片
- `train=False`：测试集，10,000 张图片
- 图片大小：32 x 32，RGB 三通道
- 类别数：10 类

数据预处理：

- 训练集：`RandomHorizontalFlip`、`ToTensor`、CIFAR-10 常用 `Normalize`
- 测试集：`ToTensor`、CIFAR-10 常用 `Normalize`

数据会下载到 `data/`，该目录不需要提交到仓库。

说明：训练脚本仍使用 `torchvision.datasets.CIFAR10` 读取和校验数据；由于默认官方源在当前网络环境中出现 SSL 握手失败，脚本将 CIFAR-10 下载 URL 指向同名数据包镜像。

## 模型结构

模型文件：`model.py`

```text
Conv2d(3 -> 32) -> ReLU -> MaxPool
Conv2d(32 -> 64) -> ReLU -> MaxPool
Conv2d(64 -> 128) -> ReLU -> MaxPool
Flatten
Linear(128*4*4 -> 256) -> ReLU -> Dropout
Linear(256 -> 10)
```

## 训练参数

- Loss：`CrossEntropyLoss`
- Optimizer：`Adam`
- Learning rate：`0.001`
- Batch size：`64`
- Epoch：先 3 轮 smoke test，再 10 轮 baseline
- Device：自动选择 `cuda` 或 `cpu`

## 评价指标

每个 epoch 输出：

- Train Loss
- Train Accuracy
- Test Loss
- Test Accuracy

输出格式：

```text
Epoch [x/y] Train Loss, Train Acc, Test Loss, Test Acc
```

## 实验结果

实验环境：

- PyTorch：`2.11.0+cu128`
- torchvision：`0.26.0+cu128`
- CUDA：可用
- GPU：`NVIDIA GeForce RTX 4080 Laptop GPU`
- 命令：`python train.py --epochs 10`
- 最佳测试集准确率：`0.7905`，即 `79.05%`
- 最佳 epoch：第 `9` 轮

10 轮训练日志：

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

## 输出文件说明

- `results/training_log.csv`：每轮训练和测试指标
- `results/loss_curve.png`：训练/测试 loss 曲线
- `results/accuracy_curve.png`：训练/测试 accuracy 曲线
- `results/confusion_matrix.png`：测试集混淆矩阵
- `checkpoints/best_model.pth`：测试集 accuracy 最好的模型参数

## 后续调参方向

- 学习率对比：`0.001`、`0.0005`、`0.0001`
- Dropout 对比：`0.3`、`0.5`
- 优化器对比：`Adam` 与 `SGD + momentum`
- 数据增强对比：加入 `RandomCrop(32, padding=4)`
- 模型结构对比：在卷积层后加入 `BatchNorm2d`

## 参考链接

- https://www.cnblogs.com/Jerry-Dong/p/8109938.html
- https://zh.d2l.ai/chapter_computer-vision/kaggle-cifar10.html
- https://docs.ultralytics.com/zh/datasets/classify/cifar10/
