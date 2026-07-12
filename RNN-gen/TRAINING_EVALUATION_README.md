# RNN 古诗生成：训练与评价工作说明

本文档记录 `HANDOFF.md` 第 7、8、13 节对应的模型训练、自动调参、生成评价和多智能体模拟盲评工作。数据清洗与固定切分由前序数据处理工作提供，本阶段不重新划分数据。

## 1. 工作内容

本阶段完成了以下任务：

- 修复模型中的未来信息泄漏，保证下一字符预测满足严格因果性。
- 在相同数据切分、随机种子和训练策略下训练三种模型。
- 完成 6 组共享超参数搜索，并用验证集选择最佳配置。
- 保存可恢复训练的完整 checkpoint，包括模型、Adam、学习率调度器、epoch 和训练历史。
- 计算 Test Loss、Perplexity、Distinct-1/2、重复率、藏头准确率和藏头完成率。
- 使用固定 prompt 和 3 个生成随机种子，为每个模型生成 27 个评价样本。
- 使用 3 个独立 AI 智能体完成模拟盲评，并计算 ICC 与 Spearman 一致性。
- 生成 JSON、CSV、图表和正式实验报告。

> 多智能体评价属于 `AI multi-rater simulated evaluation`，不能表述为真实真人盲评。

## 2. 数据与控制变量

正式实验统一使用 `data/cleaning_analysis/` 下的文本级固定切分：

| 数据集 | 文件 | 样本数 |
|---|---|---:|
| 训练集 | `peot_cleaned_train.txt` | 47,851 |
| 验证集 | `peot_cleaned_valid.txt` | 5,880 |
| 测试集 | `peot_cleaned_test.txt` | 6,074 |

控制变量如下：

- `split_strategy = text`
- `use_author_weighted_sampling = false`
- 随机种子：`123`
- 最大序列长度：`125`
- 预训练 embedding 维度：`768`
- 最大训练轮数：`20`
- Early Stopping：`patience=5`，`min_delta=1e-4`
- 生成参数：`temperature=0.8`，`top_p=0.9`
- 最佳 epoch 只根据验证集选择，测试集不参与模型选择。

## 3. 正式模型

| 实验 ID | 模型结构 |
|---|---|
| `poetry1_causal` | Embedding → 单层 UniLSTM → Linear |
| `poetry2_causal` | Embedding → 双层 UniLSTM → Linear |
| `poetry3_causal` | Embedding → 单向残差 Encoder/Decoder → 因果 Bahdanau Attention → Linear |

`PoetryModel` 已由双向 LSTM 改为单向 LSTM。`PoetryModel3` 的编码器改为单向结构，并对 attention 使用下三角 mask，使当前位置只能访问当前位置及之前的信息。单元测试验证：修改未来 token 不会改变历史位置的 logits。

## 4. 环境准备

```powershell
pip install -r requirements.txt
```

首次运行预训练 embedding 初始化时，会下载：

```text
KoichiYasuoka/roberta-classical-chinese-base-char
```

正式实验运行设备：

```text
NVIDIA GeForce RTX 5070 Laptop GPU
```

## 5. 自动调参

运行命令：

```powershell
python tune.py --trials 6 --epochs 5
```

调参只读取训练集和验证集，不读取测试指标。候选配置如下：

| Trial | Hidden | Learning Rate | Effective Batch | Best Valid Loss |
|---:|---:|---:|---:|---:|
| 1 | 256 | 0.0005 | 32 | 6.0876 |
| 2 | 256 | 0.0010 | 64 | 6.0065 |
| 3 | 512 | 0.0005 | 64 | 6.0915 |
| 4 | 512 | 0.0010 | 32 | **5.5801** |
| 5 | 512 | 0.0030 | 32 | 5.5807 |
| 6 | 768 | 0.0005 | 32 | 5.5994 |

最终选择 Trial 4：

```text
hidden_dim = 512
learning_rate = 0.001
effective_batch_size = 32
```

完整记录见：

- `results/tuning/trials.csv`
- `results/tuning/best_config.json`

## 6. 正式训练

运行全部正式实验：

```powershell
python experiment.py run-suite --suite handoff_full
```

训练中断后恢复：

```powershell
python experiment.py run-suite --suite handoff_full --resume
```

单独训练一个模型：

```powershell
python experiment.py run --experiment poetry3_causal
```

共享训练策略：

| 参数 | 值 |
|---|---:|
| Label Smoothing | 0.1 |
| Gradient Clipping | 5.0 |
| Weight Decay | 0.0001 |
| LR Scheduler | Cosine Annealing |
| Scheduled Sampling | 1.0 → 0.2 |
| Effective Batch Size | 32 |
| Max Epoch | 20 |

训练程序支持梯度累积。出现 CUDA OOM 时，可降低 micro batch 并增加累积步数，同时保持 effective batch 不变。

完整 checkpoint 包含：

- `model_state_dict`
- `optimizer_state_dict`
- `scheduler_state_dict`
- `epoch` 与 `global_step`
- 最佳验证损失与最佳 epoch
- Early Stopping 状态
- Scheduled Sampling 概率
- 完整训练历史
- CPU/CUDA 随机数状态

checkpoint 文件体积较大，未上传至 GitHub，保存在本地 `checkpoints/`。

## 7. 自动评价

重新加载最佳 checkpoint 进行评价：

```powershell
python experiment.py evaluate --suite handoff_full
```

评价指标：

- Test Loss
- Perplexity：`exp(test_loss)`
- Distinct-1 / Distinct-2
- 二元组重复率
- 藏头准确率
- 藏头完成率

Distinct 和重复率只统计续写 prompt 后的生成内容，并去除空白和标点。

固定 prompt 包含 5 个续写任务和 4 个藏头任务；使用 `123`、`456`、`789` 三个生成种子，因此每个模型生成 27 个样本。

## 8. 多智能体模拟盲评

三个独立 AI 智能体分别从不同角度评价全部 81 个生成样本：

- Rater 1：偏重古典诗词语言与流畅性。
- Rater 2：偏重意境、审美和整体文学效果。
- Rater 3：偏重提示相关性、藏头正确性和结构完整性。

每个智能体都对以下四个维度给出 1–5 分整数评分：

- 流畅性
- 意境
- 相关性
- 结构完整性

盲评输入不包含模型名称、checkpoint、自动指标、旧 AI 分数或模型映射。三个智能体各评价 81 条，共得到 243 条评分。

汇总内容包括：

- 模型与维度均值
- 样本标准差
- 95% 置信区间
- ICC(2,1) 与 ICC(2,k)
- 评价者两两 Spearman 等级相关
- 高分歧样本标记

一致性结果：

| 维度 | ICC(2,1) | ICC(2,k) |
|---|---:|---:|
| 流畅性 | 0.642 | 0.843 |
| 意境 | 0.701 | 0.875 |
| 相关性 | 0.801 | 0.924 |
| 结构完整性 | 0.661 | 0.854 |

两两 Spearman 系数为 `0.857–0.889`，本次没有出现任一维度最大分差达到 3 分的高分歧样本。

## 9. 最终结果

### 9.1 自动指标

| 模型 | Best Epoch | Test Loss | PPL | Distinct-1 | Distinct-2 | 重复率 | 藏头准确率 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `poetry1_causal` | 7 | 5.3340 | 207.27 | 0.5865 | 0.9701 | 0.0089 | 81.25% |
| `poetry2_causal` | 12 | 5.3157 | 203.50 | 0.5795 | 0.9910 | 0.0000 | **95.83%** |
| `poetry3_causal` | 11 | **4.7916** | **120.49** | 0.5760 | 0.9897 | 0.0000 | 91.67% |

### 9.2 三智能体模拟评价

| 模型 | 流畅性 | 意境 | 相关性 | 结构完整性 | 综合分 |
|---|---:|---:|---:|---:|---:|
| `poetry1_causal` | 2.43 ± 0.69 | 2.64 ± 0.76 | 3.22 ± 1.10 | 3.33 ± 1.16 | 2.91 |
| `poetry2_causal` | 2.63 ± 0.80 | 2.75 ± 0.81 | 3.60 ± 1.14 | 3.62 ± 0.97 | 3.15 |
| `poetry3_causal` | **2.86 ± 0.63** | **3.00 ± 0.77** | **3.72 ± 1.00** | **3.85 ± 0.88** | **3.36** |

综合来看：

- `poetry3_causal` 的 Test Loss、PPL 和多智能体综合评分最佳，说明因果 Attention 改进有效。
- `poetry2_causal` 的藏头准确率最高，结构约束执行最稳定。
- `poetry1_causal` 参数结构最简单，但整体指标低于另外两种模型。

## 10. 结果文件

| 路径 | 内容 |
|---|---|
| `results/final_handoff_report.md` | 正式实验报告 |
| `results/poetry*_causal.json` | 参数、训练历史、自动指标和生成记录 |
| `results/poetry*_causal_loss.csv` | 逐 epoch loss |
| `results/loss_comparison.png` | 训练与验证损失曲线 |
| `results/metric_comparison.png` | 生成指标对比图 |
| `results/ai_multi_rater/model_summary.csv` | 三智能体模型评分汇总 |
| `results/ai_multi_rater/agreement_report.json` | ICC 与 Spearman 一致性结果 |
| `results/ai_multi_rater/rater_*_scores.csv` | 三个智能体原始评分 |

## 11. 测试与验收

运行完整测试：

```powershell
python -m pytest -q --basetemp=.pytest_verify_tmp
```

最终验收结果：

```text
62 passed
```

测试覆盖模型因果性、attention mask、自动指标、checkpoint 恢复、梯度累积、调参选择、盲评数据隔离、评分校验、ICC、Spearman 和报告生成。

## 12. 限制

- 正式训练只使用一个训练随机种子，尚未报告跨 seed 方差。
- 多智能体评分只能作为 AI 模拟评价，不能替代真实真人盲评。
- checkpoint 和完整生成文本未上传 GitHub，需要在本地实验目录中查看。
- 自动指标不能完全代表诗歌的文学质量，应结合生成样本和定性评价综合判断。

