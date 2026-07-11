# 项目交接说明

## 1. 交接目的

本文件用于将已经完成的数据清洗、切分、缓存和采样能力交接给负责模型改进、训练和评价的组员。

你们后续的主实验应在同一份增强数据上比较：

```text
原始 LSTM 基线
vs
改进模型
```

不要将清洗前后的数据训练差异作为主实验。数据清洗是整个增强实验的统一底座，所有主实验必须使用相同的清洗后数据和相同的数据切分。

## 2. 开始前操作

安装数据处理依赖：

```powershell
pip install -r requirements_data.txt
```

如需重新生成清洗数据、图表、切分和权重：

```powershell
python data_cleaning_analysis.py
```

生成结果统一位于：

```text
data/cleaning_analysis/
```

不要手动编辑该目录中的训练文本、CSV 或 `.npz` 缓存。修改清洗规则后，应重新运行 `data_cleaning_analysis.py`。

## 3. 主实验使用的数据

主实验默认使用文本级固定切分：

```text
data/cleaning_analysis/peot_cleaned_train.txt
data/cleaning_analysis/peot_cleaned_valid.txt
data/cleaning_analysis/peot_cleaned_test.txt
```

当前样本数量：

| 集合 | 样本数 |
|---|---:|
| train | 47851 |
| valid | 5880 |
| test | 6074 |

`main.py` 已经通过 `process.get_split_data()` 自动读取这些文件，不需要再调用旧的随机切分函数。

## 4. 主实验配置

首次训练原始 LSTM 基线或改进模型时，在 `config.py` 中设置：

```python
self.split_strategy = 'text'
self.use_author_weighted_sampling = False

self.do_train = True
self.do_test = True
self.do_predict = False
self.do_load_model = False
```

训练完成后，如需加载最佳 checkpoint 做生成或测试：

```python
self.do_train = False
self.do_test = True
self.do_predict = True
self.do_load_model = True
```

主实验控制变量必须一致：

```text
训练/验证/测试数据
随机种子
epoch 数
batch size
学习率
词表与 max_len
生成策略
评价指标实现
```

允许变化的变量应只包括模型结构或明确列出的训练策略，例如 LSTM 层数、dropout、梯度裁剪、采样策略等。

## 5. 数据加载接口

训练入口已经完成以下逻辑：

```text
训练集建立词表
验证集和测试集未见字映射为 UNK
添加 SOP / EOP
post-padding 到 config.max_len
缓存保存 max_len 和版本号
```

因此，不要在模型代码中重新建立词表或重新划分数据。

若修改：

```python
config.max_len
```

加载器会自动检测缓存元数据并重建对应 `.npz` 文件。

## 6. 可选扩展实验

### 6.1 作者再平衡采样

用于考察作者长尾分布是否影响模型训练。

配置：

```python
self.split_strategy = 'text'
self.use_author_weighted_sampling = True
```

训练端会自动读取：

```text
data/cleaning_analysis/train_sample_weights.csv
```

并使用 `WeightedRandomSampler`。

该实验应与普通随机采样作为消融对照，不应替代主实验。

### 6.2 作者级 Holdout

用于考察模型对未见作者风格的泛化，而不是同分布文本预测能力。

配置：

```python
self.split_strategy = 'author_holdout'
self.use_author_weighted_sampling = False
```

对应数据规模：

| 集合 | 样本数 |
|---|---:|
| train | 48636 |
| valid | 4351 |
| test | 6818 |

已验证已知作者在 train、valid、test 之间的重叠数均为 `0`。

注意：作者级 holdout 的 Perplexity/Loss 不应直接与文本级切分的结果横向比较，因为测试分布不同。

## 7. 训练同学需要回传的结果

每个模型实验至少回传：

```text
模型名称和结构
数据切分策略
是否启用作者加权采样
随机种子
全部超参数
每个 epoch 的 train loss 和 valid loss
最佳 epoch
最终 test loss
Perplexity
训练耗时和运行设备
模型 checkpoint 路径
若干固定 prompt 的生成结果
```

建议保存为：

```text
checkpoints/<experiment_name>.pt
results/<experiment_name>.json
results/<experiment_name>_loss.csv
```

## 8. 评价同学需要使用的输入

评价同学应接收每个模型在同一组固定 prompt 下的生成结果，并计算统一指标：

```text
Test Loss
Perplexity
Distinct-1
Distinct-2
重复率
藏头准确率（如启用藏头生成）
人工评价：流畅性、意境、相关性、结构完整性
```

评价函数和人工评价标准必须对所有模型保持一致。

## 9. 不应执行的操作

```text
不要重新随机划分训练、验证和测试数据。
不要使用旧的 data/peot.txt 或 data/tang.npz 作为主实验数据。
不要用测试集选择最佳 epoch。
不要用验证集或测试集重新建词表。
不要手动修改 train_sample_weights.csv 的行顺序。
不要把作者级 holdout 与文本级切分结果直接当成同一难度比较。
```

## 10. 数据处理部分已提供的产物

| 文件或目录 | 用途 |
|---|---|
| `bias_report.md` | 数据清洗、偏见分析和统计结论 |
| `cleaned_metadata.csv` | 训练样本与原始诗歌的可追溯 metadata |
| `dropped_records.csv` | 被删除样本/片段和删除原因 |
| `peot_cleaned_*.txt` | 文本级主实验数据 |
| `peot_author_holdout_*.txt` | 作者级泛化实验数据 |
| `train_sample_weights.csv` | 文本级作者再平衡权重 |
| `author_holdout_train_sample_weights.csv` | 作者级 holdout 权重 |
| `tang_cleaned_splits.npz` | 文本级数值化缓存 |
| `tang_author_holdout_splits.npz` | 作者级数值化缓存 |
| `figures/` | 报告与答辩可用图表 |

## 11. 当前分工边界

数据处理部分已经完成：

```text
清洗、繁简转换、异常标记清理、扩展古汉字保留
精确去重、长诗句界分片、偏见统计、可视化
文本级和作者级数据切分、作者采样权重
词表泄漏控制、缓存一致性、训练入口接入、单元测试
```

后续仍需要模型、训练和评价部分完成：

```text
复现原始 LSTM 基线
实现并训练改进模型
在相同增强数据上进行公平比较
完成自动指标和人工评价
将实验结果合并进最终报告
```

详细的数据处理设计、实际统计和改动原因见 [EXPERIMENT_ENHANCEMENT.md](EXPERIMENT_ENHANCEMENT.md)。

---

## 12. 模型架构改进（第二轮交接）

### 12.1 三种可用模型

| 模型 | 文件 | 架构 |
|---|---|---|
| **PoetryModel** | `model.py:7` | Embedding → BiLSTM(1层, bidir) → Linear |
| **PoetryModel2** | `model.py:32` | Embedding → UniLSTM(2层) → Linear |
| **PoetryModel3** | `model.py:53` | Embedding(预训练) → BiLSTM-Encoder(2层+残差) → UniLSTM-Decoder(2层+残差) → Bahdanau Attention → Linear |

### 12.2 PoetryModel3 详细架构

```
Input tokens
    │
Embedding (vocab_size → 768, Classical Chinese RoBERTa 预训练)
    │
┌─ Encoder ─────────────────────────────────────────┐
│  BiLSTM Layer1 (768→512, bidir)  + Linear残差      │
│  BiLSTM Layer2 (1024→512, bidir) + Linear残差      │
└──────────────────────────────────────────────────┘
    │  enc_out: (batch, seq_len, 1024)
┌─ Decoder ─────────────────────────────────────────┐
│  UniLSTM Layer1 (1024→512)        + Linear残差     │
│  UniLSTM Layer2 (512→512)         + 恒等残差      │
└──────────────────────────────────────────────────┘
    │  dec_out: (batch, seq_len, 512)
┌─ Attention ───────────────────────────────────────┐
│  Bahdanau Additive: attn(dec_out, enc_out)        │
│  Context = softmax(W·tanh([dec; enc])) · enc_out  │
└──────────────────────────────────────────────────┘
    │  combined: (batch, seq_len, 512+1024)
Linear → vocab_size
```

### 12.3 预训练 Embedding

使用 HuggingFace 模型 `KoichiYasuoka/roberta-classical-chinese-base-char` 初始化 Embedding 层：
- 古典中文/文言文预训练的 RoBERTa，与唐诗数据分布匹配
- Embedding 维度 768，首次运行时自动下载（~400MB）
- 仅在 `config.do_load_model = False`（全新训练）时加载
- 训练中 embedding 不冻结，继续微调

### 12.4 训练改进

| 改进项 | 参数 | 位置 |
|---|---|---|
| Label Smoothing | 0.1 | `config.py` |
| Gradient Clipping | max_norm=5.0 | `config.py` |
| Cosine LR Scheduler | T_max=总步数 | `config.py` |
| Scheduled Sampling | 1.0 → 0.2 线性衰减 | `config.py` |
| Early Stopping | patience=5, min_delta=1e-4 | `config.py`, `main.py:70-80` |
| MPS 自动检测 | Apple Silicon GPU | `main.py` |

### 12.5 推理改进

- **Temperature sampling**: `config.temperature = 0.8`，控制生成多样性
- **Top-p (nucleus) sampling**: `config.top_p = 0.9`，只从累积概率前 90% 的 token 中采样
- 实现位置：`main.py` 的 `_sample_token()` 方法

### 12.6 新增文件

| 文件 | 用途 |
|---|---|
| `requirements.txt` | 全部依赖（torch, transformers, matplotlib 等） |
| `checkpoints/` | 模型保存目录（当前仅有占位文件） |

---

## 13. 下一轮任务：评价与调参（第三轮交接）

### 13.1 待完成工作

**评价指标选取与实现：**

需要在 `main.py` 中新增评价模块，至少包含：

| 指标 | 说明 |
|---|---|
| **Test Loss / Perplexity** | 已有 test() 计算 loss，PPL = exp(loss) |
| **Distinct-1 / Distinct-2** | unigram/bigram 去重率，衡量生成多样性 |
| **重复率** | 生成结果中重复 n-gram 的比例 |
| **藏头准确率** | 藏头诗中藏头字是否正确出现在句首 |
| **人工评价维度** | 流畅性、意境、相关性、结构完整性（需设计评分表） |

**自动化调参：**

建议新增 `tune.py`，对关键超参数做 grid search 或 Bayesian optimization（如 Optuna）：

```python
# 建议搜索空间
param_space = {
    'hidden_dim': [256, 512, 768],
    'num_epoch': [15, 20, 30],
    'lr': [5e-4, 1e-3, 3e-3],
    'batch_size': [64, 128],
    'temperature': [0.6, 0.8, 1.0],
    'top_p': [0.8, 0.9, 0.95],
}
```

**三个模型完整训练与对比：**

对 `PoetryModel`、`PoetryModel2`、`PoetryModel3` 分别训练并记录：

```text
模型名称和结构描述
全部超参数
每个 epoch 的 train loss 和 valid loss
最佳 epoch 的 test loss / Perplexity
Distinct-1 / Distinct-2
重复率
藏头准确率（使用固定 prompt 列表）
训练耗时和设备
若干固定 prompt 的生成结果（续写 + 藏头诗）
```

### 13.2 如何切换模型

在 `config.py` 中新增 `model_type`：

```python
self.model_type = 'poetry3'  # 'poetry1' / 'poetry2' / 'poetry3'
```

然后在 `main.py` 中根据该配置选择模型即可（建议实现工厂函数）。

### 13.3 结果存放约定

```text
checkpoints/<model_name>_<experiment_id>.pt      # 模型权重
results/<model_name>_<experiment_id>.json         # 超参数 + 指标
results/<model_name>_<experiment_id>_loss.csv     # 训练 loss 曲线
results/<model_name>_<experiment_id>_generated.txt # 固定 prompt 生成结果
```

### 13.4 固定 Prompt 列表（建议）

续写生成：
```text
丽日照残春
春风得意马蹄疾
大漠孤烟直
两个黄鹂鸣翠柳
床前明月光
```

藏头诗：
```text
深度学习
人工智能
春暖花开
千古风流
```

### 13.5 注意事项

- 所有模型必须使用**同一份数据切分**（`peot_cleaned_*.txt`），不要重新划分
- 控制变量：随机种子固定为 `123`、相同 batch_size、相同 max_len
- PoetryModel3 训练前会自动下载 RoBERTa 预训练模型（首次需联网）
- 不要用测试集选择最佳 epoch，用验证集
- 横向比较必须在同一个 split_strategy 下进行
- 当前 `config.py` 中 `do_train=True`，跑 `python main.py` 即可开始训练 PoetryModel3
