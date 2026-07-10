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
