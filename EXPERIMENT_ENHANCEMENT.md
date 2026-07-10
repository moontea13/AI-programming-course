# 原版唐诗生成实验增强说明

## 1. 文档目的

本项目原始目标是使用 PyTorch LSTM 进行字符级唐诗生成。本次改动不改变“唐诗生成”这一任务定义，而是为原版实验补齐可靠的数据处理、实验切分、训练控制和可复现性基础。

本增强方案的主实验逻辑是：

```text
统一的增强数据集
    -> 复现原始 LSTM 基线
    -> 训练改进模型
    -> 使用统一指标评价模型改进
```

数据清洗前后的训练对比可以作为附加消融实验，但不是本项目的主实验目标。主实验中，原始模型和改进模型必须使用相同的清洗后数据、相同的切分和相同的训练配置。

## 2. 原始项目存在的问题

原始项目能完成基本训练和生成，但存在以下影响实验可信度的问题：

| 类别 | 原始问题 | 风险 |
|---|---|---|
| 数据输入 | 直接使用 `data/peot.txt` 或旧缓存 `data/tang.npz` | 无法证明数据来源、清洗规则和切分方式 |
| 数据清洗 | 只做基础括号删除和繁简转换 | 注释、编号、异体字和异常符号可能污染正文 |
| 超长样本 | 长于 `max_len` 的诗在编码时被直接截断 | 模型学习到不完整序列 |
| 数据划分 | `test_data = data[:train_total]` | 训练集和测试集重叠，测试结果无效 |
| 词表构建 | 全量数据直接建词表 | 验证集、测试集信息泄漏进训练词表 |
| 缓存 | 只根据文件是否存在读取 `.npz` | 改变 `max_len` 后可能继续使用旧形状数据 |
| 作者分布 | 高产作者占据大量样本，作者长尾明显 | 模型更偏向高频作者风格 |
| 依赖 | `process.py` 顶层导入 OpenCC | 未安装 OpenCC 时训练入口无法启动 |

## 3. 总体数据管线

```text
data/tang/*.json
    -> data_cleaning_analysis.py
    -> 清洗、繁简转换、去重、长诗句界分片、偏见分析
    -> data/cleaning_analysis/
         -> 文本级 train/valid/test
         -> 作者级 holdout train/valid/test
         -> 作者采样权重
         -> CSV、图表和 Markdown 报告
    -> process.get_split_data()
    -> 数值化缓存 .npz
    -> main.py 训练 / 验证 / 测试
```

## 4. 数据清洗与分析改动

核心实现在 `data_cleaning_analysis.py`。

### 4.1 原始数据溯源与可追溯 metadata

原始 JSON 的每条记录包含：

```text
source_file、id、author、title、paragraphs
```

清洗后生成的 `cleaned_metadata.csv` 保留：

```text
parent_id、training_id、author、title、source_kind、
chunk_index、chunk_count、原始长度、清洗后长度、体裁、切分标签
```

作用：

- 每个训练样本都可追溯到原始诗歌。
- 长诗分片可以定位到父诗和分片编号。
- 被删除样本及删除原因保存在 `dropped_records.csv`，避免黑箱式清洗。

### 4.2 繁简转换

使用 `opencc-python-reimplemented` 的 `t2s` 转换：

```text
風 -> 风
無 -> 无
雲 -> 云
```

作用：

- 减少同义异形字造成的词表稀疏。
- 与简体中文用户输入保持一致。
- 降低模型把繁简字形误当成不同语义 token 的概率。

当前最终清洗报告显示转换器为：

```text
converter = opencc
```

### 4.3 编辑注释、异常符号和编号清理

清洗规则删除或规范化：

```text
（...）、(...)、{...}、[...]、【...】、〖...〗、〔...〕、〈...〉、《...》
数字、横线、空白符、半角标点
```

只保留：

```text
CJK 汉字（含 Extension A、兼容汉字及扩展区）
中文标点：，。！？；：
```

作用：

- 去除校勘说明、章节编号、书名和编辑符号。
- 避免 `〖一〗` 这类标记在外层括号被删除后，内部“一”误进入诗歌正文。
- 保留 `䍦`、`䙰` 等合法古汉字，避免误当乱码删除。

最终统计：

```text
移除的非正文符号类型：48
移除的非正文符号总次数：59916
```

### 4.4 精确去重

清洗并完成繁简统一后，使用正文精确匹配去重。

作用：

- 减少重复文本被反复训练。
- 降低同一首诗进入不同数据集合造成的数据泄漏风险。

当前删除的完全重复文本数：

```text
875
```

说明：近重复文本尚未自动删除。不同版本唐诗只差少数字时可能具有文献价值，因此近重复更适合先标记、后人工复核。

### 4.5 长诗句界分片回收

原始项目会对超出 `max_len` 的诗直接截断。增强方案中，超过 123 个汉字的诗优先按：

```text
。！？；
```

拆分为不超过上限的片段，不截断单句本身。

每个分片保留：

```text
parent_id
training_id
chunk_index
chunk_count
source_kind = long_poem_chunk
```

同一原诗的所有分片使用父诗 id 进行数据划分，因此不会跨训练、验证和测试集合。

实际结果：

| 指标 | 数值 |
|---|---:|
| 原始超长诗记录数 | 2475 |
| 成功回收的超长诗记录数 | 2474 |
| 新增长诗训练分片数 | 6088 |
| 无法按句界回收的超长句数 | 1 |
| 过短尾部分片数 | 2 |

作用：

- 避免截断造成不完整训练目标。
- 保留古风、排律等长诗中的局部语言结构。
- 将原诗记录保留率提升到 97.53%。

副作用：

- 分片会削弱跨片段的长程上下文。
- 不能把长诗分片训练的效果解释为完整长诗生成能力。

### 4.6 作者、长度、体裁和主题偏见分析

脚本输出：

```text
author_top20.csv
author_tail_stats.csv
length_distribution.csv
form_distribution.csv
fine_form_distribution.csv
title_keyword_distribution.csv
char_frequency_top100.csv
```

当前主要发现：

| 指标 | 数值 |
|---|---:|
| 作者总数 | 3597 |
| 只出现 1 首诗的作者数 | 1865 |
| 单样本作者占作者总数比例 | 51.85% |
| Top 10 作者样本占比 | 20.62% |
| Top 20 作者样本占比 | 30.68% |
| 不详/无名作者样本占比 | 2.86% |
| 五言样本数 | 30684 |
| 七言样本数 | 22106 |

解释：

- 作者分布具有明显长尾，高频作者的语言风格更容易被模型学习。
- 五言、七言和中短篇诗歌占主导，模型对其他体裁和长篇结构的覆盖较弱。
- 标题关键词主要集中在山水、赠寄酬答、季节、送别、夜月、宗教等传统主题。

这些结论用于解释模型生成结果的覆盖范围，不应夸大为模型必然“模仿某一个作者”。

### 4.7 可视化

自动生成图表：

```text
data/cleaning_analysis/figures/author_top15.png
data/cleaning_analysis/figures/length_distribution.png
data/cleaning_analysis/figures/fine_form_distribution.png
data/cleaning_analysis/figures/vocabulary_before_after.png
```

作用：

- 为报告和答辩提供直观证据。
- 展示作者长尾、长度结构、体裁比例和清洗前后词表变化。

## 5. 数据切分与泄漏控制

### 5.1 文本级固定切分

默认实验使用父诗 id 的 MD5 哈希进行确定性切分：

```text
train: 47851
valid: 5880
test: 6074
```

作用：

- 切分结果可复现。
- 同一原诗的所有长诗分片固定落在同一个集合。
- 适合评估模型在同一唐诗数据分布中的未见文本预测能力。

### 5.2 作者级 Holdout 切分

新增严格的作者级划分：

```text
train: 48636
valid: 4351
test: 6818
```

规则：

- 已知作者：同一作者的全部诗固定放入一个集合。
- 不详、无名氏等元数据不可靠的作者：按原诗 id 哈希，避免把不相关诗歌当作同一作者。

验证结果：

```text
已知作者 train-valid 重叠：0
已知作者 train-test 重叠：0
已知作者 valid-test 重叠：0
```

作用：

- 评估模型对未见作者风格的泛化能力。
- 不应直接与文本级切分的困惑度横向比较，因为两者测试分布不同。

## 6. 作者再平衡采样

作者长尾不均衡不能通过简单的 `1 / count` 权重解决，因为单样本作者可能被极端重复采样。

本项目使用：

```text
raw_weight = 1 / sqrt(author_train_sample_count)
clipped_weight = clip(raw_weight, 0.05, 0.50)
sample_weight = clipped_weight / training_sample_mean(clipped_weight)
```

输出：

```text
author_weights.csv
train_sample_weights.csv
author_holdout_weights.csv
author_holdout_train_sample_weights.csv
```

默认文本级切分的权重范围：

```text
最小样本权重：0.3892
最大样本权重：3.8919
最大/最小比：10.0
```

`main.py` 使用 `WeightedRandomSampler` 支持该策略。默认关闭，便于保留普通随机采样的基线：

```python
config.use_author_weighted_sampling = False
```

启用作者再平衡：

```python
config.use_author_weighted_sampling = True
```

作者再平衡应作为扩展或消融实验。主实验中，不应预先假设加权采样一定带来更低困惑度或更好生成质量。

## 7. 训练数据加载和缓存改动

核心实现在 `process.py`。

### 7.1 训练集独立构建词表

`get_split_data()`：

1. 读取当前切分策略下的 train/valid/test 文本。
2. 仅用训练集建立 `word2idx`。
3. 验证集、测试集中的未见字映射为 `UNK`。
4. 添加 `SOP`、`EOP`。
5. 使用 post-padding 编码为统一长度矩阵。

作用：

- 避免验证集、测试集字符提前进入训练词表。
- 保持训练、验证和测试的边界清晰。

### 7.2 缓存一致性

数值化缓存 `.npz` 现在保存：

```text
cache_version
max_len
train_data / valid_data / test_data
word2idx / idx2word
```

缓存读取时同时检查：

```text
源文本修改时间
cache_version
config.max_len
```

作用：

- 数据文件变更时自动重建缓存。
- 将 `max_len` 从 125 改为 100 时，不会误用旧的 `(N, 125)` 数据。

### 7.3 OpenCC 依赖隔离

`process.py` 不再顶层导入 OpenCC。OpenCC 只在历史 `_parseRawData()` 函数被调用时导入。

作用：

- 训练主流程读取已经生成的清洗数据时，不依赖 OpenCC。
- 保留旧函数作为历史基线，并通过 `DEPRECATED` 注释明确当前主流程不使用它。

## 8. 训练入口改动

核心实现在 `main.py`。

### 8.1 修复原始测试集 Bug

原始代码：

```python
train_data = data[:train_total]
test_data = data[:train_total]
```

修复为：

```python
test_data = data[train_total:]
```

实际主流程现在不再依赖这个随机切分函数，而是使用数据清洗阶段生成的固定切分。

### 8.2 训练、验证、测试职责分离

当前流程：

```text
train: 参数更新
valid: 每个 epoch 选择最优 checkpoint
test: 训练结束后仅做最终评价
```

作用：

- 不再用测试集选择最佳模型。
- 避免在训练过程中反复查看测试结果导致测试集过拟合。

### 8.3 训练配置改进

已加入：

```text
Adam 的 weight_decay
torch.load(..., map_location=device)
测试损失按 batch 平均
WeightedRandomSampler 可选接入
```

另外，生成函数由错误的全局 `model` 改为 `self.model`，使 `Trainer` 可以被独立实例化使用。

## 9. 与训练实验的衔接

本文件说明数据处理的技术设计和实际结论。模型、训练和评价组员应使用 [HANDOFF.md](HANDOFF.md) 中的交接流程，其中包括：

```text
主实验数据与配置
文本级和作者级 holdout 的适用范围
作者加权采样的启用方式
必须保持一致的控制变量
训练和评价结果的回传格式
禁止操作列表
```

## 10. 依赖、测试和复现

安装数据处理依赖：

```powershell
pip install -r requirements_data.txt
```

生成全部清洗产物：

```powershell
python data_cleaning_analysis.py
```

运行单元测试：

```powershell
python -m unittest discover -s tests -v
```

当前测试覆盖：

```text
括号注释清理
编辑性 〖〗/〈〉/〔〕 清理
标点规范化
CJK 扩展汉字保留
体裁判断
长诗句界分片
文本切分确定性
作者级切分一致性
温和作者权重上限
max_len 缓存失效
```

## 11. 当前完成状态

已完成：

```text
数据清洗与可追溯记录
偏见诊断与可视化
长诗回收
数据泄漏控制
作者再平衡训练接口
训练数据和缓存接入
单元测试
```

数据处理部分已经可以作为增强实验的统一底座。后续模型、训练和评价任务及对应交付物见 [HANDOFF.md](HANDOFF.md)。
