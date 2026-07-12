# RNN-gen：基于 PyTorch 的古诗词生成

基于 LSTM Encoder-Decoder + Attention 的古诗词生成，支持续写和藏头诗。

## 快速开始

```bash
pip install -r requirements.txt
python main.py
```

首次运行会自动下载 Classical Chinese RoBERTa 预训练模型（~400MB）。

## 项目结构

```
RNN-gen/
├── config.py                    # 全部训练/推理配置
├── main.py                      # 训练入口 + Trainer
├── model.py                     # PoetryModel / PoetryModel2 / PoetryModel3
├── process.py                   # 数据加载、词表构建、padding
├── utils.py                     # 随机种子、日志、预训练embedding初始化
├── data_cleaning_analysis.py    # 数据清洗、偏见分析、切分
├── requirements.txt             # 全部依赖
├── data/
│   ├── tang/                    # 原始全唐诗 JSON
│   ├── peot.txt                 # 清洗后全量数据
│   └── cleaning_analysis/       # 固定切分 + 统计报告 + 图表
├── checkpoints/                 # 模型保存
└── tests/                       # 单元测试
```

## 三种模型

| 模型 | 架构 |
|---|---|
| **PoetryModel** | Embedding → UniLSTM(1层) → Linear |
| **PoetryModel2** | Embedding → UniLSTM(2层) → Linear |
| **PoetryModel3** | Embedding(预训练) → UniLSTM-Encoder(2层+残差) → UniLSTM-Decoder(2层+残差) → 因果 Bahdanau Attention → Linear |

当前默认使用 PoetryModel3。

## 主要改进

### 模型结构
- Encoder-Decoder 架构：双向编码 + 单向解码
- 每层 LSTM 加残差连接，防止深层退化
- Bahdanau Attention：解码时动态关注编码器各位置

### 预训练 Embedding
- 使用 `KoichiYasuoka/roberta-classical-chinese-base-char` 初始化
- 文言文预训练的 RoBERTa，与唐诗分布匹配

### 训练策略
- Label Smoothing (0.1)
- Gradient Clipping (max_norm=5.0)
- Cosine Annealing LR Scheduler
- Scheduled Sampling (teacher forcing 1.0 → 0.2)
- Early Stopping (patience=5)

### 推理
- Temperature + Top-p (nucleus) sampling
- 支持续写生成和藏头诗两种模式

## 配置说明

`config.py` 关键参数：

```python
# 训练/推理开关
self.do_train = True       # 训练
self.do_test = True        # 测试集评估
self.do_predict = True     # 生成示例
self.do_load_model = False # 是否加载已有checkpoint

# 模型超参
self.embedding_dim = 768   # 预训练embedding维度
self.hidden_dim = 512
self.num_epoch = 20
self.batch_size = 128

# 生成控制
self.temperature = 0.8     # 越小越保守，越大越随机
self.top_p = 0.9           # nucleus sampling 阈值

# 早停
self.early_stopping = True
self.patience = 5
```

## 数据

使用 `data/cleaning_analysis/` 下的固定切分：

| 集合 | 样本数 |
|---:|---:|
| train | 47,851 |
| valid | 5,880 |
| test | 6,074 |

词表只从训练集构建，未见字映射为 UNK。

## 数据清洗

```bash
python data_cleaning_analysis.py
```

输出：清洗后语料、统计报告、可视化图表、固定切分文件。

## 参考

基于 [pytorch-book](https://github.com/chenyuntc/pytorch-book) 第九章改进，主要变更：
- 重构代码架构，Encoder-Decoder + Attention
- 预训练 Classical Chinese RoBERTa embedding
- Scheduled sampling + early stopping
- batch_first、padding不计损失
- 繁简转换、数据清洗、偏见分析

完整文档见 [EXPERIMENT_ENHANCEMENT.md](EXPERIMENT_ENHANCEMENT.md) 和 [HANDOFF.md](HANDOFF.md)。

## 可复现实验

旧的五轮快速实验已归档。HANDOFF 第 7、8、13 节正式实验按以下顺序执行：

```powershell
python tune.py --trials 6 --epochs 5
python experiment.py run-suite --suite handoff_full
python experiment.py ai-score --suite handoff_full
python experiment.py report --suite handoff_full
```

训练中断后，从各实验最后完成的 epoch 恢复：

```powershell
python experiment.py run-suite --suite handoff_full --resume
```

也可以单独运行或重新评价一个模型：

```powershell
python experiment.py run --experiment poetry3_causal
python experiment.py evaluate --suite handoff_full
python experiment.py report --suite handoff_full
```

完整 checkpoint 保存于 `checkpoints/`，JSON、逐 epoch loss、固定 prompt、调参记录、AI 模拟盲评、对比图和最终报告保存于 `results/`。AI 模拟评价不等同于真人盲评。
