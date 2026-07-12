# HANDOFF 第 7、8、13 节正式实验报告

> 三种正式模型均使用严格因果结构、相同固定切分和共享调参配置。

## 自动评价结果

| 实验 | 最佳 epoch | Valid Loss | Test Loss | PPL | Distinct-1 | Distinct-2 | 重复率 | 藏头准确率 | 藏头完成率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| poetry1_causal | 7 | 5.3183 | 5.3340 | 207.2699 | 0.5865 | 0.9701 | 0.0089 | 0.8125 | 0.8125 |
| poetry2_causal | 12 | 5.2979 | 5.3157 | 203.4984 | 0.5795 | 0.9910 | 0.0000 | 0.9583 | 0.9583 |
| poetry3_causal | 11 | 4.7712 | 4.7916 | 120.4890 | 0.5760 | 0.9897 | 0.0000 | 0.9167 | 0.9167 |

## 模型结构与共享超参数

| 实验 | 模型结构 | hidden | lr | effective batch | seed | max epochs |
|---|---|---:|---:|---:|---:|---:|
| poetry1_causal | Embedding -> UniLSTM(1 layer) -> Linear | 512 | 0.001 | 32 | 123 | 20 |
| poetry2_causal | Embedding -> UniLSTM(2 layers) -> Linear | 512 | 0.001 | 32 | 123 | 20 |
| poetry3_causal | Embedding -> causal residual Encoder/Decoder -> masked Bahdanau Attention -> Linear | 512 | 0.001 | 32 | 123 | 20 |

## 自动调参

- 选择 trial：4
- 最佳验证损失：5.5801
- 共享配置：hidden_dim=512, lr=0.001, effective_batch_size=32

| 调参试验 | hidden | lr | effective batch | 最佳 epoch | Best Valid Loss | 耗时（秒） |
|---|---:|---:|---:|---:|---:|---:|
| trial 1 | 256 | 0.0005 | 32 | 3 | 6.0876 | 229.6 |
| trial 2 | 256 | 0.001 | 64 | 4 | 6.0065 | 212.4 |
| trial 3 | 512 | 0.0005 | 64 | 4 | 6.0915 | 311.1 |
| trial 4 | 512 | 0.001 | 32 | 4 | 5.5801 | 325.2 |
| trial 5 | 512 | 0.003 | 32 | 4 | 5.5807 | 324.8 |
| trial 6 | 768 | 0.0005 | 32 | 4 | 5.5994 | 494.2 |

## 全部共享超参数

```json
{
  "recipe": "handoff_full",
  "embedding_dim": 768,
  "hidden_dim": 512,
  "use_pretrained_embeddings": true,
  "label_smoothing": 0.1,
  "grad_clip": 5.0,
  "lr_scheduler": "cosine",
  "scheduled_sampling": true,
  "split_strategy": "text",
  "use_author_weighted_sampling": false,
  "seed": 123,
  "num_epoch": 20,
  "batch_size": 32,
  "effective_batch_size": 32,
  "gradient_accumulation_steps": 1,
  "lr": 0.001,
  "weight_decay": 0.0001,
  "max_len": 125,
  "max_gen_len": 200,
  "temperature": 0.8,
  "top_p": 0.9,
  "ss_start_prob": 1.0,
  "ss_end_prob": 0.2,
  "early_stopping": true,
  "patience": 5,
  "min_delta": 0.0001
}
```

## 训练耗时与设备

| 实验 | 训练耗时（分钟） | 设备 |
|---|---:|---|
| poetry1_causal | 11.27 | NVIDIA GeForce RTX 5070 Laptop GPU |
| poetry2_causal | 19.17 | NVIDIA GeForce RTX 5070 Laptop GPU |
| poetry3_causal | 55.79 | NVIDIA GeForce RTX 5070 Laptop GPU |

## 三智能体模拟盲评

| 实验 | 流畅性 | 意境 | 相关性 | 结构完整性 | 平均分 | 高分歧样本 |
|---|---:|---:|---:|---:|---:|---:|
| poetry1_causal | 2.43 ± 0.69 | 2.64 ± 0.76 | 3.22 ± 1.10 | 3.33 ± 1.16 | 2.91 | 0 |
| poetry2_causal | 2.63 ± 0.80 | 2.75 ± 0.81 | 3.60 ± 1.14 | 3.62 ± 0.97 | 3.15 | 0 |
| poetry3_causal | 2.86 ± 0.63 | 3.00 ± 0.77 | 3.72 ± 1.00 | 3.85 ± 0.88 | 3.36 | 0 |

三智能体平均分最高的是 `poetry3_causal`（3.36/5）。

> evaluator_type=`AI multi-rater simulated evaluation`；三个智能体独立盲评，但不是真人盲评。

### 评价者一致性

| 维度 | ICC(2,1) | ICC(2,k) | 结论 |
|---|---:|---:|---|
| fluency | 0.642 | 0.843 | 可用于辅助比较 |
| imagery | 0.701 | 0.875 | 可用于辅助比较 |
| relevance | 0.801 | 0.924 | 可用于辅助比较 |
| structure | 0.661 | 0.854 | 可用于辅助比较 |

两两 Spearman 等级相关：
- `rater_01__rater_02`: 0.857
- `rater_01__rater_03`: 0.889
- `rater_02__rater_03`: 0.886

## 附录：单一启发式 AI 评价

| 实验 | 流畅性 | 意境 | 相关性 | 结构完整性 | 平均分 |
|---|---:|---:|---:|---:|---:|
| poetry1_causal | 4.48 | 2.81 | 4.67 | 4.11 | 4.02 |
| poetry2_causal | 4.19 | 2.89 | 4.93 | 4.33 | 4.08 |
| poetry3_causal | 3.78 | 2.81 | 4.85 | 4.48 | 3.98 |

当前 AI 定性平均分最高的是 `poetry2_causal`（4.08/5）。

> evaluator_type=`AI simulated evaluation`；评分时隐藏模型名称，但不是真人盲评。

## 实验口径

- 三个正式实验共享固定切分、随机种子、训练策略和调参选出的超参数，仅改变模型结构。
- 调参仅使用训练集和验证集；测试集只在载入最佳 checkpoint 后评估一次。
- 旧五轮泄漏实验完整归档于 `results/archive/quick_leaky_suite/` 和 `checkpoints/archive/quick_leaky_suite/`，不参与正式排名。

## 结果解释限制

- 正式模型已通过未来 token 不改变历史 logits 的因果性测试。
- AI 模拟评分用于统一辅助比较，不能替代真实人工评价。
- 生成质量应结合固定 prompt、Distinct、重复率、藏头准确率和定性评价判断，不能仅按 PPL 排名。
- 单训练 seed 的结果不包含跨 seed 方差。

## Checkpoint 路径

- `poetry1_causal`: best=`D:\temp\4\AI-programming-course\RNN-gen\checkpoints\poetry1_causal.best.pt`, last=`D:\temp\4\AI-programming-course\RNN-gen\checkpoints\poetry1_causal.last.pt`
- `poetry2_causal`: best=`D:\temp\4\AI-programming-course\RNN-gen\checkpoints\poetry2_causal.best.pt`, last=`D:\temp\4\AI-programming-course\RNN-gen\checkpoints\poetry2_causal.last.pt`
- `poetry3_causal`: best=`D:\temp\4\AI-programming-course\RNN-gen\checkpoints\poetry3_causal.best.pt`, last=`D:\temp\4\AI-programming-course\RNN-gen\checkpoints\poetry3_causal.last.pt`
