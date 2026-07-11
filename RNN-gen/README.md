# pytorch_peot_rnn
基于pytorch_rnn的古诗词生成

# 说明
config.py里面含有训练、测试、预测的参数，更改后运行：
```python
python main.py
```

# 预测结果
```python
if config.do_predict:
	result = trainer.generate('丽日照残春')
	print("".join(result))
	result = trainer.gen_acrostic('深度学习')
	print("".join(result))
	
丽日照残春，
风光摇落时。
不知花发意，
不得见春风。

深山高下有余灵，万里无人见钓矶。
度日茱萸人不得，一枝不得不相见。
学舞一枝花落叶，不知何处是君王。
习书不见金闺后，应是君王赐手间。
```

# 参考
> https://github.com/chenyuntc/pytorch-book<br>
其中第九章的古诗词生成，修改了以下地方：<br>
1、重构了代码架构；<br>
2、增加了数据集生成的过程；<br>
3、RNN网络改为batch_first；<br>
4、计算损失时不计算padding部分；<br>

# 数据清洗与实验切分

先生成清洗后的语料、统计报告、图表和实验切分：

```python
python data_cleaning_analysis.py
```

输出目录为 `data/cleaning_analysis/`，其中包括：

```text
peot_cleaned_train.txt / peot_cleaned_valid.txt / peot_cleaned_test.txt
peot_author_holdout_train.txt / peot_author_holdout_valid.txt / peot_author_holdout_test.txt
train_sample_weights.csv
author_holdout_train_sample_weights.csv
bias_report.md
figures/
```

在 `config.py` 中选择实验设置：

```python
# 文本级切分：评估同一唐诗分布中的未见文本
config.split_strategy = 'text'

# 作者级 holdout：评估未见作者风格上的泛化
config.split_strategy = 'author_holdout'

# 作者再平衡采样：与普通随机采样做对照实验
config.use_author_weighted_sampling = True
```

词表只使用训练集构建；验证集和测试集的未见字会映射为 `UNK`，避免数据泄漏。

完整的改动说明、数据统计、实验配置和复现步骤见 [EXPERIMENT_ENHANCEMENT.md](EXPERIMENT_ENHANCEMENT.md)。

模型、训练和评价组员请阅读 [HANDOFF.md](HANDOFF.md)。
