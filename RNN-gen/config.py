class Config:
    def __init__(self):
        self.data_dir = './data/'
        self.cleaned_data_dir = self.data_dir + 'cleaning_analysis/'
        self.data_path = self.cleaned_data_dir + 'peot_cleaned.txt'
        self.train_data_path = self.cleaned_data_dir + 'peot_cleaned_train.txt'
        self.valid_data_path = self.cleaned_data_dir + 'peot_cleaned_valid.txt'
        self.test_data_path = self.cleaned_data_dir + 'peot_cleaned_test.txt'
        self.author_holdout_train_data_path = self.cleaned_data_dir + 'peot_author_holdout_train.txt'
        self.author_holdout_valid_data_path = self.cleaned_data_dir + 'peot_author_holdout_valid.txt'
        self.author_holdout_test_data_path = self.cleaned_data_dir + 'peot_author_holdout_test.txt'
        self.pickle_path = self.cleaned_data_dir + 'tang_cleaned.npz'
        self.split_pickle_path = self.cleaned_data_dir + 'tang_cleaned_splits.npz'
        self.author_holdout_split_pickle_path = self.cleaned_data_dir + 'tang_author_holdout_splits.npz'
        self.train_sample_weights_path = self.cleaned_data_dir + 'train_sample_weights.csv'
        self.author_holdout_train_sample_weights_path = (
            self.cleaned_data_dir + 'author_holdout_train_sample_weights.csv'
        )
        # "text" measures in-distribution generalization; "author_holdout" measures unseen-author generalization.
        self.split_strategy = 'text'
        self.use_author_weighted_sampling = False
        self.load_path = './checkpoints/peot.pt'
        self.save_path = './checkpoints/peot.pt'

        self.do_train = True
        self.do_test = True
        self.do_predict = True
        self.do_load_model = False

        self.num_epoch = 20
        self.batch_size = 128
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.max_gen_len = 200
        self.max_len = 125
        self.embedding_dim = 768   # 匹配 RoBERTa classical Chinese 预训练维度
        self.hidden_dim = 512

        # --- 生成控制 ---
        self.temperature = 0.8    # 越大越随机，越小越保守
        self.top_p = 0.9          # nucleus sampling 阈值

        # --- Scheduled Sampling ---
        self.scheduled_sampling = True
        self.ss_start_prob = 1.0  # 初始 teacher forcing 概率
        self.ss_end_prob = 0.2    # 最终 teacher forcing 概率

        # --- 训练正则化 ---
        self.grad_clip = 5.0           # 梯度裁剪最大范数
        self.label_smoothing = 0.1     # 标签平滑

        # --- 学习率调度 ---
        self.lr_scheduler = 'cosine'   # 'cosine' / 'plateau' / None
