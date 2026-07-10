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

        self.do_train = False
        self.do_test = False
        self.do_predict = True
        self.do_load_model = True

        self.num_epoch = 20
        self.batch_size = 128
        self.lr = 1e-3
        self.weight_decay = 1e-4
        self.max_gen_len = 200
        self.max_len = 125
        self.embedding_dim = 300
        self.hidden_dim = 256
