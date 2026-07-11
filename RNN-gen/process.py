import re
import os
import json
import csv
import numpy as np

from config import Config

SPLIT_CACHE_VERSION = 1


def _parseRawData(author=None, constrain=None, src='./data/tang/', category="poet.tang"):
    """
    DEPRECATED: 历史数据解析基线。当前训练数据由
    data_cleaning_analysis.py 生成，并通过 get_split_data() 加载。

    code from https://github.com/justdark/pytorch-poetry-gen/blob/master/dataHandler.py
    处理json文件，返回诗歌内容
    @param: author： 作者名字
    @param: constrain: 长度限制
    @param: src: json 文件存放路径
    @param: category: 类别，有poet.song 和 poet.tang

    在此基础上，新增了将繁体转简体，以及存储文件的功能
    返回 data：list
        ['床前明月光，疑是地上霜，举头望明月，低头思故乡。',
         '一去二三里，烟村四五家，亭台六七座，八九十支花。',
        .........
        ]
    """
    import opencc

    def sentenceParse(para):
        # para 形如 "-181-村橋路不端，數里就迴湍。積壤連涇脉，高林上笋竿。早嘗甘蔗淡，
        # 生摘琵琶酸。（「琵琶」，嚴壽澄校《張祜詩集》云：疑「枇杷」之誤。）
        # 好是去塵俗，煙花長一欄。"
        result, number = re.subn(u"（.*）", "", para)
        result, number = re.subn(u"{.*}", "", result)
        result, number = re.subn(u"《.*》", "", result)
        result, number = re.subn(u"《.*》", "", result)
        result, number = re.subn(u"[\]\[]", "", result)
        r = ""
        for s in result:
            if s not in set('0123456789-'):
                r += s
        r, number = re.subn(u"。。", u"。", r)
        return r

    def handleJson(file):
        # print file
        rst = []
        data = json.loads(open(file).read())
        for poetry in data:
            pdata = ""
            if (author is not None and poetry.get("author") != author):
                continue
            p = poetry.get("paragraphs")
            flag = False
            for s in p:
                sp = re.split(u"[，！。]", s)
                for tr in sp:
                    if constrain is not None and len(tr) != constrain and len(tr) != 0:
                        flag = True
                        break
                    if flag:
                        break
            if flag:
                continue
            for sentence in poetry.get("paragraphs"):
                pdata += sentence
            pdata = sentenceParse(pdata)
            if pdata != "":
                rst.append(pdata)
        return rst

    data = []
    cc = opencc.OpenCC('t2s')
    for filename in os.listdir(src):
        if filename.startswith(category):
            peots = handleJson(src + filename)
            for peot in peots:
                peot = cc.convert(peot)
                data.append(peot)
    with open('./data/peot.txt', 'w') as fp:
        fp.write("\n".join(data))
    return data


def pad_sequences(sequences,
                  maxlen=None,
                  dtype='int32',
                  padding='pre',
                  truncating='pre',
                  value=0.):
    """
    code from keras
    Pads each sequence to the same length (length of the longest sequence).
    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.
    Supports post-padding and pre-padding (default).
    Arguments:
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    Returns:
        x: numpy array with dimensions (number_of_sequences, maxlen)
    Raises:
        ValueError: in case of invalid values for `truncating` or `padding`,
            or in case of invalid shape for a `sequences` entry.
    """
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    lengths = []
    for x in sequences:
        if not hasattr(x, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))
        lengths.append(len(x))

    num_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:  # pylint: disable=g-explicit-length-test
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((num_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if not len(s):  # pylint: disable=g-explicit-length-test
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]  # pylint: disable=invalid-unary-operand-type
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                'Shape of sample %s of sequence at position %s is different from '
                'expected shape %s'
                % (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def _read_poems(path):
    with open(path, 'r', encoding='utf-8') as fp:
        return [line.strip() for line in fp if line.strip()]


def _build_vocab(sentences):
    words = {word for sentence in sentences for word in sentence}
    word2idx = {_word: _ix + 4 for _ix, _word in enumerate(words)}
    # PAD:0 UNK:1 SOP:2（开始标识符） EOP:3（终止标识符）
    word2idx['PAD'] = 0
    word2idx['UNK'] = 1
    word2idx['SOP'] = 2
    word2idx['EOP'] = 3
    idx2word = {_ix: _word for _word, _ix in word2idx.items()}
    return word2idx, idx2word


def _encode_poems(sentences, word2idx, max_len):
    encoded = []
    for sentence in sentences:
        tokens = ['SOP'] + list(sentence) + ['EOP']
        encoded.append([word2idx.get(token, word2idx['UNK']) for token in tokens])
    return pad_sequences(
        encoded,
        maxlen=max_len,
        padding='post',
        truncating='post',
        value=0,
    )


def _split_paths(config):
    if config.split_strategy == 'text':
        return (
            config.train_data_path,
            config.valid_data_path,
            config.test_data_path,
            config.split_pickle_path,
            config.train_sample_weights_path,
        )
    if config.split_strategy == 'author_holdout':
        return (
            config.author_holdout_train_data_path,
            config.author_holdout_valid_data_path,
            config.author_holdout_test_data_path,
            config.author_holdout_split_pickle_path,
            config.author_holdout_train_sample_weights_path,
        )
    raise ValueError("split_strategy must be 'text' or 'author_holdout'")


def _cache_is_current(cache_path, source_paths, max_len):
    if not os.path.exists(cache_path):
        return False
    cache_mtime = os.path.getmtime(cache_path)
    if not all(os.path.getmtime(path) <= cache_mtime for path in source_paths):
        return False
    with np.load(cache_path, allow_pickle=True) as archive:
        if 'cache_version' not in archive.files or 'max_len' not in archive.files:
            return False
        return (
            int(archive['cache_version'].item()) == SPLIT_CACHE_VERSION
            and int(archive['max_len'].item()) == max_len
        )


def get_split_data(config):
    """
    加载数据清洗脚本生成的固定 train/valid/test 划分。

    词表仅由训练集建立；验证集和测试集中的未见字映射为 UNK，
    避免测试集信息进入训练词表。
    """
    train_path, valid_path, test_path, cache_path, _ = _split_paths(config)
    source_paths = [train_path, valid_path, test_path]
    missing_paths = [path for path in source_paths if not os.path.exists(path)]
    if missing_paths:
        raise FileNotFoundError(
            'Missing cleaned split files: {}. Run `python data_cleaning_analysis.py` first.'.format(
                ', '.join(missing_paths)
            )
        )

    if _cache_is_current(cache_path, source_paths, config.max_len):
        archive = np.load(cache_path, allow_pickle=True)
        return (
            archive['train_data'],
            archive['valid_data'],
            archive['test_data'],
            archive['word2idx'].item(),
            archive['idx2word'].item(),
        )

    train_sentences = _read_poems(train_path)
    valid_sentences = _read_poems(valid_path)
    test_sentences = _read_poems(test_path)
    word2idx, idx2word = _build_vocab(train_sentences)

    train_data = _encode_poems(train_sentences, word2idx, config.max_len)
    valid_data = _encode_poems(valid_sentences, word2idx, config.max_len)
    test_data = _encode_poems(test_sentences, word2idx, config.max_len)

    cache_dir = os.path.dirname(cache_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(
        cache_path,
        train_data=train_data,
        valid_data=valid_data,
        test_data=test_data,
        word2idx=word2idx,
        idx2word=idx2word,
        max_len=np.array(config.max_len),
        cache_version=np.array(SPLIT_CACHE_VERSION),
    )
    return train_data, valid_data, test_data, word2idx, idx2word


def get_train_sample_weights(config, expected_count):
    """Load sample weights aligned with the active training split."""
    _, _, _, _, weight_path = _split_paths(config)
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            'Missing sample-weight file: {}. Run `python data_cleaning_analysis.py` first.'.format(
                weight_path
            )
        )
    with open(weight_path, 'r', encoding='utf-8-sig', newline='') as fp:
        weights = [float(row['sample_weight']) for row in csv.DictReader(fp)]
    if len(weights) != expected_count:
        raise ValueError(
            'Weight count {} does not match train sample count {}. Re-run the cleaning pipeline.'.format(
                len(weights), expected_count
            )
        )
    return np.asarray(weights, dtype=np.float64)


def get_data(config):
    if _cache_is_current(config.pickle_path, [config.data_path], config.max_len):
        data = np.load(config.pickle_path, allow_pickle=True)
        data, word2idx, idx2word = data['data'], data['word2idx'].item(), data['idx2word'].item()
        return data, word2idx, idx2word

    data = _read_poems(config.data_path)
    word2idx, idx2word = _build_vocab(data)
    pad_data = _encode_poems(data, word2idx, config.max_len)
    cache_dir = os.path.dirname(config.pickle_path)
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(
        config.pickle_path,
        data=pad_data,
        word2idx=word2idx,
        idx2word=idx2word,
        max_len=np.array(config.max_len),
        cache_version=np.array(SPLIT_CACHE_VERSION),
    )
    return pad_data, word2idx, idx2word


if __name__ == '__main__':
    # 1、读取json文件，提取古诗并简体化，最后存储
    # data = _parseRawData()
    # print(data[:10])

    # 2、获取映射
    config = Config()
    data, word2idx, idx2word = get_data(config)
    print(data[0])
    print(word2idx)
    print(idx2word)
