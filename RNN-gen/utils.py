import random
import numpy as np
import torch
import logging


def set_seed(seed=123):
    """
    设置随机数种子，保证实验可重现
    :param seed:
    :return:
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_logger(log_path):
    """
    配置log
    :param log_path:s
    :return:
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 由于每调用一次set_logger函数，就会创建一个handler，会造成重复打印的问题，因此需要判断root logger中是否已有该handler
    if not any(handler.__class__ == logging.FileHandler for handler in logger.handlers):
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not any(handler.__class__ == logging.StreamHandler for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def init_pretrained_embeddings(word2idx, idx2word,
                               model_name='KoichiYasuoka/roberta-classical-chinese-base-char'):
    """Initialize embedding weights from pre-trained Classical Chinese RoBERTa.

    Returns a weight tensor of shape (vocab_size, 768) suitable for
    nn.Embedding.from_pretrained().
    """
    import logging
    logger = logging.getLogger(__name__)
    from transformers import AutoTokenizer, AutoModel

    logger.info('Loading Classical Chinese RoBERTa: %s ...', model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    emb_layer = model.get_input_embeddings()
    pretrained = emb_layer.weight.detach().cpu().numpy()
    pretrained_dim = pretrained.shape[1]
    logger.info('Pre-trained embedding dim = %d', pretrained_dim)

    vocab_size = len(word2idx)
    embedding_weights = np.random.normal(
        scale=0.02, size=(vocab_size, pretrained_dim)
    ).astype(np.float32)

    # PAD
    embedding_weights[0] = 0.0

    matched = 0
    multi_token = 0
    for idx in range(4, vocab_size):
        char = idx2word[idx]
        ids = tokenizer.encode(char, add_special_tokens=False)
        if len(ids) == 0:
            continue
        if len(ids) == 1:
            embedding_weights[idx] = pretrained[ids[0]]
            matched += 1
        else:
            # Multi-token character → average the embeddings
            embedding_weights[idx] = pretrained[ids].mean(axis=0)
            multi_token += 1

    total = vocab_size - 4
    logger.info(
        'Embedding init: %d/%d matched (%.1f%%), %d multi-token',
        matched, total, 100 * matched / max(total, 1), multi_token
    )
    return torch.tensor(embedding_weights), pretrained_dim