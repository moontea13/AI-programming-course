import torch
import random
import logging
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np

from config import Config
from process import get_split_data, get_train_sample_weights
from model import PoetryModel, PoetryModel2, PoetryModel3
from utils import set_seed, set_logger, init_pretrained_embeddings

logger = logging.getLogger(__name__)


def split_train_test(data, train_ratio=0.8, shuffle=True):
    if shuffle:
        random.shuffle(data)
    total = len(data)
    train_total = int(total * train_ratio)
    train_data = data[:train_total]
    test_data = data[train_total:]
    print('总共有数据{}条'.format(total))
    print('划分后，训练集{}条'.format(train_total))
    print('划分后，测试集{}条'.format(total - train_total))
    return train_data, test_data


class Trainer:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.ss_prob = config.ss_start_prob
        self.ss_decay = (config.ss_start_prob - config.ss_end_prob) / config.num_epoch

    def train(self, train_loader, valid_loader=None):
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )

        scheduler = None
        if self.config.lr_scheduler == 'cosine':
            total_steps = len(train_loader) * self.config.num_epoch
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
        elif self.config.lr_scheduler == 'plateau' and valid_loader is not None:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)

        global_step = 0
        best_valid_loss = float("inf")
        best_epoch = None
        total_step = len(train_loader) * self.config.num_epoch
        for epoch in range(1, self.config.num_epoch + 1):
            total_loss = 0.
            for train_step, train_data in enumerate(train_loader):
                self.model.train()
                train_data = train_data.long().to(self.config.device)
                input = train_data[:, :-1]
                target = train_data[:, 1:]

                # Scheduled Sampling: first pass with teacher forcing
                output, _ = self.model(input)

                if self.config.scheduled_sampling and self.ss_prob < 1.0:
                    # Replace some input tokens with model's own predictions
                    with torch.no_grad():
                        pred_ids = output.view(input.size(0), input.size(1), -1).argmax(dim=-1)
                        mask = torch.rand(input.size(), device=input.device) > self.ss_prob
                        # Don't replace SOP (position 0)
                        mask[:, 0] = False
                        noisy_input = input.clone()
                        noisy_input[mask] = pred_ids[mask]
                    output, _ = self.model(noisy_input)

                active = (input > 0).view(-1)
                active_output = output[active]
                active_target = target.contiguous().view(-1)[active]
                loss = self.criterion(active_output, active_target)
                total_loss = total_loss + loss.item()

                optimizer.zero_grad()
                loss.backward()

                # Gradient clipping
                if self.config.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.config.grad_clip
                    )

                optimizer.step()

                if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
                    scheduler.step()

                logger.info('epoch:{} step:{}/{} loss:{:.4f} lr:{:.2e} ss_prob:{:.2f}'.format(
                    epoch, global_step, total_step, loss.item(),
                    optimizer.param_groups[0]['lr'], self.ss_prob
                ))
                global_step += 1

            logger.info('epoch:{} total_loss:{:.4f}'.format(epoch, total_loss))

            if valid_loader is not None:
                valid_loss = self.test(valid_loader)
                if valid_loss < best_valid_loss:
                    torch.save(self.model.state_dict(), self.config.save_path)
                    best_valid_loss = valid_loss
                    best_epoch = epoch
                logger.info('epoch:{} valid_loss:{:.4f}'.format(epoch, valid_loss))

                if scheduler is not None and isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(valid_loss)

            self.ss_prob = max(self.config.ss_end_prob, self.ss_prob - self.ss_decay)

        if best_epoch is None:
            torch.save(self.model.state_dict(), self.config.save_path)
            best_epoch = self.config.num_epoch
            best_valid_loss = total_loss
        logger.info('====================')
        logger.info('在第{}个epoch验证损失最小为：{:.4f}'.format(best_epoch, best_valid_loss))

    def test(self, test_loader):
        self.model.eval()
        total_loss = 0.
        with torch.no_grad():
            for test_step, test_data in enumerate(test_loader):
                test_data = test_data.long().to(self.config.device)
                input = test_data[:, :-1]
                target = test_data[:, 1:]
                output, _ = self.model(input)
                active = (input > 0).view(-1)
                active_output = output[active]
                active_target = target.contiguous().view(-1)[active]
                loss = self.criterion(active_output, active_target)
                total_loss = total_loss + loss.item()
        return total_loss / max(len(test_loader), 1)

    def _sample_token(self, last_output):
        """Temperature + top-p (nucleus) sampling over the last position."""
        logits = last_output / self.config.temperature

        # Top-p filtering
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > self.config.top_p
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = False

        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).item()

    def generate(self, start_words, prefix_words=None):
        results = list(start_words)
        start_word_len = len(start_words)

        if prefix_words:
            for word in prefix_words:
                results = [word] + results

        for i in range(self.config.max_gen_len):
            input_seq = ['SOP'] + results
            input_ids = [self.config.word2idx.get(w, self.config.word2idx['UNK']) for w in input_seq]
            input = torch.tensor([input_ids]).long().to(self.config.device)
            output, _ = self.model(input)
            last_output = output[-1:, :]

            if i < start_word_len:
                w = results[i]
            else:
                top_index = self._sample_token(last_output)
                w = self.config.idx2word[top_index]
                results.append(w)
            if w == 'EOP':
                del results[-1]
                break
        return results

    def gen_acrostic(self, start_words, prefix_words=None):
        results = []
        start_word_len = len(start_words)
        index = 0
        pre_word = 'SOP'

        if prefix_words:
            for word in prefix_words:
                results.append(word)

        for i in range(self.config.max_gen_len):
            input_seq = ['SOP'] + results
            input_ids = [self.config.word2idx.get(w, self.config.word2idx['UNK']) for w in input_seq]
            input = torch.tensor([input_ids]).long().to(self.config.device)
            output, _ = self.model(input)
            last_output = output[-1:, :]

            if pre_word in {u'。', u'！', 'SOP'}:
                if index == start_word_len:
                    break
                else:
                    w = start_words[index]
                    index += 1
            else:
                top_index = self._sample_token(last_output)
                w = self.config.idx2word[top_index]
            results.append(w)
            pre_word = w
        return results


if __name__ == '__main__':
    config = Config()
    set_seed(123)
    set_logger('./main.log')
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    config.device = device

    train_data, valid_data, test_data, word2idx, idx2word = get_split_data(config)
    config.word2idx = word2idx
    config.idx2word = idx2word

    if config.do_train:
        train_data = torch.from_numpy(train_data)
        train_sampler = None
        if config.use_author_weighted_sampling:
            sample_weights = get_train_sample_weights(config, len(train_data))
            train_sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(sample_weights),
                replacement=True,
            )
        train_loader = DataLoader(
            train_data,
            batch_size=config.batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=2,
        )
        valid_data = torch.from_numpy(valid_data)
        valid_loader = DataLoader(
            valid_data,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2,
        )

    if config.do_test:
        test_data = torch.from_numpy(test_data)
        test_loader = DataLoader(
            test_data,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2,
        )

    model = PoetryModel3(len(word2idx), config.embedding_dim, config.hidden_dim)

    if not config.do_load_model:
        pretrained_weight, pretrained_dim = init_pretrained_embeddings(word2idx, idx2word)
        if pretrained_dim != config.embedding_dim:
            config.embedding_dim = pretrained_dim
        model.embeddings = model.embeddings.from_pretrained(pretrained_weight, freeze=False)

    if config.do_load_model:
        print('加载已训练好的模型。。。')
        model.load_state_dict(torch.load(config.load_path, map_location=device))

    model.to(device)

    trainer = Trainer(model, config)
    if config.do_train:
        trainer.train(train_loader, valid_loader)

    if config.do_test:
        if config.do_train:
            model.load_state_dict(torch.load(config.save_path, map_location=device))
        test_loss = trainer.test(test_loader)
        logger.info('final_test_loss:{}'.format(test_loss))

    if config.do_predict:
        result = trainer.generate('丽日照残春')
        print("".join(result))
        result = trainer.gen_acrostic('深度学习')
        print("".join(result))
