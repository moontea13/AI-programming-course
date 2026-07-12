# coding:utf8
import torch
import torch.nn as nn
import torch.nn.functional as F


class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim,
                            self.hidden_dim,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)
        self.linear = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input, hidden=None):
        batch_size, seq_len = input.size()
        if hidden is None:
            h_0 = input.data.new(1, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(1, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        embeds = self.embeddings(input)
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.linear(output.contiguous().view(seq_len * batch_size, -1))
        return output, hidden


class PoetryModel2(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel2, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=2, batch_first=True)
        self.linear1 = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input, hidden=None):
        batch_size, seq_len = input.size()
        if hidden is None:
            h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
        else:
            h_0, c_0 = hidden
        embeds = self.embeddings(input)
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.linear1(output.contiguous().view(seq_len * batch_size, -1))
        return output, hidden


class PoetryModel3(nn.Module):
    """Encoder-Decoder with residual LSTM layers + Bahdanau attention.

    Embedding -> UniLSTM-Encoder (2-layer, residual)
              -> UniLSTM-Decoder (2-layer, residual)
              -> Causal Attention(dec_out, enc_out) -> Concat -> Linear -> vocab
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel3, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)

        # --- Encoder: 2-layer unidirectional LSTM ---
        self.enc_lstm1 = nn.LSTM(embedding_dim, hidden_dim, num_layers=1,
                                 batch_first=True, bidirectional=False)
        self.enc_res1 = nn.Linear(embedding_dim, hidden_dim)

        self.enc_lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers=1,
                                 batch_first=True, bidirectional=False)
        self.enc_res2 = nn.Linear(hidden_dim, hidden_dim)

        # --- Decoder: 2-layer unidirectional LSTM ---
        self.dec_lstm1 = nn.LSTM(hidden_dim, hidden_dim, num_layers=1,
                                 batch_first=True, bidirectional=False)
        self.dec_res1 = nn.Linear(hidden_dim, hidden_dim)

        self.dec_lstm2 = nn.LSTM(hidden_dim, hidden_dim, num_layers=1,
                                 batch_first=True, bidirectional=False)

        # --- Bahdanau Attention ---
        self.attn_W = nn.Linear(hidden_dim + hidden_dim, hidden_dim)
        self.attn_v = nn.Linear(hidden_dim, 1)

        # --- Output ---
        self.linear = nn.Linear(hidden_dim + hidden_dim, vocab_size)

    def _attention(self, dec_out, enc_out):
        decoder_weight = self.attn_W.weight[:, :self.hidden_dim]
        encoder_weight = self.attn_W.weight[:, self.hidden_dim:]
        decoder_projection = F.linear(dec_out, decoder_weight, self.attn_W.bias)
        encoder_projection = F.linear(enc_out, encoder_weight)
        scores = self.attn_v(
            torch.tanh(
                decoder_projection.unsqueeze(2) + encoder_projection.unsqueeze(1)
            )
        ).squeeze(-1)
        causal_mask = torch.triu(
            torch.ones(
                dec_out.size(1),
                enc_out.size(1),
                dtype=torch.bool,
                device=scores.device,
            ),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.bmm(attn_weights, enc_out)
        return context

    def forward(self, input, hidden=None):
        batch_size, seq_len = input.size()
        embeds = self.embeddings(input)

        # Encoder layer 1 + residual
        enc1, _ = self.enc_lstm1(embeds)
        enc1 = enc1 + self.enc_res1(embeds)

        # Encoder layer 2 + residual
        enc2, _ = self.enc_lstm2(enc1)
        enc2 = enc2 + self.enc_res2(enc1)

        # Decoder layer 1 + residual
        dec1, _ = self.dec_lstm1(enc2)
        dec1 = dec1 + self.dec_res1(enc2)

        # Decoder layer 2 + residual
        dec2, _ = self.dec_lstm2(dec1)
        dec2 = dec2 + dec1

        # Attention over encoder outputs
        context = self._attention(dec2, enc2)

        # Concat decoder output with attention context
        combined = torch.cat([dec2, context], dim=-1)
        output = self.linear(combined.contiguous().view(seq_len * batch_size, -1))
        return output, None


if __name__ == '__main__':
    input = torch.tensor([[2, 13, 15, 20], [2, 13, 15, 20]]).long()
    model = PoetryModel2(10000, 300, 256)
    ouput, hidden = model(input)
