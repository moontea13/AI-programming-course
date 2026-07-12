import torch
import torch.nn.functional as F

import model as model_module
from model import PoetryModel, PoetryModel2, PoetryModel3


def test_poetry3_attention_matches_reference_without_four_dimensional_concat(monkeypatch):
    torch.manual_seed(123)
    model = PoetryModel3(vocab_size=20, embedding_dim=6, hidden_dim=4)
    decoder = torch.randn(2, 3, 4)
    encoder_dim = model.attn_W.in_features - model.hidden_dim
    encoder = torch.randn(2, 5, encoder_dim)

    decoder_expanded = decoder.unsqueeze(2).expand(-1, -1, encoder.size(1), -1)
    encoder_expanded = encoder.unsqueeze(1).expand(-1, decoder.size(1), -1, -1)
    reference_input = torch.cat([decoder_expanded, encoder_expanded], dim=-1)
    reference_scores = model.attn_v(torch.tanh(model.attn_W(reference_input))).squeeze(-1)
    causal_mask = torch.triu(torch.ones(3, 5, dtype=torch.bool), diagonal=1)
    reference_scores = reference_scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
    reference = torch.bmm(F.softmax(reference_scores, dim=-1), encoder)

    original_cat = torch.cat

    def reject_four_dimensional_concat(tensors, *args, **kwargs):
        if tensors[0].dim() == 4:
            raise AssertionError("attention must not materialize a 4D concatenated tensor")
        return original_cat(tensors, *args, **kwargs)

    monkeypatch.setattr(model_module.torch, "cat", reject_four_dimensional_concat)

    actual = model._attention(decoder, encoder)

    assert torch.allclose(actual, reference, atol=1e-6, rtol=1e-5)


@torch.no_grad()
@torch.inference_mode()
def _assert_future_tokens_do_not_change_past_logits(model):
    model.eval()
    prefix = torch.tensor([[1, 2, 3]])
    first = torch.cat([prefix, torch.tensor([[4, 5, 6]])], dim=1)
    second = torch.cat([prefix, torch.tensor([[7, 8, 9]])], dim=1)

    first_logits, _ = model(first)
    second_logits, _ = model(second)
    first_logits = first_logits.view(1, first.size(1), -1)
    second_logits = second_logits.view(1, second.size(1), -1)

    assert torch.allclose(first_logits[:, :3], second_logits[:, :3], atol=1e-6, rtol=1e-5)


def test_all_official_models_are_strictly_causal():
    torch.manual_seed(123)
    for model_class in (PoetryModel, PoetryModel2, PoetryModel3):
        _assert_future_tokens_do_not_change_past_logits(
            model_class(vocab_size=12, embedding_dim=6, hidden_dim=4)
        )
