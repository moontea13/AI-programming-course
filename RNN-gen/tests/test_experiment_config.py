import pytest

from experiment_config import EXPERIMENTS, HANDOFF_FULL_SUITE, get_experiment
from experiment_core import build_model
from model import PoetryModel, PoetryModel2, PoetryModel3


@pytest.mark.parametrize(
    ("experiment_id", "model_class"),
    [
        ("poetry1_controlled", PoetryModel),
        ("poetry2_controlled", PoetryModel2),
        ("poetry3_controlled", PoetryModel3),
        ("poetry2_legacy_recipe", PoetryModel2),
    ],
)
def test_core_suite_defines_expected_models(experiment_id, model_class):
    spec = get_experiment(experiment_id)
    model = build_model(spec, vocab_size=100)

    assert isinstance(model, model_class)
    assert spec.num_epoch == 5
    assert spec.batch_size == 32
    assert spec.split_strategy == "text"
    assert spec.use_author_weighted_sampling is False


def test_controlled_experiments_share_training_recipe():
    controlled = [spec for spec in EXPERIMENTS.values() if spec.recipe == "controlled"]

    assert len(controlled) == 3
    assert {(spec.embedding_dim, spec.hidden_dim) for spec in controlled} == {(768, 512)}
    assert all(spec.use_pretrained_embeddings for spec in controlled)
    assert all(spec.label_smoothing == 0.1 for spec in controlled)
    assert all(spec.grad_clip == 5.0 for spec in controlled)
    assert all(spec.lr_scheduler == "cosine" for spec in controlled)
    assert all(spec.scheduled_sampling for spec in controlled)


def test_legacy_recipe_matches_original_training_style():
    spec = get_experiment("poetry2_legacy_recipe")

    assert spec.embedding_dim == 300
    assert spec.hidden_dim == 256
    assert spec.use_pretrained_embeddings is False
    assert spec.label_smoothing == 0.0
    assert spec.grad_clip == 0.0
    assert spec.lr_scheduler is None
    assert spec.scheduled_sampling is False


def test_unknown_experiment_is_rejected():
    with pytest.raises(KeyError, match="Unknown experiment"):
        get_experiment("missing")


@pytest.mark.parametrize(
    ("experiment_id", "model_class", "architecture"),
    [
        ("poetry1_causal", PoetryModel, "Embedding -> UniLSTM(1 layer) -> Linear"),
        ("poetry2_causal", PoetryModel2, "Embedding -> UniLSTM(2 layers) -> Linear"),
        (
            "poetry3_causal",
            PoetryModel3,
            "Embedding -> causal residual Encoder/Decoder -> masked Bahdanau Attention -> Linear",
        ),
    ],
)
def test_handoff_full_suite_defines_official_causal_models(
    experiment_id, model_class, architecture
):
    spec = get_experiment(experiment_id)

    assert experiment_id in HANDOFF_FULL_SUITE
    assert isinstance(build_model(spec, vocab_size=100), model_class)
    assert spec.architecture == architecture
    assert spec.num_epoch == 20
    assert spec.effective_batch_size == 32
    assert spec.seed == 123


def test_handoff_full_suite_uses_one_shared_training_recipe():
    specs = [get_experiment(experiment_id) for experiment_id in HANDOFF_FULL_SUITE]

    shared_fields = {
        (
            spec.embedding_dim,
            spec.hidden_dim,
            spec.effective_batch_size,
            spec.lr,
            spec.weight_decay,
            spec.label_smoothing,
            spec.grad_clip,
            spec.lr_scheduler,
            spec.scheduled_sampling,
        )
        for spec in specs
    }
    assert len(shared_fields) == 1
