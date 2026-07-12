from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExperimentSpec:
    experiment_id: str
    model_type: str
    recipe: str
    architecture: str
    embedding_dim: int
    hidden_dim: int
    use_pretrained_embeddings: bool
    label_smoothing: float
    grad_clip: float
    lr_scheduler: str | None
    scheduled_sampling: bool
    split_strategy: str = "text"
    use_author_weighted_sampling: bool = False
    seed: int = 123
    num_epoch: int = 5
    batch_size: int = 32
    effective_batch_size: int = 32
    gradient_accumulation_steps: int = 1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_len: int = 125
    max_gen_len: int = 200
    temperature: float = 0.8
    top_p: float = 0.9
    ss_start_prob: float = 1.0
    ss_end_prob: float = 0.2
    early_stopping: bool = True
    patience: int = 5
    min_delta: float = 1e-4

    def to_dict(self):
        return asdict(self)


def _controlled(experiment_id, model_type):
    architectures = {
        "poetry1": "Embedding -> UniLSTM(1 layer) -> Linear",
        "poetry2": "Embedding -> UniLSTM(2 layers) -> Linear",
        "poetry3": "Embedding -> causal residual Encoder/Decoder -> masked Bahdanau Attention -> Linear",
    }
    return ExperimentSpec(
        experiment_id=experiment_id,
        model_type=model_type,
        recipe="controlled",
        architecture=architectures[model_type],
        embedding_dim=768,
        hidden_dim=512,
        use_pretrained_embeddings=True,
        label_smoothing=0.1,
        grad_clip=5.0,
        lr_scheduler="cosine",
        scheduled_sampling=True,
    )


EXPERIMENTS = {
    "poetry1_controlled": _controlled("poetry1_controlled", "poetry1"),
    "poetry2_controlled": _controlled("poetry2_controlled", "poetry2"),
    "poetry3_controlled": _controlled("poetry3_controlled", "poetry3"),
    "poetry2_legacy_recipe": ExperimentSpec(
        experiment_id="poetry2_legacy_recipe",
        model_type="poetry2",
        recipe="legacy",
        architecture="Embedding -> UniLSTM(2 layers) -> Linear",
        embedding_dim=300,
        hidden_dim=256,
        use_pretrained_embeddings=False,
        label_smoothing=0.0,
        grad_clip=0.0,
        lr_scheduler=None,
        scheduled_sampling=False,
        early_stopping=False,
    ),
    "poetry1_causal": ExperimentSpec(
        experiment_id="poetry1_causal",
        model_type="poetry1",
        recipe="handoff_full",
        architecture="Embedding -> UniLSTM(1 layer) -> Linear",
        embedding_dim=768,
        hidden_dim=512,
        use_pretrained_embeddings=True,
        label_smoothing=0.1,
        grad_clip=5.0,
        lr_scheduler="cosine",
        scheduled_sampling=True,
        num_epoch=20,
    ),
    "poetry2_causal": ExperimentSpec(
        experiment_id="poetry2_causal",
        model_type="poetry2",
        recipe="handoff_full",
        architecture="Embedding -> UniLSTM(2 layers) -> Linear",
        embedding_dim=768,
        hidden_dim=512,
        use_pretrained_embeddings=True,
        label_smoothing=0.1,
        grad_clip=5.0,
        lr_scheduler="cosine",
        scheduled_sampling=True,
        num_epoch=20,
    ),
    "poetry3_causal": ExperimentSpec(
        experiment_id="poetry3_causal",
        model_type="poetry3",
        recipe="handoff_full",
        architecture="Embedding -> causal residual Encoder/Decoder -> masked Bahdanau Attention -> Linear",
        embedding_dim=768,
        hidden_dim=512,
        use_pretrained_embeddings=True,
        label_smoothing=0.1,
        grad_clip=5.0,
        lr_scheduler="cosine",
        scheduled_sampling=True,
        num_epoch=20,
    ),
}

CORE_QUICK_SUITE = (
    "poetry2_legacy_recipe",
    "poetry1_controlled",
    "poetry2_controlled",
    "poetry3_controlled",
)

HANDOFF_FULL_SUITE = (
    "poetry1_causal",
    "poetry2_causal",
    "poetry3_causal",
)


def get_experiment(experiment_id):
    try:
        return EXPERIMENTS[experiment_id]
    except KeyError as exc:
        raise KeyError("Unknown experiment: {}".format(experiment_id)) from exc
