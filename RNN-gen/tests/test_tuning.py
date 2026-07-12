from tune import TUNING_CANDIDATES, select_best_trial


def test_tuning_uses_six_approved_candidates():
    assert TUNING_CANDIDATES == (
        {"hidden_dim": 256, "lr": 5e-4, "effective_batch_size": 32},
        {"hidden_dim": 256, "lr": 1e-3, "effective_batch_size": 64},
        {"hidden_dim": 512, "lr": 5e-4, "effective_batch_size": 64},
        {"hidden_dim": 512, "lr": 1e-3, "effective_batch_size": 32},
        {"hidden_dim": 512, "lr": 3e-3, "effective_batch_size": 32},
        {"hidden_dim": 768, "lr": 5e-4, "effective_batch_size": 32},
    )


def test_tuning_selects_lowest_validation_loss_then_runtime_then_hidden_size():
    trials = [
        {"trial": 1, "best_valid_loss": 1.0, "train_seconds": 20.0, "hidden_dim": 256},
        {"trial": 2, "best_valid_loss": 0.8, "train_seconds": 30.0, "hidden_dim": 512},
        {"trial": 3, "best_valid_loss": 0.8, "train_seconds": 25.0, "hidden_dim": 768},
        {"trial": 4, "best_valid_loss": 0.8, "train_seconds": 25.0, "hidden_dim": 256},
    ]

    assert select_best_trial(trials)["trial"] == 4


def test_tuning_selection_does_not_accept_test_metrics():
    trial = {
        "trial": 1,
        "best_valid_loss": 0.8,
        "train_seconds": 20.0,
        "hidden_dim": 256,
        "test_loss": 0.1,
    }

    assert select_best_trial([trial])["best_valid_loss"] == 0.8
