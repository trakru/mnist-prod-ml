import pytest
from model.train import objective
from optuna.trial import FixedTrial
import torch


# Fixture to create a fixed trial object
@pytest.fixture
def fixed_trial():
    return FixedTrial(
        {
            "batch_size": 32,
            "lr": 0.1,
            "optimizer": "SGD",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
    )


# Test to check if the objective function returns a valid accuracy
def test_objective(fixed_trial):
    accuracy = objective(fixed_trial)
    assert 0 <= accuracy <= 1, "Accuracy must be between 0 and 1"
