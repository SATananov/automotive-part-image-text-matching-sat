from __future__ import annotations

import torch

from src.models import ImageRelationCNN, MultimodalRelationCNN, MultimodalRelationCNNNoAuxiliary, TextRelationMLP, count_trainable_parameters
from src.train import set_seed


def test_parameter_counts() -> None:
    assert count_trainable_parameters(TextRelationMLP(342)) == 24131
    assert count_trainable_parameters(ImageRelationCNN()) == 21459
    assert count_trainable_parameters(MultimodalRelationCNN(342)) == 58579
    assert count_trainable_parameters(MultimodalRelationCNNNoAuxiliary(342)) == 57923


def test_seed_44_shared_initialization_is_identical() -> None:
    set_seed(44)
    with_aux = MultimodalRelationCNN(342)
    set_seed(44)
    without_aux = MultimodalRelationCNNNoAuxiliary(342)
    assert set(without_aux.state_dict()).issubset(with_aux.state_dict())
    assert all(torch.equal(with_aux.state_dict()[key], value) for key, value in without_aux.state_dict().items())
