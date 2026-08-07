from __future__ import annotations

import numpy as np
import pandas as pd
import torch

import src.v4_training as training
from src.v4_training import (
    ImageOnlyClassifierV4,
    MultimodalClassifierV4,
    TextOnlyClassifierV4,
    evaluate_classifier_v4,
)


def test_development_loader_requests_only_train_and_validation(monkeypatch) -> None:
    calls: list[str] = []

    def fake_load(split: str) -> pd.DataFrame:
        calls.append(split)
        return pd.DataFrame({"split": [split]})

    monkeypatch.setattr(training, "load_relations_v4", fake_load)
    train, validation = training.load_development_relations_v4()
    assert calls == ["train", "validation"]
    assert train.iloc[0]["split"] == "train"
    assert validation.iloc[0]["split"] == "validation"


def test_simple_v4_classifier_shapes() -> None:
    image = torch.zeros(4, 512)
    text = torch.zeros(4, 32)

    assert TextOnlyClassifierV4(32)(image, text).shape == (4, 3)
    assert ImageOnlyClassifierV4(512)(image, text).shape == (4, 3)
    assert MultimodalClassifierV4(512, 32, auxiliary_heads=False)(
        image, text
    ).shape == (4, 3)

    relation, image_category, text_category = MultimodalClassifierV4(
        512, 32, auxiliary_heads=True
    )(image, text)
    assert relation.shape == (4, 3)
    assert image_category.shape == (4, 50)
    assert text_category.shape == (4, 50)


def test_evaluation_returns_valid_metrics() -> None:
    image = np.zeros((6, 512), dtype=np.float32)
    text = np.zeros((6, 16), dtype=np.float32)
    relation = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64)
    categories = np.zeros(6, dtype=np.int64)
    loader = training._make_loader(
        image,
        text,
        relation,
        categories,
        categories,
        batch_size=3,
        shuffle=False,
    )
    model = TextOnlyClassifierV4(16)
    accuracy, macro_f1, matrix = evaluate_classifier_v4(
        model, loader, torch.device("cpu")
    )
    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= macro_f1 <= 1.0
    assert len(matrix) == 3
    assert all(len(row) == 3 for row in matrix)
