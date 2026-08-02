from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

from src.data import PROJECT_ROOT
from src.models import MultimodalRelationCNNNoAuxiliary
from src.verify import EXPECTED_SEED_44_INITIAL_STATE_SHA256, state_dict_sha256


def test_ablation_summary() -> None:
    summary = json.loads((PROJECT_ROOT / "results/ablation/auxiliary_loss_ablation_summary.json").read_text())
    assert summary["no_auxiliary_aggregate"]["seeds"] == [43, 44, 45]
    assert summary["no_auxiliary_aggregate"]["runs"] == 3
    assert summary["no_auxiliary_aggregate"]["all_runs_collapsed_to_one_predicted_class"] is True
    assert np.isclose(summary["validation_accuracy_gap"], 0.45208333333333334)
    assert np.isclose(summary["validation_macro_f1_gap"], 0.6206176526591394)
    assert summary["test_split_used"] is False


def test_each_ablation_checkpoint_and_predictions() -> None:
    for seed in (43, 44, 45):
        directory = PROJECT_ROOT / "results/ablation" / f"seed_{seed}"
        metrics = json.loads((directory / "metrics.json").read_text())
        predictions = pd.read_csv(directory / "validation_predictions.csv")
        payload = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=False)
        model = MultimodalRelationCNNNoAuxiliary(payload["text_dimension"])
        model.load_state_dict(payload["state_dict"], strict=True)
        assert payload["seed"] == seed
        assert payload["test_split_used"] is False
        assert predictions.predicted_label.nunique() == 1
        assert np.isclose(accuracy_score(predictions.true_label, predictions.predicted_label), 1 / 3)
        assert np.isclose(f1_score(predictions.true_label, predictions.predicted_label, average="macro"), 1 / 6)
        assert metrics["collapsed_to_one_predicted_class"] is True

def test_seed_44_saved_initial_state_is_portable_and_intact() -> None:
    payload = torch.load(
        PROJECT_ROOT / "results/ablation/seed_44/checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = MultimodalRelationCNNNoAuxiliary(payload["text_dimension"])
    saved_initial_state = payload["initial_state_dict"]
    reference_state = model.state_dict()
    assert set(saved_initial_state) == set(reference_state)
    assert all(saved_initial_state[key].shape == reference_state[key].shape for key in reference_state)
    assert state_dict_sha256(saved_initial_state) == EXPECTED_SEED_44_INITIAL_STATE_SHA256

