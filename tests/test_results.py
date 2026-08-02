from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from src.data import PROJECT_ROOT, file_sha256
from src.evaluate import evaluate_validation


def test_selected_checkpoint_hash() -> None:
    assert file_sha256(PROJECT_ROOT / "models/selected_multimodal_model.pt") == "bce8a98fc96140294f3043fd845e1a7b6f491c67869a6c775877cd6ca1aa2a17"


def test_validation_metrics_recompute() -> None:
    predictions = pd.read_csv(PROJECT_ROOT / "results/validation/selected_model_predictions.csv")
    assert int(predictions.is_correct.sum()) == 377
    assert np.isclose(accuracy_score(predictions.true_label, predictions.predicted_label), 0.7854166666666667)
    assert np.isclose(f1_score(predictions.true_label, predictions.predicted_label, average="macro"), 0.787284319325806)


def test_selected_checkpoint_reproduces_saved_validation_predictions() -> None:
    result = evaluate_validation()
    assert result["status"] == "PASS_SELECTED_MODEL_VALIDATION_REPRODUCED"
    assert result["correct_predictions"] == 377
    assert result["test_split_used"] is False


def test_frozen_final_metrics_recompute_without_inference() -> None:
    predictions = pd.read_csv(PROJECT_ROOT / "results/final_test/test_predictions.csv")
    metrics = json.loads((PROJECT_ROOT / "results/final_test/test_metrics.json").read_text())
    assert int(predictions.is_correct.sum()) == 354
    assert np.isclose(accuracy_score(predictions.true_label, predictions.predicted_label), 0.7375)
    assert np.isclose(f1_score(predictions.true_label, predictions.predicted_label, average="macro"), 0.7382299830250852)
    assert metrics["authorization_consumed"] is True
    assert metrics["further_tuning_permitted"] is False



def test_validation_category_rule_diagnostic() -> None:
    diagnostic = pd.read_csv(PROJECT_ROOT / "results/validation/category_rule_diagnostic.csv")
    metrics = json.loads(
        (PROJECT_ROOT / "results/validation/category_rule_diagnostic.json").read_text()
    )
    assert len(diagnostic) == 480
    assert diagnostic["image_id"].nunique() == 80
    assert int(diagnostic["learned_relation_correct"].sum()) == 377
    assert int(diagnostic["category_rule_correct"].sum()) == 350
    assert np.isclose(metrics["predicted_category_rule"]["accuracy"], 0.7291666666666666)
    assert np.isclose(metrics["predicted_category_rule"]["macro_f1"], 0.7324044215433009)
    assert np.isclose(metrics["auxiliary_category_predictions"]["image_category_accuracy"], 0.6625)
    assert np.isclose(metrics["auxiliary_category_predictions"]["text_category_accuracy"], 1.0)
    assert metrics["oracle_category_rule"]["accuracy"] == 1.0
    assert metrics["test_split_used"] is False
    assert metrics["new_training_performed"] is False
