from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.data import LABELS, PROJECT_ROOT, file_sha256, validate_frozen_test_evidence
from src.models import (
    MultimodalRelationCNN,
    MultimodalRelationCNNNoAuxiliary,
    count_trainable_parameters,
)
from src.train import set_seed

EXPECTED_CHECKPOINT_SHA256 = "bce8a98fc96140294f3043fd845e1a7b6f491c67869a6c775877cd6ca1aa2a17"
EXPECTED_TEST_LOCK_SHA256 = "162de671f97c43e3f5afe6c25f577e65228d1f759b699ff93a0d74d9726764a5"
EXPECTED_SEED_44_INITIAL_STATE_SHA256 = "a6d63d1c4cece2056cb58f55a5c1a4ed1c5f7f2ad5ee35d2bc52f64fff97f11d"


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    """Return a platform-independent digest of a saved PyTorch state dictionary."""
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(",".join(str(size) for size in tensor.shape).encode("ascii"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def verify_validation() -> dict[str, object]:
    predictions = pd.read_csv(PROJECT_ROOT / "results/validation/selected_model_predictions.csv")
    metrics = json.loads((PROJECT_ROOT / "results/validation/selected_model_metrics.json").read_text())
    accuracy = float(accuracy_score(predictions.true_label, predictions.predicted_label))
    macro_f1 = float(f1_score(predictions.true_label, predictions.predicted_label, average="macro", zero_division=0))
    assert len(predictions) == 480
    assert predictions.image_id.nunique() == 80
    assert int(predictions.is_correct.sum()) == 377
    assert np.isclose(accuracy, metrics["accuracy"])
    assert np.isclose(macro_f1, metrics["macro_f1"])
    assert metrics["seed"] == 44
    return {"correct": 377, "rows": 480, "accuracy": accuracy, "macro_f1": macro_f1}


def verify_validation_diagnostic() -> dict[str, object]:
    predictions = pd.read_csv(
        PROJECT_ROOT / "results/validation/category_rule_diagnostic.csv"
    )
    metrics = json.loads(
        (PROJECT_ROOT / "results/validation/category_rule_diagnostic.json").read_text()
    )
    learned_accuracy = float(
        accuracy_score(predictions.true_label, predictions.learned_relation_prediction)
    )
    learned_macro_f1 = float(
        f1_score(
            predictions.true_label,
            predictions.learned_relation_prediction,
            average="macro",
            zero_division=0,
        )
    )
    rule_accuracy = float(
        accuracy_score(predictions.true_label, predictions.category_rule_prediction)
    )
    rule_macro_f1 = float(
        f1_score(
            predictions.true_label,
            predictions.category_rule_prediction,
            average="macro",
            zero_division=0,
        )
    )
    assert len(predictions) == 480
    assert predictions.image_id.nunique() == 80
    assert int(predictions.learned_relation_correct.sum()) == 377
    assert int(predictions.category_rule_correct.sum()) == 350
    assert np.isclose(learned_accuracy, metrics["learned_relation_head"]["accuracy"])
    assert np.isclose(learned_macro_f1, metrics["learned_relation_head"]["macro_f1"])
    assert np.isclose(rule_accuracy, metrics["predicted_category_rule"]["accuracy"])
    assert np.isclose(rule_macro_f1, metrics["predicted_category_rule"]["macro_f1"])
    assert metrics["oracle_category_rule"]["accuracy"] == 1.0
    assert metrics["test_split_used"] is False
    assert metrics["new_training_performed"] is False
    return {
        "learned_relation_accuracy": learned_accuracy,
        "learned_relation_macro_f1": learned_macro_f1,
        "predicted_category_rule_accuracy": rule_accuracy,
        "predicted_category_rule_macro_f1": rule_macro_f1,
        "image_category_accuracy": metrics["auxiliary_category_predictions"][
            "image_category_accuracy"
        ],
        "text_category_accuracy": metrics["auxiliary_category_predictions"][
            "text_category_accuracy"
        ],
        "test_split_used": False,
        "new_training_performed": False,
    }


def verify_final_test() -> dict[str, object]:
    predictions = pd.read_csv(PROJECT_ROOT / "results/final_test/test_predictions.csv")
    metrics = json.loads((PROJECT_ROOT / "results/final_test/test_metrics.json").read_text())
    execution = json.loads((PROJECT_ROOT / "results/final_test/execution_state.json").read_text())
    consumption = json.loads((PROJECT_ROOT / "results/final_test/authorization_consumption.json").read_text())
    accuracy = float(accuracy_score(predictions.true_label, predictions.predicted_label))
    macro_f1 = float(f1_score(predictions.true_label, predictions.predicted_label, average="macro", zero_division=0))
    assert len(predictions) == 480
    assert predictions.image_id.nunique() == 80
    assert int(predictions.is_correct.sum()) == 354
    assert np.isclose(accuracy, 0.7375)
    assert np.isclose(macro_f1, 0.7382299830250852)
    assert np.isclose(accuracy, metrics["accuracy"])
    assert np.isclose(macro_f1, metrics["macro_f1"])
    assert metrics["authorization_consumed"] is True
    assert metrics["authorized_evaluations_completed"] == 1
    assert metrics["further_tuning_permitted"] is False
    assert execution.get("test_evaluation_executed") is True
    assert consumption.get("authorization_consumed") is True
    saved_confusion = pd.read_csv(PROJECT_ROOT / "results/final_test/test_confusion_matrix.csv", index_col=0).to_numpy()
    actual_confusion = confusion_matrix(predictions.true_label, predictions.predicted_label, labels=list(LABELS))
    assert np.array_equal(saved_confusion, actual_confusion)
    return {"correct": 354, "rows": 480, "accuracy": accuracy, "macro_f1": macro_f1, "inference_rerun": False}


def verify_models() -> dict[str, object]:
    checkpoint_path = PROJECT_ROOT / "models/selected_multimodal_model.pt"
    assert file_sha256(checkpoint_path) == EXPECTED_CHECKPOINT_SHA256
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    selected = MultimodalRelationCNN(int(payload["text_dimension"]))
    selected.load_state_dict(payload["state_dict"], strict=True)
    no_aux = MultimodalRelationCNNNoAuxiliary(int(payload["text_dimension"]))
    assert count_trainable_parameters(selected) == 58579
    assert count_trainable_parameters(no_aux) == 57923
    return {"selected_checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256, "selected_parameters": 58579, "no_auxiliary_parameters": 57923}


def verify_ablation() -> dict[str, object]:
    comparison = pd.read_csv(PROJECT_ROOT / "results/ablation/auxiliary_loss_ablation_comparison.csv")
    summary = json.loads((PROJECT_ROOT / "results/ablation/auxiliary_loss_ablation_summary.json").read_text())
    assert summary["no_auxiliary_aggregate"]["seeds"] == [43, 44, 45]
    assert summary["no_auxiliary_aggregate"]["runs"] == 3
    assert summary["no_auxiliary_aggregate"]["all_runs_collapsed_to_one_predicted_class"] is True
    assert summary["test_split_used"] is False
    assert np.isclose(summary["validation_accuracy_gap"], 0.45208333333333334)
    assert np.isclose(summary["validation_macro_f1_gap"], 0.6206176526591394)
    assert len(comparison) == 4
    for seed in (43, 44, 45):
        directory = PROJECT_ROOT / "results/ablation" / f"seed_{seed}"
        metrics = json.loads((directory / "metrics.json").read_text())
        predictions = pd.read_csv(directory / "validation_predictions.csv")
        checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=False)
        model = MultimodalRelationCNNNoAuxiliary(int(checkpoint["text_dimension"]))
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        assert checkpoint["seed"] == seed
        assert checkpoint["test_split_used"] is False
        assert len(predictions) == 480
        accuracy = accuracy_score(predictions.true_label, predictions.predicted_label)
        macro_f1 = f1_score(predictions.true_label, predictions.predicted_label, average="macro", zero_division=0)
        assert np.isclose(accuracy, metrics["validation_accuracy"])
        assert np.isclose(macro_f1, metrics["validation_macro_f1"])
        assert np.isclose(accuracy, 1 / 3)
        assert np.isclose(macro_f1, 1 / 6)
        assert predictions.predicted_label.nunique() == 1
    set_seed(44)
    with_aux = MultimodalRelationCNN(342)
    set_seed(44)
    without_aux = MultimodalRelationCNNNoAuxiliary(342)
    assert all(torch.equal(with_aux.state_dict()[key], value) for key, value in without_aux.state_dict().items())
    checkpoint44 = torch.load(PROJECT_ROOT / "results/ablation/seed_44/checkpoint.pt", map_location="cpu", weights_only=False)
    saved_initial_state = checkpoint44["initial_state_dict"]
    reference_state = without_aux.state_dict()
    assert set(saved_initial_state) == set(reference_state)
    assert all(saved_initial_state[key].shape == reference_state[key].shape for key in reference_state)
    initial_state_sha256 = state_dict_sha256(saved_initial_state)
    assert initial_state_sha256 == EXPECTED_SEED_44_INITIAL_STATE_SHA256
    assert summary["seed_44_shared_initialization"]["initial_state_sha256"] == initial_state_sha256
    return {
        "seeds": [43, 44, 45],
        "runs": 3,
        "all_collapsed": True,
        "test_split_used": False,
        "seed_44_shared_initialization": True,
        "seed_44_initial_state_sha256": initial_state_sha256,
        "verification_mode": "saved-state digest; portable across operating systems",
    }


def verify_notebook() -> dict[str, object]:
    notebooks = sorted(PROJECT_ROOT.glob("*.ipynb"))
    assert [path.name for path in notebooks] == ["project.ipynb"]
    notebook = nbformat.read(notebooks[0], as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code
    assert all(cell.execution_count is not None for cell in code)
    errors = [out for cell in code for out in cell.outputs if out.output_type == "error"]
    assert not errors
    return {"notebooks": 1, "cells": len(notebook.cells), "code_cells": len(code), "executed_code_cells": len(code), "errors": 0}


def verify_hygiene() -> dict[str, object]:
    """Verify distributable project content while tolerating local developer state.

    A working copy may legitimately contain ``.venv``, ``.git`` and runtime
    caches after it is opened in VS Code. These paths are excluded from the
    distributable-content scan instead of being treated as project defects.
    """
    forbidden_names = {"project_v3.ipynb", "project_final.ipynb"}
    local_environment_dirs = {".git", ".venv"}
    runtime_cache_dirs = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}
    excluded_dirs = local_environment_dirs | runtime_cache_dirs

    distributable_paths: list[str] = []
    observed_local_dirs: set[str] = set()
    observed_runtime_dirs: set[str] = set()

    for root, directories, files in os.walk(PROJECT_ROOT):
        root_path = Path(root)
        retained_directories: list[str] = []
        for directory in directories:
            relative = (root_path / directory).relative_to(PROJECT_ROOT).as_posix()
            if directory in local_environment_dirs:
                observed_local_dirs.add(relative)
            elif directory in runtime_cache_dirs:
                observed_runtime_dirs.add(relative)
            else:
                retained_directories.append(directory)
                distributable_paths.append(relative)
        directories[:] = retained_directories
        distributable_paths.extend(
            (root_path / filename).relative_to(PROJECT_ROOT).as_posix()
            for filename in files
        )

    assert not forbidden_names.intersection({Path(path).name for path in distributable_paths})
    assert not any("dataset_v2" in path.lower() for path in distributable_paths)
    return {
        "dataset_v2_paths": 0,
        "competing_notebooks": 0,
        "local_environment_dirs_ignored": sorted(observed_local_dirs),
        "runtime_cache_dirs_ignored": sorted(observed_runtime_dirs),
        "distributable_paths_checked": len(distributable_paths),
    }


def verify_hash_manifest() -> dict[str, object]:
    path = PROJECT_ROOT / "evidence/hashes.sha256"
    if not path.is_file():
        return {"present": False, "verified": 0}
    count = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        target = PROJECT_ROOT / relative
        assert target.is_file(), relative
        assert file_sha256(target) == expected, relative
        count += 1
    return {"present": True, "verified": count}


def run_verification(*, full_hashes: bool) -> dict[str, object]:
    lock = json.loads((PROJECT_ROOT / "data/locked_test/dataset_v3/test_lock.json").read_text())
    assert lock["test_lock_sha256"] == EXPECTED_TEST_LOCK_SHA256
    return {
        "status": "PASS_UNIFIED_PROJECT_VERIFIED",
        "data": validate_frozen_test_evidence(verify_hashes=full_hashes),
        "models": verify_models(),
        "validation": verify_validation(),
        "validation_diagnostic": verify_validation_diagnostic(),
        "ablation": verify_ablation(),
        "final_test": verify_final_test(),
        "notebook": verify_notebook(),
        "hygiene": verify_hygiene(),
        "hash_manifest": verify_hash_manifest(),
        "test_lock_sha256": EXPECTED_TEST_LOCK_SHA256,
        "new_final_test_inference_performed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the unified project and preserved evidence.")
    parser.add_argument("--full-hashes", action="store_true", help="Recompute all 640 image SHA-256 hashes.")
    args = parser.parse_args()
    print(json.dumps(run_verification(full_hashes=args.full_hashes), indent=2))


if __name__ == "__main__":
    main()
