from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from src.data import (
    CATEGORIES,
    FAMILIES,
    IMAGE_SIZE,
    LABELS,
    PROJECT_ROOT,
    load_images,
    load_relations,
)
from src.models import MultimodalRelationCNN

CHECKPOINT_PATH = PROJECT_ROOT / "models" / "selected_multimodal_model.pt"
SAVED_PREDICTIONS = PROJECT_ROOT / "results" / "validation" / "selected_model_predictions.csv"
DIAGNOSTIC_PREDICTIONS = (
    PROJECT_ROOT / "results" / "validation" / "category_rule_diagnostic.csv"
)
DIAGNOSTIC_METRICS = (
    PROJECT_ROOT / "results" / "validation" / "category_rule_diagnostic.json"
)


def relation_from_categories(image_category: str, text_category: str) -> str:
    """Apply the deterministic Dataset V3 relation rule."""
    if image_category not in FAMILIES or text_category not in FAMILIES:
        raise ValueError("Unknown automotive-part category")
    if image_category == text_category:
        return "MATCH"
    if FAMILIES[image_category] == FAMILIES[text_category]:
        return "PARTIAL_MATCH"
    return "MISMATCH"


def _run_validation_inference(
    checkpoint_path: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, str]:
    train = load_relations("train")
    validation = load_relations("validation")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm="l2",
    )
    vectorizer.fit(train["description"])
    validation_text = (
        vectorizer.transform(validation["description"])
        .toarray()
        .astype(np.float32)
    )
    validation_images = load_images(validation, IMAGE_SIZE).astype(np.float32) / 255.0
    validation_images = np.transpose(validation_images, (0, 3, 1, 2))

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = MultimodalRelationCNN(
        int(payload["text_dimension"]),
        number_of_part_categories=len(CATEGORIES),
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()

    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(validation_images).float(),
            torch.from_numpy(validation_text).float(),
        ),
        batch_size=32,
        shuffle=False,
    )
    relation_batches: list[np.ndarray] = []
    image_category_batches: list[np.ndarray] = []
    text_category_batches: list[np.ndarray] = []
    with torch.no_grad():
        for image_batch, text_batch in loader:
            relation_logits, image_logits, text_logits = model(image_batch, text_batch)
            relation_batches.append(relation_logits.argmax(dim=1).cpu().numpy())
            image_category_batches.append(image_logits.argmax(dim=1).cpu().numpy())
            text_category_batches.append(text_logits.argmax(dim=1).cpu().numpy())

    return (
        validation,
        np.concatenate(relation_batches),
        np.concatenate(image_category_batches),
        np.concatenate(text_category_batches),
        str(payload["model_slug"]),
    )


def build_category_rule_diagnostic(
    validation: pd.DataFrame,
    relation_indices: np.ndarray,
    image_category_indices: np.ndarray,
    text_category_indices: np.ndarray,
    *,
    model_slug: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Decompose the selected validation prediction without using the test split."""
    category_names = np.asarray(CATEGORIES)
    relation_names = np.asarray(LABELS)
    learned_predictions = relation_names[relation_indices]
    predicted_image_categories = category_names[image_category_indices]
    predicted_text_categories = category_names[text_category_indices]
    category_rule_predictions = np.asarray(
        [
            relation_from_categories(image_category, text_category)
            for image_category, text_category in zip(
                predicted_image_categories,
                predicted_text_categories,
            )
        ]
    )
    oracle_rule_predictions = np.asarray(
        [
            relation_from_categories(image_category, text_category)
            for image_category, text_category in zip(
                validation["part_category"],
                validation["text_category"],
            )
        ]
    )
    true_labels = validation["label"].to_numpy()
    if not np.array_equal(oracle_rule_predictions, true_labels):
        raise ValueError("The oracle category rule does not reproduce Dataset V3 labels")

    diagnostic = validation[
        [
            "sample_id",
            "image_id",
            "part_category",
            "text_category",
            "description",
            "label",
            "image_path",
        ]
    ].copy()
    diagnostic = diagnostic.rename(columns={"label": "true_label"})
    diagnostic["learned_relation_prediction"] = learned_predictions
    diagnostic["predicted_image_category"] = predicted_image_categories
    diagnostic["predicted_text_category"] = predicted_text_categories
    diagnostic["category_rule_prediction"] = category_rule_predictions
    diagnostic["oracle_rule_prediction"] = oracle_rule_predictions
    diagnostic["learned_relation_correct"] = (
        diagnostic["true_label"] == diagnostic["learned_relation_prediction"]
    )
    diagnostic["category_rule_correct"] = (
        diagnostic["true_label"] == diagnostic["category_rule_prediction"]
    )

    learned_accuracy = float(accuracy_score(true_labels, learned_predictions))
    learned_macro_f1 = float(
        f1_score(true_labels, learned_predictions, average="macro", zero_division=0)
    )
    category_rule_accuracy = float(
        accuracy_score(true_labels, category_rule_predictions)
    )
    category_rule_macro_f1 = float(
        f1_score(
            true_labels,
            category_rule_predictions,
            average="macro",
            zero_division=0,
        )
    )
    metrics: dict[str, object] = {
        "status": "PASS_VALIDATION_CATEGORY_RULE_DIAGNOSTIC",
        "model_slug": model_slug,
        "split": "validation",
        "rows": int(len(validation)),
        "independent_images": int(validation["image_id"].nunique()),
        "learned_relation_head": {
            "correct_predictions": int(np.sum(true_labels == learned_predictions)),
            "accuracy": learned_accuracy,
            "macro_f1": learned_macro_f1,
        },
        "auxiliary_category_predictions": {
            "image_category_accuracy": float(
                accuracy_score(
                    validation["part_category"],
                    predicted_image_categories,
                )
            ),
            "text_category_accuracy": float(
                accuracy_score(
                    validation["text_category"],
                    predicted_text_categories,
                )
            ),
        },
        "predicted_category_rule": {
            "correct_predictions": int(
                np.sum(true_labels == category_rule_predictions)
            ),
            "accuracy": category_rule_accuracy,
            "macro_f1": category_rule_macro_f1,
        },
        "oracle_category_rule": {
            "purpose": "dataset-construction check, not a deployable model",
            "correct_predictions": int(np.sum(true_labels == oracle_rule_predictions)),
            "accuracy": float(accuracy_score(true_labels, oracle_rule_predictions)),
            "macro_f1": float(
                f1_score(
                    true_labels,
                    oracle_rule_predictions,
                    average="macro",
                    zero_division=0,
                )
            ),
        },
        "learned_head_minus_predicted_rule": {
            "accuracy": learned_accuracy - category_rule_accuracy,
            "macro_f1": learned_macro_f1 - category_rule_macro_f1,
        },
        "interpretation": (
            "The rule diagnostic uses the selected model's auxiliary category "
            "predictions. It is a decomposition of the same model, not an "
            "independent baseline. The oracle result is 1.0 by dataset construction."
        ),
        "test_manifest_read": False,
        "test_images_read": False,
        "test_split_used": False,
        "test_evaluation_executed": False,
        "new_training_performed": False,
    }
    return diagnostic, metrics


def write_validation_diagnostic(
    checkpoint_path: Path = CHECKPOINT_PATH,
) -> dict[str, object]:
    validation, relation, image_categories, text_categories, model_slug = (
        _run_validation_inference(checkpoint_path)
    )
    diagnostic, metrics = build_category_rule_diagnostic(
        validation,
        relation,
        image_categories,
        text_categories,
        model_slug=model_slug,
    )
    DIAGNOSTIC_PREDICTIONS.parent.mkdir(parents=True, exist_ok=True)
    diagnostic.to_csv(DIAGNOSTIC_PREDICTIONS, index=False)
    DIAGNOSTIC_METRICS.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def evaluate_validation(checkpoint_path: Path = CHECKPOINT_PATH) -> dict[str, object]:
    """Evaluate the frozen selected checkpoint on validation only.

    The final test is intentionally unavailable here. Its one-time result is
    verified from preserved predictions by ``python -m src.verify``.
    """
    validation, relation, image_categories, text_categories, model_slug = (
        _run_validation_inference(checkpoint_path)
    )
    predicted_labels = np.asarray(LABELS)[relation]
    true_labels = validation["label"].to_numpy()

    saved = pd.read_csv(SAVED_PREDICTIONS)
    if not np.array_equal(predicted_labels, saved["predicted_label"].to_numpy()):
        raise ValueError("Validation inference does not match saved canonical predictions")

    diagnostic, diagnostic_metrics = build_category_rule_diagnostic(
        validation,
        relation,
        image_categories,
        text_categories,
        model_slug=model_slug,
    )
    saved_diagnostic = pd.read_csv(DIAGNOSTIC_PREDICTIONS)
    pd.testing.assert_frame_equal(
        diagnostic,
        saved_diagnostic,
        check_dtype=False,
    )
    saved_diagnostic_metrics = json.loads(DIAGNOSTIC_METRICS.read_text())
    if diagnostic_metrics != saved_diagnostic_metrics:
        raise ValueError("Validation diagnostic does not match its saved metrics")

    result = {
        "status": "PASS_SELECTED_MODEL_VALIDATION_REPRODUCED",
        "model_slug": model_slug,
        "correct_predictions": int(np.sum(true_labels == predicted_labels)),
        "total_predictions": len(validation),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(
            f1_score(true_labels, predicted_labels, average="macro", zero_division=0)
        ),
        "saved_predictions_match": True,
        "category_rule_diagnostic": diagnostic_metrics,
        "test_manifest_read": False,
        "test_images_read": False,
        "test_split_used": False,
        "test_evaluation_executed": False,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce the selected validation result."
    )
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument(
        "--write-diagnostic",
        action="store_true",
        help="Regenerate the saved validation-only category-rule diagnostic.",
    )
    args = parser.parse_args()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    if args.write_diagnostic:
        result = write_validation_diagnostic(args.checkpoint)
    else:
        result = evaluate_validation(args.checkpoint)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
