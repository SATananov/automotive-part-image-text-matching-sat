from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.v4_data import (
    LABELS,
    PROJECT_ROOT,
    encode_categories_v4,
    encode_relation_labels_v4,
    load_image_manifest_v4,
    load_relations_v4,
)
from src.v4_training import MultimodalClassifierV4, set_seed_v4
from tools.run_dataset_v4_validation import choose_device, extract_image_features

RESULT_DIR = PROJECT_ROOT / "results" / "dataset_v4"
VALIDATION_SUMMARY = RESULT_DIR / "step03_validation_summary.json"
FINAL_SUMMARY = RESULT_DIR / "step04_final_test_summary.json"
FINAL_PREDICTIONS = RESULT_DIR / "step04_final_test_predictions.csv"
TEST_LOCK = PROJECT_ROOT / "data" / "locked_test" / "dataset_v4" / "test_lock.json"
TEST_RELATIONS = PROJECT_ROOT / "data" / "locked_test" / "dataset_v4" / "test_relations.csv"
CACHE_DIR = PROJECT_ROOT / ".cache" / "dataset_v4"
EXPECTED_VALIDATION_SHA256 = "3e7bd14a0aecee3a574fe9e51a22a01d956bf3be10a7260e872dd61598118769"
MAX_TEXT_FEATURES = 512


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_validation_choice() -> dict[str, object]:
    if not VALIDATION_SUMMARY.is_file():
        raise FileNotFoundError(VALIDATION_SUMMARY)
    actual_hash = file_sha256(VALIDATION_SUMMARY)
    if actual_hash != EXPECTED_VALIDATION_SHA256:
        raise ValueError(
            "The frozen Step 03 validation summary has changed. "
            f"Expected {EXPECTED_VALIDATION_SHA256}, got {actual_hash}."
        )
    summary = json.loads(VALIDATION_SUMMARY.read_text(encoding="utf-8"))
    if summary.get("status") != "PASS_DATASET_V4_VALIDATION_EXPERIMENTS":
        raise ValueError("Step 03 validation status is not PASS")
    if summary.get("development_only") is not True:
        raise ValueError("Step 03 must be development-only")
    if summary.get("final_test_read") is not False:
        raise ValueError("Step 03 reports that the final test was already read")
    if summary.get("smoke") is not False:
        raise ValueError("The frozen Step 03 result must not be a smoke run")
    if summary.get("pretrained_resnet18") is not True:
        raise ValueError("The selected validation run must use pretrained ResNet18")
    if summary.get("selected_model") != "multimodal_auxiliary":
        raise ValueError("Unexpected frozen model selection")
    return summary


def load_test_lock() -> dict[str, object]:
    lock = json.loads(TEST_LOCK.read_text(encoding="utf-8"))
    if lock.get("locked") is not True or lock.get("created_before_training") is not True:
        raise ValueError("Dataset V4 final-test lock is not valid")
    expected = str(lock.get("test_relations_sha256", ""))
    actual = file_sha256(TEST_RELATIONS)
    if actual != expected:
        raise ValueError("Locked final-test relation hash does not match test_lock.json")
    return lock


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def fit_final_text_features(
    development: pd.DataFrame,
    final_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, int]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=MAX_TEXT_FEATURES,
        sublinear_tf=True,
    )
    development_text = vectorizer.fit_transform(
        development["description"].astype(str)
    )
    final_text = vectorizer.transform(final_test["description"].astype(str))
    return (
        development_text.toarray().astype(np.float32),
        final_text.toarray().astype(np.float32),
        len(vectorizer.get_feature_names_out()),
    )


def make_relation_arrays(
    relations: pd.DataFrame,
    text_features: np.ndarray,
    image_features: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image_matrix = np.stack(
        [image_features[str(image_id)] for image_id in relations["image_id"]]
    ).astype(np.float32)
    return (
        image_matrix,
        np.asarray(text_features, dtype=np.float32),
        encode_relation_labels_v4(relations["label"]),
        encode_categories_v4(relations["part_category"]),
        encode_categories_v4(relations["text_category"]),
    )


def make_final_test_arrays(
    relations: pd.DataFrame,
    text_features: np.ndarray,
    image_features: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build final-test inputs without feeding category ground truth to the model."""
    image_matrix = np.stack(
        [image_features[str(image_id)] for image_id in relations["image_id"]]
    ).astype(np.float32)
    dummy_categories = np.zeros(len(relations), dtype=np.int64)
    return (
        image_matrix,
        np.asarray(text_features, dtype=np.float32),
        encode_relation_labels_v4(relations["label"]),
        dummy_categories,
        dummy_categories.copy(),
    )


def make_loader(
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    image, text, relation, image_category, text_category = arrays
    dataset = TensorDataset(
        torch.from_numpy(image),
        torch.from_numpy(text),
        torch.from_numpy(relation),
        torch.from_numpy(image_category),
        torch.from_numpy(text_category),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_fixed_epochs(
    model: MultimodalClassifierV4,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    auxiliary_weight: float,
    device: torch.device,
) -> list[float]:
    loader = make_loader(arrays, batch_size=batch_size, shuffle=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    losses: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        sample_count = 0
        for image, text, relation, image_category, text_category in loader:
            image = image.to(device)
            text = text.to(device)
            relation = relation.to(device)
            image_category = image_category.to(device)
            text_category = text_category.to(device)

            optimizer.zero_grad()
            relation_logits, image_logits, text_logits = model(image, text)
            loss = criterion(relation_logits, relation)
            loss = loss + auxiliary_weight * criterion(image_logits, image_category)
            loss = loss + auxiliary_weight * criterion(text_logits, text_category)
            loss.backward()
            optimizer.step()

            current = int(relation.shape[0])
            total_loss += float(loss.item()) * current
            sample_count += current

        mean_loss = total_loss / max(sample_count, 1)
        losses.append(mean_loss)
        print(f"  epoch={epoch}, train_loss={mean_loss:.4f}")
    return losses


def evaluate_once(
    model: MultimodalClassifierV4,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, float, float, list[list[int]], dict[str, dict[str, float | int]]]:
    loader = make_loader(arrays, batch_size=batch_size, shuffle=False)
    model.eval()
    truth: list[int] = []
    predicted: list[int] = []
    with torch.no_grad():
        for image, text, relation, _, _ in loader:
            relation_logits, _, _ = model(image.to(device), text.to(device))
            truth.extend(relation.numpy().tolist())
            predicted.extend(relation_logits.argmax(dim=1).cpu().numpy().tolist())

    truth_array = np.asarray(truth, dtype=np.int64)
    predicted_array = np.asarray(predicted, dtype=np.int64)
    accuracy = float(accuracy_score(truth_array, predicted_array))
    macro_f1 = float(f1_score(truth_array, predicted_array, average="macro"))
    matrix = confusion_matrix(
        truth_array, predicted_array, labels=list(range(len(LABELS)))
    ).astype(int).tolist()
    precision, recall, f1, support = precision_recall_fscore_support(
        truth_array,
        predicted_array,
        labels=list(range(len(LABELS))),
        zero_division=0,
    )
    per_label = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(LABELS)
    }
    return predicted_array, accuracy, macro_f1, matrix, per_label


def check_only() -> None:
    summary = load_frozen_validation_choice()
    lock = load_test_lock()
    print("PASS_DATASET_V4_FINAL_TEST_POLICY_CHECK")
    print(f"Frozen model:       {summary['selected_model']}")
    print(f"Frozen best epoch:  {summary['selected_best_epoch']}")
    print(f"Validation SHA-256: {EXPECTED_VALIDATION_SHA256}")
    print(f"Test-lock SHA-256:  {lock['test_relations_sha256']}")
    print("Final test relations loaded: False")


def run_final_test(args: argparse.Namespace) -> dict[str, object]:
    if not args.confirm_final_test:
        raise SystemExit(
            "Final test not opened. Run again with --confirm-final-test only after "
            "the evaluation code has been committed and model selection is frozen."
        )
    if FINAL_SUMMARY.exists() or FINAL_PREDICTIONS.exists():
        raise FileExistsError(
            "Dataset V4 final-test result already exists. This runner is intentionally one-time."
        )

    frozen = load_frozen_validation_choice()
    lock = load_test_lock()
    train = load_relations_v4("train")
    validation = load_relations_v4("validation")
    final_test = load_relations_v4("test", allow_locked_test=True)
    development = pd.concat([train, validation], ignore_index=True)

    development_text, final_text, text_dimension = fit_final_text_features(
        development, final_test
    )
    manifest = load_image_manifest_v4()
    development_ids = set(development["image_id"].astype(str))
    final_ids = set(final_test["image_id"].astype(str))
    device = choose_device()

    development_features = extract_image_features(
        manifest,
        development_ids,
        batch_size=args.feature_batch_size,
        device=device,
        pretrained=True,
        cache_path=CACHE_DIR / "resnet18_train_validation_features.npz",
    )
    final_features = extract_image_features(
        manifest,
        final_ids,
        batch_size=args.feature_batch_size,
        device=device,
        pretrained=True,
        cache_path=CACHE_DIR / "resnet18_final_test_features.npz",
    )

    development_arrays = make_relation_arrays(
        development, development_text, development_features
    )
    final_arrays = make_final_test_arrays(final_test, final_text, final_features)
    image_dimension = int(development_arrays[0].shape[1])

    epochs = int(frozen["selected_best_epoch"])
    batch_size = int(frozen["batch_size"])
    learning_rate = float(frozen["learning_rate"])
    auxiliary_weight = float(frozen["auxiliary_weight"])
    seed = int(frozen["seed"]) + 3

    set_seed_v4(seed)
    model = MultimodalClassifierV4(
        image_dimension,
        text_dimension,
        auxiliary_heads=True,
    )
    print(
        f"Training frozen final model for {epochs} epochs on train + validation..."
    )
    training_losses = train_fixed_epochs(
        model,
        development_arrays,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        auxiliary_weight=auxiliary_weight,
        device=device,
    )

    print("Evaluating locked final test exactly once...")
    predictions, accuracy, macro_f1, matrix, per_label = evaluate_once(
        model,
        final_arrays,
        batch_size=batch_size,
        device=device,
    )

    prediction_table = final_test[
        ["sample_id", "image_id", "part_category", "text_category", "label"]
    ].copy()
    prediction_table["prediction"] = [LABELS[index] for index in predictions]
    prediction_table["correct"] = prediction_table["label"].eq(
        prediction_table["prediction"]
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    prediction_table.to_csv(FINAL_PREDICTIONS, index=False, lineterminator="\n")
    summary: dict[str, object] = {
        "status": "PASS_DATASET_V4_FINAL_TEST_EVALUATED",
        "final_test_read": True,
        "new_final_test_inference_performed": True,
        "post_test_tuning_allowed": False,
        "selected_model": frozen["selected_model"],
        "selection_source": "frozen Step 03 validation result",
        "validation_summary_sha256": EXPECTED_VALIDATION_SHA256,
        "selected_validation_accuracy": frozen["selected_validation_accuracy"],
        "selected_validation_macro_f1": frozen["selected_validation_macro_f1"],
        "final_training_data": "train + validation",
        "final_training_relation_rows": len(development),
        "final_training_independent_images": int(development["image_id"].nunique()),
        "final_training_epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "auxiliary_weight": auxiliary_weight,
        "seed": seed,
        "device": str(device),
        "pretrained_resnet18": True,
        "text_max_features": MAX_TEXT_FEATURES,
        "text_features": text_dimension,
        "image_features": image_dimension,
        "final_test_relation_rows": len(final_test),
        "final_test_independent_images": int(final_test["image_id"].nunique()),
        "test_lock_sha256": lock["test_relations_sha256"],
        "final_test_accuracy": accuracy,
        "final_test_macro_f1": macro_f1,
        "labels": list(LABELS),
        "confusion_matrix": matrix,
        "per_label": per_label,
        "training_losses": training_losses,
        "predictions_sha256": file_sha256(FINAL_PREDICTIONS),
        "evaluation_code_git_head": git_head(),
    }
    FINAL_SUMMARY.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the one-time Dataset V4 locked final-test evaluation."
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check frozen validation and lock metadata without loading test relations.",
    )
    parser.add_argument(
        "--confirm-final-test",
        action="store_true",
        help="Explicitly allow the one-time locked final-test evaluation.",
    )
    parser.add_argument("--feature-batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_only:
        check_only()
        return
    summary = run_final_test(args)
    print("")
    print(summary["status"])
    print(f"Selected model:       {summary['selected_model']}")
    print(f"Final test rows:      {summary['final_test_relation_rows']}")
    print(f"Independent images:   {summary['final_test_independent_images']}")
    print(f"Final test accuracy:  {float(summary['final_test_accuracy']):.4f}")
    print(f"Final test macro F1:  {float(summary['final_test_macro_f1']):.4f}")
    print("Post-test tuning:     NOT ALLOWED")


if __name__ == "__main__":
    main()
