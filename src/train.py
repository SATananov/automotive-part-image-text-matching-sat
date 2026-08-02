from __future__ import annotations

import argparse
import copy
import json
import platform
import random
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import (
    CATEGORIES,
    IMAGE_SIZE,
    LABELS,
    PROJECT_ROOT,
    encode_labels,
    load_images,
    load_relations,
    validate_development_dataset,
)
from src.evaluation import grouped_paired_randomization
from src.models import ImageRelationCNN, MultimodalRelationCNN, TextRelationMLP

BASE_RANDOM_STATE = 42
SELECTED_MODEL_SEED = 44
MAX_EPOCHS = 80
PATIENCE = 10
BATCH_SIZE = 32
BOOTSTRAP_REPEATS = 2000
AUXILIARY_LOSS_WEIGHT = 0.40
CANONICAL_TORCH_VERSION = "2.10.0"
MAIN_MODEL_SLUG = "torch_multimodal_dataset_v3"


def require_canonical_torch_version() -> None:
    actual = torch.__version__.split("+", 1)[0]
    if actual != CANONICAL_TORCH_VERSION:
        raise RuntimeError(
            f"This project pins PyTorch {CANONICAL_TORCH_VERSION}; "
            f"current version is {torch.__version__}."
        )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def simple_image_features(data: pd.DataFrame) -> np.ndarray:
    return load_images(data, size=(16, 16)).reshape(len(data), -1).astype(np.float32) / 255.0


def torch_images(data: pd.DataFrame) -> np.ndarray:
    images = load_images(data, IMAGE_SIZE).astype(np.float32) / 255.0
    return np.transpose(images, (0, 3, 1, 2))


def augment_images(images: torch.Tensor) -> torch.Tensor:
    result = images.clone()
    flip_mask = torch.rand(len(result), device=result.device) < 0.5
    result[flip_mask] = torch.flip(result[flip_mask], dims=[3])
    brightness = 0.9 + 0.2 * torch.rand((len(result), 1, 1, 1), device=result.device)
    return torch.clamp(result * brightness, 0.0, 1.0)


def grouped_bootstrap_interval(
    true: np.ndarray,
    predicted: np.ndarray,
    groups: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    repeats: int = BOOTSTRAP_REPEATS,
    seed: int = BASE_RANDOM_STATE,
) -> tuple[float, float]:
    unique_groups = np.asarray(pd.unique(groups))
    indices = {group: np.flatnonzero(groups == group) for group in unique_groups}
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        rows = np.concatenate([indices[group] for group in sampled])
        values[index] = metric(true[rows], predicted[rows])
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def score_predictions(
    validation: pd.DataFrame,
    predicted: np.ndarray,
    *,
    model: str,
    model_slug: str,
    modality: str,
    training_data: str,
    image_category_accuracy: float | None = None,
    text_category_accuracy: float | None = None,
) -> dict[str, object]:
    true = validation["label"].to_numpy()
    groups = validation["image_id"].to_numpy()
    accuracy = float(accuracy_score(true, predicted))
    macro_f1 = float(f1_score(true, predicted, average="macro", zero_division=0))
    accuracy_ci = grouped_bootstrap_interval(
        true,
        predicted,
        groups,
        lambda y_true, y_pred: float(accuracy_score(y_true, y_pred)),
    )
    macro_f1_ci = grouped_bootstrap_interval(
        true,
        predicted,
        groups,
        lambda y_true, y_pred: float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        seed=BASE_RANDOM_STATE + 1,
    )
    return {
        "model": model,
        "model_slug": model_slug,
        "modality": modality,
        "training_data": training_data,
        "validation_accuracy": accuracy,
        "validation_macro_f1": macro_f1,
        "accuracy_ci_low": accuracy_ci[0],
        "accuracy_ci_high": accuracy_ci[1],
        "macro_f1_ci_low": macro_f1_ci[0],
        "macro_f1_ci_high": macro_f1_ci[1],
        "correct_predictions": int(np.sum(true == predicted)),
        "total_predictions": int(len(true)),
        "validation_images": int(validation["image_id"].nunique()),
        "validation_image_category_accuracy": image_category_accuracy,
        "validation_text_category_accuracy": text_category_accuracy,
    }


def prediction_rows(
    validation: pd.DataFrame,
    predicted: np.ndarray,
    *,
    model: str,
    model_slug: str,
) -> pd.DataFrame:
    columns = [
        "sample_id",
        "image_id",
        "part_group_id",
        "object_group_id",
        "part_category",
        "text_category",
        "source",
        "description",
        "label",
        "image_path",
    ]
    out = validation[columns].copy().rename(columns={"label": "true_label"})
    out["predicted_label"] = predicted
    out["is_correct"] = out["true_label"].eq(out["predicted_label"])
    out["model"] = model
    out["model_slug"] = model_slug
    return out[
        [
            "sample_id",
            "image_id",
            "part_group_id",
            "object_group_id",
            "part_category",
            "text_category",
            "source",
            "description",
            "true_label",
            "predicted_label",
            "is_correct",
            "image_path",
            "model",
            "model_slug",
        ]
    ]


def prepare_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> dict[str, object]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm="l2",
    )
    train_text_sparse = vectorizer.fit_transform(train["description"])
    validation_text_sparse = vectorizer.transform(validation["description"])
    category_names = tuple(sorted(CATEGORIES))
    category_to_index = {name: index for index, name in enumerate(category_names)}
    return {
        "vectorizer": vectorizer,
        "train_text_sparse": train_text_sparse,
        "validation_text_sparse": validation_text_sparse,
        "train_text": train_text_sparse.toarray().astype(np.float32),
        "validation_text": validation_text_sparse.toarray().astype(np.float32),
        "train_images": torch_images(train),
        "validation_images": torch_images(validation),
        "y_train": encode_labels(train["label"]),
        "y_validation": encode_labels(validation["label"]),
        "category_names": category_names,
        "train_image_categories": train["part_category"].map(category_to_index).to_numpy(np.int64),
        "train_text_categories": train["text_category"].map(category_to_index).to_numpy(np.int64),
        "validation_image_categories": validation["part_category"].map(category_to_index).to_numpy(np.int64),
        "validation_text_categories": validation["text_category"].map(category_to_index).to_numpy(np.int64),
    }


def evaluate_single_input(
    model: nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float, np.ndarray]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(features, labels), batch_size=BATCH_SIZE, shuffle=False)
    total_loss = 0.0
    correct = 0
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for x_batch, y_batch in loader:
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            total_loss += float(loss.item()) * len(y_batch)
            predicted = logits.argmax(dim=1)
            correct += int((predicted == y_batch).sum().item())
            predictions.append(predicted.cpu().numpy())
    return total_loss / len(labels), correct / len(labels), np.concatenate(predictions)


def fit_single_input_model(
    model: nn.Module,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    *,
    seed: int,
    augment_image_batches: bool,
) -> tuple[np.ndarray, pd.DataFrame]:
    x_train = torch.from_numpy(train_features).float()
    y_train = torch.from_numpy(train_labels).long()
    x_validation = torch.from_numpy(validation_features).float()
    y_validation = torch.from_numpy(validation_labels).long()
    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    stale_epochs = 0
    rows: list[dict[str, float | int]] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        for x_batch, y_batch in loader:
            if augment_image_batches:
                x_batch = augment_images(x_batch)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(y_batch)
            correct += int((logits.argmax(dim=1) == y_batch).sum().item())
        val_loss, val_accuracy, _ = evaluate_single_input(model, x_validation, y_validation)
        rows.append(
            {
                "epoch": epoch,
                "loss": total_loss / len(y_train),
                "accuracy": correct / len(y_train),
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
            }
        )
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break

    model.load_state_dict(best_state)
    _, _, predicted = evaluate_single_input(model, x_validation, y_validation)
    return predicted, pd.DataFrame(rows)


def evaluate_multimodal(
    model: MultimodalRelationCNN,
    images: torch.Tensor,
    texts: torch.Tensor,
    relations: torch.Tensor,
    image_categories: torch.Tensor,
    text_categories: torch.Tensor,
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(images, texts, relations, image_categories, text_categories),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    totals = {"loss": 0.0, "relation": 0, "image_category": 0, "text_category": 0}
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for image_batch, text_batch, relation_batch, image_cat_batch, text_cat_batch in loader:
            relation_logits, image_logits, text_logits = model(image_batch, text_batch)
            relation_loss = criterion(relation_logits, relation_batch)
            image_loss = criterion(image_logits, image_cat_batch)
            text_loss = criterion(text_logits, text_cat_batch)
            loss = relation_loss + AUXILIARY_LOSS_WEIGHT * (image_loss + text_loss)
            totals["loss"] += float(loss.item()) * len(relation_batch)
            relation_pred = relation_logits.argmax(dim=1)
            totals["relation"] += int((relation_pred == relation_batch).sum().item())
            totals["image_category"] += int((image_logits.argmax(dim=1) == image_cat_batch).sum().item())
            totals["text_category"] += int((text_logits.argmax(dim=1) == text_cat_batch).sum().item())
            predictions.append(relation_pred.cpu().numpy())
    count = len(relations)
    return (
        {
            "loss": totals["loss"] / count,
            "relation_accuracy": totals["relation"] / count,
            "image_category_accuracy": totals["image_category"] / count,
            "text_category_accuracy": totals["text_category"] / count,
        },
        np.concatenate(predictions),
    )


def fit_multimodal_model(
    model: MultimodalRelationCNN,
    features: dict[str, object],
    *,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, float]]:
    train_tensors = [
        torch.from_numpy(features["train_images"]).float(),
        torch.from_numpy(features["train_text"]).float(),
        torch.from_numpy(features["y_train"]).long(),
        torch.from_numpy(features["train_image_categories"]).long(),
        torch.from_numpy(features["train_text_categories"]).long(),
    ]
    validation_tensors = [
        torch.from_numpy(features["validation_images"]).float(),
        torch.from_numpy(features["validation_text"]).float(),
        torch.from_numpy(features["y_validation"]).long(),
        torch.from_numpy(features["validation_image_categories"]).long(),
        torch.from_numpy(features["validation_text_categories"]).long(),
    ]
    loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    stale_epochs = 0
    rows: list[dict[str, float | int]] = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        totals = {"loss": 0.0, "relation": 0, "image_category": 0, "text_category": 0}
        for image_batch, text_batch, relation_batch, image_cat_batch, text_cat_batch in loader:
            image_batch = augment_images(image_batch)
            optimizer.zero_grad(set_to_none=True)
            relation_logits, image_logits, text_logits = model(image_batch, text_batch)
            relation_loss = criterion(relation_logits, relation_batch)
            image_loss = criterion(image_logits, image_cat_batch)
            text_loss = criterion(text_logits, text_cat_batch)
            loss = relation_loss + AUXILIARY_LOSS_WEIGHT * (image_loss + text_loss)
            loss.backward()
            optimizer.step()
            totals["loss"] += float(loss.item()) * len(relation_batch)
            totals["relation"] += int((relation_logits.argmax(dim=1) == relation_batch).sum().item())
            totals["image_category"] += int((image_logits.argmax(dim=1) == image_cat_batch).sum().item())
            totals["text_category"] += int((text_logits.argmax(dim=1) == text_cat_batch).sum().item())

        validation_metrics, _ = evaluate_multimodal(model, *validation_tensors)
        count = len(train_tensors[2])
        rows.append(
            {
                "epoch": epoch,
                "loss": totals["loss"] / count,
                "accuracy": totals["relation"] / count,
                "image_category_accuracy": totals["image_category"] / count,
                "text_category_accuracy": totals["text_category"] / count,
                "val_loss": validation_metrics["loss"],
                "val_accuracy": validation_metrics["relation_accuracy"],
                "val_image_category_accuracy": validation_metrics["image_category_accuracy"],
                "val_text_category_accuracy": validation_metrics["text_category_accuracy"],
            }
        )
        if validation_metrics["loss"] < best_loss - 1e-4:
            best_loss = validation_metrics["loss"]
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break

    model.load_state_dict(best_state)
    final_metrics, predicted = evaluate_multimodal(model, *validation_tensors)
    return predicted, pd.DataFrame(rows), final_metrics


def save_model_summary(model: nn.Module, path: Path) -> None:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    path.write_text(
        f"{model}\n\nTrainable parameters: {trainable:,}\nTotal parameters: {total:,}\n",
        encoding="utf-8",
        newline="\n",
    )


def save_checkpoint(
    model: nn.Module,
    path: Path,
    *,
    model_slug: str,
    seed: int,
    text_dimension: int | None,
) -> None:
    torch.save(
        {
            "model_slug": model_slug,
            "dataset_version": "Dataset V3",
            "labels": list(LABELS),
            "categories": list(sorted(CATEGORIES)),
            "seed": seed,
            "text_dimension": text_dimension,
            "state_dict": copy.deepcopy(model.state_dict()),
        },
        path,
    )


def run_training(output: Path) -> dict[str, object]:
    require_canonical_torch_version()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)
    (output / "models").mkdir()

    validate_development_dataset(verify_hashes=False)
    train = load_relations("train")
    validation = load_relations("validation")
    features = prepare_features(train, validation)
    labels = np.asarray(LABELS)
    results: list[dict[str, object]] = []
    prediction_tables: list[pd.DataFrame] = []

    def record(
        name: str,
        slug: str,
        modality: str,
        training_data: str,
        predicted: np.ndarray,
        image_category_accuracy: float | None = None,
        text_category_accuracy: float | None = None,
    ) -> None:
        predicted = np.asarray(predicted)
        results.append(
            score_predictions(
                validation,
                predicted,
                model=name,
                model_slug=slug,
                modality=modality,
                training_data=training_data,
                image_category_accuracy=image_category_accuracy,
                text_category_accuracy=text_category_accuracy,
            )
        )
        prediction_tables.append(
            prediction_rows(validation, predicted, model=name, model_slug=slug)
        )

    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(np.zeros((len(train), 1)), train["label"])
    record(
        "Majority baseline",
        "majority",
        "none",
        "Dataset V3 train relations",
        majority.predict(np.zeros((len(validation), 1))),
    )

    text_baseline = LogisticRegression(max_iter=3000, random_state=BASE_RANDOM_STATE)
    text_baseline.fit(features["train_text_sparse"], train["label"])
    record(
        "TF-IDF + Logistic Regression",
        "tfidf_logistic_regression",
        "text",
        "Dataset V3 train relations",
        text_baseline.predict(features["validation_text_sparse"]),
    )

    flat_train = simple_image_features(train)
    flat_validation = simple_image_features(validation)
    image_baseline = LogisticRegression(max_iter=3000, random_state=BASE_RANDOM_STATE)
    image_baseline.fit(flat_train, train["label"])
    record(
        "Image pixels + Logistic Regression",
        "image_logistic_regression",
        "image",
        "Dataset V3 train relations",
        image_baseline.predict(flat_validation),
    )

    combined_train = sparse.hstack(
        [sparse.csr_matrix(flat_train), features["train_text_sparse"]], format="csr"
    )
    combined_validation = sparse.hstack(
        [sparse.csr_matrix(flat_validation), features["validation_text_sparse"]],
        format="csr",
    )
    combined_baseline = LogisticRegression(max_iter=4000, random_state=BASE_RANDOM_STATE)
    combined_baseline.fit(combined_train, train["label"])
    record(
        "Image + text Logistic Regression",
        "image_text_logistic_regression",
        "image + text",
        "Dataset V3 train relations",
        combined_baseline.predict(combined_validation),
    )

    text_dimension = int(features["train_text"].shape[1])
    set_seed(BASE_RANDOM_STATE)
    text_model = TextRelationMLP(text_dimension)
    text_pred, text_history = fit_single_input_model(
        text_model,
        features["train_text"],
        features["y_train"],
        features["validation_text"],
        features["y_validation"],
        seed=BASE_RANDOM_STATE,
        augment_image_batches=False,
    )
    text_history.to_csv(output / "torch_text_dataset_v3_training_history.csv", index=False)
    save_model_summary(text_model, output / "torch_text_dataset_v3_architecture.txt")
    record(
        "PyTorch neural text model",
        "torch_text_dataset_v3",
        "text",
        "Dataset V3 train relations",
        labels[text_pred],
    )

    set_seed(BASE_RANDOM_STATE + 1)
    image_model = ImageRelationCNN()
    image_pred, image_history = fit_single_input_model(
        image_model,
        features["train_images"],
        features["y_train"],
        features["validation_images"],
        features["y_validation"],
        seed=BASE_RANDOM_STATE + 1,
        augment_image_batches=True,
    )
    image_history.to_csv(output / "torch_cnn_image_dataset_v3_training_history.csv", index=False)
    save_model_summary(image_model, output / "torch_cnn_image_dataset_v3_architecture.txt")
    record(
        "PyTorch CNN image model",
        "torch_cnn_image_dataset_v3",
        "image",
        "Dataset V3 train relations",
        labels[image_pred],
    )

    set_seed(SELECTED_MODEL_SEED)
    multimodal_model = MultimodalRelationCNN(
        text_dimension,
        number_of_part_categories=len(CATEGORIES),
    )
    multimodal_pred, multimodal_history, auxiliary_metrics = fit_multimodal_model(
        multimodal_model,
        features,
        seed=SELECTED_MODEL_SEED,
    )
    multimodal_history.to_csv(
        output / "torch_multimodal_dataset_v3_training_history.csv", index=False
    )
    save_model_summary(multimodal_model, output / "torch_multimodal_dataset_v3_architecture.txt")
    save_checkpoint(
        multimodal_model,
        output / "models" / "selected_multimodal_model.pt",
        model_slug=MAIN_MODEL_SLUG,
        seed=SELECTED_MODEL_SEED,
        text_dimension=text_dimension,
    )
    record(
        "PyTorch multimodal CNN + text MLP",
        MAIN_MODEL_SLUG,
        "image + text",
        "480 curated Dataset V3 train images",
        labels[multimodal_pred],
        auxiliary_metrics["image_category_accuracy"],
        auxiliary_metrics["text_category_accuracy"],
    )

    comparison = pd.DataFrame(results).sort_values(
        ["validation_macro_f1", "validation_accuracy", "model"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)
    all_predictions = pd.concat(prediction_tables, ignore_index=True)
    selected_predictions = all_predictions.loc[
        all_predictions["model_slug"].eq(MAIN_MODEL_SLUG)
    ].reset_index(drop=True)
    comparison.to_csv(output / "model_comparison.csv", index=False)
    all_predictions.to_csv(output / "all_model_predictions.csv", index=False)
    selected_predictions.to_csv(output / "selected_model_predictions.csv", index=False)

    per_category = []
    for category, group in selected_predictions.groupby("part_category", sort=True):
        per_category.append(
            {
                "part_category": category,
                "samples": len(group),
                "independent_images": group["image_id"].nunique(),
                "accuracy": accuracy_score(group["true_label"], group["predicted_label"]),
                "macro_f1": f1_score(
                    group["true_label"], group["predicted_label"], average="macro", zero_division=0
                ),
            }
        )
    pd.DataFrame(per_category).to_csv(output / "per_category.csv", index=False)

    paired = pd.DataFrame(
        [
            grouped_paired_randomization(
                all_predictions, MAIN_MODEL_SLUG, "image_text_logistic_regression"
            ),
            grouped_paired_randomization(all_predictions, MAIN_MODEL_SLUG, "majority"),
        ]
    )
    paired.to_csv(output / "paired_comparisons.csv", index=False)

    selected = comparison.loc[comparison["model_slug"].eq(MAIN_MODEL_SLUG)].iloc[0]
    run_info = {
        "status": "PASS_RETRAINED_DEVELOPMENT_RESULTS",
        "dataset": "Dataset V3",
        "random_state": BASE_RANDOM_STATE,
        "selected_model_seed": SELECTED_MODEL_SEED,
        "auxiliary_loss_weight": AUXILIARY_LOSS_WEIGHT,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "training_rows": len(train),
        "validation_rows": len(validation),
        "main_validation_accuracy": float(selected.validation_accuracy),
        "main_validation_macro_f1": float(selected.validation_macro_f1),
        "test_manifest_read": False,
        "test_images_read": False,
        "test_split_used": False,
        "test_evaluation_executed": False,
    }
    (output / "run_info.json").write_text(
        json.dumps(run_info, indent=2) + "\n", encoding="utf-8"
    )
    return run_info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce the seven-model Dataset V3 development comparison."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results" / "retrained",
        help="New output directory. Existing directories are never overwritten.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    result = run_training(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
