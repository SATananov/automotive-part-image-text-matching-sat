from __future__ import annotations

import argparse
import copy
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import (
    CATEGORIES,
    LABELS,
    PROJECT_ROOT,
    encode_labels,
    load_images,
    load_relations,
    validate_development_dataset,
)
from src.models import (
    MultimodalRelationCNNNoAuxiliary,
    count_total_parameters,
    count_trainable_parameters,
)
from src.train import (
    BATCH_SIZE,
    CANONICAL_TORCH_VERSION,
    IMAGE_SIZE,
    MAX_EPOCHS,
    PATIENCE,
    augment_images,
    prediction_rows,
    require_canonical_torch_version,
    set_seed,
)

ALLOWED_SEEDS = (43, 44, 45)
MODEL_SLUG = "torch_multimodal_no_auxiliary_dataset_v3"


def prepare_features(
    train: pd.DataFrame,
    validation: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        norm="l2",
    )
    train_text = vectorizer.fit_transform(train["description"]).toarray().astype(np.float32)
    validation_text = vectorizer.transform(validation["description"]).toarray().astype(np.float32)

    train_images = load_images(train, IMAGE_SIZE).astype(np.float32) / 255.0
    validation_images = load_images(validation, IMAGE_SIZE).astype(np.float32) / 255.0
    train_images = np.transpose(train_images, (0, 3, 1, 2))
    validation_images = np.transpose(validation_images, (0, 3, 1, 2))
    return (
        train_images,
        train_text,
        encode_labels(train["label"]),
        validation_images,
        validation_text,
        encode_labels(validation["label"]),
        int(train_text.shape[1]),
    )


def evaluate(
    model: MultimodalRelationCNNNoAuxiliary,
    images: torch.Tensor,
    texts: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[float, float, np.ndarray]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(images, texts, labels),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )
    total_loss = 0.0
    correct = 0
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for image_batch, text_batch, label_batch in loader:
            logits = model(image_batch, text_batch)
            loss = criterion(logits, label_batch)
            total_loss += float(loss.item()) * len(label_batch)
            predicted = logits.argmax(dim=1)
            correct += int((predicted == label_batch).sum().item())
            predictions.append(predicted.cpu().numpy())
    return (
        total_loss / len(labels),
        correct / len(labels),
        np.concatenate(predictions),
    )


def train_model(
    model: MultimodalRelationCNNNoAuxiliary,
    train_images: np.ndarray,
    train_text: np.ndarray,
    train_labels: np.ndarray,
    validation_images: np.ndarray,
    validation_text: np.ndarray,
    validation_labels: np.ndarray,
    *,
    seed: int,
) -> tuple[np.ndarray, pd.DataFrame, float]:
    train_tensors = (
        torch.from_numpy(train_images).float(),
        torch.from_numpy(train_text).float(),
        torch.from_numpy(train_labels).long(),
    )
    validation_tensors = (
        torch.from_numpy(validation_images).float(),
        torch.from_numpy(validation_text).float(),
        torch.from_numpy(validation_labels).long(),
    )
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
        total_loss = 0.0
        correct = 0
        for image_batch, text_batch, label_batch in loader:
            image_batch = augment_images(image_batch)
            optimizer.zero_grad(set_to_none=True)
            logits = model(image_batch, text_batch)
            loss = criterion(logits, label_batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(label_batch)
            correct += int((logits.argmax(dim=1) == label_batch).sum().item())

        validation_loss, validation_accuracy, _ = evaluate(
            model, *validation_tensors
        )
        rows.append(
            {
                "epoch": epoch,
                "loss": total_loss / len(train_tensors[2]),
                "accuracy": correct / len(train_tensors[2]),
                "val_loss": validation_loss,
                "val_accuracy": validation_accuracy,
            }
        )
        if validation_loss < best_loss - 1e-4:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break

    model.load_state_dict(best_state)
    final_loss, _, predicted = evaluate(model, *validation_tensors)
    return predicted, pd.DataFrame(rows), final_loss


def run_ablation(seed: int, output: Path) -> dict[str, object]:
    if seed not in ALLOWED_SEEDS:
        raise ValueError(f"Seed must be one of {ALLOWED_SEEDS}, found {seed}")
    require_canonical_torch_version()
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    output.mkdir(parents=True)

    development = validate_development_dataset(verify_hashes=False)
    train = load_relations("train")
    validation = load_relations("validation")
    (
        train_images,
        train_text,
        train_labels,
        validation_images,
        validation_text,
        validation_labels,
        text_dimension,
    ) = prepare_features(train, validation)

    set_seed(seed)
    model = MultimodalRelationCNNNoAuxiliary(text_dimension)
    initial_state = copy.deepcopy(model.state_dict())
    predicted_indices, history, validation_loss = train_model(
        model,
        train_images,
        train_text,
        train_labels,
        validation_images,
        validation_text,
        validation_labels,
        seed=seed,
    )
    predicted_labels = np.asarray(LABELS)[predicted_indices]
    true_labels = validation["label"].to_numpy()
    accuracy = float(accuracy_score(true_labels, predicted_labels))
    macro_f1 = float(
        f1_score(true_labels, predicted_labels, average="macro", zero_division=0)
    )
    distribution = {
        label: int(np.sum(predicted_labels == label)) for label in LABELS
    }

    predictions = prediction_rows(
        validation,
        predicted_labels,
        model=f"Relation-only ablation, seed {seed}",
        model_slug=MODEL_SLUG,
    )
    predictions.insert(0, "seed", seed)
    predictions.to_csv(
        output / "validation_predictions.csv", index=False, lineterminator="\n"
    )
    history.to_csv(output / "training_history.csv", index=False, lineterminator="\n")

    checkpoint = {
        "model_slug": MODEL_SLUG,
        "dataset_version": "Dataset V3",
        "labels": list(LABELS),
        "categories": list(sorted(CATEGORIES)),
        "seed": seed,
        "text_dimension": text_dimension,
        "auxiliary_category_losses": False,
        "test_manifest_read": False,
        "test_images_read": False,
        "test_split_used": False,
        "initial_state_dict": initial_state,
        "state_dict": copy.deepcopy(model.state_dict()),
    }
    torch.save(checkpoint, output / "checkpoint.pt")

    trainable = count_trainable_parameters(model)
    total = count_total_parameters(model)
    (output / "architecture.txt").write_text(
        f"{model}\n\nTrainable parameters: {trainable:,}\n"
        f"Total parameters: {total:,}\n",
        encoding="utf-8",
        newline="\n",
    )

    result: dict[str, object] = {
        "status": "PASS_AUXILIARY_LOSS_ABLATION_RUN",
        "experiment": "relation_only_no_auxiliary",
        "model_slug": MODEL_SLUG,
        "seed": seed,
        "auxiliary_category_losses": False,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "training_rows": len(train),
        "validation_rows": len(validation),
        "validation_images": int(validation["image_id"].nunique()),
        "correct_predictions": int(np.sum(true_labels == predicted_labels)),
        "total_predictions": len(validation),
        "validation_accuracy": accuracy,
        "validation_macro_f1": macro_f1,
        "validation_loss": validation_loss,
        "epochs_recorded": len(history),
        "predicted_class_distribution": distribution,
        "predicted_classes": int(sum(count > 0 for count in distribution.values())),
        "collapsed_to_one_predicted_class": sum(count > 0 for count in distribution.values()) == 1,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "test_manifest_read": False,
        "test_images_read": False,
        "test_split_used": False,
        "test_evaluation_executed": False,
        "development_validation_status": development["status"],
    }
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one controlled validation-only auxiliary-loss ablation."
    )
    parser.add_argument("--seed", type=int, required=True, choices=ALLOWED_SEEDS)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    result = run_ablation(args.seed, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
