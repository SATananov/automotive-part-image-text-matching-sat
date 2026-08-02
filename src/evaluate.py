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

from src.data import CATEGORIES, IMAGE_SIZE, LABELS, PROJECT_ROOT, load_images, load_relations
from src.models import MultimodalRelationCNN

CHECKPOINT_PATH = PROJECT_ROOT / "models" / "selected_multimodal_model.pt"
SAVED_PREDICTIONS = PROJECT_ROOT / "results" / "validation" / "selected_model_predictions.csv"


def evaluate_validation(checkpoint_path: Path = CHECKPOINT_PATH) -> dict[str, object]:
    """Evaluate the frozen selected checkpoint on validation only.

    The final test is intentionally unavailable here. Its one-time result is
    verified from preserved predictions by ``python -m src.verify``.
    """
    train = load_relations("train")
    validation = load_relations("validation")
    vectorizer = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), sublinear_tf=True, norm="l2"
    )
    vectorizer.fit(train["description"])
    validation_text = vectorizer.transform(validation["description"]).toarray().astype(np.float32)
    validation_images = load_images(validation, IMAGE_SIZE).astype(np.float32) / 255.0
    validation_images = np.transpose(validation_images, (0, 3, 1, 2))

    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = MultimodalRelationCNN(
        int(payload["text_dimension"]), number_of_part_categories=len(CATEGORIES)
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
    predicted: list[np.ndarray] = []
    with torch.no_grad():
        for image_batch, text_batch in loader:
            relation_logits, _, _ = model(image_batch, text_batch)
            predicted.append(relation_logits.argmax(dim=1).cpu().numpy())
    predicted_labels = np.asarray(LABELS)[np.concatenate(predicted)]
    true_labels = validation["label"].to_numpy()

    saved = pd.read_csv(SAVED_PREDICTIONS)
    if not np.array_equal(predicted_labels, saved["predicted_label"].to_numpy()):
        raise ValueError("Validation inference does not match saved canonical predictions")

    result = {
        "status": "PASS_SELECTED_MODEL_VALIDATION_REPRODUCED",
        "model_slug": payload["model_slug"],
        "correct_predictions": int(np.sum(true_labels == predicted_labels)),
        "total_predictions": len(validation),
        "accuracy": float(accuracy_score(true_labels, predicted_labels)),
        "macro_f1": float(
            f1_score(true_labels, predicted_labels, average="macro", zero_division=0)
        ),
        "saved_predictions_match": True,
        "test_manifest_read": False,
        "test_images_read": False,
        "test_split_used": False,
        "test_evaluation_executed": False,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce the selected validation result.")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    args = parser.parse_args()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    print(json.dumps(evaluate_validation(args.checkpoint), indent=2))


if __name__ == "__main__":
    main()
