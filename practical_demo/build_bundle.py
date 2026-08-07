from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
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

DEMO_ROOT = Path(__file__).resolve().parent

EXPECTED_VALIDATION_SHA256 = (
    "41e88e8ac40f03c38229f52bc13a893bc0cb0a414b6b2087890c655d8ca55cd4"
)

MODEL_DIR = DEMO_ROOT / "models"
MODEL_PATH = MODEL_DIR / "dataset_v4_deployment_classifier.pt"
VECTORIZER_PATH = MODEL_DIR / "dataset_v4_deployment_tfidf.pkl"
METADATA_PATH = MODEL_DIR / "dataset_v4_deployment_metadata.json"

CACHE_DIR = PROJECT_ROOT / ".cache" / "dataset_v4"

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
            "Frozen Step 03 validation summary has changed. "
            f"Expected {EXPECTED_VALIDATION_SHA256}, got {actual_hash}."
        )

    summary = json.loads(
        VALIDATION_SUMMARY.read_text(encoding="utf-8")
    )

    if summary.get("status") != "PASS_DATASET_V4_VALIDATION_EXPERIMENTS":
        raise ValueError("Frozen Step 03 validation status is not PASS.")

    if summary.get("development_only") is not True:
        raise ValueError("Step 03 must be development-only.")

    if summary.get("final_test_read") is not False:
        raise ValueError(
            "Step 03 unexpectedly reports final-test access."
        )

    if summary.get("smoke") is not False:
        raise ValueError("Frozen Step 03 result must not be a smoke run.")

    if summary.get("pretrained_resnet18") is not True:
        raise ValueError(
            "Frozen selection must use pretrained ResNet18."
        )

    if summary.get("selected_model") != "multimodal_auxiliary":
        raise ValueError("Unexpected frozen Dataset V4 model.")

    return summary


def fit_deployment_text_features(
    development: pd.DataFrame,
) -> tuple[TfidfVectorizer, np.ndarray]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=MAX_TEXT_FEATURES,
        sublinear_tf=True,
    )

    development_text = vectorizer.fit_transform(
        development["description"].astype(str)
    )

    return (
        vectorizer,
        development_text.toarray().astype(np.float32),
    )


def make_relation_arrays(
    relations: pd.DataFrame,
    text_features: np.ndarray,
    image_features: dict[str, np.ndarray],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    image_matrix = np.stack(
        [
            image_features[str(image_id)]
            for image_id in relations["image_id"]
        ]
    ).astype(np.float32)

    return (
        image_matrix,
        np.asarray(text_features, dtype=np.float32),
        encode_relation_labels_v4(relations["label"]),
        encode_categories_v4(relations["part_category"]),
        encode_categories_v4(relations["text_category"]),
    )


def make_loader(
    arrays: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    *,
    batch_size: int,
) -> DataLoader:
    image, text, relation, image_category, text_category = arrays

    dataset = TensorDataset(
        torch.from_numpy(image),
        torch.from_numpy(text),
        torch.from_numpy(relation),
        torch.from_numpy(image_category),
        torch.from_numpy(text_category),
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )


def train_fixed_epochs(
    model: MultimodalClassifierV4,
    arrays: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    auxiliary_weight: float,
    device: torch.device,
) -> list[float]:
    loader = make_loader(
        arrays,
        batch_size=batch_size,
    )

    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    criterion = nn.CrossEntropyLoss()

    losses: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        sample_count = 0

        for (
            image,
            text,
            relation,
            image_category,
            text_category,
        ) in loader:
            image = image.to(device)
            text = text.to(device)
            relation = relation.to(device)
            image_category = image_category.to(device)
            text_category = text_category.to(device)

            optimizer.zero_grad()

            relation_logits, image_logits, text_logits = model(
                image,
                text,
            )

            loss = criterion(
                relation_logits,
                relation,
            )

            loss = loss + auxiliary_weight * criterion(
                image_logits,
                image_category,
            )

            loss = loss + auxiliary_weight * criterion(
                text_logits,
                text_category,
            )

            loss.backward()
            optimizer.step()

            current = int(relation.shape[0])

            total_loss += float(loss.item()) * current
            sample_count += current

        mean_loss = total_loss / max(sample_count, 1)
        losses.append(mean_loss)

        print(
            f"epoch={epoch}/{epochs}, "
            f"train_loss={mean_loss:.4f}"
        )

    return losses


def build_bundle(*, overwrite: bool) -> dict[str, object]:
    outputs = [
        MODEL_PATH,
        VECTORIZER_PATH,
        METADATA_PATH,
    ]

    existing = [
        path for path in outputs
        if path.exists()
    ]

    if existing and not overwrite:
        raise FileExistsError(
            "Deployment bundle already exists. "
            "Use --overwrite only intentionally."
        )

    frozen = load_frozen_validation_choice()

    # DEVELOPMENT DATA ONLY.
    # Locked final-test relations are never loaded here.
    train = load_relations_v4("train")
    validation = load_relations_v4("validation")

    development = pd.concat(
        [train, validation],
        ignore_index=True,
    )

    vectorizer, development_text = (
        fit_deployment_text_features(development)
    )

    text_dimension = len(
        vectorizer.get_feature_names_out()
    )

    manifest = load_image_manifest_v4()

    development_ids = set(
        development["image_id"].astype(str)
    )

    device = choose_device()

    print("Device:", device)
    print(
        "Development relation rows:",
        len(development),
    )
    print(
        "Development independent images:",
        len(development_ids),
    )
    print("Locked final-test relations loaded: False")

    development_features = extract_image_features(
        manifest,
        development_ids,
        batch_size=64,
        device=device,
        pretrained=True,
        cache_path=(
            CACHE_DIR
            / "resnet18_train_validation_features.npz"
        ),
    )

    development_arrays = make_relation_arrays(
        development,
        development_text,
        development_features,
    )

    image_dimension = int(
        development_arrays[0].shape[1]
    )

    epochs = int(
        frozen["selected_best_epoch"]
    )

    batch_size = int(
        frozen["batch_size"]
    )

    learning_rate = float(
        frozen["learning_rate"]
    )

    auxiliary_weight = float(
        frozen["auxiliary_weight"]
    )

    # Same already-frozen final-training seed policy.
    seed = int(frozen["seed"]) + 3

    set_seed_v4(seed)

    model = MultimodalClassifierV4(
        image_dimension,
        text_dimension,
        auxiliary_heads=True,
    )

    print()
    print("Frozen deployment configuration")
    print("Model: multimodal_auxiliary")
    print("Epochs:", epochs)
    print("Batch size:", batch_size)
    print("Learning rate:", learning_rate)
    print("Auxiliary weight:", auxiliary_weight)
    print("Seed:", seed)

    losses = train_fixed_epochs(
        model,
        development_arrays,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        auxiliary_weight=auxiliary_weight,
        device=device,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    cpu_state = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }

    torch.save(
        cpu_state,
        MODEL_PATH,
    )

    with VECTORIZER_PATH.open("wb") as stream:
        pickle.dump(
            vectorizer,
            stream,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    metadata: dict[str, object] = {
        "status": "PASS_DATASET_V4_DEPLOYMENT_BUNDLE_CREATED",
        "purpose": (
            "Practical application demonstration. "
            "Not new benchmark evidence."
        ),
        "training_data": "Dataset V4 train + validation only",
        "locked_final_test_relations_loaded": False,
        "new_final_test_inference_performed": False,
        "post_test_tuning_performed": False,
        "configuration_source": (
            "Frozen Dataset V4 validation selection"
        ),
        "validation_summary_sha256": (
            EXPECTED_VALIDATION_SHA256
        ),
        "selected_model": "multimodal_auxiliary",
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "auxiliary_weight": auxiliary_weight,
        "seed": seed,
        "pretrained_resnet18": True,
        "text_max_features": MAX_TEXT_FEATURES,
        "text_features": text_dimension,
        "image_features": image_dimension,
        "labels": list(LABELS),
        "development_relation_rows": len(development),
        "development_independent_images": int(
            development["image_id"].nunique()
        ),
        "training_losses": losses,
        "classifier_file": MODEL_PATH.name,
        "tfidf_file": VECTORIZER_PATH.name,
        "classifier_sha256": file_sha256(MODEL_PATH),
        "tfidf_sha256": file_sha256(VECTORIZER_PATH),
    }

    METADATA_PATH.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a practical Dataset V4 deployment bundle "
            "using development data and the frozen configuration."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing deployment bundle.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metadata = build_bundle(
        overwrite=args.overwrite
    )

    print()
    print(
        "PASS_DATASET_V4_DEPLOYMENT_BUNDLE_CREATED"
    )
    print(
        "Locked final-test relations loaded:",
        metadata["locked_final_test_relations_loaded"],
    )
    print(
        "New final-test inference performed:",
        metadata["new_final_test_inference_performed"],
    )
    print(
        "Classifier:",
        MODEL_PATH,
    )
    print(
        "TF-IDF:",
        VECTORIZER_PATH,
    )


if __name__ == "__main__":
    main()
