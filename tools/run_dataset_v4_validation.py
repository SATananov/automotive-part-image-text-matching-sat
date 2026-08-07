from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from src.v4_data import (
    LABELS,
    PROJECT_ROOT,
    category_to_index_v4,
    encode_categories_v4,
    encode_relation_labels_v4,
    fit_text_vectorizer_v4,
    image_transform_v4,
    load_image_manifest_v4,
)
from src.v4_training import (
    ImageOnlyClassifierV4,
    MultimodalClassifierV4,
    TextOnlyClassifierV4,
    load_development_relations_v4,
    set_seed_v4,
    train_classifier_v4,
)

DEFAULT_SEED = 44
CACHE_DIR = PROJECT_ROOT / ".cache" / "dataset_v4"
RESULT_DIR = PROJECT_ROOT / "results" / "dataset_v4"


class ImageFeatureDatasetV4(Dataset[tuple[str, torch.Tensor]]):
    def __init__(self, rows: pd.DataFrame) -> None:
        self.rows = rows.reset_index(drop=True)
        self.transform = image_transform_v4(training=False)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        row = self.rows.iloc[index]
        path = PROJECT_ROOT / str(row["image_path"])
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return str(row["image_id"]), tensor


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def choose_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_smoke_images(
    manifest: pd.DataFrame,
    *,
    train_images: int = 80,
    validation_images: int = 40,
    seed: int,
) -> tuple[set[str], set[str]]:
    train = manifest.loc[manifest["split"].eq("train")]
    validation = manifest.loc[manifest["split"].eq("validation")]
    train_ids = set(
        train.sample(n=min(train_images, len(train)), random_state=seed)["image_id"]
    )
    validation_ids = set(
        validation.sample(
            n=min(validation_images, len(validation)), random_state=seed
        )["image_id"]
    )
    return train_ids, validation_ids


def extract_image_features(
    manifest: pd.DataFrame,
    image_ids: set[str],
    *,
    batch_size: int,
    device: torch.device,
    pretrained: bool,
    cache_path: Path | None,
) -> dict[str, np.ndarray]:
    selected = manifest.loc[manifest["image_id"].isin(image_ids)].copy()
    selected = selected.sort_values("image_id").reset_index(drop=True)
    if len(selected) != len(image_ids):
        raise ValueError("Some requested development images are missing from the manifest")

    if cache_path is not None and cache_path.is_file():
        cached = np.load(cache_path, allow_pickle=False)
        cached_ids = cached["image_ids"].astype(str)
        cached_features = cached["features"].astype(np.float32)
        if set(cached_ids) == image_ids:
            return {
                image_id: cached_features[index]
                for index, image_id in enumerate(cached_ids)
            }

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval().to(device)

    loader = DataLoader(
        ImageFeatureDatasetV4(selected),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    output_ids: list[str] = []
    output_features: list[np.ndarray] = []
    with torch.no_grad():
        for ids, images in loader:
            features = model(images.to(device)).cpu().numpy().astype(np.float32)
            output_ids.extend(list(ids))
            output_features.append(features)

    matrix = np.concatenate(output_features, axis=0)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            image_ids=np.asarray(output_ids),
            features=matrix,
        )
    return {image_id: matrix[index] for index, image_id in enumerate(output_ids)}


def relation_arrays(
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


def run(args: argparse.Namespace) -> dict[str, object]:
    set_seed_v4(args.seed)
    train, validation = load_development_relations_v4()
    manifest = load_image_manifest_v4()

    if args.smoke:
        train_ids, validation_ids = select_smoke_images(manifest, seed=args.seed)
        train = train.loc[train["image_id"].isin(train_ids)].reset_index(drop=True)
        validation = validation.loc[
            validation["image_id"].isin(validation_ids)
        ].reset_index(drop=True)
    else:
        train_ids = set(train["image_id"].astype(str))
        validation_ids = set(validation["image_id"].astype(str))

    vectorizer, train_text, validation_text = fit_text_vectorizer_v4(
        train,
        validation,
        max_features=args.max_text_features,
    )
    text_dimension = len(vectorizer.get_feature_names_out())

    all_image_ids = train_ids | validation_ids
    cache_path = None
    if not args.smoke:
        cache_path = CACHE_DIR / "resnet18_train_validation_features.npz"
    device = choose_device()
    image_features = extract_image_features(
        manifest,
        all_image_ids,
        batch_size=args.feature_batch_size,
        device=device,
        pretrained=not args.no_pretrained,
        cache_path=cache_path,
    )

    train_arrays = relation_arrays(train, train_text, image_features)
    validation_arrays = relation_arrays(validation, validation_text, image_features)
    image_dimension = int(train_arrays[0].shape[1])

    model_builders = [
        (
            "text_only",
            lambda: TextOnlyClassifierV4(text_dimension),
        ),
        (
            "image_only",
            lambda: ImageOnlyClassifierV4(image_dimension),
        ),
        (
            "multimodal_no_auxiliary",
            lambda: MultimodalClassifierV4(
                image_dimension, text_dimension, auxiliary_heads=False
            ),
        ),
        (
            "multimodal_auxiliary",
            lambda: MultimodalClassifierV4(
                image_dimension, text_dimension, auxiliary_heads=True
            ),
        ),
    ]

    results: list[dict[str, object]] = []
    for offset, (name, build_model) in enumerate(model_builders):
        set_seed_v4(args.seed + offset)
        print(f"Training {name}...")
        result, _ = train_classifier_v4(
            name,
            build_model(),
            train_arrays,
            validation_arrays,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
            auxiliary_weight=args.auxiliary_weight,
        )
        results.append(
            {
                "name": result.name,
                "best_epoch": result.best_epoch,
                "validation_accuracy": result.accuracy,
                "validation_macro_f1": result.macro_f1,
                "confusion_matrix": result.confusion_matrix,
                "history": result.history,
            }
        )
        print(
            f"  best epoch={result.best_epoch}, "
            f"accuracy={result.accuracy:.4f}, macro_f1={result.macro_f1:.4f}"
        )

    ranked = sorted(
        results,
        key=lambda item: (
            float(item["validation_macro_f1"]),
            float(item["validation_accuracy"]),
        ),
        reverse=True,
    )
    selected = ranked[0]
    summary: dict[str, object] = {
        "status": "PASS_DATASET_V4_VALIDATION_EXPERIMENTS",
        "development_only": True,
        "final_test_read": False,
        "seed": args.seed,
        "device": str(device),
        "pretrained_resnet18": not args.no_pretrained,
        "train_relation_rows": len(train),
        "validation_relation_rows": len(validation),
        "train_independent_images": int(train["image_id"].nunique()),
        "validation_independent_images": int(validation["image_id"].nunique()),
        "text_features": text_dimension,
        "image_features": image_dimension,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "auxiliary_weight": args.auxiliary_weight,
        "labels": list(LABELS),
        "models": results,
        "selected_by": "highest validation macro F1, then validation accuracy",
        "selected_model": selected["name"],
        "selected_best_epoch": selected["best_epoch"],
        "selected_validation_accuracy": selected["validation_accuracy"],
        "selected_validation_macro_f1": selected["validation_macro_f1"],
        "smoke": bool(args.smoke),
    }

    if not args.smoke:
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = RESULT_DIR / "step03_validation_summary.json"
        output_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        history_rows: list[dict[str, object]] = []
        for model in results:
            for row in model["history"]:
                history_rows.append({"model": model["name"], **row})
        pd.DataFrame(history_rows).to_csv(
            RESULT_DIR / "step03_training_history.csv",
            index=False,
            lineterminator="\n",
        )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Dataset V4 validation-only model experiments."
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--feature-batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--auxiliary-weight", type=float, default=0.15)
    parser.add_argument("--max-text-features", type=int, default=512)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Use an untrained ResNet18. Intended only for smoke tests.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use a small development subset and do not save result files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args)
    print("")
    print(summary["status"])
    print(f"Development only: {summary['development_only']}")
    print(f"Final test read:   {summary['final_test_read']}")
    print(f"Selected model:    {summary['selected_model']}")
    print(
        "Selected validation: "
        f"accuracy={float(summary['selected_validation_accuracy']):.4f}, "
        f"macro_f1={float(summary['selected_validation_macro_f1']):.4f}"
    )


if __name__ == "__main__":
    main()
