from __future__ import annotations

import argparse

import torch

from src.v4_data import (
    fit_text_vectorizer_v4,
    load_relations_v4,
    validate_dataset_v4,
)
from src.v4_models import MultimodalRelationModelV4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate the Dataset V4 foundation.")
    parser.add_argument("--full-hashes", action="store_true")
    parser.add_argument("--include-locked-test", action="store_true")
    parser.add_argument("--skip-model-smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = validate_dataset_v4(
        verify_hashes=args.full_hashes,
        include_locked_test=args.include_locked_test,
    )
    print(result["status"])
    print(f"Images:      {result['images']}")
    print(f"Categories:  {result['categories']}")
    print(f"Families:    {result['families']}")
    print(f"Split images: {result['split_images']}")
    print(
        "Relation rows:",
        {split: values["rows"] for split, values in result["relations"].items()},
    )
    print("Maximum overlap:", max(result["overlap"].values(), default=0))
    print("Test lock SHA-256:", result["test_lock_sha256"])

    if not args.skip_model_smoke:
        train = load_relations_v4("train").head(64)
        validation = load_relations_v4("validation").head(32)
        _, train_text, _ = fit_text_vectorizer_v4(train, validation)
        model = MultimodalRelationModelV4(
            train_text.shape[1],
            pretrained_image_encoder=False,
            freeze_image_backbone=True,
        )
        model.eval()
        with torch.no_grad():
            relation, image_category, text_category = model(
                torch.zeros(2, 3, 224, 224),
                torch.from_numpy(train_text[:2]),
            )
        if relation.shape != (2, 3):
            raise RuntimeError(f"Unexpected relation shape: {relation.shape}")
        if image_category.shape != (2, 50) or text_category.shape != (2, 50):
            raise RuntimeError("Unexpected category output shape")
        print("Model smoke test: PASS")


if __name__ == "__main__":
    main()
