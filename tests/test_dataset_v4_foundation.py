from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from src.v4_data import (
    DatasetV4Relations,
    category_names_v4,
    fit_text_vectorizer_v4,
    load_image_manifest_v4,
    load_relations_v4,
    validate_dataset_v4,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_dataset_v4_counts_and_uniqueness() -> None:
    manifest = load_image_manifest_v4()
    assert len(manifest) == 9239
    assert manifest["sha256"].nunique() == 9239
    assert manifest.groupby("split").size().to_dict() == {
        "test": 1388,
        "train": 6463,
        "validation": 1388,
    }
    assert len(category_names_v4()) == 50


def test_dataset_v4_explicit_license_record_matches_manifest() -> None:
    licenses = pd.read_csv(PROJECT_ROOT / "data/licenses_dataset_v4.csv")
    manifest = load_image_manifest_v4()
    assert len(licenses) == 1
    assert licenses.loc[0, "source_url"] == "https://www.kaggle.com/datasets/gpiosenka/car-parts-40-classes"
    assert licenses.loc[0, "recorded_license"] == "Apache-2.0"
    assert int(licenses.loc[0, "selected_images"]) == len(manifest) == 9239
    recorded_categories = set(str(licenses.loc[0, "selected_categories"]).split(";"))
    assert recorded_categories == set(manifest["part_category"].unique())
    assert licenses.loc[0, "manifest_path"] == "data/manifests/dataset_v4/images.csv"


def test_dataset_v4_development_boundaries_and_balancing() -> None:
    result = validate_dataset_v4(verify_hashes=False, include_locked_test=False)
    assert result["status"] == "PASS_DATASET_V4_FOUNDATION"
    assert result["locked_test_relations_read"] is False
    assert all(value == 0 for value in result["overlap"].values())
    assert result["relations"]["train"]["image_side_balanced"] is True
    assert result["relations"]["train"]["text_category_balanced"] is True
    assert result["relations"]["train"]["exact_description_balanced"] is True


def test_dataset_v4_locked_test_requires_permission() -> None:
    with pytest.raises(PermissionError):
        load_relations_v4("test")
    assert len(load_relations_v4("test", allow_locked_test=True)) == 9030


def test_dataset_v4_text_vectorizer_and_lazy_dataset() -> None:
    train = load_relations_v4("train").head(24)
    validation = load_relations_v4("validation").head(12)
    _, train_text, validation_text = fit_text_vectorizer_v4(train, validation)
    assert train_text.shape[0] == len(train)
    assert validation_text.shape[0] == len(validation)
    assert train_text.dtype.name == "float32"

    dataset = DatasetV4Relations(train, train_text, training=False)
    image, text, relation, image_category, text_category = dataset[0]
    assert image.shape == (3, 224, 224)
    assert text.shape == (train_text.shape[1],)
    assert image.dtype == torch.float32
    assert relation.ndim == image_category.ndim == text_category.ndim == 0
