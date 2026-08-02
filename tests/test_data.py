from __future__ import annotations

import pandas as pd
import pytest

from src.data import PROJECT_ROOT, load_image_manifest, load_relations, validate_development_dataset, validate_frozen_test_evidence


def test_dataset_v3_counts_and_uniqueness() -> None:
    manifest = load_image_manifest()
    assert len(manifest) == 640
    assert manifest["sha256"].nunique() == 640
    assert manifest.groupby("split").size().to_dict() == {"test": 80, "train": 480, "validation": 80}


def test_development_dataset_has_no_leakage() -> None:
    result = validate_development_dataset(verify_hashes=False)
    assert result["status"] == "PASS_DEVELOPMENT_DATASET"
    assert result["locked_test_relations_read"] is False
    assert all(value == 0 for value in result["overlap"].values())


def test_locked_test_requires_explicit_permission() -> None:
    with pytest.raises(PermissionError):
        load_relations("test")
    assert len(load_relations("test", allow_locked_test=True)) == 480


def test_full_split_boundaries() -> None:
    result = validate_frozen_test_evidence(verify_hashes=False)
    assert result["status"] == "PASS_FROZEN_TEST_EVIDENCE"
    assert result["split_relation_rows"] == {"train": 2880, "validation": 480, "test": 480}
    assert all(value == 0 for value in result["overlap"].values())



def test_active_license_record_matches_dataset_v3_source() -> None:
    licenses = pd.read_csv(PROJECT_ROOT / "data/licenses.csv")
    manifest = load_image_manifest()
    assert len(licenses) == 1
    assert licenses.loc[0, "source_url"] == "https://www.kaggle.com/datasets/gpiosenka/car-parts-40-classes"
    assert licenses.loc[0, "recorded_license"] == "Apache-2.0"
    assert int(licenses.loc[0, "selected_images"]) == len(manifest) == 640
    assert set(manifest["source_dataset"]) == {"gpiosenka/car-parts-40-classes"}
    assert not licenses.astype(str).apply(lambda column: column.str.contains("Wikimedia", case=False)).any().any()


def test_single_input_sides_are_relation_balanced() -> None:
    for split in ("train", "validation"):
        relations = load_relations(split)
        per_image = relations.groupby(["image_id", "label"]).size().unstack(fill_value=0)
        per_text = relations.groupby(["description", "label"]).size().unstack(fill_value=0)
        assert per_image.nunique(axis=1).eq(1).all()
        assert per_text.nunique(axis=1).eq(1).all()
