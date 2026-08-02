from __future__ import annotations

import pytest

from src.data import load_image_manifest, load_relations, validate_development_dataset, validate_frozen_test_evidence


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
