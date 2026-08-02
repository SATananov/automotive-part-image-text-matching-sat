from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_DIR / "manifests" / "images.csv"
RELATION_DIR = DATA_DIR / "relations"
LOCKED_TEST_DIR = DATA_DIR / "locked_test" / "dataset_v3"
LOCKED_TEST_RELATIONS = LOCKED_TEST_DIR / "test_relations.csv"
LOCKED_TEST_PREFIX = "data/locked_test/"

LABELS = ("MATCH", "MISMATCH", "PARTIAL_MATCH")
CATEGORIES = (
    "alternator",
    "brake_disc",
    "brake_pad",
    "coil_spring",
    "headlight",
    "oil_filter",
    "starter",
    "taillight",
)
FAMILIES = {
    "alternator": "engine_support",
    "oil_filter": "engine_support",
    "starter": "engine_support",
    "brake_disc": "chassis",
    "brake_pad": "chassis",
    "coil_spring": "chassis",
    "headlight": "lighting",
    "taillight": "lighting",
}
EXPECTED_IMAGES = {"train": 480, "validation": 80, "test": 80}
EXPECTED_IMAGES_PER_CATEGORY = {"train": 60, "validation": 10, "test": 10}
EXPECTED_RELATION_ROWS = {"train": 2880, "validation": 480, "test": 480}
IMAGE_SIZE = (48, 48)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_image_manifest() -> pd.DataFrame:
    data = pd.read_csv(MANIFEST_PATH)
    required = {
        "image_id",
        "image_group_id",
        "part_category",
        "split",
        "rank_within_category",
        "image_path",
        "sha256",
        "source_dataset",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing image manifest columns: {missing}")
    return data


def load_relations(split: str, *, allow_locked_test: bool = False) -> pd.DataFrame:
    """Load relation rows.

    Training code can load only ``train`` and ``validation``. Reading the locked
    test relation table requires an explicit acknowledgement and is reserved for
    frozen-result verification, never model development.
    """
    if split not in EXPECTED_RELATION_ROWS:
        raise ValueError(f"Unknown split: {split}")
    if split == "test":
        if not allow_locked_test:
            raise PermissionError(
                "The final-test table is locked. Use saved final-test artifacts; "
                "only verification code may pass allow_locked_test=True."
            )
        path = LOCKED_TEST_RELATIONS
    else:
        path = RELATION_DIR / f"{split}.csv"

    data = pd.read_csv(path)
    required = {
        "sample_id",
        "image_id",
        "part_group_id",
        "object_group_id",
        "image_path",
        "part_family",
        "part_category",
        "text_category",
        "description",
        "label",
        "source",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing relation columns in {path.name}: {missing}")
    return data


def encode_labels(values: pd.Series | np.ndarray) -> np.ndarray:
    mapping = {label: index for index, label in enumerate(LABELS)}
    encoded = pd.Series(values).map(mapping)
    if encoded.isna().any():
        unknown = sorted(set(pd.Series(values)) - set(mapping))
        raise ValueError(f"Unknown labels: {unknown}")
    return encoded.to_numpy(np.int64)


def load_images(data: pd.DataFrame, size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Load RGB images while caching repeated paths in a relation table."""
    cache: dict[str, np.ndarray] = {}
    rows: list[np.ndarray] = []
    for relative_path in data["image_path"].astype(str):
        normalized = relative_path.replace("\\", "/")
        if normalized not in cache:
            path = PROJECT_ROOT / normalized
            if not path.is_file():
                raise FileNotFoundError(path)
            with Image.open(path) as image:
                cache[normalized] = np.asarray(
                    image.convert("RGB").resize(size, Image.Resampling.BILINEAR),
                    dtype=np.float32,
                )
        rows.append(cache[normalized])
    return np.stack(rows)


def _validate_relation_semantics(split: str, relations: pd.DataFrame) -> None:
    if len(relations) != EXPECTED_RELATION_ROWS[split]:
        raise ValueError(f"Unexpected {split} relation count: {len(relations)}")
    if relations["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample IDs in {split}")
    if relations["image_id"].nunique() != EXPECTED_IMAGES[split]:
        raise ValueError(f"Unexpected independent image count in {split}")
    if set(relations["label"]) != set(LABELS):
        raise ValueError(f"Unexpected labels in {split}")
    if set(relations["part_category"]) != set(CATEGORIES):
        raise ValueError(f"Unexpected part categories in {split}")
    if set(relations["text_category"]) != set(CATEGORIES):
        raise ValueError(f"Unexpected text categories in {split}")
    expected_source = {"dataset_v3_final_test"} if split == "test" else {"dataset_v3"}
    if set(relations["source"]) != expected_source:
        raise ValueError(f"Unexpected source values in {split}")

    per_image_label = relations.groupby(["image_id", "label"]).size().unstack(fill_value=0)
    per_image_label = per_image_label.reindex(columns=sorted(LABELS), fill_value=0)
    if not (per_image_label == 2).all().all():
        raise ValueError(f"Each {split} image must have two rows per label")

    match = relations.loc[relations["label"].eq("MATCH")]
    partial = relations.loc[relations["label"].eq("PARTIAL_MATCH")]
    mismatch = relations.loc[relations["label"].eq("MISMATCH")]
    if not match["part_category"].eq(match["text_category"]).all():
        raise ValueError(f"Invalid MATCH rows in {split}")
    if not partial.apply(
        lambda row: row.part_category != row.text_category
        and FAMILIES[row.part_category] == FAMILIES[row.text_category],
        axis=1,
    ).all():
        raise ValueError(f"Invalid PARTIAL_MATCH rows in {split}")
    if not mismatch.apply(
        lambda row: FAMILIES[row.part_category] != FAMILIES[row.text_category],
        axis=1,
    ).all():
        raise ValueError(f"Invalid MISMATCH rows in {split}")


def _validate_manifest(*, verify_hashes: bool) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    manifest = load_image_manifest()
    if len(manifest) != sum(EXPECTED_IMAGES.values()):
        raise ValueError(f"Expected 640 images, found {len(manifest)}")
    for column in ("image_id", "image_group_id", "image_path", "sha256"):
        if manifest[column].duplicated().any():
            raise ValueError(f"Duplicate image manifest field: {column}")
    if set(manifest["part_category"]) != set(CATEGORIES):
        raise ValueError("Unexpected image categories")

    split_rows: dict[str, pd.DataFrame] = {}
    for split in EXPECTED_IMAGES:
        selected = manifest.loc[manifest["split"].eq(split)].copy()
        split_rows[split] = selected
        if len(selected) != EXPECTED_IMAGES[split]:
            raise ValueError(f"Unexpected {split} image count: {len(selected)}")
        counts = selected["part_category"].value_counts()
        if set(counts.index) != set(CATEGORIES) or not (
            counts == EXPECTED_IMAGES_PER_CATEGORY[split]
        ).all():
            raise ValueError(f"Unbalanced {split} image categories: {counts.to_dict()}")
        for row in selected.itertuples(index=False):
            path = PROJECT_ROOT / str(row.image_path)
            if not path.is_file():
                raise FileNotFoundError(path)
            if verify_hashes and file_sha256(path) != str(row.sha256):
                raise ValueError(f"Image hash mismatch: {row.image_id}")
    return manifest, split_rows


def _manifest_overlap(split_rows: dict[str, pd.DataFrame]) -> dict[str, int]:
    overlap: dict[str, int] = {}
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        for column in ("image_id", "image_group_id", "image_path", "sha256"):
            count = len(set(split_rows[left][column]) & set(split_rows[right][column]))
            overlap[f"{left}_{right}_{column}"] = count
            if count:
                raise ValueError(f"Leakage detected: {left}/{right} overlap in {column}")
    return overlap


def validate_development_dataset(*, verify_hashes: bool = False) -> dict[str, object]:
    """Validate train/validation data without opening locked-test relation rows."""
    manifest, split_rows = _validate_manifest(verify_hashes=verify_hashes)
    overlap = _manifest_overlap(split_rows)
    relations = {
        split: load_relations(split)
        for split in ("train", "validation")
    }
    for split, table in relations.items():
        _validate_relation_semantics(split, table)
        paths = table["image_path"].astype(str).str.replace("\\", "/", regex=False)
        if paths.str.startswith(LOCKED_TEST_PREFIX).any():
            raise ValueError(f"Locked-test path found in {split}")

    description_overlap = len(
        set(relations["train"]["description"])
        & set(relations["validation"]["description"])
    )
    overlap["train_validation_description"] = description_overlap
    if description_overlap:
        raise ValueError("Description overlap detected between train and validation")

    return {
        "status": "PASS_DEVELOPMENT_DATASET",
        "images": len(manifest),
        "unique_hashes": int(manifest["sha256"].nunique()),
        "split_images": {split: len(rows) for split, rows in split_rows.items()},
        "development_relation_rows": {
            split: len(rows) for split, rows in relations.items()
        },
        "categories": len(CATEGORIES),
        "overlap": overlap,
        "hashes_verified": verify_hashes,
        "locked_test_relations_read": False,
    }


def validate_frozen_test_evidence(*, verify_hashes: bool = False) -> dict[str, object]:
    """Explicitly validate the frozen test data and all three split boundaries."""
    manifest, split_rows = _validate_manifest(verify_hashes=verify_hashes)
    overlap = _manifest_overlap(split_rows)
    relations = {
        "train": load_relations("train"),
        "validation": load_relations("validation"),
        "test": load_relations("test", allow_locked_test=True),
    }
    for split, table in relations.items():
        _validate_relation_semantics(split, table)
    test_paths = relations["test"]["image_path"].astype(str).str.replace(
        "\\", "/", regex=False
    )
    if not test_paths.str.startswith(LOCKED_TEST_PREFIX).all():
        raise ValueError("Every test row must point into the locked-test directory")

    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        description_overlap = len(
            set(relations[left]["description"]) & set(relations[right]["description"])
        )
        overlap[f"{left}_{right}_description"] = description_overlap
        if description_overlap:
            raise ValueError(f"Description overlap detected between {left} and {right}")

    return {
        "status": "PASS_FROZEN_TEST_EVIDENCE",
        "images": len(manifest),
        "unique_hashes": int(manifest["sha256"].nunique()),
        "split_images": {split: len(rows) for split, rows in split_rows.items()},
        "split_relation_rows": {split: len(rows) for split, rows in relations.items()},
        "categories": len(CATEGORIES),
        "overlap": overlap,
        "hashes_verified": verify_hashes,
        "locked_test_relations_read": True,
        "purpose": "integrity_verification_only",
    }
