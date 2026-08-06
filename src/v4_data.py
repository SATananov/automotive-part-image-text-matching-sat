from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_V4_ROOT = PROJECT_ROOT / "data"
MANIFEST_PATH = DATASET_V4_ROOT / "manifests" / "dataset_v4" / "images.csv"
AUDIT_PATH = DATASET_V4_ROOT / "manifests" / "dataset_v4" / "dataset_audit.json"
CATEGORY_FAMILIES_PATH = (
    DATASET_V4_ROOT / "manifests" / "dataset_v4" / "category_families.csv"
)
RELATION_DIR = DATASET_V4_ROOT / "relations" / "dataset_v4"
LOCKED_TEST_DIR = DATASET_V4_ROOT / "locked_test" / "dataset_v4"
LOCKED_TEST_RELATIONS = LOCKED_TEST_DIR / "test_relations.csv"
LOCKED_TEST_LOCK = LOCKED_TEST_DIR / "test_lock.json"
LOCKED_TEST_PREFIX = "data/locked_test/dataset_v4/"

LABELS = ("MATCH", "MISMATCH", "PARTIAL_MATCH")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABELS)}
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset_v4_audit() -> dict[str, object]:
    if not AUDIT_PATH.is_file():
        raise FileNotFoundError(AUDIT_PATH)
    return json.loads(AUDIT_PATH.read_text(encoding="utf-8"))


def load_category_families_v4() -> pd.DataFrame:
    data = pd.read_csv(CATEGORY_FAMILIES_PATH)
    required = {"part_category", "display_name", "part_family"}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing category-family columns: {missing}")
    if len(data) != 50 or data["part_category"].nunique() != 50:
        raise ValueError("Dataset V4 must contain exactly 50 unique categories")
    return data.sort_values("part_category").reset_index(drop=True)


def category_names_v4() -> tuple[str, ...]:
    return tuple(load_category_families_v4()["part_category"].astype(str))


def category_to_index_v4() -> dict[str, int]:
    return {name: index for index, name in enumerate(category_names_v4())}


def family_map_v4() -> dict[str, str]:
    table = load_category_families_v4()
    return dict(zip(table["part_category"], table["part_family"], strict=True))


def load_image_manifest_v4() -> pd.DataFrame:
    data = pd.read_csv(MANIFEST_PATH)
    required = {
        "image_id",
        "image_group_id",
        "part_category",
        "part_family",
        "split",
        "rank_within_category",
        "image_path",
        "sha256",
        "width",
        "height",
        "mode",
        "source_dataset",
        "source_category",
        "source_original_split",
        "source_archive_path",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing Dataset V4 manifest columns: {missing}")
    return data


def load_relations_v4(split: str, *, allow_locked_test: bool = False) -> pd.DataFrame:
    if split not in {"train", "validation", "test"}:
        raise ValueError(f"Unknown split: {split}")
    if split == "test":
        if not allow_locked_test:
            raise PermissionError(
                "Dataset V4 final-test relations are locked. Development code may "
                "load only train and validation data."
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
        "text_family",
        "description",
        "label",
        "source",
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"Missing Dataset V4 relation columns in {path.name}: {missing}")
    return data


def encode_relation_labels_v4(values: pd.Series | np.ndarray) -> np.ndarray:
    encoded = pd.Series(values).map(LABEL_TO_INDEX)
    if encoded.isna().any():
        unknown = sorted(set(pd.Series(values).astype(str)) - set(LABEL_TO_INDEX))
        raise ValueError(f"Unknown relation labels: {unknown}")
    return encoded.to_numpy(np.int64)


def encode_categories_v4(values: pd.Series | np.ndarray) -> np.ndarray:
    mapping = category_to_index_v4()
    encoded = pd.Series(values).map(mapping)
    if encoded.isna().any():
        unknown = sorted(set(pd.Series(values).astype(str)) - set(mapping))
        raise ValueError(f"Unknown Dataset V4 categories: {unknown}")
    return encoded.to_numpy(np.int64)


def image_transform_v4(*, training: bool) -> Callable[[Image.Image], torch.Tensor]:
    steps: list[Callable] = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if training:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=6),
                transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.08),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


def fit_text_vectorizer_v4(
    train_relations: pd.DataFrame,
    validation_relations: pd.DataFrame,
    *,
    max_features: int = 2048,
) -> tuple[TfidfVectorizer, np.ndarray, np.ndarray]:
    """Fit TF-IDF on training text only and transform train/validation text."""
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=max_features,
        sublinear_tf=True,
    )
    train_sparse = vectorizer.fit_transform(train_relations["description"].astype(str))
    validation_sparse = vectorizer.transform(
        validation_relations["description"].astype(str)
    )
    return (
        vectorizer,
        train_sparse.toarray().astype(np.float32),
        validation_sparse.toarray().astype(np.float32),
    )


class DatasetV4Relations(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Lazy image-text relation dataset for Dataset V4.

    Images are opened only when a sample is requested. Text features are supplied
    as a dense float32 matrix produced by ``fit_text_vectorizer_v4``.
    """

    def __init__(
        self,
        relations: pd.DataFrame,
        text_features: np.ndarray,
        *,
        training: bool,
    ) -> None:
        if len(relations) != len(text_features):
            raise ValueError("Relation rows and text feature rows must have equal length")
        if text_features.ndim != 2:
            raise ValueError("Text features must be a two-dimensional matrix")
        self.relations = relations.reset_index(drop=True).copy()
        self.text_features = np.asarray(text_features, dtype=np.float32)
        self.transform = image_transform_v4(training=training)
        self.relation_labels = encode_relation_labels_v4(self.relations["label"])
        self.image_categories = encode_categories_v4(self.relations["part_category"])
        self.text_categories = encode_categories_v4(self.relations["text_category"])

    def __len__(self) -> int:
        return len(self.relations)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.relations.iloc[index]
        relative = str(row["image_path"]).replace("\\", "/")
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            image_tensor = self.transform(image.convert("RGB"))
        return (
            image_tensor,
            torch.from_numpy(self.text_features[index]),
            torch.tensor(self.relation_labels[index], dtype=torch.long),
            torch.tensor(self.image_categories[index], dtype=torch.long),
            torch.tensor(self.text_categories[index], dtype=torch.long),
        )


def _validate_relation_table(split: str, table: pd.DataFrame) -> dict[str, object]:
    categories = set(category_names_v4())
    families = family_map_v4()
    if table["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample IDs in Dataset V4 {split}")
    if set(table["label"]) != set(LABELS):
        raise ValueError(f"Unexpected labels in Dataset V4 {split}")
    if set(table["part_category"]) != categories:
        raise ValueError(f"Unexpected image categories in Dataset V4 {split}")
    if set(table["text_category"]) != categories:
        raise ValueError(f"Unexpected text categories in Dataset V4 {split}")

    per_image = table.groupby(["image_id", "label"]).size().unstack(fill_value=0)
    per_image = per_image.reindex(columns=list(LABELS), fill_value=0)
    if not per_image.nunique(axis=1).eq(1).all() or not per_image.min(axis=1).ge(1).all():
        raise ValueError(f"Image-side labels are not balanced in Dataset V4 {split}")

    per_text = table.groupby(["text_category", "label"]).size().unstack(fill_value=0)
    per_text = per_text.reindex(columns=list(LABELS), fill_value=0)
    if not per_text.nunique(axis=1).eq(1).all():
        raise ValueError(f"Text-category labels are not balanced in Dataset V4 {split}")

    per_description = table.groupby(["description", "label"]).size().unstack(fill_value=0)
    per_description = per_description.reindex(columns=list(LABELS), fill_value=0)
    if not per_description.nunique(axis=1).eq(1).all():
        raise ValueError(f"Exact descriptions are not balanced in Dataset V4 {split}")

    match = table[table["label"].eq("MATCH")]
    partial = table[table["label"].eq("PARTIAL_MATCH")]
    mismatch = table[table["label"].eq("MISMATCH")]
    if not match["part_category"].eq(match["text_category"]).all():
        raise ValueError(f"Invalid MATCH semantics in Dataset V4 {split}")
    if not (
        partial["part_category"].ne(partial["text_category"])
        & partial["part_family"].eq(partial["text_family"])
    ).all():
        raise ValueError(f"Invalid PARTIAL_MATCH semantics in Dataset V4 {split}")
    if not mismatch["part_family"].ne(mismatch["text_family"]).all():
        raise ValueError(f"Invalid MISMATCH semantics in Dataset V4 {split}")
    if not table["part_category"].map(families).eq(table["part_family"]).all():
        raise ValueError(f"Image family mapping mismatch in Dataset V4 {split}")
    if not table["text_category"].map(families).eq(table["text_family"]).all():
        raise ValueError(f"Text family mapping mismatch in Dataset V4 {split}")

    return {
        "rows": len(table),
        "independent_images": int(table["image_id"].nunique()),
        "label_counts": {
            key: int(value) for key, value in table["label"].value_counts().items()
        },
        "image_side_balanced": True,
        "text_category_balanced": True,
        "exact_description_balanced": True,
    }


def validate_dataset_v4(
    *,
    verify_hashes: bool = False,
    include_locked_test: bool = False,
) -> dict[str, object]:
    audit = load_dataset_v4_audit()
    manifest = load_image_manifest_v4()
    categories = set(category_names_v4())
    expected_splits = {
        key: int(value) for key, value in dict(audit["split_images"]).items()
    }

    if len(manifest) != int(audit["images"]):
        raise ValueError("Dataset V4 manifest count does not match its audit")
    for column in ("image_id", "image_group_id", "image_path", "sha256"):
        if manifest[column].duplicated().any():
            raise ValueError(f"Duplicate Dataset V4 manifest field: {column}")
    if set(manifest["part_category"]) != categories:
        raise ValueError("Dataset V4 manifest categories do not match category metadata")

    split_rows: dict[str, pd.DataFrame] = {}
    for split, expected in expected_splits.items():
        selected = manifest.loc[manifest["split"].eq(split)].copy()
        split_rows[split] = selected
        if len(selected) != expected:
            raise ValueError(f"Unexpected Dataset V4 {split} image count: {len(selected)}")
        for row in selected.itertuples(index=False):
            path = PROJECT_ROOT / str(row.image_path)
            if not path.is_file():
                raise FileNotFoundError(path)
            if verify_hashes and file_sha256(path) != str(row.sha256):
                raise ValueError(f"Dataset V4 image hash mismatch: {row.image_id}")

    overlap: dict[str, int] = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for column in ("image_id", "image_group_id", "image_path", "sha256"):
            value = len(set(split_rows[left][column]) & set(split_rows[right][column]))
            overlap[f"{left}_{right}_{column}"] = value
            if value:
                raise ValueError(f"Dataset V4 leakage in {column}: {left}/{right}")

    relation_tables = {
        "train": load_relations_v4("train"),
        "validation": load_relations_v4("validation"),
    }
    if include_locked_test:
        relation_tables["test"] = load_relations_v4("test", allow_locked_test=True)

    relation_summary = {
        split: _validate_relation_table(split, table)
        for split, table in relation_tables.items()
    }
    descriptions = {split: set(table["description"]) for split, table in relation_tables.items()}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if left in descriptions and right in descriptions:
            value = len(descriptions[left] & descriptions[right])
            overlap[f"{left}_{right}_description"] = value
            if value:
                raise ValueError(f"Dataset V4 description overlap: {left}/{right}")

    lock = json.loads(LOCKED_TEST_LOCK.read_text(encoding="utf-8"))
    if not lock.get("locked") or not lock.get("created_before_training"):
        raise ValueError("Dataset V4 final-test lock is not valid")
    if file_sha256(LOCKED_TEST_RELATIONS) != lock.get("test_relations_sha256"):
        raise ValueError("Dataset V4 final-test relation hash does not match its lock")

    return {
        "status": "PASS_DATASET_V4_FOUNDATION",
        "images": len(manifest),
        "unique_sha256": int(manifest["sha256"].nunique()),
        "categories": len(categories),
        "families": int(load_category_families_v4()["part_family"].nunique()),
        "split_images": expected_splits,
        "relations": relation_summary,
        "overlap": overlap,
        "hashes_verified": verify_hashes,
        "locked_test_relations_read": include_locked_test,
        "test_lock_sha256": lock["test_relations_sha256"],
    }
