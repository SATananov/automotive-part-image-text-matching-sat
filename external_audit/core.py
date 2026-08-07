from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from practical_demo.deployment import (
    CLASSIFIER_PATH,
    METADATA_PATH,
    TFIDF_PATH,
    load_deployment_metadata,
    verify_deployment_artifacts,
)
from src.v4_data import (
    CATEGORY_FAMILIES_PATH,
    MANIFEST_PATH,
    PROJECT_ROOT,
    load_category_families_v4,
)

AUDIT_ROOT = PROJECT_ROOT / "external_audit"
PROTOCOL_PATH = AUDIT_ROOT / "protocol.json"
CATEGORY_PLAN_PATH = AUDIT_ROOT / "category_plan.csv"
PROVENANCE_PATH = AUDIT_ROOT / "provenance.json"
IMAGES_ROOT = AUDIT_ROOT / "images"
MANIFEST_OUT = AUDIT_ROOT / "external_manifest.csv"
RELATIONS_OUT = AUDIT_ROOT / "external_relations.csv"
LOCK_PATH = AUDIT_ROOT / "external_lock.json"
RESULT_DIR = AUDIT_ROOT / "results"
PREDICTIONS_PATH = RESULT_DIR / "external_predictions.csv"
SUMMARY_PATH = RESULT_DIR / "external_summary.json"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

REQUIRED_PLAN_COLUMNS = {
    "part_category",
    "display_name",
    "part_family",
    "partial_partner",
    "mismatch_partner",
    "clean_text",
    "natural_text",
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_protocol() -> dict[str, Any]:
    data = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    required = {
        "version",
        "status",
        "official_dataset_v4_unchanged",
        "uses_locked_final_test",
        "training_allowed",
        "post_audit_tuning_allowed",
        "target_categories",
        "target_families",
        "images_per_category",
        "target_images",
        "text_modes",
        "labels",
        "rows_per_image",
        "target_relation_rows",
        "lock_before_inference",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"External audit protocol is missing fields: {missing}")
    if data["official_dataset_v4_unchanged"] is not True:
        raise ValueError("External audit must leave Dataset V4 unchanged.")
    if data["uses_locked_final_test"] is not False:
        raise ValueError("External audit must not use the locked final test.")
    if data["training_allowed"] is not False:
        raise ValueError("External audit must not allow training.")
    if data["post_audit_tuning_allowed"] is not False:
        raise ValueError("External audit must not allow post-audit tuning.")
    return data


def load_category_plan() -> pd.DataFrame:
    table = pd.read_csv(CATEGORY_PLAN_PATH)
    missing = sorted(REQUIRED_PLAN_COLUMNS - set(table.columns))
    if missing:
        raise ValueError(f"External category plan is missing columns: {missing}")
    return table


def validate_plan() -> dict[str, Any]:
    protocol = load_protocol()
    plan = load_category_plan()

    if len(plan) != int(protocol["target_categories"]):
        raise ValueError("External audit category count does not match protocol.")
    if plan["part_category"].duplicated().any():
        raise ValueError("External audit category plan contains duplicates.")

    reference = load_category_families_v4()
    reference_family = dict(
        zip(
            reference["part_category"].astype(str),
            reference["part_family"].astype(str),
            strict=True,
        )
    )

    planned = set(plan["part_category"].astype(str))
    unknown = sorted(planned - set(reference_family))
    if unknown:
        raise ValueError(f"External audit uses unknown Dataset V4 categories: {unknown}")

    actual_families = set(plan["part_family"].astype(str))
    if len(actual_families) != int(protocol["target_families"]):
        raise ValueError("External audit family count does not match protocol.")

    per_family = plan.groupby("part_family")["part_category"].nunique()
    if not per_family.eq(2).all():
        raise ValueError("External audit must use exactly two categories per family.")

    rows = plan.set_index("part_category")
    for row in plan.itertuples(index=False):
        category = str(row.part_category)
        family = str(row.part_family)

        if reference_family[category] != family:
            raise ValueError(f"Family mismatch for {category}.")

        partial = str(row.partial_partner)
        mismatch = str(row.mismatch_partner)

        if partial not in planned or mismatch not in planned:
            raise ValueError(f"Partner category outside plan for {category}.")
        if partial == category or mismatch == category:
            raise ValueError(f"Partner category equals source category for {category}.")
        if str(rows.loc[partial, "part_family"]) != family:
            raise ValueError(f"PARTIAL_MATCH partner has wrong family for {category}.")
        if str(rows.loc[mismatch, "part_family"]) == family:
            raise ValueError(f"MISMATCH partner has same family for {category}.")

    if plan["partial_partner"].nunique() != len(plan):
        raise ValueError("Partial-partner mapping must be one-to-one.")
    if plan["mismatch_partner"].nunique() != len(plan):
        raise ValueError("Mismatch-partner mapping must be one-to-one.")

    expected_images = (
        int(protocol["target_categories"])
        * int(protocol["images_per_category"])
    )
    if expected_images != int(protocol["target_images"]):
        raise ValueError("External audit target image count is inconsistent.")

    expected_rows = (
        expected_images
        * int(protocol["rows_per_image"])
    )
    if expected_rows != int(protocol["target_relation_rows"]):
        raise ValueError("External audit target relation count is inconsistent.")

    return {
        "status": "PASS_EXTERNAL_AUDIT_V1_PLAN",
        "categories": len(plan),
        "families": len(actual_families),
        "images_per_category": int(protocol["images_per_category"]),
        "target_images": expected_images,
        "target_relation_rows": expected_rows,
    }


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def verify_lock() -> dict[str, Any]:
    if not LOCK_PATH.is_file():
        raise FileNotFoundError(
            "External audit is not locked yet. Run "
            "`python -m external_audit.prepare --lock` only after the "
            "independent image set and provenance are complete."
        )

    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    if lock.get("locked") is not True:
        raise ValueError("External audit lock is not marked locked.")
    if lock.get("created_before_external_inference") is not True:
        raise ValueError("External audit lock chronology is invalid.")
    if lock.get("official_locked_final_test_loaded") is not False:
        raise ValueError("External audit unexpectedly reports final-test access.")
    if lock.get("training_allowed") is not False:
        raise ValueError("External audit lock unexpectedly allows training.")
    if lock.get("post_audit_tuning_allowed") is not False:
        raise ValueError("External audit lock unexpectedly allows tuning.")

    expected_files = {
        "protocol_sha256": PROTOCOL_PATH,
        "category_plan_sha256": CATEGORY_PLAN_PATH,
        "provenance_sha256": PROVENANCE_PATH,
        "external_manifest_sha256": MANIFEST_OUT,
        "external_relations_sha256": RELATIONS_OUT,
        "dataset_v4_manifest_sha256": MANIFEST_PATH,
        "category_families_sha256": CATEGORY_FAMILIES_PATH,
        "deployment_metadata_sha256": METADATA_PATH,
    }

    for key, path in expected_files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = file_sha256(path)
        if actual != str(lock.get(key, "")):
            raise ValueError(f"External audit lock hash mismatch: {key}")

    metadata = load_deployment_metadata()
    verify_deployment_artifacts(metadata)

    if file_sha256(CLASSIFIER_PATH) != str(
        lock.get("deployment_classifier_sha256", "")
    ):
        raise ValueError("Frozen deployment classifier hash mismatch.")
    if file_sha256(TFIDF_PATH) != str(
        lock.get("deployment_tfidf_sha256", "")
    ):
        raise ValueError("Frozen deployment TF-IDF hash mismatch.")

    return lock
