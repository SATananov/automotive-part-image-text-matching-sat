from __future__ import annotations

import argparse
import json
from typing import Any

import pandas as pd
from PIL import Image

from external_audit.core import (
    AUDIT_ROOT,
    CATEGORY_FAMILIES_PATH,
    CATEGORY_PLAN_PATH,
    CLASSIFIER_PATH,
    IMAGE_EXTENSIONS,
    IMAGES_ROOT,
    LOCK_PATH,
    MANIFEST_OUT,
    MANIFEST_PATH,
    METADATA_PATH,
    PROTOCOL_PATH,
    PROVENANCE_PATH,
    RELATIONS_OUT,
    TFIDF_PATH,
    file_sha256,
    git_head,
    load_category_plan,
    load_protocol,
    validate_plan,
)
from practical_demo.deployment import (
    load_deployment_metadata,
    verify_deployment_artifacts,
)
from src.v4_data import load_image_manifest_v4


def load_provenance() -> dict[str, Any]:
    if not PROVENANCE_PATH.is_file():
        raise FileNotFoundError(
            "Create external_audit/provenance.json from provenance_template.json "
            "and complete it before locking the audit."
        )

    data = json.loads(PROVENANCE_PATH.read_text(encoding="utf-8"))
    text = json.dumps(data, sort_keys=True)

    if "FILL_BEFORE_LOCK" in text:
        raise ValueError("External audit provenance is incomplete.")
    if data.get("independent_from_dataset_v4_source") is not True:
        raise ValueError("External image source must be independent from Dataset V4.")
    if data.get("redistribution_allowed") is not True:
        raise ValueError(
            "Public-repository External Audit V1 requires images that may be redistributed."
        )
    return data


def collect_manifest() -> pd.DataFrame:
    protocol = load_protocol()
    plan = load_category_plan()
    provenance = load_provenance()

    if not IMAGES_ROOT.is_dir():
        raise FileNotFoundError(IMAGES_ROOT)

    categories = list(plan["part_category"].astype(str))
    planned = set(categories)

    unexpected_dirs = sorted(
        path.name
        for path in IMAGES_ROOT.iterdir()
        if path.is_dir() and path.name not in planned
    )
    if unexpected_dirs:
        raise ValueError(
            f"Unexpected category folders in external audit: {unexpected_dirs}"
        )

    family_by_category = dict(
        zip(
            plan["part_category"].astype(str),
            plan["part_family"].astype(str),
            strict=True,
        )
    )

    rows: list[dict[str, Any]] = []
    expected_per_category = int(protocol["images_per_category"])

    for category in categories:
        folder = IMAGES_ROOT / category
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Missing external image category folder: {folder}"
            )

        paths = sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if len(paths) != expected_per_category:
            raise ValueError(
                f"{category}: expected exactly {expected_per_category} images, "
                f"found {len(paths)}."
            )

        for index, path in enumerate(paths, start=1):
            digest = file_sha256(path)

            with Image.open(path) as image:
                width, height = image.size
                mode = str(image.mode)
                image.verify()

            image_id = (
                f"external_v1_{category}_{index:02d}_{digest[:12]}"
            )
            rows.append(
                {
                    "image_id": image_id,
                    "part_category": category,
                    "part_family": family_by_category[category],
                    "image_path": path.relative_to(
                        AUDIT_ROOT.parent
                    ).as_posix(),
                    "sha256": digest,
                    "width": int(width),
                    "height": int(height),
                    "mode": mode,
                    "source_type": str(provenance.get("source_type", "")),
                }
            )

    manifest = pd.DataFrame(rows).sort_values(
        ["part_category", "image_id"]
    ).reset_index(drop=True)

    if len(manifest) != int(protocol["target_images"]):
        raise ValueError("External manifest image count does not match protocol.")
    if manifest["image_id"].duplicated().any():
        raise ValueError("Duplicate external image IDs.")
    if manifest["sha256"].duplicated().any():
        raise ValueError(
            "Duplicate external image bytes detected. "
            "Each audit image must be unique."
        )

    v4 = load_image_manifest_v4()
    overlap = set(manifest["sha256"].astype(str)) & set(
        v4["sha256"].astype(str)
    )
    if overlap:
        raise ValueError(
            "External audit contains exact image bytes already present in Dataset V4."
        )

    return manifest


def build_relations(manifest: pd.DataFrame) -> pd.DataFrame:
    protocol = load_protocol()
    plan = load_category_plan().set_index("part_category")

    rows: list[dict[str, Any]] = []

    for image in manifest.itertuples(index=False):
        image_category = str(image.part_category)
        image_family = str(image.part_family)

        targets = {
            "MATCH": image_category,
            "PARTIAL_MATCH": str(
                plan.loc[image_category, "partial_partner"]
            ),
            "MISMATCH": str(
                plan.loc[image_category, "mismatch_partner"]
            ),
        }

        for label, text_category in targets.items():
            text_family = str(
                plan.loc[text_category, "part_family"]
            )
            for text_mode in ("clean", "natural"):
                description = str(
                    plan.loc[
                        text_category,
                        f"{text_mode}_text",
                    ]
                )
                sample_id = (
                    f"{image.image_id}__{text_mode}__"
                    f"{label.lower()}"
                )
                rows.append(
                    {
                        "sample_id": sample_id,
                        "image_id": str(image.image_id),
                        "image_path": str(image.image_path),
                        "part_category": image_category,
                        "part_family": image_family,
                        "text_category": text_category,
                        "text_family": text_family,
                        "description": description,
                        "text_mode": text_mode,
                        "label": label,
                        "source": "external_audit_v1",
                    }
                )

    table = pd.DataFrame(rows).sort_values(
        ["image_id", "text_mode", "label"]
    ).reset_index(drop=True)

    if len(table) != int(protocol["target_relation_rows"]):
        raise ValueError("External relation count does not match protocol.")
    if table["sample_id"].duplicated().any():
        raise ValueError("Duplicate external relation sample IDs.")

    expected_labels = {"MATCH", "PARTIAL_MATCH", "MISMATCH"}
    if set(table["label"]) != expected_labels:
        raise ValueError("External relation labels are incomplete.")

    label_counts = table["label"].value_counts()
    if label_counts.nunique() != 1:
        raise ValueError("External relation labels are not balanced.")

    mode_counts = table["text_mode"].value_counts()
    if mode_counts.nunique() != 1:
        raise ValueError("External clean/natural text modes are not balanced.")

    per_image = (
        table.groupby(["image_id", "label"])
        .size()
        .unstack(fill_value=0)
    )
    if not per_image.eq(2).all().all():
        raise ValueError(
            "Every external image must have two rows per relation label."
        )

    match = table[table["label"].eq("MATCH")]
    partial = table[table["label"].eq("PARTIAL_MATCH")]
    mismatch = table[table["label"].eq("MISMATCH")]

    if not match["part_category"].eq(match["text_category"]).all():
        raise ValueError("Invalid external MATCH semantics.")
    if not (
        partial["part_category"].ne(partial["text_category"])
        & partial["part_family"].eq(partial["text_family"])
    ).all():
        raise ValueError("Invalid external PARTIAL_MATCH semantics.")
    if not mismatch["part_family"].ne(mismatch["text_family"]).all():
        raise ValueError("Invalid external MISMATCH semantics.")

    per_text = (
        table.groupby(["text_category", "label"])
        .size()
        .unstack(fill_value=0)
    )
    if not per_text.nunique(axis=1).eq(1).all():
        raise ValueError(
            "External text categories are not balanced across relation labels."
        )

    return table


def create_lock() -> dict[str, Any]:
    validate_plan()

    outputs = [MANIFEST_OUT, RELATIONS_OUT, LOCK_PATH]
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "External audit lock artifacts already exist. "
            "Do not overwrite a locked audit."
        )

    manifest = collect_manifest()
    relations = build_relations(manifest)

    metadata = load_deployment_metadata()
    verify_deployment_artifacts(metadata)

    manifest.to_csv(
        MANIFEST_OUT,
        index=False,
        lineterminator="\n",
    )
    relations.to_csv(
        RELATIONS_OUT,
        index=False,
        lineterminator="\n",
    )

    lock: dict[str, Any] = {
        "status": "LOCKED_EXTERNAL_AUDIT_V1",
        "locked": True,
        "created_before_external_inference": True,
        "official_locked_final_test_loaded": False,
        "training_allowed": False,
        "post_audit_tuning_allowed": False,
        "external_inference_performed": False,
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "category_plan_sha256": file_sha256(CATEGORY_PLAN_PATH),
        "provenance_sha256": file_sha256(PROVENANCE_PATH),
        "external_manifest_sha256": file_sha256(MANIFEST_OUT),
        "external_relations_sha256": file_sha256(RELATIONS_OUT),
        "dataset_v4_manifest_sha256": file_sha256(MANIFEST_PATH),
        "category_families_sha256": file_sha256(CATEGORY_FAMILIES_PATH),
        "deployment_metadata_sha256": file_sha256(METADATA_PATH),
        "deployment_classifier_sha256": file_sha256(CLASSIFIER_PATH),
        "deployment_tfidf_sha256": file_sha256(TFIDF_PATH),
        "deployment_selected_model": metadata["selected_model"],
        "deployment_training_data": metadata["training_data"],
        "external_images": int(len(manifest)),
        "external_relation_rows": int(len(relations)),
        "git_head_before_external_inference": git_head(),
    }

    LOCK_PATH.write_text(
        json.dumps(lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate or lock External Robustness Audit V1. "
            "This command never performs model inference."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--check-plan",
        action="store_true",
        help="Validate the frozen category/text protocol without requiring images.",
    )
    group.add_argument(
        "--lock",
        action="store_true",
        help="Create the external manifest, relation table and pre-inference lock.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check_plan:
        summary = validate_plan()
        print(summary["status"])
        print("Categories:", summary["categories"])
        print("Families:", summary["families"])
        print("Target external images:", summary["target_images"])
        print("Target relation rows:", summary["target_relation_rows"])
        print("External inference performed: False")
        return

    lock = create_lock()
    print(lock["status"])
    print("External images:", lock["external_images"])
    print("External relation rows:", lock["external_relation_rows"])
    print("Official locked final test loaded: False")
    print("External inference performed: False")
    print("Post-audit tuning: NOT ALLOWED")


if __name__ == "__main__":
    main()
