from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import shutil
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath

import pandas as pd
from PIL import Image

DATASET_ROOT = "car parts 50/"
DATASET_NAME = "gpiosenka_car_parts_50"
SEED = 44
VALIDATION_FRACTION = 0.15
TEST_FRACTION = 0.15

SOURCE_TO_CATEGORY = {
    "AIR COMPRESSOR": "air_compressor",
    "ALTERNATOR": "alternator",
    "BATTERY": "battery",
    "BRAKE CALIPER": "brake_caliper",
    "BRAKE PAD": "brake_pad",
    "BRAKE ROTOR": "brake_rotor",
    "CAMSHAFT": "camshaft",
    "CARBERATOR": "carburetor",
    "CLUTCH PLATE": "clutch_plate",
    "COIL SPRING": "coil_spring",
    "CRANKSHAFT": "crankshaft",
    "CYLINDER HEAD": "cylinder_head",
    "DISTRIBUTOR": "distributor",
    "ENGINE BLOCK": "engine_block",
    "ENGINE VALVE": "engine_valve",
    "FUEL INJECTOR": "fuel_injector",
    "FUSE BOX": "fuse_box",
    "GAS CAP": "gas_cap",
    "HEADLIGHTS": "headlight",
    "IDLER ARM": "idler_arm",
    "IGNITION COIL": "ignition_coil",
    "INSTRUMENT CLUSTER": "instrument_cluster",
    "LEAF SPRING": "leaf_spring",
    "LOWER CONTROL ARM": "lower_control_arm",
    "MUFFLER": "muffler",
    "OIL FILTER": "oil_filter",
    "OIL PAN": "oil_pan",
    "OIL PRESSURE SENSOR": "oil_pressure_sensor",
    "OVERFLOW TANK": "overflow_tank",
    "OXYGEN SENSOR": "oxygen_sensor",
    "PISTON": "piston",
    "PRESSURE PLATE": "pressure_plate",
    "RADIATOR": "radiator",
    "RADIATOR FAN": "radiator_fan",
    "RADIATOR HOSE": "radiator_hose",
    "RADIO": "radio",
    "RIM": "rim",
    "SHIFT KNOB": "shift_knob",
    "SIDE MIRROR": "side_mirror",
    "SPARK PLUG": "spark_plug",
    "SPOILER": "spoiler",
    "STARTER": "starter",
    "TAILLIGHTS": "taillight",
    "THERMOSTAT": "thermostat",
    "TORQUE CONVERTER": "torque_converter",
    "TRANSMISSION": "transmission",
    "VACUUM BRAKE BOOSTER": "vacuum_brake_booster",
    "VALVE LIFTER": "valve_lifter",
    "WATER PUMP": "water_pump",
    "WINDOW REGULATOR": "window_regulator",
}

FAMILIES = {
    "alternator": "electrical_starting_charging",
    "battery": "electrical_starting_charging",
    "fuse_box": "electrical_starting_charging",
    "starter": "electrical_starting_charging",
    "carburetor": "ignition_and_fuel",
    "distributor": "ignition_and_fuel",
    "fuel_injector": "ignition_and_fuel",
    "ignition_coil": "ignition_and_fuel",
    "spark_plug": "ignition_and_fuel",
    "brake_caliper": "braking",
    "brake_pad": "braking",
    "brake_rotor": "braking",
    "vacuum_brake_booster": "braking",
    "coil_spring": "suspension_steering_wheels",
    "idler_arm": "suspension_steering_wheels",
    "leaf_spring": "suspension_steering_wheels",
    "lower_control_arm": "suspension_steering_wheels",
    "rim": "suspension_steering_wheels",
    "camshaft": "engine_internal",
    "crankshaft": "engine_internal",
    "cylinder_head": "engine_internal",
    "engine_block": "engine_internal",
    "engine_valve": "engine_internal",
    "piston": "engine_internal",
    "valve_lifter": "engine_internal",
    "oil_filter": "engine_lubrication",
    "oil_pan": "engine_lubrication",
    "oil_pressure_sensor": "engine_lubrication",
    "air_compressor": "cooling_and_climate",
    "overflow_tank": "cooling_and_climate",
    "radiator": "cooling_and_climate",
    "radiator_fan": "cooling_and_climate",
    "radiator_hose": "cooling_and_climate",
    "thermostat": "cooling_and_climate",
    "water_pump": "cooling_and_climate",
    "clutch_plate": "drivetrain",
    "pressure_plate": "drivetrain",
    "torque_converter": "drivetrain",
    "transmission": "drivetrain",
    "headlight": "lighting",
    "taillight": "lighting",
    "gas_cap": "body_and_exterior",
    "side_mirror": "body_and_exterior",
    "spoiler": "body_and_exterior",
    "window_regulator": "body_and_exterior",
    "instrument_cluster": "interior_controls",
    "radio": "interior_controls",
    "shift_knob": "interior_controls",
    "muffler": "exhaust_and_emissions",
    "oxygen_sensor": "exhaust_and_emissions",
}

DISPLAY_NAMES = {category: category.replace("_", " ") for category in FAMILIES}

TEMPLATES = {
    "train": (
        "This listing identifies the component as {name}.",
        "The description names the automotive part as {name}.",
    ),
    "validation": (
        "The product text refers to the component category {name}.",
        "This catalogue description identifies the part as {name}.",
    ),
    "test": (
        "The submitted listing describes the item as {name}.",
        "The accompanying text presents the automotive part as {name}.",
    ),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(value: str) -> int:
    return SEED + int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:8], 16)


def inspect_archive(archive_path: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)

    records: list[dict[str, object]] = []
    all_hashes: dict[str, str] = {}
    duplicate_hashes: defaultdict[str, list[str]] = defaultdict(list)
    invalid: list[dict[str, str]] = []
    source_split_counts: Counter[str] = Counter()
    source_category_counts: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()

    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("The source ZIP failed its CRC integrity test.")

        unsafe = []
        for info in archive.infolist():
            pure = PurePosixPath(info.filename.replace("\\", "/"))
            if pure.is_absolute() or ".." in pure.parts:
                unsafe.append(info.filename)
        if unsafe:
            raise ValueError(f"Unsafe ZIP member paths: {unsafe[:5]}")

        names = sorted(
            info.filename
            for info in archive.infolist()
            if info.filename.startswith(DATASET_ROOT)
            and info.filename.lower().endswith(".jpg")
        )
        if not names:
            raise ValueError(f"No JPEG images found below {DATASET_ROOT!r}")

        for source_path in names:
            parts = source_path.split("/")
            if len(parts) != 4:
                raise ValueError(f"Unexpected source path structure: {source_path}")
            source_split = parts[1]
            source_category = parts[2]
            if source_split not in {"train", "valid", "test"}:
                raise ValueError(f"Unexpected original split: {source_split}")
            if source_category not in SOURCE_TO_CATEGORY:
                raise ValueError(f"Unknown source category: {source_category}")

            raw = archive.read(source_path)
            digest = sha256_bytes(raw)
            duplicate_hashes[digest].append(source_path)
            all_hashes[source_path] = digest

            try:
                with Image.open(io.BytesIO(raw)) as image:
                    width, height = image.size
                    mode = image.mode
                    image.verify()
            except Exception as exc:  # pragma: no cover - source-specific failure path
                invalid.append({"path": source_path, "error": repr(exc)})
                continue

            source_split_counts[source_split] += 1
            source_category_counts[source_category] += 1
            modes[mode] += 1
            dimensions[f"{width}x{height}"] += 1
            records.append(
                {
                    "source_archive_path": source_path,
                    "source_original_split": source_split,
                    "source_category": source_category,
                    "part_category": SOURCE_TO_CATEGORY[source_category],
                    "sha256": digest,
                    "width": width,
                    "height": height,
                    "mode": mode,
                }
            )

    duplicates = {
        digest: paths for digest, paths in duplicate_hashes.items() if len(paths) > 1
    }
    if invalid:
        raise ValueError(f"Invalid source images: {invalid[:5]}")
    if duplicates:
        raise ValueError(f"Exact duplicate source images: {list(duplicates.items())[:3]}")
    if len(records) != 9239:
        raise ValueError(f"Expected 9239 canonical images, found {len(records)}")
    if set(SOURCE_TO_CATEGORY.values()) != set(FAMILIES):
        missing = sorted(set(SOURCE_TO_CATEGORY.values()) ^ set(FAMILIES))
        raise ValueError(f"Category/family mapping mismatch: {missing}")

    audit = {
        "status": "PASS_SOURCE_ARCHIVE",
        "archive_name": archive_path.name,
        "archive_sha256": sha256_file(archive_path),
        "canonical_root": DATASET_ROOT,
        "images": len(records),
        "categories": len(set(record["part_category"] for record in records)),
        "unique_sha256": len(set(all_hashes.values())),
        "invalid_images": len(invalid),
        "exact_duplicate_groups": len(duplicates),
        "source_split_counts": dict(sorted(source_split_counts.items())),
        "source_category_counts": dict(sorted(source_category_counts.items())),
        "dimensions": dict(sorted(dimensions.items())),
        "modes": dict(sorted(modes.items())),
        "excluded_legacy_root": "car parts/",
        "excluded_pretrained_models": [
            "car parts 50/EfficientNetB0-50-(224 X 224)- 97.99.h5",
            "car parts/EfficientNetB2-40-(224 X 224)- 96.90.h5",
        ],
    }
    return records, audit


def assign_splits(records: list[dict[str, object]]) -> list[dict[str, object]]:
    by_category: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        by_category[str(record["part_category"])].append(record)

    output: list[dict[str, object]] = []
    for category in sorted(by_category):
        rows = sorted(by_category[category], key=lambda row: str(row["sha256"]))
        rng = random.Random(stable_seed(category))
        rng.shuffle(rows)
        count = len(rows)
        validation_count = round(count * VALIDATION_FRACTION)
        test_count = round(count * TEST_FRACTION)
        train_count = count - validation_count - test_count
        assignments = (
            ["test"] * test_count
            + ["validation"] * validation_count
            + ["train"] * train_count
        )
        for rank, (record, split) in enumerate(zip(rows, assignments, strict=True), start=1):
            enriched = dict(record)
            enriched["split"] = split
            enriched["rank_within_category"] = rank
            output.append(enriched)
    return output


def destination_for(project_root: Path, record: dict[str, object]) -> tuple[Path, str, str]:
    category = str(record["part_category"])
    split = str(record["split"])
    digest = str(record["sha256"])
    rank = int(record["rank_within_category"])
    image_id = f"v4_{category}_{split}_{rank:04d}_{digest[:16]}"
    filename = f"{image_id}.jpg"
    if split == "test":
        relative = Path("data") / "locked_test" / "dataset_v4" / "images" / category / filename
    else:
        relative = Path("data") / "images" / "dataset_v4" / split / category / filename
    return project_root / relative, relative.as_posix(), image_id


def write_images_and_manifest(
    archive_path: Path,
    project_root: Path,
    assigned: list[dict[str, object]],
) -> pd.DataFrame:
    targets = [
        project_root / "data" / "images" / "dataset_v4",
        project_root / "data" / "locked_test" / "dataset_v4",
        project_root / "data" / "manifests" / "dataset_v4",
        project_root / "data" / "relations" / "dataset_v4",
        project_root / "evidence" / "dataset_v4",
    ]
    for target in targets:
        if target.exists():
            shutil.rmtree(target)

    manifest_rows: list[dict[str, object]] = []
    with zipfile.ZipFile(archive_path) as archive:
        for record in sorted(
            assigned,
            key=lambda row: (
                {"train": 0, "validation": 1, "test": 2}[str(row["split"])],
                str(row["part_category"]),
                int(row["rank_within_category"]),
            ),
        ):
            target, relative, image_id = destination_for(project_root, record)
            target.parent.mkdir(parents=True, exist_ok=True)
            raw = archive.read(str(record["source_archive_path"]))
            if sha256_bytes(raw) != record["sha256"]:
                raise ValueError(f"Source bytes changed: {record['source_archive_path']}")
            target.write_bytes(raw)
            manifest_rows.append(
                {
                    "image_id": image_id,
                    "image_group_id": f"sha256_{record['sha256']}",
                    "part_category": record["part_category"],
                    "part_family": FAMILIES[str(record["part_category"])],
                    "split": record["split"],
                    "rank_within_category": record["rank_within_category"],
                    "image_path": relative,
                    "sha256": record["sha256"],
                    "width": record["width"],
                    "height": record["height"],
                    "mode": record["mode"],
                    "source_dataset": DATASET_NAME,
                    "source_category": record["source_category"],
                    "source_original_split": record["source_original_split"],
                    "source_archive_path": record["source_archive_path"],
                }
            )

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = project_root / "data" / "manifests" / "dataset_v4" / "images.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False, lineterminator="\n")
    return manifest


def description(split: str, category: str, template_index: int) -> str:
    return TEMPLATES[split][template_index].format(name=DISPLAY_NAMES[category])


def relation_rows_for_split(manifest: pd.DataFrame, split: str) -> pd.DataFrame:
    selected = manifest.loc[manifest["split"].eq(split)].copy()
    categories = sorted(FAMILIES)
    family_members: defaultdict[str, list[str]] = defaultdict(list)
    for category, family in FAMILIES.items():
        family_members[family].append(category)
    for family in family_members:
        family_members[family].sort()

    rows: list[dict[str, object]] = []
    for index, image in enumerate(selected.itertuples(index=False), start=0):
        image_category = str(image.part_category)
        image_family = FAMILIES[image_category]
        partial_candidates = [
            candidate for candidate in family_members[image_family] if candidate != image_category
        ]
        mismatch_candidates = [
            candidate for candidate in categories if FAMILIES[candidate] != image_family
        ]
        if not partial_candidates or not mismatch_candidates:
            raise ValueError(f"Insufficient relation candidates for {image_category}")

        digest_offset = int(str(image.sha256)[:12], 16)
        choices = {
            "MATCH": [image_category, image_category],
            "PARTIAL_MATCH": [
                partial_candidates[(index * 2 + offset) % len(partial_candidates)]
                for offset in range(2)
            ],
            "MISMATCH": [
                mismatch_candidates[(digest_offset + index + offset * 17) % len(mismatch_candidates)]
                for offset in range(2)
            ],
        }
        for label in ("MATCH", "PARTIAL_MATCH", "MISMATCH"):
            for template_index, text_category in enumerate(choices[label]):
                sample_id = f"{image.image_id}_{label.lower()}_{template_index + 1}"
                rows.append(
                    {
                        "sample_id": sample_id,
                        "image_id": image.image_id,
                        "part_group_id": image.image_group_id,
                        "object_group_id": image.image_group_id,
                        "image_path": image.image_path,
                        "part_family": image_family,
                        "part_category": image_category,
                        "text_category": text_category,
                        "text_family": FAMILIES[text_category],
                        "description": description(split, text_category, template_index),
                        "label": label,
                        "source": "dataset_v4_final_test" if split == "test" else "dataset_v4",
                    }
                )
    return pd.DataFrame(rows)


def validate_generated(project_root: Path, manifest: pd.DataFrame, relations: dict[str, pd.DataFrame]) -> dict[str, object]:
    if len(manifest) != 9239:
        raise ValueError(f"Unexpected generated image count: {len(manifest)}")
    if manifest["image_id"].duplicated().any():
        raise ValueError("Duplicate generated image IDs")
    if manifest["image_path"].duplicated().any():
        raise ValueError("Duplicate generated image paths")
    if manifest["sha256"].duplicated().any():
        raise ValueError("Duplicate generated image hashes")
    if set(manifest["part_category"]) != set(FAMILIES):
        raise ValueError("Generated categories do not match the family map")

    split_counts = manifest["split"].value_counts().to_dict()
    expected_split_counts = {"train": 6463, "validation": 1388, "test": 1388}
    if split_counts != expected_split_counts:
        raise ValueError(f"Unexpected split counts: {split_counts}")

    overlap: dict[str, int] = {}
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for column in ("image_id", "image_group_id", "image_path", "sha256"):
            value = len(
                set(manifest.loc[manifest["split"].eq(left), column])
                & set(manifest.loc[manifest["split"].eq(right), column])
            )
            overlap[f"{left}_{right}_{column}"] = value
            if value:
                raise ValueError(f"Leakage in {column}: {left}/{right}")

    relation_summary: dict[str, object] = {}
    descriptions: dict[str, set[str]] = {}
    for split, table in relations.items():
        expected_rows = expected_split_counts[split] * 6
        if len(table) != expected_rows:
            raise ValueError(f"Unexpected {split} relation count: {len(table)}")
        if table["sample_id"].duplicated().any():
            raise ValueError(f"Duplicate sample IDs in {split}")
        if table["image_id"].nunique() != expected_split_counts[split]:
            raise ValueError(f"Unexpected independent image count in {split}")
        label_counts = table["label"].value_counts().to_dict()
        expected_per_label = expected_split_counts[split] * 2
        if label_counts != {
            "MATCH": expected_per_label,
            "PARTIAL_MATCH": expected_per_label,
            "MISMATCH": expected_per_label,
        }:
            raise ValueError(f"Unbalanced labels in {split}: {label_counts}")

        per_image = table.groupby(["image_id", "label"]).size().unstack(fill_value=0)
        if not (per_image[["MATCH", "PARTIAL_MATCH", "MISMATCH"]] == 2).all().all():
            raise ValueError(f"Each {split} image must have two rows per label")

        match = table[table["label"].eq("MATCH")]
        partial = table[table["label"].eq("PARTIAL_MATCH")]
        mismatch = table[table["label"].eq("MISMATCH")]
        if not match["part_category"].eq(match["text_category"]).all():
            raise ValueError(f"Invalid MATCH semantics in {split}")
        if not (
            partial["part_category"].ne(partial["text_category"])
            & partial["part_family"].eq(partial["text_family"])
        ).all():
            raise ValueError(f"Invalid PARTIAL_MATCH semantics in {split}")
        if not mismatch["part_family"].ne(mismatch["text_family"]).all():
            raise ValueError(f"Invalid MISMATCH semantics in {split}")

        descriptions[split] = set(table["description"])
        relation_summary[split] = {
            "rows": len(table),
            "independent_images": int(table["image_id"].nunique()),
            "label_counts": label_counts,
            "unique_descriptions": int(table["description"].nunique()),
            "text_category_counts": {
                key: int(value)
                for key, value in table["text_category"].value_counts().sort_index().items()
            },
        }

    description_overlap = {
        "train_validation": len(descriptions["train"] & descriptions["validation"]),
        "train_test": len(descriptions["train"] & descriptions["test"]),
        "validation_test": len(descriptions["validation"] & descriptions["test"]),
    }
    if any(description_overlap.values()):
        raise ValueError(f"Description overlap: {description_overlap}")

    # Verify every extracted file still matches its recorded source digest.
    missing = 0
    mismatch = 0
    for row in manifest.itertuples(index=False):
        path = project_root / str(row.image_path)
        if not path.is_file():
            missing += 1
        elif sha256_file(path) != str(row.sha256):
            mismatch += 1
    if missing or mismatch:
        raise ValueError(f"Extracted image verification failed: missing={missing}, mismatch={mismatch}")

    category_counts = (
        manifest.groupby(["part_category", "split"]).size().unstack(fill_value=0)
        .reindex(columns=["train", "validation", "test"])
        .sort_index()
    )
    return {
        "status": "PASS_DATASET_V4_BUILT",
        "seed": SEED,
        "split_policy": {
            "train": "remainder after rounded 15% validation and rounded 15% test per category",
            "validation_fraction": VALIDATION_FRACTION,
            "test_fraction": TEST_FRACTION,
        },
        "images": len(manifest),
        "unique_sha256": int(manifest["sha256"].nunique()),
        "categories": int(manifest["part_category"].nunique()),
        "families": len(set(FAMILIES.values())),
        "split_images": {key: int(value) for key, value in split_counts.items()},
        "category_split_counts": {
            category: {split: int(value) for split, value in row.items()}
            for category, row in category_counts.to_dict(orient="index").items()
        },
        "relation_summary": relation_summary,
        "overlap": overlap,
        "description_overlap": description_overlap,
        "extracted_images_missing": missing,
        "extracted_image_hash_mismatches": mismatch,
        "final_test_locked_before_training": True,
    }


def write_metadata(project_root: Path, source_audit: dict[str, object], generated_audit: dict[str, object]) -> None:
    evidence_dir = project_root / "evidence" / "dataset_v4"
    manifest_dir = project_root / "data" / "manifests" / "dataset_v4"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    (evidence_dir / "source_archive_audit.json").write_text(
        json.dumps(source_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (manifest_dir / "dataset_audit.json").write_text(
        json.dumps(generated_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    family_rows = [
        {
            "part_category": category,
            "display_name": DISPLAY_NAMES[category],
            "part_family": FAMILIES[category],
        }
        for category in sorted(FAMILIES)
    ]
    pd.DataFrame(family_rows).to_csv(
        manifest_dir / "category_families.csv",
        index=False,
        lineterminator="\n",
    )

    test_relations = project_root / "data" / "locked_test" / "dataset_v4" / "test_relations.csv"
    test_lock = {
        "dataset": "dataset_v4",
        "locked": True,
        "purpose": "One final evaluation after model selection; not for development or tuning.",
        "test_relations_path": test_relations.relative_to(project_root).as_posix(),
        "test_relations_sha256": sha256_file(test_relations),
        "test_images": generated_audit["split_images"]["test"],
        "test_relation_rows": generated_audit["relation_summary"]["test"]["rows"],
        "created_before_training": True,
    }
    (test_relations.parent / "test_lock.json").write_text(
        json.dumps(test_lock, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build(archive_path: Path, project_root: Path) -> dict[str, object]:
    records, source_audit = inspect_archive(archive_path)
    assigned = assign_splits(records)
    manifest = write_images_and_manifest(archive_path, project_root, assigned)

    relations = {
        split: relation_rows_for_split(manifest, split)
        for split in ("train", "validation", "test")
    }
    relation_dir = project_root / "data" / "relations" / "dataset_v4"
    relation_dir.mkdir(parents=True, exist_ok=True)
    relations["train"].to_csv(relation_dir / "train.csv", index=False, lineterminator="\n")
    relations["validation"].to_csv(
        relation_dir / "validation.csv", index=False, lineterminator="\n"
    )
    test_dir = project_root / "data" / "locked_test" / "dataset_v4"
    test_dir.mkdir(parents=True, exist_ok=True)
    relations["test"].to_csv(test_dir / "test_relations.csv", index=False, lineterminator="\n")

    generated_audit = validate_generated(project_root, manifest, relations)
    write_metadata(project_root, source_audit, generated_audit)
    return {"source": source_audit, "dataset_v4": generated_audit}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the 50-category Dataset V4 safely.")
    parser.add_argument("--archive", required=True, type=Path, help="Original Kaggle ZIP")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project repository root",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build(args.archive.resolve(), args.project_root.resolve())
    summary = result["dataset_v4"]
    print("PASS_DATASET_V4_BUILT")
    print(f"Images:      {summary['images']}")
    print(f"Categories:  {summary['categories']}")
    print(f"Families:    {summary['families']}")
    print(f"Train:       {summary['split_images']['train']}")
    print(f"Validation:  {summary['split_images']['validation']}")
    print(f"Locked test: {summary['split_images']['test']}")
    print(
        "Relation rows:",
        {split: data["rows"] for split, data in summary["relation_summary"].items()},
    )


if __name__ == "__main__":
    main()
