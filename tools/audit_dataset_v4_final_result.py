from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
from sklearn.metrics import accuracy_score, f1_score

from src.v4_data import LABELS, load_image_manifest_v4, load_relations_v4
from src.v4_training import MultimodalClassifierV4
from tools.run_dataset_v4_final_test import (
    EXPECTED_VALIDATION_SHA256,
    fit_final_text_features,
    make_final_test_arrays,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "results" / "dataset_v4"
PREDICTIONS_PATH = RESULT_DIR / "step04_final_test_predictions.csv"
SUMMARY_PATH = RESULT_DIR / "step04_final_test_summary.json"
AUDIT_PATH = RESULT_DIR / "step04_sanity_audit.json"
CANDIDATES_PATH = RESULT_DIR / "step04_sanity_near_duplicate_candidates.csv"
TEST_LOCK_PATH = PROJECT_ROOT / "data" / "locked_test" / "dataset_v4" / "test_lock.json"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def perceptual_hash(path: Path) -> int:
    with Image.open(path) as image:
        gray = image.convert("L").resize((32, 32), Image.Resampling.LANCZOS)
        values = np.asarray(gray, dtype=np.float32)
    low = dctn(values, type=2, norm="ortho")[:8, :8].reshape(-1)
    median = float(np.median(low[1:]))
    result = 0
    for bit in low > median:
        result = (result << 1) | int(bit)
    return result


def difference_hash(path: Path) -> int:
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        values = np.asarray(gray, dtype=np.int16)
    bits = (values[:, 1:] > values[:, :-1]).reshape(-1)
    result = 0
    for bit in bits:
        result = (result << 1) | int(bit)
    return result


def hamming(left: int, right: int) -> int:
    return (int(left) ^ int(right)).bit_count()


def severity(phash_distance: int, dhash_distance: int) -> str | None:
    if phash_distance <= 4 and dhash_distance <= 6:
        return "very_strict"
    if phash_distance <= 6 and dhash_distance <= 8:
        return "strict"
    if phash_distance <= 8 and dhash_distance <= 10:
        return "screen"
    return None


def exact_split_overlap(manifest: pd.DataFrame) -> dict[str, int]:
    result: dict[str, int] = {}
    for left, right in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        for column in ("image_id", "image_group_id", "image_path", "sha256"):
            a = set(manifest.loc[manifest["split"].eq(left), column].astype(str))
            b = set(manifest.loc[manifest["split"].eq(right), column].astype(str))
            result[f"{left}_{right}_{column}"] = len(a & b)
    return result


def compute_hashes(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    total = len(manifest)
    for index, row in enumerate(manifest.itertuples(index=False), start=1):
        path = PROJECT_ROOT / str(row.image_path)
        rows.append(
            {
                "image_id": str(row.image_id),
                "split": str(row.split),
                "part_category": str(row.part_category),
                "source_original_split": str(row.source_original_split),
                "source_archive_path": str(row.source_archive_path),
                "image_path": str(row.image_path),
                "phash": perceptual_hash(path),
                "dhash": difference_hash(path),
            }
        )
        if index % 1000 == 0 or index == total:
            print(f"  perceptual hashes: {index}/{total}")
    return pd.DataFrame(rows)


def near_duplicate_candidates(hash_table: pd.DataFrame) -> pd.DataFrame:
    pairs: list[dict[str, object]] = []
    comparisons = (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    )
    for category, category_rows in hash_table.groupby("part_category", sort=True):
        for left_split, right_split in comparisons:
            left = category_rows.loc[category_rows["split"].eq(left_split)]
            right = category_rows.loc[category_rows["split"].eq(right_split)]
            for left_row in left.itertuples(index=False):
                for right_row in right.itertuples(index=False):
                    phash_distance = hamming(left_row.phash, right_row.phash)
                    if phash_distance > 8:
                        continue
                    dhash_distance = hamming(left_row.dhash, right_row.dhash)
                    level = severity(phash_distance, dhash_distance)
                    if level is None:
                        continue
                    pairs.append(
                        {
                            "severity": level,
                            "part_category": category,
                            "left_split": left_split,
                            "right_split": right_split,
                            "left_image_id": left_row.image_id,
                            "right_image_id": right_row.image_id,
                            "left_source_original_split": left_row.source_original_split,
                            "right_source_original_split": right_row.source_original_split,
                            "left_source_archive_path": left_row.source_archive_path,
                            "right_source_archive_path": right_row.source_archive_path,
                            "left_image_path": left_row.image_path,
                            "right_image_path": right_row.image_path,
                            "phash_distance": phash_distance,
                            "dhash_distance": dhash_distance,
                        }
                    )
    if not pairs:
        return pd.DataFrame(
            columns=[
                "severity",
                "part_category",
                "left_split",
                "right_split",
                "left_image_id",
                "right_image_id",
                "phash_distance",
                "dhash_distance",
            ]
        )
    order = {"very_strict": 0, "strict": 1, "screen": 2}
    result = pd.DataFrame(pairs)
    result["_order"] = result["severity"].map(order)
    result = result.sort_values(
        ["_order", "phash_distance", "dhash_distance", "part_category"]
    ).drop(columns="_order")
    return result.reset_index(drop=True)


def metric_block(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty:
        return {"rows": 0, "independent_images": 0, "accuracy": None, "macro_f1": None}
    return {
        "rows": int(len(frame)),
        "independent_images": int(frame["image_id"].nunique()),
        "accuracy": float(accuracy_score(frame["label"], frame["prediction"])),
        "macro_f1": float(
            f1_score(frame["label"], frame["prediction"], labels=list(LABELS), average="macro")
        ),
    }


def duplicate_sensitivity(
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
) -> dict[str, object]:
    result: dict[str, object] = {}
    threshold_levels = {
        "very_strict": {"very_strict"},
        "strict": {"very_strict", "strict"},
        "screen": {"very_strict", "strict", "screen"},
    }
    dev_test = candidates.loc[candidates["right_split"].eq("test")].copy()
    for name, levels in threshold_levels.items():
        flagged = set(dev_test.loc[dev_test["severity"].isin(levels), "right_image_id"].astype(str))
        kept = predictions.loc[~predictions["image_id"].astype(str).isin(flagged)].copy()
        result[name] = {
            "flagged_test_images": len(flagged),
            "flagged_test_image_fraction": len(flagged) / predictions["image_id"].nunique(),
            "metrics_after_excluding_flagged_images": metric_block(kept),
        }
    return result


def image_level_metrics(predictions: pd.DataFrame) -> dict[str, object]:
    frame = predictions.copy()
    frame["correct_bool"] = frame["label"].eq(frame["prediction"])
    grouped = frame.groupby("image_id")["correct_bool"].agg(["mean", "size"])
    return {
        "independent_images": int(len(grouped)),
        "equal_weight_mean_relation_accuracy": float(grouped["mean"].mean()),
        "median_relation_accuracy": float(grouped["mean"].median()),
        "minimum_relation_accuracy": float(grouped["mean"].min()),
        "perfect_images": int(grouped["mean"].eq(1.0).sum()),
        "perfect_image_fraction": float(grouped["mean"].eq(1.0).mean()),
        "relations_per_image_min": int(grouped["size"].min()),
        "relations_per_image_median": float(grouped["size"].median()),
        "relations_per_image_max": int(grouped["size"].max()),
    }


def category_metrics(predictions: pd.DataFrame) -> dict[str, object]:
    output: dict[str, object] = {}
    for category, frame in predictions.groupby("part_category", sort=True):
        output[str(category)] = metric_block(frame)
    return output


def code_leakage_checks() -> dict[str, object]:
    signature = inspect.signature(MultimodalClassifierV4.forward)
    forward_inputs = [name for name in signature.parameters if name != "self"]

    toy_relations = pd.DataFrame(
        {
            "image_id": ["toy"],
            "label": ["MATCH"],
            "part_category": ["battery"],
            "text_category": ["starter"],
        }
    )
    toy_text = np.zeros((1, 2), dtype=np.float32)
    toy_image = {"toy": np.zeros(3, dtype=np.float32)}
    arrays = make_final_test_arrays(toy_relations, toy_text, toy_image)
    dummy_image_categories = arrays[3]
    dummy_text_categories = arrays[4]

    text_source = inspect.getsource(fit_final_text_features)
    compact_text_source = "".join(text_source.split())
    text_fit_on_development_only = (
        "fit_transform(development" in compact_text_source
        and "transform(final_test" in compact_text_source
        and "fit_transform(final_test" not in compact_text_source
    )

    return {
        "model_forward_inputs": forward_inputs,
        "model_receives_only_image_and_text": forward_inputs == ["image", "text"],
        "final_test_image_category_targets_are_dummy_zero": bool(
            np.all(dummy_image_categories == 0)
        ),
        "final_test_text_category_targets_are_dummy_zero": bool(
            np.all(dummy_text_categories == 0)
        ),
        "tfidf_fit_on_development_only": text_fit_on_development_only,
        "expected_validation_summary_sha256": EXPECTED_VALIDATION_SHA256,
    }


def verify_predictions_against_locked_relations(predictions: pd.DataFrame) -> dict[str, object]:
    locked = load_relations_v4("test", allow_locked_test=True)
    columns = ["sample_id", "image_id", "part_category", "text_category", "label"]
    left = predictions[columns].sort_values("sample_id").reset_index(drop=True)
    right = locked[columns].sort_values("sample_id").reset_index(drop=True)
    return {
        "same_number_of_rows": len(left) == len(right),
        "same_sample_ids_and_ground_truth_columns": left.equals(right),
    }


def relation_text_overlap() -> dict[str, int]:
    train = set(load_relations_v4("train")["description"].astype(str))
    validation = set(load_relations_v4("validation")["description"].astype(str))
    test = set(load_relations_v4("test", allow_locked_test=True)["description"].astype(str))
    return {
        "train_validation": len(train & validation),
        "train_test": len(train & test),
        "validation_test": len(validation & test),
    }


def main() -> None:
    if not PREDICTIONS_PATH.is_file() or not SUMMARY_PATH.is_file():
        raise FileNotFoundError("Frozen Step 04 result files are required for this audit")

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    predictions = pd.read_csv(PREDICTIONS_PATH)
    manifest = load_image_manifest_v4()
    lock = json.loads(TEST_LOCK_PATH.read_text(encoding="utf-8"))

    recomputed = metric_block(predictions)
    if file_sha256(PREDICTIONS_PATH) != str(summary["predictions_sha256"]):
        raise ValueError("Prediction CSV SHA-256 does not match the frozen final summary")
    if file_sha256(PROJECT_ROOT / "data" / "locked_test" / "dataset_v4" / "test_relations.csv") != str(lock["test_relations_sha256"]):
        raise ValueError("Locked test relation SHA-256 changed")
    if abs(float(summary["final_test_accuracy"]) - float(recomputed["accuracy"])) > 1e-12:
        raise ValueError("Stored and recomputed final-test accuracy differ")
    if abs(float(summary["final_test_macro_f1"]) - float(recomputed["macro_f1"])) > 1e-12:
        raise ValueError("Stored and recomputed final-test macro F1 differ")

    exact_overlap = exact_split_overlap(manifest)
    if any(exact_overlap.values()):
        raise ValueError(f"Exact split leakage detected: {exact_overlap}")

    print("Computing perceptual hashes for Dataset V4 images...")
    hash_table = compute_hashes(manifest)
    print("Screening cross-split near-duplicate candidates...")
    candidates = near_duplicate_candidates(hash_table)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(CANDIDATES_PATH, index=False, lineterminator="\n")

    source_mix = pd.crosstab(
        manifest["source_original_split"], manifest["split"]
    ).reindex(index=["train", "valid", "test"], columns=["train", "validation", "test"], fill_value=0)

    severity_counts = {
        level: int(candidates["severity"].eq(level).sum())
        for level in ("very_strict", "strict", "screen")
    }
    test_candidate_counts = {}
    for level, included in {
        "very_strict": {"very_strict"},
        "strict": {"very_strict", "strict"},
        "screen": {"very_strict", "strict", "screen"},
    }.items():
        ids = set(
            candidates.loc[
                candidates["right_split"].eq("test") & candidates["severity"].isin(included),
                "right_image_id",
            ].astype(str)
        )
        test_candidate_counts[level] = len(ids)

    result: dict[str, object] = {
        "status": "PASS_WITH_NEAR_DUPLICATE_WARNING" if len(candidates) else "PASS_NO_NEAR_DUPLICATE_CANDIDATES",
        "audit_scope": "diagnostic only; no model training and no new final-test inference",
        "frozen_result": {
            "summary_sha256": file_sha256(SUMMARY_PATH),
            "predictions_sha256": file_sha256(PREDICTIONS_PATH),
            "stored_status": summary.get("status"),
            "post_test_tuning_allowed": summary.get("post_test_tuning_allowed"),
            "metrics_recomputed": recomputed,
        },
        "prediction_file_vs_locked_relations": verify_predictions_against_locked_relations(predictions),
        "code_leakage_checks": code_leakage_checks(),
        "exact_split_overlap": exact_overlap,
        "description_overlap": relation_text_overlap(),
        "image_level_metrics": image_level_metrics(predictions),
        "category_metrics": category_metrics(predictions),
        "source_original_split_vs_v4_split": {
            str(index): {str(column): int(value) for column, value in row.items()}
            for index, row in source_mix.to_dict(orient="index").items()
        },
        "near_duplicate_screen": {
            "method": "64-bit pHash + 64-bit dHash within the same category; candidates require visual review",
            "candidate_pairs_total": int(len(candidates)),
            "severity_pair_counts": severity_counts,
            "unique_final_test_images_flagged_cumulative": test_candidate_counts,
            "sensitivity": duplicate_sensitivity(predictions, candidates),
            "candidate_csv": CANDIDATES_PATH.relative_to(PROJECT_ROOT).as_posix(),
        },
        "interpretation": {
            "high_accuracy_is_not_proof_of_leakage": True,
            "task_is_structured": "Text explicitly names a part category; the relation target depends on category/family agreement.",
            "near_duplicate_candidates_make_random_image_level_split_less_independent": bool(len(candidates)),
            "deployment_claim": "The result is an evaluation on this constructed benchmark, not evidence of production readiness.",
        },
    }

    AUDIT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("")
    print(result["status"])
    print(f"Frozen accuracy:     {float(recomputed['accuracy']):.4f}")
    print(f"Frozen macro F1:     {float(recomputed['macro_f1']):.4f}")
    print(f"Exact split overlap: {max(exact_overlap.values())}")
    print(f"Near-dup pairs:      {len(candidates)}")
    print(f"Very strict pairs:   {severity_counts['very_strict']}")
    print(f"Strict-only pairs:   {severity_counts['strict']}")
    print(f"Screen-only pairs:   {severity_counts['screen']}")
    print(f"Flagged test images (very strict): {test_candidate_counts['very_strict']}")
    print(f"Flagged test images (strict):      {test_candidate_counts['strict']}")
    print(f"Flagged test images (screen):      {test_candidate_counts['screen']}")
    print("New final-test inference performed: False")
    print(f"Audit report: {AUDIT_PATH.relative_to(PROJECT_ROOT).as_posix()}")
    print(f"Candidates:   {CANDIDATES_PATH.relative_to(PROJECT_ROOT).as_posix()}")


if __name__ == "__main__":
    main()
