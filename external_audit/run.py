from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.nn import functional as F

from external_audit.core import (
    MANIFEST_OUT,
    PREDICTIONS_PATH,
    PROJECT_ROOT,
    RELATIONS_OUT,
    RESULT_DIR,
    SUMMARY_PATH,
    file_sha256,
    git_head,
    verify_lock,
)
from practical_demo.deployment import DatasetV4DeploymentPredictor
from src.v4_data import LABELS, LABEL_TO_INDEX


def metric_block(table: pd.DataFrame) -> dict[str, float | int]:
    truth = table["label"].map(LABEL_TO_INDEX).to_numpy(np.int64)
    predicted = table["prediction"].map(LABEL_TO_INDEX).to_numpy(np.int64)
    return {
        "rows": int(len(table)),
        "accuracy": float(accuracy_score(truth, predicted)),
        "macro_f1": float(f1_score(truth, predicted, average="macro")),
    }


def run_external_audit() -> dict[str, Any]:
    lock = verify_lock()

    if PREDICTIONS_PATH.exists() or SUMMARY_PATH.exists():
        raise FileExistsError(
            "External Audit V1 results already exist. "
            "This runner is intentionally one-time."
        )

    manifest = pd.read_csv(MANIFEST_OUT)
    relations = pd.read_csv(RELATIONS_OUT)

    predictor = DatasetV4DeploymentPredictor()

    image_cache: dict[str, np.ndarray] = {}
    for row in manifest.itertuples(index=False):
        path = PROJECT_ROOT / str(row.image_path)
        with Image.open(path) as image:
            image_cache[str(row.image_id)] = (
                predictor.extract_image_feature(image)[0]
            )

    text_cache: dict[str, tuple[np.ndarray, int]] = {}
    for description in sorted(
        set(relations["description"].astype(str))
    ):
        matrix, recognized = predictor.transform_text(description)
        text_cache[description] = (matrix[0], recognized)

    image_matrix = np.stack(
        [
            image_cache[str(image_id)]
            for image_id in relations["image_id"]
        ]
    ).astype(np.float32)
    text_matrix = np.stack(
        [
            text_cache[str(description)][0]
            for description in relations["description"]
        ]
    ).astype(np.float32)
    recognized = np.asarray(
        [
            text_cache[str(description)][1]
            for description in relations["description"]
        ],
        dtype=np.int64,
    )

    all_scores: list[np.ndarray] = []
    batch_size = 128

    predictor.classifier.eval()
    with torch.no_grad():
        for start in range(0, len(relations), batch_size):
            stop = min(start + batch_size, len(relations))
            image_batch = torch.from_numpy(
                image_matrix[start:stop]
            ).to(predictor.device)
            text_batch = torch.from_numpy(
                text_matrix[start:stop]
            ).to(predictor.device)

            output = predictor.classifier(
                image_batch,
                text_batch,
            )
            relation_logits = output[0] if isinstance(output, tuple) else output
            scores = F.softmax(relation_logits, dim=1)
            all_scores.append(
                scores.cpu().numpy().astype(float)
            )

    score_matrix = np.concatenate(all_scores, axis=0)
    predicted_index = score_matrix.argmax(axis=1)
    predicted_labels = [
        LABELS[int(index)]
        for index in predicted_index
    ]

    predictions = relations.copy()
    predictions["prediction"] = predicted_labels
    predictions["correct"] = predictions["label"].eq(
        predictions["prediction"]
    )
    predictions["model_score"] = score_matrix.max(axis=1)
    predictions["recognized_text_features"] = recognized

    for index, label in enumerate(LABELS):
        predictions[f"score_{label.lower()}"] = score_matrix[:, index]

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
        lineterminator="\n",
    )

    overall = metric_block(predictions)
    by_text_mode = {
        mode: metric_block(group.copy())
        for mode, group in predictions.groupby("text_mode")
    }
    by_label = {
        label: {
            "rows": int(len(group)),
            "accuracy": float(group["correct"].mean()),
        }
        for label, group in predictions.groupby("label")
    }

    per_category = {
        category: {
            "rows": int(len(group)),
            "accuracy": float(group["correct"].mean()),
        }
        for category, group in predictions.groupby("part_category")
    }

    summary: dict[str, Any] = {
        "status": "EXTERNAL_AUDIT_V1_EVALUATED",
        "secondary_evidence_only": True,
        "official_dataset_v4_result_changed": False,
        "official_locked_final_test_loaded": False,
        "training_performed": False,
        "post_audit_tuning_allowed": False,
        "model_source": (
            "Frozen practical_demo deployment bundle built from "
            "Dataset V4 train + validation only"
        ),
        "external_images": int(
            predictions["image_id"].nunique()
        ),
        "external_relation_rows": int(len(predictions)),
        "overall": overall,
        "by_text_mode": by_text_mode,
        "by_label": by_label,
        "per_image_category": per_category,
        "zero_recognized_text_feature_rows": int(
            (recognized == 0).sum()
        ),
        "predictions_sha256": file_sha256(PREDICTIONS_PATH),
        "external_lock_sha256": file_sha256(
            PROJECT_ROOT
            / "external_audit"
            / "external_lock.json"
        ),
        "lock_external_relations_sha256": lock[
            "external_relations_sha256"
        ],
        "deployment_classifier_sha256": lock[
            "deployment_classifier_sha256"
        ],
        "deployment_tfidf_sha256": lock[
            "deployment_tfidf_sha256"
        ],
        "evaluation_code_git_head": git_head(),
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run External Robustness Audit V1 once on the already locked "
            "external image-text relation set."
        )
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify the external lock and frozen deployment bundle without inference.",
    )
    parser.add_argument(
        "--confirm-external-audit",
        action="store_true",
        help="Explicitly allow the one-time external robustness inference.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.check_only:
        lock = verify_lock()
        print("PASS_EXTERNAL_AUDIT_V1_LOCK_CHECK")
        print("External images:", lock["external_images"])
        print("External relation rows:", lock["external_relation_rows"])
        print("Official locked final test loaded: False")
        print("External inference performed: False")
        return

    if not args.confirm_external_audit:
        raise SystemExit(
            "External audit not run. Lock and commit the external dataset first, "
            "then use --confirm-external-audit exactly once."
        )

    summary = run_external_audit()
    print("")
    print(summary["status"])
    print("External images:", summary["external_images"])
    print("External relation rows:", summary["external_relation_rows"])
    print(
        "External accuracy:",
        f"{float(summary['overall']['accuracy']):.4f}",
    )
    print(
        "External macro F1:",
        f"{float(summary['overall']['macro_f1']):.4f}",
    )
    print("Post-audit tuning: NOT ALLOWED")


if __name__ == "__main__":
    main()
