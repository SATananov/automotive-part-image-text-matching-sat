from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from external_audit.core import (
    LOCK_PATH,
    PREDICTIONS_PATH,
    SUMMARY_PATH,
    file_sha256,
    validate_plan,
    verify_lock,
)
from src.v4_data import LABEL_TO_INDEX


def verify_source_isolation() -> None:
    root = Path(__file__).resolve().parent
    forbidden = [
        'load_relations_v4("test")',
        "allow_locked_test",
        "--confirm-final-test",
        "run_dataset_v4_final_test",
    ]
    for name in ("prepare.py", "run.py"):
        source = (root / name).read_text(encoding="utf-8")
        for token in forbidden:
            if token in source:
                raise AssertionError(
                    f"Forbidden official final-test token in {name}: {token}"
                )


def verify_results() -> None:
    lock = verify_lock()

    if not PREDICTIONS_PATH.is_file() or not SUMMARY_PATH.is_file():
        raise FileNotFoundError(
            "External audit result files are incomplete."
        )

    summary = json.loads(
        SUMMARY_PATH.read_text(encoding="utf-8")
    )
    if summary.get("status") != "EXTERNAL_AUDIT_V1_EVALUATED":
        raise ValueError("Unexpected external audit result status.")
    if summary.get("secondary_evidence_only") is not True:
        raise ValueError("External result must remain secondary evidence.")
    if summary.get("official_dataset_v4_result_changed") is not False:
        raise ValueError("External audit must not change official Dataset V4.")
    if summary.get("official_locked_final_test_loaded") is not False:
        raise ValueError("External audit unexpectedly reports final-test access.")
    if summary.get("training_performed") is not False:
        raise ValueError("External audit unexpectedly reports training.")
    if summary.get("post_audit_tuning_allowed") is not False:
        raise ValueError("External audit unexpectedly allows tuning.")

    if file_sha256(PREDICTIONS_PATH) != summary.get(
        "predictions_sha256"
    ):
        raise ValueError("External prediction hash mismatch.")

    predictions = pd.read_csv(PREDICTIONS_PATH)
    truth = predictions["label"].map(LABEL_TO_INDEX)
    predicted = predictions["prediction"].map(LABEL_TO_INDEX)

    accuracy = float(accuracy_score(truth, predicted))
    macro_f1 = float(f1_score(truth, predicted, average="macro"))

    recorded = summary["overall"]
    if abs(accuracy - float(recorded["accuracy"])) > 1e-12:
        raise ValueError("External accuracy mismatch.")
    if abs(macro_f1 - float(recorded["macro_f1"])) > 1e-12:
        raise ValueError("External macro F1 mismatch.")

    if file_sha256(
        Path(__file__).resolve().parent / "external_lock.json"
    ) != summary.get("external_lock_sha256"):
        raise ValueError("External lock hash recorded in result has changed.")

    if summary.get("lock_external_relations_sha256") != lock.get(
        "external_relations_sha256"
    ):
        raise ValueError("External relation lineage mismatch.")


def main() -> None:
    plan = validate_plan()
    verify_source_isolation()

    print("External audit protocol:", plan["status"])
    print("Categories:", plan["categories"])
    print("Families:", plan["families"])
    print("Target images:", plan["target_images"])
    print("Target relation rows:", plan["target_relation_rows"])
    print("Official locked final-test code isolation: PASS")

    if not LOCK_PATH.is_file():
        if PREDICTIONS_PATH.exists() or SUMMARY_PATH.exists():
            raise ValueError(
                "External result exists without an external lock."
            )
        print("External audit lock: NOT CREATED YET")
        print("External inference performed: False")
        print("")
        print("PASS_EXTERNAL_AUDIT_V1_PROTOCOL_READY")
        return

    lock = verify_lock()
    print("External audit lock: PASS")
    print("Locked images:", lock["external_images"])
    print("Locked relation rows:", lock["external_relation_rows"])

    if PREDICTIONS_PATH.exists() or SUMMARY_PATH.exists():
        verify_results()
        print("External result verification: PASS")
        print("")
        print("PASS_EXTERNAL_AUDIT_V1_RESULTS_VERIFIED")
    else:
        print("External inference performed: False")
        print("")
        print("PASS_EXTERNAL_AUDIT_V1_LOCKED_NOT_EVALUATED")


if __name__ == "__main__":
    main()
