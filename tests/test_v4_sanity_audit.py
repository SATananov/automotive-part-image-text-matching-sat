from __future__ import annotations

import numpy as np
import pandas as pd

from tools.audit_dataset_v4_final_result import (
    code_leakage_checks,
    hamming,
    metric_block,
    severity,
)


def test_hamming_distance() -> None:
    assert hamming(0b0000, 0b0000) == 0
    assert hamming(0b0000, 0b1111) == 4
    assert hamming(0b1010, 0b0011) == 2


def test_near_duplicate_severity_thresholds() -> None:
    assert severity(4, 6) == "very_strict"
    assert severity(6, 8) == "strict"
    assert severity(8, 10) == "screen"
    assert severity(9, 10) is None
    assert severity(8, 11) is None


def test_metric_block() -> None:
    frame = pd.DataFrame(
        {
            "image_id": ["a", "a", "b"],
            "label": ["MATCH", "MISMATCH", "PARTIAL_MATCH"],
            "prediction": ["MATCH", "MATCH", "PARTIAL_MATCH"],
        }
    )
    result = metric_block(frame)
    assert result["rows"] == 3
    assert result["independent_images"] == 2
    assert np.isclose(result["accuracy"], 2 / 3)


def test_code_leakage_checks() -> None:
    checks = code_leakage_checks()
    assert checks["model_receives_only_image_and_text"] is True
    assert checks["final_test_image_category_targets_are_dummy_zero"] is True
    assert checks["final_test_text_category_targets_are_dummy_zero"] is True
    assert checks["tfidf_fit_on_development_only"] is True
