from __future__ import annotations

import pandas as pd
import pytest

from src.v4_data import load_relations_v4
from tools import run_dataset_v4_final_test as final_runner


def test_locked_test_still_requires_explicit_permission() -> None:
    with pytest.raises(PermissionError):
        load_relations_v4("test")


def test_final_runner_refuses_without_confirmation() -> None:
    args = final_runner.argparse.Namespace(
        check_only=False,
        confirm_final_test=False,
        feature_batch_size=64,
    )
    with pytest.raises(SystemExit, match="Final test not opened"):
        final_runner.run_final_test(args)


def test_final_text_vectorizer_fits_only_development_text() -> None:
    development = pd.DataFrame({"description": ["brake disc", "oil filter"]})
    final_test = pd.DataFrame({"description": ["brake filter unseenword"]})
    development_text, final_text, dimension = final_runner.fit_final_text_features(
        development, final_test
    )
    assert development_text.shape == (2, dimension)
    assert final_text.shape == (1, dimension)
    assert dimension > 0

def test_clean_clone_validation_hash_and_check_only(capsys: pytest.CaptureFixture[str]) -> None:
    actual_hash = final_runner.file_sha256(final_runner.VALIDATION_SUMMARY)
    assert actual_hash == final_runner.EXPECTED_VALIDATION_SHA256
    assert actual_hash == "41e88e8ac40f03c38229f52bc13a893bc0cb0a414b6b2087890c655d8ca55cd4"

    final_runner.check_only()
    output = capsys.readouterr().out
    assert "PASS_DATASET_V4_FINAL_TEST_POLICY_CHECK" in output
