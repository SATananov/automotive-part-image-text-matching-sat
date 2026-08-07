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
