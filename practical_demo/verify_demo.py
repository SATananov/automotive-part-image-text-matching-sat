from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from practical_demo.deployment import (
    CLASSIFIER_PATH,
    METADATA_PATH,
    TFIDF_PATH,
    load_deployment_metadata,
    verify_deployment_artifacts,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    print("=== PRACTICAL DEMO VERIFICATION ===")

    metadata = load_deployment_metadata()

    assert metadata["status"] == (
        "PASS_DATASET_V4_DEPLOYMENT_BUNDLE_CREATED"
    )

    assert metadata["training_data"] == (
        "Dataset V4 train + validation only"
    )

    assert metadata["locked_final_test_relations_loaded"] is False
    assert metadata["new_final_test_inference_performed"] is False
    assert metadata["post_test_tuning_performed"] is False

    assert CLASSIFIER_PATH.is_file()
    assert TFIDF_PATH.is_file()
    assert METADATA_PATH.is_file()

    verify_deployment_artifacts(metadata)

    state = torch.load(
        CLASSIFIER_PATH,
        map_location="cpu",
        weights_only=True,
    )

    assert isinstance(state, dict)
    assert state

    builder_path = (
        PROJECT_ROOT
        / "practical_demo"
        / "build_bundle.py"
    )

    assert builder_path.is_file()

    builder_source = builder_path.read_text(
        encoding="utf-8"
    )

    forbidden = [
        'load_relations_v4("test")',
        "allow_locked_test",
        "--confirm-final-test",
        "run_dataset_v4_final_test",
    ]

    for token in forbidden:
        assert token not in builder_source

    assert 'load_relations_v4("train")' in builder_source
    assert 'load_relations_v4("validation")' in builder_source

    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "practical_demo.predict",
            "--help",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert cli.returncode == 0

    print("Deployment artifacts: PASS")
    print("Artifact SHA-256 verification: PASS")
    print("Development-only training policy: PASS")
    print("Locked final-test relations loaded: False")
    print("New final-test inference performed: False")
    print("CLI startup: PASS")
    print()
    print("PASS_PRACTICAL_DEMO")


if __name__ == "__main__":
    main()
