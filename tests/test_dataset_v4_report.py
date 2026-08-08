from __future__ import annotations

from pathlib import Path

import nbformat


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_readme_uses_dataset_v4_as_official_result() -> None:
    text = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    required = [
        "9,239",
        "50 automotive-part categories",
        "41,742",
        "9,030",
        "multimodal_auxiliary",
        "0.9540",
        "0.9541",
        "8,615 / 9,030",
        "not evidence that the system is production-ready",
    ]
    for item in required:
        assert item in text


def test_core_docs_are_updated_for_v4() -> None:
    methodology = (PROJECT_ROOT / "docs" / "methodology.md").read_text(encoding="utf-8")
    provenance = (PROJECT_ROOT / "docs" / "provenance.md").read_text(encoding="utf-8")
    policy = (PROJECT_ROOT / "docs" / "test_policy.md").read_text(encoding="utf-8")
    data_readme = (PROJECT_ROOT / "data" / "README.md").read_text(encoding="utf-8")

    assert "9,239" in methodology and "50" in methodology
    assert "ee269fb85db3c53630b68aa0e302197daceb81fa6c1f9e63140722dc0e9ebd6e" in provenance
    assert "79bae874c5b365200e7562b1a3a1d8ee28e8ea3fccfab86d4034b89332260ef2" in policy
    assert "Dataset V4" in data_readme


def test_official_notebook_is_executed_v4_report() -> None:
    path = PROJECT_ROOT / "project.ipynb"
    notebook = nbformat.read(path, as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]

    assert len(code_cells) == 14
    assert all(cell.execution_count is not None for cell in code_cells)
    assert not [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]

    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    source = "\n".join(cell.source for cell in code_cells)
    assert "Dataset V4" in markdown
    assert "95.4%" in markdown
    assert "External robustness audit" in markdown
    assert "0.6722" in markdown
    assert "0.6001" in markdown
    assert "completely separate external dataset" not in markdown
    assert "--confirm-final-test" not in source
    assert "run_dataset_v4_final_test" not in source


def test_notebook_reads_frozen_results_instead_of_training() -> None:
    notebook = nbformat.read(PROJECT_ROOT / "project.ipynb", as_version=4)
    source = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "code"
    )
    assert "step03_validation_summary.json" in source
    assert "step04_final_test_predictions.csv" in source
    assert "step04_sanity_audit.json" in source
    assert "train_classifier_v4" not in source
    assert "fit(" not in source


def test_dataset_v4_submission_hash_manifest_is_explicit() -> None:
    text = (PROJECT_ROOT / "evidence" / "hashes_dataset_v4_submission.sha256").read_text(
        encoding="utf-8"
    )
    assert "project.ipynb" in text
    assert "data/manifests/dataset_v4/images.csv" in text
    assert "results/dataset_v4/step04_final_test_summary.json" in text
    assert "external_audit/results/external_summary.json" in text
