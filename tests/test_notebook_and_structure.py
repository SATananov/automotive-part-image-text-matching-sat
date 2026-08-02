from __future__ import annotations

from pathlib import Path

import nbformat

from src import verify as verify_module
from src.data import PROJECT_ROOT


def test_one_executed_official_notebook() -> None:
    notebooks = sorted(PROJECT_ROOT.glob("*.ipynb"))
    assert [path.name for path in notebooks] == ["project.ipynb"]
    notebook = nbformat.read(notebooks[0], as_version=4)
    code = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert len(code) == 14
    assert all(cell.execution_count is not None for cell in code)
    assert not [out for cell in code for out in cell.outputs if out.output_type == "error"]


def test_obsolete_project_paths_are_absent() -> None:
    paths = [path.relative_to(PROJECT_ROOT).as_posix().lower() for path in PROJECT_ROOT.rglob("*")]
    assert not any("dataset_v2" in path for path in paths)
    assert not (PROJECT_ROOT / "project_v3.ipynb").exists()
    assert not (PROJECT_ROOT / "project_final.ipynb").exists()


def test_hygiene_ignores_local_git_venv_and_caches(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    (tmp_path / ".venv" / "Lib" / "site-packages").mkdir(parents=True)
    (tmp_path / "src" / "__pycache__").mkdir(parents=True)
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / ".ipynb_checkpoints").mkdir()
    (tmp_path / "src" / "models.py").write_text("# distributable file\n", encoding="utf-8")

    monkeypatch.setattr(verify_module, "PROJECT_ROOT", tmp_path)
    result = verify_module.verify_hygiene()

    assert result["dataset_v2_paths"] == 0
    assert result["competing_notebooks"] == 0
    assert result["local_environment_dirs_ignored"] == [".git", ".venv"]
    assert result["runtime_cache_dirs_ignored"] == [
        ".ipynb_checkpoints",
        ".pytest_cache",
        "src/__pycache__",
    ]
