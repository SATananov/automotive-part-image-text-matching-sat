from pathlib import Path

import pandas as pd

from external_audit.core import (
    CATEGORY_PLAN_PATH,
    load_protocol,
    validate_plan,
)


def test_external_audit_v1_protocol_shape() -> None:
    summary = validate_plan()
    assert summary["status"] == "PASS_EXTERNAL_AUDIT_V1_PLAN"
    assert summary["categories"] == 12
    assert summary["families"] == 6
    assert summary["images_per_category"] == 10
    assert summary["target_images"] == 120
    assert summary["target_relation_rows"] == 720


def test_external_audit_v1_partner_mappings_are_balanced() -> None:
    plan = pd.read_csv(CATEGORY_PLAN_PATH)

    assert plan["part_category"].nunique() == 12
    assert plan["partial_partner"].nunique() == 12
    assert plan["mismatch_partner"].nunique() == 12

    family = dict(
        zip(
            plan["part_category"],
            plan["part_family"],
            strict=True,
        )
    )

    for row in plan.itertuples(index=False):
        assert row.partial_partner != row.part_category
        assert row.mismatch_partner != row.part_category
        assert family[row.partial_partner] == row.part_family
        assert family[row.mismatch_partner] != row.part_family


def test_external_audit_v1_isolation_from_official_final_test() -> None:
    protocol = load_protocol()
    assert protocol["official_dataset_v4_unchanged"] is True
    assert protocol["uses_locked_final_test"] is False
    assert protocol["training_allowed"] is False
    assert protocol["post_audit_tuning_allowed"] is False

    root = Path(__file__).resolve().parents[1] / "external_audit"
    forbidden = [
        'load_relations_v4("test")',
        "allow_locked_test",
        "--confirm-final-test",
        "run_dataset_v4_final_test",
    ]

    for filename in ("prepare.py", "run.py"):
        source = (root / filename).read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source
