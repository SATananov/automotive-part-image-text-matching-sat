# External Robustness Audit V1

## Why this exists

Dataset V4 is a controlled benchmark from one public source collection and uses
structured descriptions. The official locked result remains **0.9540 accuracy /
0.9541 macro F1** and is not changed by this audit.

External Audit V1 is a separate secondary evaluation designed to ask a harder
question:

> How well does the already frozen deployment model behave on independent
> photographs and on text that is less template-like?

The audit must not be used for model selection, training, threshold tuning or
rewriting the official Dataset V4 result.

## Frozen V1 design

The protocol is defined before any external inference:

- 12 categories;
- 6 functional families;
- 2 categories per family;
- 10 independent images per category;
- 120 external images total;
- two text modes: `clean` and `natural`;
- `MATCH`, `PARTIAL_MATCH`, and `MISMATCH`;
- two rows per relation label for every image;
- 6 relation rows per image;
- 720 relation rows total.

The category plan is in `category_plan.csv`.

The design uses a one-to-one PARTIAL partner and a one-to-one MISMATCH partner,
so the text categories remain balanced across the three relation labels.

## Model used

The audit uses the already saved `practical_demo` deployment bundle.

That bundle was built with:

- Dataset V4 train + validation only;
- the already frozen model type and epoch policy;
- no locked final-test relation access;
- no new final-test inference.

External Audit V1 performs **inference only**. It does not train a model.

## Phase A - protocol scaffold

Validate the frozen plan:

```powershell
python -m external_audit.prepare --check-plan
python -m external_audit.verify
```

At this phase there are no external scores.

## Phase B - independent images and lock

Place exactly 10 images in each category folder under `images/`.

Use `provenance_template.json` to create `provenance.json` and document where
the external images came from and why they can be redistributed.

Then create the pre-inference lock:

```powershell
python -m external_audit.prepare --lock
python -m external_audit.run --check-only
python -m external_audit.verify
```

The lock records SHA-256 hashes for:

- protocol;
- category plan;
- provenance;
- external image manifest;
- external relation table;
- Dataset V4 manifest and category-family table;
- frozen deployment metadata, classifier, and TF-IDF artifact.

**Commit the locked audit before external inference.**

## Phase C - one external evaluation

Only after the Phase B lock has been committed:

```powershell
python -m external_audit.run --confirm-external-audit
python -m external_audit.verify
```

The result is saved under `external_audit/results/`.

No tuning is allowed after seeing the result. A low external score is still a
valid scientific result because the purpose is to measure domain/text shift,
not to protect the 95.4% Dataset V4 number.

## Interpretation

The external result is not a replacement for the official locked final test.

The correct reporting is:

- Dataset V4 locked test: controlled benchmark evidence;
- External Audit V1: separate robustness/domain-shift evidence.

This distinction must remain explicit in the README and report.
