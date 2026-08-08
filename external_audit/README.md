# External Robustness Audit V1.1

## Why this exists

Dataset V4 is a controlled benchmark from one public source collection and uses
structured descriptions. The official locked result remains **0.9540 accuracy /
0.9541 macro F1** and is not changed by this audit.

External Audit V1.1 is a separate secondary evaluation designed to ask a harder
question:

> How well does the already frozen deployment model behave on independent
> photographs and on text that is less template-like?

The audit must not be used for model selection, training, threshold tuning or
rewriting the official Dataset V4 result.

## Protocol amendment V1.1

Before any external inference, a source-feasibility review found that generic
Wikimedia Commons search results were too semantically noisy for several of
the original 12 image categories. The protocol was therefore narrowed to six
categories in three paired functional families, using dedicated Commons
categories and preserving balanced text-category use across relation labels.

The amendment chronology and rationale are recorded in
`protocol_amendment_v1_1.md`.

## Frozen V1.1 design

The protocol is defined before any external inference:

- 6 categories;
- 3 functional families;
- 2 categories per family;
- 10 independent images per category;
- 60 external images total;
- two text modes: `clean` and `natural`;
- `MATCH`, `PARTIAL_MATCH`, and `MISMATCH`;
- two rows per relation label for every image;
- 6 relation rows per image;
- 360 relation rows total.

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

External Audit V1.1 performs **inference only**. It does not train a model.

## Phase A - protocol scaffold

Validate the frozen plan:

```powershell
python -m external_audit.prepare --check-plan
python -m external_audit.verify
```

At this phase there are no external scores.

## Phase B - independent images and lock

Place exactly 10 images in each of the six V1.1 category folders under `images/`.

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

## Observed V1.1 result

The one-time external evaluation was completed only after the V1.1 protocol
and the 60-image / 360-relation dataset had been locked, committed, and pushed.

| Condition | Accuracy | Macro F1 |
| --- | ---: | ---: |
| Overall external audit | **0.6722** | **0.6725** |
| Clean text | **0.7444** | **0.7458** |
| Natural text | **0.6000** | **0.6001** |

The audit records **60 relation rows
with zero recognized TF-IDF features**. Together with the clean-versus-natural
gap, this shows that the frozen text representation is sensitive to wording
outside its development vocabulary.

The category-level accuracy also varies substantially, so the external result
should be treated as a small robustness audit, not as proof of performance on
all real automotive-part photographs.

Detailed frozen results are in `results/README.md`.

## Interpretation

The external result is not a replacement for the official locked final test.

The correct reporting is:

- Dataset V4 locked test: controlled benchmark evidence;
- External Audit V1.1: separate robustness/domain-shift evidence.

This distinction must remain explicit in the README and report.
