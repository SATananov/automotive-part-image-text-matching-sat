# External Robustness Audit V1.1 results

## Status

This is the frozen one-time result of the pre-committed External Audit V1.1.
The protocol and 60-image external dataset were committed before evaluation.

- External inference runs: **1**
- Training during this audit: **no**
- Post-audit tuning: **not allowed**
- Official Dataset V4 result changed: **no**
- Official locked final-test relations loaded: **no**

## Main result

| Metric | Result |
| --- | ---: |
| External images | **60** |
| Image-text relation rows | **360** |
| Accuracy | **0.6722** |
| Macro F1 | **0.6725** |

This secondary result is intentionally reported exactly as observed.
It is lower than the official Dataset V4 locked-test result
(0.9540 accuracy / 0.9541 macro F1), showing a substantial robustness gap
under independent-image and wording shift.

## Clean vs natural text

| Text mode | Rows | Accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| clean | 180 | **0.7444** | **0.7458** |
| natural | 180 | **0.6000** | **0.6001** |

The natural-text condition is harder for the frozen TF-IDF representation.
The audit records **60 rows with zero recognized TF-IDF features**, providing direct evidence that lexical
coverage is an important robustness limitation.

## Accuracy by true relation label

| True label | Rows | Accuracy |
| --- | ---: | ---: |
| `MATCH` | 120 | 0.7500 |
| `MISMATCH` | 120 | 0.6333 |
| `PARTIAL_MATCH` | 120 | 0.6333 |

## Accuracy by external image category

| Image category | Rows | Accuracy |
| --- | ---: | ---: |
| `alternator` | 60 | 0.6333 |
| `headlight` | 60 | 0.5667 |
| `ignition_coil` | 60 | 0.5833 |
| `spark_plug` | 60 | 0.8833 |
| `starter` | 60 | 0.9500 |
| `taillight` | 60 | 0.4167 |

The category spread is large. `starter` and `spark_plug` transfer much
better than `taillight`, `headlight`, and `ignition_coil` on this small
independent audit.

## Confusion matrix

Label order: `['MATCH', 'MISMATCH', 'PARTIAL_MATCH']`

```text
true MATCH          [90, 19, 11]
true MISMATCH       [32, 76, 12]
true PARTIAL_MATCH  [24, 20, 76]
```

## Reproducibility

- Predictions SHA-256: `b0d41bb071e0d5781b74e815f66a60eae392977949b173694e1427e60ebbace9`
- External lock SHA-256: `13ca9aca951060d6eac5bea6859da48e08da4de861f996a784782cc3d8519bbe`
- Evaluation-code Git HEAD: `4ae64053c5a7993320f9b46396ba5e6f71dec541`

Frozen result artifacts:

- `external_predictions.csv`
- `external_summary.json`

No retraining, threshold tuning, image replacement, text rewriting, or
second external evaluation is allowed after observing this result.
