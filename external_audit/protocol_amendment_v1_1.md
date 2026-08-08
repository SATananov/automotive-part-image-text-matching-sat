# External Robustness Audit V1.1 protocol amendment

## Status

This amendment is made **before any External Audit inference**.

- External predictions produced before amendment: **none**
- External score observed before amendment: **none**
- Model output used to choose the amended categories: **no**
- Official Dataset V4 result changed: **no**
- Official locked final-test relations accessed for this amendment: **no**

## Why V1 was amended

The original V1 protocol proposed 12 external image categories and 120 images.
Before any external inference, a source-feasibility review of the independent
Wikimedia Commons candidate pool showed that generic search retrieval was too
semantically noisy for several categories. Forcing ten weak or incorrectly
depicted images into each category would test dataset-label quality rather than
model robustness.

The protocol is therefore narrowed before evaluation instead of accepting
lower-quality external labels.

## V1.1 frozen scope

V1.1 uses six image categories in three functional-family pairs:

| Image category | Family | PARTIAL partner | Dedicated Commons category |
| --- | --- | --- | --- |
| alternator | electrical_starting_charging | starter | Category:Automobile alternators |
| starter | electrical_starting_charging | alternator | Category:Electric starter motors |
| headlight | lighting | taillight | Category:Automobile headlamps |
| taillight | lighting | headlight | Category:Automobile rear lights |
| spark_plug | ignition_and_fuel | ignition_coil | Category:Spark plugs |
| ignition_coil | ignition_and_fuel | spark_plug | Category:Ignition coils |

The one-to-one cross-family MISMATCH mapping is:

- alternator -> headlight
- starter -> taillight
- headlight -> spark_plug
- taillight -> ignition_coil
- spark_plug -> alternator
- ignition_coil -> starter

This means every V1.1 text category appears exactly once as a MATCH target,
once as a PARTIAL_MATCH target and once as a MISMATCH target at the category
mapping level.

## Size

- 6 categories
- 3 functional families
- 10 independent images per category
- 60 images total
- clean + natural text
- MATCH + PARTIAL_MATCH + MISMATCH
- 6 relation rows per image
- 360 relation rows total

## Selection and lock discipline

Images must come from the dedicated Wikimedia Commons categories above and
must have reusable license metadata. Each final image must retain its Commons
source page, author/creator, license, license URL and SHA-256.

Selection may use only visual category correctness and reasonable diversity.
Model predictions are forbidden during selection.

The selected 60-image dataset and 360-row relation table must be hashed,
locked, committed and pushed **before** the one-time external inference.

No post-audit tuning is allowed.
