# Dataset and result provenance

## Public image source

Dataset V3 is a manually curated subset of the public Kaggle dataset `gpiosenka/car-parts-40-classes`. The recorded upstream license is Apache-2.0. The original source split labels were ignored; this project created its own deterministic grouped train, validation, and final-test split.

The exact per-file source metadata and SHA-256 hashes are stored in `data/manifests/images.csv`. The preserved license registry is `data/licenses.csv`.

## Original result lineage

The Dataset V3 images, relation tables, selected checkpoint, validation result, and one-time locked final-test result originate from:

- repository: `SATananov/automotive-part-image-text-matching`;
- authoritative branch: `dataset-v3`;
- original evidence checkpoint: `76f62ef91faa1e544c9074a91e80d86ed71a99fa`;
- later reporting/verification state: `d9ead2a8ef6637a3e86713b98433fc1ca1389158`.

The compact active source layout was informed by:

- repository: `SATananov/automotive-part-multimodal-classification`;
- clean architecture source commit: `a9717986b2b73d089f29ce3a619bbe9e08494ef6`.

The second repository re-packaged the same Dataset V3 result lineage; it is not treated as an independent replication.

## Unified-project work

The following are explicitly later work in this repository:

- consolidation into one active source tree;
- one official README, methodology, and notebook;
- removal of Dataset V2 and duplicate entry points;
- portable hash-based verification;
- the relation-only auxiliary-loss ablation with seeds 43, 44, and 45;
- its checkpoints, histories, predictions, comparison CSV, and summary JSON.

No claim is made that the ablation was completed before the original final-test evaluation. It is a validation-only explanatory follow-up and does not change the frozen selected checkpoint or final-test score.
