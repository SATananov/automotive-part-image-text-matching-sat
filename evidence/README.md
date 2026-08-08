# Evidence hash manifests

Two hash manifests are intentionally kept because this repository preserves the
earlier Dataset V3 proof of concept as project history.

- `hashes.sha256` is the compatibility manifest used by the preserved Dataset V3
  verifier. Its listed paths are refreshed after documentation/report polishing
  so `python -m src.verify` remains valid.
- `hashes_dataset_v4_submission.sha256` is the explicit final Dataset V4
  submission inventory. It covers current repository artifacts while excluding
  the large `data/images/` trees.

Dataset V4 image bytes are not duplicated in the submission hash inventory
because every one of the 9,239 official Dataset V4 images already has a SHA-256
value in `data/manifests/dataset_v4/images.csv`. The original source archive is
independently recorded in `evidence/dataset_v4/source_archive_audit.json`.
