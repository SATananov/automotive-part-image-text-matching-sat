# Data layout

The official final experiment is Dataset V4.

```text
data/
├── images/dataset_v4/
│   ├── train/                 # 6,463 images
│   ├── validation/            # 1,388 images
│   └── test/                  # 1,388 locked-test images
├── manifests/dataset_v4/
│   ├── images.csv             # image paths, categories, splits and SHA-256 hashes
│   ├── category_families.csv  # 50 categories mapped to 12 functional families
│   └── dataset_audit.json     # split and relation-construction checks
├── relations/dataset_v4/
│   ├── train.csv              # 41,742 relation rows
│   └── validation.csv         # 9,030 relation rows
└── locked_test/dataset_v4/
    ├── test_relations.csv     # 9,030 frozen relation rows
    └── test_lock.json         # locked-test metadata and SHA-256
```

The source audit is stored in `evidence/dataset_v4/source_archive_audit.json`.

Dataset V4 uses 9,239 unique source images from 50 categories. Exact image-hash overlap and exact description overlap across train, validation, and test are zero.

The repository also retains Dataset V3 files from the earlier 8-category proof of concept. They are historical material; the official final report uses Dataset V4.

## Licence records

- `licenses_dataset_v4.csv` is the dataset-level licence record for the official
  Dataset V4 submission.
- `licenses.csv` is retained for the earlier Dataset V3 project-history verifier.
- Per-image Dataset V4 source paths and SHA-256 values are recorded in
  `manifests/dataset_v4/images.csv`.
