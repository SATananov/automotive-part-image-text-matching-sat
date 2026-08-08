# Dataset V4 provenance

## Source

Dataset V4 uses images from the public Kaggle dataset **50 Types of Car Parts - Image Classification** by G. Piosenka:

<https://www.kaggle.com/datasets/gpiosenka/car-parts-40-classes>

The source page records the dataset licence as Apache-2.0.

## Dataset-level licence record

The official Dataset V4 dataset-level licence record is stored in:

```text
data/licenses_dataset_v4.csv
```

The older `data/licenses.csv` record is retained only for Dataset V3 project
history and its preserved verifier. Per-image Dataset V4 provenance remains in
`data/manifests/dataset_v4/images.csv`.

## Source archive audit

The original downloaded archive is recorded by:

```text
evidence/dataset_v4/source_archive_audit.json
```

The source archive SHA-256 recorded during Dataset V4 construction is:

```text
ee269fb85db3c53630b68aa0e302197daceb81fa6c1f9e63140722dc0e9ebd6e
```

The archive contains a `car parts 50/` dataset and an older `car parts/` version. Dataset V4 uses only `car parts 50/`.

The initial source audit found:

- 9,239 images in the 50-category dataset;
- 50 categories;
- 224 × 224 image size;
- no unreadable images;
- 9,239 unique SHA-256 image hashes.

## Project manifest

Every Dataset V4 image is recorded in:

```text
data/manifests/dataset_v4/images.csv
```

The manifest includes:

- project image ID;
- image-group ID;
- part category and functional family;
- project split;
- project image path;
- SHA-256 hash;
- width, height and colour mode;
- source dataset/category;
- source archive path and original source split.

This allows every project image to be traced back to the downloaded source archive.

## Project split

I did not use the source train/validation/test folders as the final project split. Dataset V4 uses its own image-level split:

| Split | Images |
|---|---:|
| Train | 6,463 |
| Validation | 1,388 |
| Locked final test | 1,388 |

The same exact image hash does not occur in more than one project split.

## Relation data

Relation tables are stored in:

```text
data/relations/dataset_v4/train.csv
data/relations/dataset_v4/validation.csv
data/locked_test/dataset_v4/test_relations.csv
```

The relation-generation audit is stored in:

```text
data/manifests/dataset_v4/dataset_audit.json
```

The final balanced relation counts are 41,742 train rows, 9,030 validation rows, and 9,030 locked final-test rows.

## Near-duplicate diagnostic

SHA-256 equality detects byte-identical files, but visually almost identical images can have different hashes. After the final result was frozen, I therefore ran a diagnostic pHash/dHash screen across project splits.

It found 93 candidate cross-split pairs. These candidates are recorded in:

```text
results/dataset_v4/step04_sanity_near_duplicate_candidates.csv
```

This is documented as a limitation. The saved-result sensitivity analysis shows that excluding all 37 final-test images flagged by the widest screen changes accuracy only from 0.9540 to 0.9536.

## Dataset V3 history

The repository also contains the earlier Dataset V3 proof of concept with 640 images and 8 categories. Dataset V3 is kept for project history, but Dataset V4 is the official final experiment and report.
