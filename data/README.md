# Data layout

- `images/dataset_v3/train/`: 480 training images.
- `images/dataset_v3/validation/`: 80 validation images.
- `relations/train.csv`: 2,880 training relation rows.
- `relations/validation.csv`: 480 validation relation rows.
- `locked_test/dataset_v3/`: 80 final-test images and 480 locked relation rows.
- `manifests/images.csv`: canonical image paths, source fields, splits, and SHA-256 hashes.
- `licenses.csv`: one dataset-level licence record for the only image source used by Dataset V3.

All 640 selected image records can be traced through `manifests/images.csv`. The final-test directory is preserved for evidence integrity. Active training and validation evaluation code do not use it.
