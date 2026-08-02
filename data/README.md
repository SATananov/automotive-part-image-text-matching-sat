# Data layout

- `images/dataset_v3/train/`: 480 training images.
- `images/dataset_v3/validation/`: 80 validation images.
- `relations/train.csv`: 2,880 training relation rows.
- `relations/validation.csv`: 480 validation relation rows.
- `locked_test/dataset_v3/`: 80 final-test images and 480 locked relation rows.
- `manifests/images.csv`: canonical image paths, source fields, splits, and SHA-256 hashes.
- `licenses.csv`: recorded source license information.

The final-test directory is preserved for evidence integrity. Active training and evaluation code do not use it.
