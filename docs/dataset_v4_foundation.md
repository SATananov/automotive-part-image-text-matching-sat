# Dataset V4 foundation

Dataset V4 uses 9,239 unique 224 × 224 images from all 50 categories in the original Kaggle source. The independent image split is 6,463 train, 1,388 validation, and 1,388 locked final-test images.

## Relation balancing correction

Before any model training, the relation tables were regenerated to prevent either single input from carrying a relation-label shortcut:

- every image occurs equally often with `MATCH`, `PARTIAL_MATCH`, and `MISMATCH`;
- every text category occurs equally often with all three labels;
- each exact text template is equally represented across all three labels;
- `MATCH` uses the same category;
- `PARTIAL_MATCH` uses a different category from the same functional family;
- `MISMATCH` uses a category from a different family.

The correction changes only the relation construction and the locked-test hash. The 9,239 image files, image split, image hashes, and category-family mapping remain unchanged. It is performed before training and before final-test evaluation.

## Model foundation

The new multimodal model uses:

- a ResNet18 image encoder with ImageNet pretrained weights during real training;
- a TF-IDF text representation followed by a small MLP;
- a relation head for the three target labels;
- helper heads for the 50 image and text categories.

The backbone is frozen during the first training stage. A later training step may unfreeze the last ResNet block for controlled fine-tuning.
