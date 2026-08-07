# Dataset V4 validation experiments

I use the training and validation splits only during this step. The locked final test is not loaded.

The image representation comes from a pretrained ResNet18 with its final classification layer removed. The ResNet18 is used as a fixed feature extractor. This keeps the experiment practical on a normal computer and is a simple transfer-learning approach.

The text representation is TF-IDF fitted on the training descriptions only. Validation text is transformed with the already fitted vectorizer.

I compare four small classifiers:

1. text-only baseline;
2. image-only baseline;
3. multimodal classifier without auxiliary category tasks;
4. multimodal classifier with image-category and text-category auxiliary tasks.

The one-sided baselines are useful checks because the relation labels are balanced for both the image side and the text side. A useful multimodal model should perform better than those baselines.

The selected model is the model with the highest validation macro F1. Validation accuracy is used only as a tie-breaker. The final test remains locked until model selection is finished.

The full run writes only two small result files under `results/dataset_v4/`: a JSON summary and a CSV training history. Cached ResNet18 features are written under `.cache/`, which is already ignored by Git.
