# Methodology

## 1. Goal

The project asks one main question:

**Can a compact multimodal model learn a three-way relation between an automotive-part image and a short text description?**

The output has three classes: `MATCH`, `PARTIAL_MATCH`, and `MISMATCH`.

The image-only and text-only models are included as sanity checks. Because the relation label is defined by comparing the two inputs, neither single input contains enough information to solve the task by itself.

## 2. Data source and preparation

The images come from the public Kaggle dataset **50 Types of Car Parts - Image Classification**. I selected 640 images from eight categories and created new text descriptions for the image-text relation task.

For every image, I created six rows:

- two descriptions from the same category (`MATCH`);
- two descriptions from a related category (`PARTIAL_MATCH`);
- two descriptions from an unrelated category (`MISMATCH`).

This gives 2,880 training rows, 480 validation rows, and 480 final-test rows.

These 3,840 rows are built from 640 independent images. Six rows share each image, so the row count must not be interpreted as 3,840 different visual examples. The dataset is large enough for a student proof of concept and a controlled comparison of the models, but it is too small and too narrow for a production system.

The text side is deliberately controlled. It contains 64 exact template variants across the splits, and each description explicitly names one of the eight categories. The experiment therefore studies controlled image-text relation classification rather than open-ended language understanding.

## 3. Train, validation, and test split

The split is made by complete image groups:

- 480 train images;
- 80 validation images;
- 80 final-test images.

I checked for overlap using image IDs, group IDs, file paths, SHA-256 hashes, and exact description text. The same image cannot appear in two splits. This is important because otherwise the model could remember an image instead of learning the relation task.

The final-test data is stored separately under `data/locked_test/`.

## 4. Relation construction and single-input balance

The relation is deterministic once the image category and text category are known:

- equal categories -> `MATCH`;
- different categories in the same functional family -> `PARTIAL_MATCH`;
- categories in different families -> `MISMATCH`.

Each image appears exactly twice with every relation label. Exact text templates are also balanced across the three labels within each development split. This prevents a model from predicting the relation from one side alone. For this reason, image-only and text-only results near one third are expected and should be interpreted as sanity checks, not as a fair test of general image or text classification ability.

## 5. Models

I compare seven approaches, starting with simple baselines:

- majority baseline;
- TF-IDF + Logistic Regression;
- image pixels + Logistic Regression;
- image and text + Logistic Regression;
- text MLP;
- image CNN;
- multimodal CNN + text MLP.

Images are resized to 48 × 48 RGB. Text is converted to TF-IDF unigram and bigram features fitted only on the training descriptions.

The selected multimodal model has:

- a small CNN for the image;
- a small MLP for the text;
- a final layer that combines both representations and predicts the relation;
- two helper outputs that predict the image category and text category during training.

Its training loss is:

```text
relation loss
+ 0.40 × image-category loss
+ 0.40 × text-category loss
```

## 6. Training

The neural models use:

- Adam optimizer;
- learning rate `0.001`;
- weight decay `0.0001`;
- batch size `32`;
- maximum `80` epochs;
- early stopping with patience `10`.

The image training data uses a horizontal flip and a small brightness change. Model selection and early stopping use validation only. The saved history files contain training and validation loss and accuracy for the neural models, and the notebook displays the learning curves.

## 7. Evaluation

I report accuracy and macro F1. Macro F1 is useful because it gives equal importance to all three relation classes.

The main comparison is made on validation. The best model is selected from these validation results. I also use:

- a confusion matrix;
- accuracy by automotive-part category;
- concrete wrong predictions;
- training and validation learning curves;
- automated unit and integrity tests.

Because six rows share each image, the original experiment also stores grouped confidence intervals and paired comparisons by image group.

## 8. Validation-only category-rule diagnostic

The selected model has helper outputs for image category and text category. I use those validation predictions in a diagnostic decomposition:

1. predict the image category and text category;
2. apply the deterministic Dataset V3 relation rule;
3. compare that result with the learned relation head.

The validation results are:

- learned relation head: accuracy 0.7854, macro F1 0.7873;
- rule applied to predicted categories: accuracy 0.7292, macro F1 0.7324;
- image-category helper accuracy: 0.6625;
- text-category helper accuracy: 1.0000;
- rule applied to true categories: accuracy and macro F1 1.0000.

The predicted-category rule is not an independent model because it uses helper outputs from the selected model. It is a diagnostic of how the selected model works. The true-category result is only a construction check and cannot be used for a new image where the true category is unknown.

The learned relation head is about 0.0563 accuracy and 0.0549 macro F1 better than the predicted-category rule. This suggests that the joint representation adds useful information beyond making two hard category decisions.

## 9. Small additional experiment

After the main comparison, I made one extra validation-only experiment. I removed the two helper category outputs and trained the remaining relation model with seeds 43, 44, and 45.

The data, encoders, optimizer, batch size, augmentation, and early stopping stayed the same. Only the helper outputs and their losses were removed.

All three runs reached accuracy 0.3333 and macro F1 0.1667 and predicted only one class. This suggests that the helper tasks stabilized optimization for this small architecture and dataset. It does not prove that every multimodal model needs the same helper tasks.

## 10. Final test

The final test was used once after the model was selected. The saved result is:

- 354/480 correct predictions;
- accuracy 0.7375;
- macro F1 0.7382.

The current notebook reads the saved test predictions for tables, plots, and error analysis. It does not run the model again on the final-test images.

## 11. Limitations

- The 3,840 rows are based on only 640 independent images.
- The images come from one public collection.
- Only eight automotive-part categories are included, and the final test contains 80 images.
- The text descriptions are fixed templates and explicitly state the category.
- The relation labels are generated from manually defined category families.
- The model is compact, uses 48 × 48 images, and is trained from scratch.
- No pretrained vision backbone, resolution comparison, or visual attribution method is included.
- The additional experiment is limited to this architecture and training setup.
- Practical use would require more independently collected images, more categories, different brands and vehicle models, difficult user photos, freer language, and a separately collected external test set.

## 12. Sources

1. G. Piosenka, [50 Types of Car Parts - Image Classification](https://www.kaggle.com/datasets/gpiosenka/car-parts-40-classes), Kaggle.
2. T. Baltrušaitis, C. Ahuja, and L.-P. Morency, [Multimodal Machine Learning: A Survey and Taxonomy](https://arxiv.org/abs/1705.09406), IEEE TPAMI, 2019.
3. Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, [Gradient-Based Learning Applied to Document Recognition](https://bottou.org/papers/lecun-98h), Proceedings of the IEEE, 1998.
