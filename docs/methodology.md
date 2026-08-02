# Methodology

## 1. Goal

The project asks one main question:

**Does combining an automotive-part image with a text description work better than using only the image or only the text?**

The output has three classes: `MATCH`, `PARTIAL_MATCH`, and `MISMATCH`.

## 2. Data source and preparation

The images come from the public Kaggle dataset **50 Types of Car Parts - Image Classification**. I selected 640 images from eight categories and created new text descriptions for the image-text relation task.

For every image, I created six rows:

- two descriptions from the same category (`MATCH`);
- two descriptions from a related category (`PARTIAL_MATCH`);
- two descriptions from an unrelated category (`MISMATCH`).

This gives 2,880 training rows, 480 validation rows, and 480 final-test rows.

## 3. Train, validation, and test split

The split is made by complete image groups:

- 480 train images;
- 80 validation images;
- 80 final-test images.

I checked for overlap using image IDs, group IDs, file paths, SHA-256 hashes, and exact description text. The same image cannot appear in two splits. This is important because otherwise the model could remember an image instead of learning the relation task.

The final-test data is stored separately under `data/locked_test/`.

## 4. Models

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

## 5. Training

The neural models use:

- Adam optimizer;
- learning rate `0.001`;
- weight decay `0.0001`;
- batch size `32`;
- maximum `80` epochs;
- early stopping with patience `10`.

The image training data uses a horizontal flip and a small brightness change. Model selection and early stopping use validation only.

## 6. Evaluation

I report accuracy and macro F1. Macro F1 is useful because it gives equal importance to all three relation classes.

The main comparison is made on validation. The best model is selected from these validation results. I also use:

- a confusion matrix;
- accuracy by automotive-part category;
- concrete wrong predictions;
- automated unit and integrity tests.

Because six rows share each image, the original experiment also stores grouped confidence intervals and paired comparisons by image group.

## 7. Small additional experiment

After the main comparison, I made one extra validation-only experiment. I removed the two helper category outputs and trained the remaining relation model with seeds 43, 44, and 45.

The data, encoders, optimizer, batch size, augmentation, and early stopping stayed the same. Only the helper outputs and their losses were removed.

All three runs reached accuracy 0.3333 and macro F1 0.1667 and predicted only one class. This suggests that the helper tasks were important for this small model. It does not prove that every multimodal model needs the same helper tasks.

## 8. Final test

The final test was used once after the model was selected. The saved result is:

- 354/480 correct predictions;
- accuracy 0.7375;
- macro F1 0.7382.

The current notebook reads the saved test predictions for tables, plots, and error analysis. It does not run the model again on the final-test images.

## 9. Limitations

- The images come from one public collection.
- Only eight automotive-part categories are included.
- The model is compact and trained from scratch.
- Text descriptions are created using a fixed relation-generation method.
- The additional experiment is limited to this architecture and training setup.
- A separately collected external dataset would be needed before practical use.

## 10. Sources

1. G. Piosenka, [50 Types of Car Parts - Image Classification](https://www.kaggle.com/datasets/gpiosenka/car-parts-40-classes), Kaggle.
2. T. Baltrušaitis, C. Ahuja, and L.-P. Morency, [Multimodal Machine Learning: A Survey and Taxonomy](https://arxiv.org/abs/1705.09406), IEEE TPAMI, 2019.
3. Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, [Gradient-Based Learning Applied to Document Recognition](https://bottou.org/papers/lecun-98h), Proceedings of the IEEE, 1998.
