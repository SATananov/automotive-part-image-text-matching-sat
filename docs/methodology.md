# Methodology - Dataset V4

## 1. Goal

The project asks one main question:

**Can a multimodal model classify the relation between an automotive-part image and a short text description better than models that see only one of the two inputs?**

The target labels are `MATCH`, `PARTIAL_MATCH`, and `MISMATCH`.

## 2. Image data

Dataset V4 uses 9,239 unique 224 × 224 RGB images from all 50 categories in the `car parts 50/` part of the public Kaggle source archive.

The image split is:

- 6,463 train images;
- 1,388 validation images;
- 1,388 locked final-test images.

The split is made at image level. SHA-256 checks show no exact image duplication across the three splits.

## 3. Relation construction

The 50 categories are grouped into 12 functional families. A relation is defined from the image category and the category named by the text:

- same category -> `MATCH`;
- different category in the same family -> `PARTIAL_MATCH`;
- category from a different family -> `MISMATCH`.

Before training, I regenerated the relation tables to remove simple one-sided shortcuts. The construction balances:

- each image across the three relation labels;
- each text category across the three labels;
- each exact text template across the three labels.

This produces:

- 41,742 train relation rows;
- 9,030 validation relation rows;
- 9,030 locked final-test relation rows.

The row counts are larger than the image counts because one image is paired with several text descriptions.

## 4. Text representation

Text is represented with TF-IDF unigram and bigram features. For the validation experiments, the vectorizer is fitted on training descriptions only and then applied to validation descriptions.

For the frozen final model, train and validation are combined as development data. A new vectorizer is fitted on development descriptions and only `transform()` is applied to final-test descriptions.

This prevents the final-test text from influencing the learned TF-IDF vocabulary or weights.

## 5. Image representation

I use ResNet18 with ImageNet pretrained weights. Its final classification layer is removed and the network is used as a fixed feature extractor.

Using a pretrained feature extractor is practical for a student project because the 9,239 images are enough for the relation experiment but still small compared with the datasets normally used to train a modern convolutional network from scratch.

## 6. Models compared on validation

I compare four small classifiers:

1. text-only;
2. image-only;
3. multimodal without auxiliary tasks;
4. multimodal with auxiliary image-category and text-category tasks.

The multimodal model combines the ResNet18 image feature vector and the TF-IDF text representation in a small neural network.

For the auxiliary version, two extra training heads predict the image and text categories. Their losses are added with weight 0.15. The final relation prediction still comes from the multimodal relation head.

## 7. Training and model selection

The validation run uses:

- Adam optimizer;
- learning rate 0.001;
- batch size 256;
- maximum 8 epochs;
- auxiliary-loss weight 0.15;
- fixed random seed recorded in the saved result;
- pretrained ResNet18 features cached locally for efficiency.

The selected model is the one with the highest validation macro F1. Validation accuracy is used only as a tie-breaker.

The results are:

| Model | Best epoch | Accuracy | Macro F1 |
|---|---:|---:|---:|
| text-only | 5 | 0.3333 | 0.3042 |
| image-only | 2 | 0.3333 | 0.3003 |
| multimodal without auxiliary tasks | 8 | 0.8622 | 0.8592 |
| multimodal with auxiliary tasks | 7 | 0.8975 | 0.8958 |

The selected model is `multimodal_auxiliary`, frozen at epoch 7.

## 8. Why the one-input baselines are near chance

The target is a relation between two inputs. The relation construction is balanced so that the image alone or the text alone does not determine whether the pair is a match.

With three balanced labels, chance accuracy is approximately one third. The text-only and image-only results near 0.333 therefore act as useful sanity checks.

## 9. Final-test protocol

The final-test evaluation code was committed before the locked test was opened.

After model selection:

1. train and validation were combined as development data;
2. the selected architecture was trained for the already frozen 7 epochs;
3. the locked final test was evaluated once;
4. predictions and a JSON summary were saved;
5. no model selection or tuning was allowed after this point.

The frozen result is:

- 8,615 / 9,030 correct relation predictions;
- accuracy 0.9540420819490587;
- macro F1 0.9540841368243022;
- 1,388 independent final-test images.

## 10. Additional sanity audit

Because final-test performance is higher than validation performance, I performed a diagnostic audit without new model inference.

The audit checks:

- exact image/hash/path/group overlap;
- exact description overlap;
- saved predictions against the locked relation table;
- model input signature;
- dummy auxiliary category targets at final-test inference;
- TF-IDF fitting scope;
- equal-weight image-level accuracy;
- perceptual near-duplicate candidates using 64-bit pHash and dHash;
- sensitivity of saved metrics after removing flagged test images.

Exact overlap is zero. Perceptual screening flags 93 cross-split candidate pairs. At the widest threshold, 37 final-test images are flagged. Removing all rows from those 37 images from the already saved predictions gives accuracy 0.953551912568306 and macro F1 0.953594492512139.

This is very close to the frozen 0.9540 / 0.9541 result, so the flagged near-duplicate candidates do not explain the high score. They are still documented as a limitation.

## 11. Error analysis

There are 415 wrong relation predictions. They are concentrated in 181 independent images, while 1,207 of 1,388 test images have all their relation rows correct.

The most common confusion is `PARTIAL_MATCH -> MATCH` (121 errors), followed by `MISMATCH -> MATCH` (100 errors). This suggests that the model sometimes treats different parts as more similar than the benchmark rule does.

The notebook reports the full confusion matrix, category-level results, error directions, and the most difficult categories.

## 12. Metrics

The two headline metrics are accuracy and macro F1.

Accuracy is:

```text
number of correct predictions / total number of predictions
```

Macro F1 is the arithmetic mean of the F1 score of the three classes. It gives each class equal importance.

## 13. Limitations

- Text explicitly names one of the 50 known part categories.
- The relation labels are generated from manually defined functional families.
- The images come from one public dataset.
- A random image-level split can still contain visually near-identical examples even when SHA-256 hashes differ.
- Several relation rows share the same image.
- The experiment does not test unknown categories or free-form customer language.
- The result is a benchmark result, not a production-readiness claim.
