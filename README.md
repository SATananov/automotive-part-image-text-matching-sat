# Automotive Part Image-Text Matching

This is my final exam project for classifying the relation between an automotive-part image and a short text description.

The model predicts one of three labels:

- `MATCH` - the image and text describe the same type of part;
- `PARTIAL_MATCH` - the parts are different but have a related function;
- `MISMATCH` - the image and text describe unrelated parts.

## Research question

**Can a compact multimodal model learn this three-way image-text relation, and what do simple single-input checks show about the need to compare both inputs?**

This is useful for checking product catalogues, uploaded listings, and warehouse descriptions where an image can be paired with the wrong text.

## Data

The project uses 640 images from eight automotive-part categories:

`alternator`, `brake_disc`, `brake_pad`, `coil_spring`, `headlight`, `oil_filter`, `starter`, and `taillight`.

| Split | Images | Image-text rows | Purpose |
|---|---:|---:|---|
| Train | 480 | 2,880 | model training |
| Validation | 80 | 480 | model comparison and selection |
| Final test | 80 | 480 | one final evaluation after model selection |

I created my own grouped split instead of using the original dataset split. I checked that the same image does not appear in more than one split by comparing image IDs, paths, groups, and SHA-256 file hashes. The final-test files are stored in a separate folder.

The text descriptions are short templates written for this controlled task. There are 64 exact text variants across the three splits, and each description explicitly names a part category. The project therefore tests controlled relation classification, not open-ended language understanding.

More information about the source and licence is available in [docs/provenance.md](docs/provenance.md).

### Are these data enough?

The tables contain 3,840 image-text rows, but the independent visual examples are the 640 images. I do not treat all rows as separate images because six rows share each image.

The data are enough for a student proof of concept and for comparing the tested models under the same split. They are not enough to claim that the system is ready for a real warehouse or online shop. A practical version would need more independently collected images, more part categories, different brands and vehicle models, difficult user photos, freer text, and a separate external test set.

## Models I compared

I started with simple checks and then added small neural networks:

1. majority-class baseline;
2. TF-IDF + Logistic Regression for text;
3. image pixels + Logistic Regression;
4. image and text + Logistic Regression;
5. text MLP;
6. image CNN;
7. multimodal CNN + text MLP.

The multimodal model combines features from the image and the text. During training, it also learns two small helper tasks: predicting the category visible in the image and the category named in the text.

### How to interpret the single-input results

Every image is paired with exactly two rows from each relation class. The text side is balanced in the same way. Therefore, an image-only or text-only model does not receive enough information to determine the relation label: the label depends on comparing the two inputs. Results near one third are expected and are used as sanity checks, not as proof that a particular unimodal architecture is weak.

## Validation results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| Majority baseline | 0.3333 | 0.1667 |
| TF-IDF + Logistic Regression | 0.3333 | 0.1667 |
| Image pixels + Logistic Regression | 0.3333 | 0.1667 |
| Image + text Logistic Regression | 0.3333 | 0.1667 |
| Text MLP | 0.3333 | 0.2923 |
| Image CNN | 0.3333 | 0.2666 |
| **Multimodal CNN + text MLP** | **0.7854** | **0.7873** |

The multimodal model was the clear winner on validation, so I selected it before looking at the final test result.

## Validation-only category-rule diagnostic

The relation labels are defined from the two part categories: equal categories mean `MATCH`, different categories in the same family mean `PARTIAL_MATCH`, and different families mean `MISMATCH`.

I used the selected model's two helper category outputs to check how much of the validation result can be reproduced by this rule:

| Validation diagnostic | Accuracy | Macro F1 |
|---|---:|---:|
| Learned relation head | 0.7854 | 0.7873 |
| Rule applied to predicted image/text categories | 0.7292 | 0.7324 |
| Rule applied to true categories | 1.0000 | 1.0000 |

The image-category helper reaches 0.6625 accuracy and the text-category helper reaches 1.0000. The predicted-category rule is a decomposition of the selected model, not an independent baseline. The true-category result is only a dataset-construction check; it is not a usable model because true categories are unavailable for a new image.

The learned relation head is about 0.0563 accuracy and 0.0549 macro F1 better than applying the rule to the model's predicted categories. This suggests that the combined representation adds useful information beyond a hard category decision, while the task is still strongly structured by the category families.

## Final result

| Evaluation | Correct predictions | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Validation | 377 / 480 | 0.7854 | 0.7873 |
| Final test | 354 / 480 | 0.7375 | 0.7382 |

The final test was used once after the model was selected. The repository keeps the saved predictions and metrics, but the normal evaluation command does not rerun the final test.

## Where the model makes mistakes

The weakest final-test categories are `headlight` with 29/60 correct predictions and `oil_filter` with 30/60. The `PARTIAL_MATCH` class is also difficult: 110/160 examples are correct, and 33 are predicted as `MISMATCH`.

Some mistakes are understandable. For example, a brake disc and a brake pad belong to the same braking system, while an alternator and a starter are both electrical engine-support parts. These examples make `PARTIAL_MATCH` harder than a clear exact match or a completely unrelated pair.

The notebook shows the training curves, model comparison, validation diagnostic, confusion matrix, category results, and several concrete wrong predictions with their images and descriptions.

## Small additional experiment

As an extra check, I trained the same multimodal model without the two helper category tasks. I ran it three times with seeds 43, 44, and 45, using only train and validation data.

All three runs reached 160/480 correct predictions, accuracy 0.3333, and macro F1 0.1667. Each run predicted only one class. In this project, the helper tasks made optimization much more successful.

This experiment is additional evidence, not a requirement of the exam. It does not prove that helper losses are always necessary, and it does not change the selected model or the final-test result.

## Project files

```text
automotive-part-image-text-matching-sat/
├── README.md
├── project.ipynb          # main report with tables, plots, and examples
├── docs/                  # methodology, data source, and test policy
├── data/                  # Dataset V3 and the separated final test
├── src/                   # data loading, models, training, and evaluation
├── models/                # selected model checkpoint
├── results/               # saved validation, diagnostic, experiment, and test results
├── evidence/              # hashes and source information
└── tests/                 # automated checks
```

## How to run

Create the environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the checks:

```powershell
python -m src.verify --full-hashes
python -m pytest -q
```

Recalculate the saved validation result and diagnostic:

```powershell
python -m src.evaluate
```

Regenerate only the saved validation diagnostic when intentionally updating it:

```powershell
python -m src.evaluate --write-diagnostic
```

Run a new development comparison in a different output folder:

```powershell
python -m src.train --output results/retrained
```

Run the additional experiment without helper tasks:

```powershell
python -m src.train_ablation --seed 44 --output results/retrained_no_helpers_seed44
```

## Limitations

- The 3,840 rows come from 640 independent images, so rows that share an image are not fully independent.
- The images come from one public dataset.
- The dataset contains only eight selected categories, with 80 images in the final test.
- The text is template-generated, explicitly names the category, and does not test free-form language understanding.
- The relation label is determined by category equality and three manually defined part families.
- The single-input checks are expected to be near chance because the label needs both inputs.
- The network is small, uses 48 × 48 images, and is trained from scratch rather than from a pretrained vision backbone.
- The result of the additional experiment applies only to this model and this dataset.
- This is an educational project, not a production warehouse system.

## Sources

1. G. Piosenka, [50 Types of Car Parts - Image Classification](https://www.kaggle.com/datasets/gpiosenka/car-parts-40-classes), Kaggle dataset used as the image source.
2. T. Baltrušaitis, C. Ahuja, and L.-P. Morency, [Multimodal Machine Learning: A Survey and Taxonomy](https://arxiv.org/abs/1705.09406), 2017/2019.
3. Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, [Gradient-Based Learning Applied to Document Recognition](https://bottou.org/papers/lecun-98h), 1998.
