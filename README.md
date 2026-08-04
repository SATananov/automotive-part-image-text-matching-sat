# Automotive Part Image-Text Matching

This is my final exam project for classifying the relation between an automotive-part image and a short text description.

My model predicts one of three labels:

* `MATCH` — the image and text describe the same type of part;
* `PARTIAL_MATCH` — the parts are different but have a related function;
* `MISMATCH` — the image and text describe unrelated parts.

## Research question

**Can a compact multimodal model learn this three-way image-text relation, and what do simple single-input checks show about the need to compare both inputs?**

I chose this problem because it can be useful for checking product catalogues, uploaded listings, and warehouse descriptions where an image may be paired with the wrong text.

## Data

I use 640 images from eight automotive-part categories:

`alternator`, `brake_disc`, `brake_pad`, `coil_spring`, `headlight`, `oil_filter`, `starter`, and `taillight`.

| Split      | Images | Image-text rows | Purpose                                    |
| ---------- | -----: | --------------: | ------------------------------------------ |
| Train      |    480 |           2,880 | model training                             |
| Validation |     80 |             480 | model comparison and selection             |
| Final test |     80 |             480 | one final evaluation after model selection |

I created my own grouped split instead of using the original dataset split. I checked that the same image does not appear in more than one split by comparing image IDs, paths, groups, and SHA-256 file hashes. I keep the final-test files in a separate folder.

I wrote short text templates for this controlled task. There are 64 exact text variants across the three splits, and every description explicitly names a part category. For this reason, my project tests controlled relation classification rather than open-ended language understanding.

More information about the data source and licence is available in [docs/provenance.md](docs/provenance.md).

### Are these data enough?

The tables contain 3,840 image-text rows, but the independent visual examples are the 640 images. I do not treat all rows as separate images because six rows share each image.

I consider the data sufficient for a student proof of concept and for comparing the tested models under the same split. However, they are not sufficient to claim that the system is ready for use in a real warehouse or online shop.

A practical version would need more independently collected images, more part categories, different brands and vehicle models, difficult user photos, freer text, and a separate external test set.

## Models I compared

I started with simple checks and then gradually added small neural networks:

1. majority-class baseline;
2. TF-IDF + Logistic Regression for text;
3. image pixels + Logistic Regression;
4. image and text + Logistic Regression;
5. text MLP;
6. image CNN;
7. multimodal CNN + text MLP.

My multimodal model combines features extracted from the image and the text.

During training, I also use two small helper tasks:

* predicting the category visible in the image;
* predicting the category named in the text.

These helper tasks are used only during training to support the main relation-classification task.

### How I interpret the single-input results

Every image is paired with exactly two rows from each relation class. The text side is balanced in the same way.

Therefore, an image-only or text-only model does not receive enough information to determine the relation label. The label depends on comparing both inputs.

For this reason, results near one third are expected from the single-input models. I use these results as sanity checks rather than as proof that a particular unimodal architecture is weak.

## Validation results

| Model                              |   Accuracy |   Macro F1 |
| ---------------------------------- | ---------: | ---------: |
| Majority baseline                  |     0.3333 |     0.1667 |
| TF-IDF + Logistic Regression       |     0.3333 |     0.1667 |
| Image pixels + Logistic Regression |     0.3333 |     0.1667 |
| Image + text Logistic Regression   |     0.3333 |     0.1667 |
| Text MLP                           |     0.3333 |     0.2923 |
| Image CNN                          |     0.3333 |     0.2666 |
| **Multimodal CNN + text MLP**      | **0.7854** | **0.7873** |

The multimodal model was the clear winner on validation. I therefore selected it before looking at the final-test result.

## Validation-only category-rule diagnostic

I define the relation labels from the two part categories:

* equal categories mean `MATCH`;
* different categories from the same functional family mean `PARTIAL_MATCH`;
* categories from different families mean `MISMATCH`.

I used the two helper category outputs of my selected model to check how much of the validation result can be reproduced by applying this rule to the predicted categories.

| Validation diagnostic                           | Accuracy | Macro F1 |
| ----------------------------------------------- | -------: | -------: |
| Learned relation head                           |   0.7854 |   0.7873 |
| Rule applied to predicted image/text categories |   0.7292 |   0.7324 |
| Rule applied to true categories                 |   1.0000 |   1.0000 |

The image-category helper reaches an accuracy of 0.6625, while the text-category helper reaches 1.0000.

I treat the predicted-category rule as a decomposition of my selected model, not as an independent baseline.

The true-category result is only a check of the dataset construction. It is not a usable model because the true image category is unavailable for a new input image.

The learned relation head is approximately 0.0563 accuracy and 0.0549 macro F1 better than applying the rule to the model's predicted categories.

This suggests that the combined representation adds useful information beyond a hard category decision, although the task remains strongly structured by the predefined category families.

## Final result

| Evaluation | Correct predictions | Accuracy | Macro F1 |
| ---------- | ------------------: | -------: | -------: |
| Validation |           377 / 480 |   0.7854 |   0.7873 |
| Final test |           354 / 480 |   0.7375 |   0.7382 |

I used the final test once, after selecting the model from the validation results.

I keep the saved final-test predictions and metrics in the repository, but the normal evaluation command does not rerun final-test inference.

## Where my model makes mistakes

The weakest final-test categories are `headlight`, with 29 out of 60 correct predictions, and `oil_filter`, with 30 out of 60 correct predictions.

The `PARTIAL_MATCH` class is also difficult. My model correctly classifies 110 out of 160 examples, while 33 examples are incorrectly predicted as `MISMATCH`.

Some mistakes are understandable because related automotive parts may have similar functions.

For example:

* a brake disc and a brake pad belong to the same braking system;
* an alternator and a starter are both electrical engine-support parts.

These examples make `PARTIAL_MATCH` more difficult than a clear exact match or a completely unrelated image-text pair.

In the notebook, I show:

* training curves;
* model comparisons;
* the validation diagnostic;
* the confusion matrix;
* category-level results;
* several concrete wrong predictions with their images and descriptions.

## Small additional experiment

As an additional check, I trained the same multimodal model without the two helper category tasks.

I ran this experiment three times with seeds 43, 44, and 45, using only the train and validation data.

All three runs reached:

* 160 out of 480 correct predictions;
* accuracy of 0.3333;
* macro F1 of 0.1667.

Each run predicted only one relation class.

In my project, the helper tasks made the model optimization much more successful.

I treat this as additional evidence rather than as a requirement of the exam. The experiment does not prove that helper losses are always necessary, and it does not change my selected model or the saved final-test result.

## Development and learning process

I started experimenting with this project near the beginning of the deep learning course and developed it gradually as I learned new concepts.

I first tested simple baseline models. I then added text, image, and multimodal neural networks. During the course, I also improved the data split, validation analysis, error analysis, experiment tracking, and reproducibility checks.

I consulted publicly available documentation, educational tutorials, research papers, and examples of similar image-text and multimodal classification problems.

These materials helped me understand the individual components used in my project, including:

* convolutional neural networks;
* text feature extraction with TF-IDF;
* Logistic Regression baselines;
* multimodal feature combination;
* neural-network training loops;
* classification metrics;
* random seeds and reproducible experiments.

I prepared the dataset split, relation-label construction, model implementation, experiments, saved results, and analysis specifically for this project.

The additional automated checks were added gradually to help me keep the data, model checkpoint, notebook, predictions, and reported results consistent.

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

Run the project checks:

```powershell
python -m src.verify --full-hashes
python -m pytest -q
```

Recalculate the saved validation result and validation diagnostic:

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

Run the additional experiment without the helper tasks:

```powershell
python -m src.train_ablation --seed 44 --output results/retrained_no_helpers_seed44
```

## Limitations

* The 3,840 rows come from 640 independent images, so rows sharing the same image are not fully independent.
* The images come from one public dataset.
* I selected only eight automotive-part categories.
* The final test contains 80 independent images.
* The text is template-generated and explicitly names the category.
* My project does not test free-form language understanding.
* The relation label is determined by category equality and three manually defined functional families.
* I expect the single-input checks to remain near chance because the relation label requires both inputs.
* My neural network is small and uses 48 × 48 images.
* I train the vision component from scratch rather than using a pretrained vision backbone.
* The result of the additional experiment applies only to this model and this dataset.
* This is an educational student project, not a production warehouse system.

## Sources

1. G. Piosenka, [50 Types of Car Parts - Image Classification](https://www.kaggle.com/datasets/gpiosenka/car-parts-40-classes), Kaggle dataset used as the image source.

2. T. Baltrušaitis, C. Ahuja, and L.-P. Morency, [Multimodal Machine Learning: A Survey and Taxonomy](https://arxiv.org/abs/1705.09406), IEEE Transactions on Pattern Analysis and Machine Intelligence, 2019.

3. Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, [Gradient-Based Learning Applied to Document Recognition](https://bottou.org/papers/lecun-98h), Proceedings of the IEEE, 1998.

4. PyTorch, [Deep Learning with PyTorch: A 60 Minute Blitz](https://docs.pytorch.org/tutorials/beginner/blitz/index.html), official beginner tutorials covering tensors, neural networks, training loops, and image classification.

5. PyTorch, [Reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html), official guidance on random seeds and reproducible experiments.

6. Scikit-learn, [Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html), official documentation used to understand TF-IDF text features.

7. Scikit-learn, [Linear Models](https://scikit-learn.org/stable/modules/linear_model.html), official documentation for the Logistic Regression baseline models.

8. Scikit-learn, [Metrics and Scoring](https://scikit-learn.org/stable/modules/model_evaluation.html), official documentation for classification accuracy, F1 score, and confusion-matrix-based evaluation.
