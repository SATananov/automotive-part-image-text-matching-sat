# Automotive Part Image-Text Matching

This is my final exam project for classifying the relation between an automotive-part image and a short text description.

The model predicts one of three labels:

- `MATCH` - the image and text describe the same type of part;
- `PARTIAL_MATCH` - the parts are different but have a related function;
- `MISMATCH` - the image and text describe unrelated parts.

## Research question

**Does a model that uses both an image and text perform better than models that use only one of them?**

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

More information about the source and licence is available in [docs/provenance.md](docs/provenance.md).

## Models I compared

I started with simple baselines and then added small neural networks:

1. majority-class baseline;
2. TF-IDF + Logistic Regression for text;
3. image pixels + Logistic Regression;
4. image and text + Logistic Regression;
5. text MLP;
6. image CNN;
7. multimodal CNN + text MLP.

The multimodal model combines features from the image and the text. During training, it also learns two small helper tasks: predicting the category visible in the image and the category named in the text.

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

## Final result

| Evaluation | Correct predictions | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Validation | 377 / 480 | 0.7854 | 0.7873 |
| Final test | 354 / 480 | 0.7375 | 0.7382 |

The final test was used once after the model was selected. The repository keeps the saved predictions and metrics, but the normal evaluation command does not rerun the final test.

## Where the model makes mistakes

The weakest final-test categories are `headlight` with 29/60 correct predictions and `oil_filter` with 30/60. The `PARTIAL_MATCH` class is also difficult: 110/160 examples are correct, and 33 are predicted as `MISMATCH`.

Some mistakes are understandable. For example, a brake disc and a brake pad belong to the same braking system, while an alternator and a starter are both electrical engine-support parts. These examples make `PARTIAL_MATCH` harder than a clear exact match or a completely unrelated pair.

The notebook shows the confusion matrix, category results, and several concrete wrong predictions with their images and descriptions.

## Small additional experiment

As an extra check, I trained the same multimodal model without the two helper category tasks. I ran it three times with seeds 43, 44, and 45, using only train and validation data.

All three runs reached 160/480 correct predictions, accuracy 0.3333, and macro F1 0.1667. Each run predicted only one class. In this project, the helper tasks made the training much more successful.

This experiment is additional evidence, not a requirement of the exam, and it does not change the selected model or the final-test result.

## Project files

```text
automotive-part-image-text-matching-sat/
├── README.md
├── project.ipynb          # main report with tables, plots, and examples
├── docs/                  # methodology, data source, and test policy
├── data/                  # Dataset V3 and the separated final test
├── src/                   # data loading, models, training, and evaluation
├── models/                # selected model checkpoint
├── results/               # saved validation, extra experiment, and test results
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

Recalculate the saved validation result:

```powershell
python -m src.evaluate
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

- The images come from one public dataset.
- The dataset contains only eight selected categories.
- The network is small and trained from scratch.
- Several text rows use the same image, so the image groups must be considered when interpreting the results.
- The result of the additional experiment applies only to this model and this dataset.
- This is an educational project, not a production warehouse system.

## Sources

1. G. Piosenka, [50 Types of Car Parts - Image Classification](https://www.kaggle.com/datasets/gpiosenka/car-parts-40-classes), Kaggle dataset used as the image source.
2. T. Baltrušaitis, C. Ahuja, and L.-P. Morency, [Multimodal Machine Learning: A Survey and Taxonomy](https://arxiv.org/abs/1705.09406), 2017/2019.
3. Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, [Gradient-Based Learning Applied to Document Recognition](https://bottou.org/papers/lecun-98h), 1998.
