# Automotive Part Image–Text Matching SAT

A compact multimodal Deep Learning project for classifying whether an automotive-part image and a text description are a **MATCH**, **MISMATCH**, or **PARTIAL_MATCH**.

This repository unifies the strongest verified parts of two earlier project lines into one clean, standalone implementation. Dataset V3, its split provenance, the selected checkpoint, validation evidence, and the one-time locked final-test result are preserved from the original experiment. The simplified code structure and the controlled auxiliary-loss ablation are presented as later consolidation work. The origin of old results is not hidden or rewritten.

## Research question and hypotheses

Can a compact image-and-text neural model classify automotive-part relations better than unimodal and simple combined baselines?

- **H1:** the complete multimodal training design will outperform the tested image-only, text-only, and simple combined baselines.
- **H2:** `PARTIAL_MATCH` will be the most difficult relation class.
- **H3:** performance will vary across automotive-part categories.
- **H4, controlled follow-up:** removing the auxiliary image-category and text-category objectives will reduce validation performance under the same training setup.

H1–H3 describe the original Dataset V3 study. H4 is a retrospective, validation-only follow-up. It was not used to replace the selected model and it did not access the final test.

## Dataset V3

The project contains 640 unique automotive-part images from eight categories:

`alternator`, `brake_disc`, `brake_pad`, `coil_spring`, `headlight`, `oil_filter`, `starter`, and `taillight`.

| Split | Images | Relation rows | Use |
|---|---:|---:|---|
| Train | 480 | 2,880 | training |
| Validation | 80 | 480 | comparison, selection, and ablation |
| Locked final test | 80 | 480 | one authorized historical evaluation only |

Each image has two rows for each relation label. The grouped split is disjoint by image ID, group ID, path, SHA-256 image hash, and exact description text.

Dataset and license details are documented in [docs/provenance.md](docs/provenance.md).

## Models

Seven development models were compared:

1. majority baseline;
2. TF-IDF + Logistic Regression;
3. image pixels + Logistic Regression;
4. image + text Logistic Regression;
5. neural text MLP;
6. image CNN;
7. multimodal CNN + text MLP.

The selected model combines a CNN image encoder, a TF-IDF text MLP, and a relation head. During training it also predicts the image category and the category described by the text:

```text
relation loss
+ 0.40 × image-category loss
+ 0.40 × text-category loss
```

| Neural model | Trainable parameters |
|---|---:|
| Text MLP | 24,131 |
| Image CNN | 21,459 |
| Multimodal with auxiliary objectives | 58,579 |
| Multimodal without auxiliary objectives | 57,923 |

## Canonical results

The selected model is `torch_multimodal_dataset_v3`, initialized with seed `44`.

| Evaluation | Correct | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Validation | 377/480 | 0.7854167 | 0.7872843 |
| Locked final test | 354/480 | 0.7375000 | 0.7382300 |

Model selection used validation only. The final-test result comes from one authorized evaluation of the already frozen checkpoint. The authorization is consumed; the saved final-test artifacts are retained for reporting and integrity verification, not for tuning or another model choice.

## Controlled auxiliary-loss ablation

The relation-only variant keeps the same image encoder, text encoder, relation head, data splits, optimizer, learning rate, weight decay, batch size, augmentation, and early-stopping rule. Only the two category heads and their losses are removed.

| Run | Seed | Auxiliary losses | Validation accuracy | Macro F1 | Predicted classes |
|---|---:|:---:|---:|---:|---:|
| Selected model | 44 | Yes | 0.7854 | 0.7873 | 3 |
| Relation-only | 43 | No | 0.3333 | 0.1667 | 1 (`MATCH`) |
| Relation-only | 44 | No | 0.3333 | 0.1667 | 1 (`MISMATCH`) |
| Relation-only | 45 | No | 0.3333 | 0.1667 | 1 (`MISMATCH`) |

All three relation-only runs collapsed to one predicted class. The mean accuracy gap is `0.4521` and the mean macro-F1 gap is `0.6206`. Under this fixed architecture and protocol, auxiliary category supervision is a material part of the successful model. This result does not prove that every multimodal architecture requires auxiliary supervision.

For seed 44, all 18 tensors shared by the with-auxiliary and no-auxiliary architectures were independently confirmed to have identical initial values.

## Repository structure

```text
automotive-part-image-text-matching-sat/
├── README.md
├── project.ipynb                 # the only official notebook
├── docs/                         # methodology, provenance, test policy
├── data/
│   ├── images/dataset_v3/        # train and validation images
│   ├── locked_test/dataset_v3/   # physically separated final test
│   ├── manifests/
│   └── relations/
├── models/                       # one selected canonical checkpoint
├── results/
│   ├── validation/
│   ├── ablation/
│   ├── final_test/
│   └── training/
├── evidence/                     # lineage, hashes, original lock records
├── src/                          # one active implementation
└── tests/                        # compact integrity and reproducibility suite
```

Dataset V2, competing notebooks, obsolete builders, duplicated checkpoints, patch files, caches, and the old Git history are intentionally excluded.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Verify the preserved project

```powershell
python -m src.verify --full-hashes
python -m pytest -q
```

Reproduce the selected model on validation only:

```powershell
python -m src.evaluate
```

Run a fresh development comparison in a new output directory:

```powershell
python -m src.train --output results/retrained
```

Run one relation-only ablation in a new directory:

```powershell
python -m src.train_ablation --seed 44 --output results/retrained_ablation_seed44
```

The active evaluation command intentionally has no final-test option. The final-test evaluator must not be rerun.

## Limitations

- All images originate from one public source collection, although the grouped split prevents exact image leakage.
- The model is small and trained from scratch rather than based on a large pretrained vision-language model.
- Six relation rows share every image, so rows are not independent; grouped uncertainty methods are used.
- `PARTIAL_MATCH` remains difficult and category performance is uneven.
- The auxiliary-loss conclusion is limited to this architecture, dataset, and training protocol.
- The system is an educational experiment and is not ready for automatic warehouse decisions.

## Result lineage

The machine-readable lineage is in [evidence/lineage.json](evidence/lineage.json). Original Dataset V3 results remain attributed to their source repository and commits. The three-seed auxiliary-loss ablation is identified separately as new unified-project work.
