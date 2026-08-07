# Automotive Part Image-Text Matching

This is my final exam project. My goal is to check whether a model can understand the relation between an automotive-part image and a short text description.

The model predicts one of three labels:

- `MATCH` - the image and text describe the same part category;
- `PARTIAL_MATCH` - they describe different categories from the same functional family;
- `MISMATCH` - they describe categories from different functional families.

## Research question

**Can a model that uses both the image and the text classify this relation better than a model that uses only one of them?**

I use a prepared benchmark for this experiment. This is an educational project and not a production system.

## Research context and baseline rationale

The official experiment uses deliberately simple components so that the main
comparison stays controlled: text-only, image-only, multimodal fusion, and
multimodal fusion with auxiliary category tasks.

This differs from modern large vision-language systems. VisualBERT uses
Transformer-based joint vision-language representations, while CLIP learns
image-text alignment through large-scale contrastive pretraining. I do not use
either as an official baseline here. The goal of this course project is to test
whether combining the two modalities helps on a fixed prepared benchmark, not
to claim a state-of-the-art vision-language architecture.

For that reason, the 95.4% final-test result is evidence only for this Dataset
V4 experiment. It is not a comparison against CLIP or another large pretrained
vision-language model.

A fuller literature position and the rationale for the selected baselines are
in [docs/research_context.md](docs/research_context.md).

## Dataset V4

For the final version I used all **9,239 unique images from 50 automotive-part categories** in the `car parts 50/` folder of the source archive. I grouped the 50 categories into 12 functional families so I could create `PARTIAL_MATCH` examples.

| Split | Independent images | Image-text relation rows | Purpose |
|---|---:|---:|---|
| Train | 6,463 | 41,742 | model training |
| Validation | 1,388 | 9,030 | model comparison and selection |
| Locked final test | 1,388 | 9,030 | one final evaluation |
| **Total images** | **9,239** |  |  |

I created the image split before creating the image-text pairs. I also balanced the three relation labels so that a model using only the image or only the text should not have an easy shortcut.

The source check found 9,239 unique SHA-256 image hashes. Exact image overlap and exact description overlap between train, validation, and test are both zero.

More details are in [docs/provenance.md](docs/provenance.md) and [docs/methodology.md](docs/methodology.md).

## Model

For the image part I used **ResNet18 with ImageNet pretrained weights** as a fixed feature extractor. I did not train ResNet18 from scratch. For the text part I used **TF-IDF unigrams and bigrams** followed by a small neural network.

I compared four simple models:

1. text-only baseline;
2. image-only baseline;
3. multimodal model without auxiliary category tasks;
4. multimodal model with auxiliary image-category and text-category tasks.

The auxiliary tasks are used only during training. During final prediction, the relation model receives only the image representation and the text representation.

## Validation results

I used validation macro F1 to compare the models. Validation accuracy was used only as a tie-breaker.

| Model | Best epoch | Validation accuracy | Validation macro F1 |
|---|---:|---:|---:|
| Text only | 5 | 0.3333 | 0.3042 |
| Image only | 2 | 0.3333 | 0.3003 |
| Multimodal, no auxiliary tasks | 8 | 0.8622 | 0.8592 |
| **Multimodal + auxiliary tasks** | **7** | **0.8975** | **0.8958** |

The text-only and image-only results are close to the three-class chance level. This makes sense because the answer depends on the relation between the two inputs.

I selected `multimodal_auxiliary` at epoch 7 before opening the locked final test.

## Locked final-test result

Before running the final test, I fixed the selected model, epoch, and evaluation code. I then trained the selected model on train + validation for the already chosen 7 epochs and evaluated the locked final test once.

| Metric | Result |
|---|---:|
| Correct relation predictions | **8,615 / 9,030** |
| Accuracy | **0.9540** |
| Macro F1 | **0.9541** |
| Independent final-test images | **1,388** |
| Post-test tuning | **Not allowed** |

The final-test score is higher than the validation score. Because 95.4% looked unusually high to me, I did extra checks instead of accepting the number without checking it.

## Extra checks for the high final score

These checks do **not** train a new model and do **not** run the final-test model again. I used the already saved predictions.

I checked the following:

- exact image/hash overlap between the splits is **0**;
- exact description overlap is **0**;
- TF-IDF is fitted only on development data and the final-test text is only transformed;
- the final relation model receives only `image` and `text` inputs;
- the saved predictions match the locked test sample IDs and labels;
- 93 pairs of visually similar cross-split images were flagged for review;
- the widest check flags 37 of the 1,388 final-test images;
- after removing those 37 images from the **saved predictions**, accuracy is still **0.9536** and macro F1 is **0.9536**;
- when every test image has equal weight, accuracy is **0.9556**.

These visually similar images do not explain the 95.4% result, but they are still a limitation of using one public image collection and a random image-level split.

## Error analysis

The final test has **415 wrong relation predictions**. These errors come from **181 of the 1,388 independent images**. The other **1,207 images (86.96%)** have all their related pairs classified correctly.

The most common errors are:

| True label | Predicted label | Errors |
|---|---|---:|
| `PARTIAL_MATCH` | `MATCH` | 121 |
| `MISMATCH` | `MATCH` | 100 |
| `MISMATCH` | `PARTIAL_MATCH` | 79 |
| `MATCH` | `PARTIAL_MATCH` | 49 |
| `PARTIAL_MATCH` | `MISMATCH` | 39 |
| `MATCH` | `MISMATCH` | 27 |

This shows that the model sometimes thinks two parts are more similar than the benchmark label says they are. Some of the harder categories are `gas_cap`, `leaf_spring`, `transmission`, `fuel_injector`, and `alternator`.

The notebook shows the confusion matrix, class results, difficult categories, error directions, and the check for visually similar images.

## Project structure

```text
automotive-part-image-text-matching-sat/
├── README.md
├── project.ipynb                       # main Dataset V4 report
├── docs/
│   ├── methodology.md
│   ├── research_context.md
│   ├── provenance.md
│   ├── test_policy.md
│   ├── dataset_v4_foundation.md
│   ├── dataset_v4_training.md
│   ├── dataset_v4_final_test.md
│   └── dataset_v4_sanity_audit.md
├── external_audit/                      # independent robustness protocol
├── data/
│   ├── images/dataset_v4/
│   ├── manifests/dataset_v4/
│   ├── relations/dataset_v4/
│   └── locked_test/dataset_v4/
├── src/                                # data, model and training helpers
├── tools/                              # dataset, training, final-test and audit scripts
├── results/dataset_v4/                 # saved V4 results
├── evidence/dataset_v4/                # source archive check
└── tests/                              # automated checks
```

The repository also keeps the older Dataset V3 files so the earlier version of the project is not lost. Dataset V4 is the final version used in `project.ipynb` and in the main results above.

## How to run the checks

Create and activate the environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the main checks:

```powershell
python -m src.check_dataset_v4
python -m pytest -q
```

Check the saved final-test policy **without running final inference**:

```powershell
python -m tools.run_dataset_v4_final_test --check-only
```

Validate the separate external-audit protocol **without external inference**:

```powershell
python -m external_audit.prepare --check-plan
python -m external_audit.verify
```

Re-run the extra data/result checks from the already saved predictions:

```powershell
python -m tools.audit_dataset_v4_final_result
```

Open the report:

```powershell
jupyter notebook project.ipynb
```

I intentionally do not include the `--confirm-final-test` command in the normal workflow because the official final-test run has already been completed and frozen.

## External robustness audit

I also keep a **separate secondary robustness protocol** in
[`external_audit/`](external_audit/README.md). It is designed to evaluate the
already frozen development-only deployment bundle on 120 independent
photographs from 12 categories and on both clean and more natural text.

This audit does not alter Dataset V4, does not read the official locked
final-test relations, and does not allow training or post-audit tuning. The
protocol is committed before any external inference. At the current scaffold
stage, no external score is reported yet.

## Limitations

- The text descriptions clearly name a part category, so this is a structured experiment.
- I create the relation labels from category equality and 12 manually defined functional families.
- All images come from one public source collection.
- The extra image check found a small number of visually similar cross-split candidates even though exact hash overlap is zero.
- One image can create several relation rows, so 9,030 test rows do not mean 9,030 independent photographs.
- I use ResNet18 as a fixed pretrained feature extractor and do not compare larger vision models.
- I use TF-IDF for text instead of a modern language model.
- I do not compare the official model with a pretrained vision-language system such as CLIP.
- The project does not test unknown part categories, free-form customer text, very difficult customer photos, several parts in one image, or a completely separate external test set.
- The 95.4% result is evidence for this experiment and is **not evidence that the system is production-ready**.

## Earlier V3 version

My first version used 640 images from 8 categories and reached 0.7375 accuracy / 0.7382 macro F1 on its final test. In Dataset V4 I increased the data to all 50 source categories, used transfer learning, balanced the relation data, used a larger locked test, and added extra checks for leakage and visually similar images.

I keep the V3 result only as project history. Dataset V4 is the final result for the submission.

## Sources

1. G. Piosenka, **50 Types of Car Parts - Image Classification**, Kaggle: <https://www.kaggle.com/datasets/gpiosenka/car-parts-40-classes>
2. K. He, X. Zhang, S. Ren, J. Sun, **Deep Residual Learning for Image Recognition**, CVPR 2016: <https://arxiv.org/abs/1512.03385>
3. T. Baltrušaitis, C. Ahuja, L.-P. Morency, **Multimodal Machine Learning: A Survey and Taxonomy**, IEEE TPAMI 2019: <https://arxiv.org/abs/1705.09406>
4. PyTorch / Torchvision, **ResNet18 documentation**: <https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html>
5. scikit-learn, **TfidfVectorizer documentation**: <https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html>

6. L. H. Li, M. Yatskar, D. Yin, C.-J. Hsieh, K.-W. Chang, **VisualBERT: A Simple and Performant Baseline for Vision and Language**, 2019: <https://arxiv.org/abs/1908.03557>
7. A. Radford et al., **Learning Transferable Visual Models From Natural Language Supervision**, ICML 2021: <https://proceedings.mlr.press/v139/radford21a.html>
## Optional personal practical demo

A separate personal visualization of a possible practical use is available in
[`practical_demo/`](practical_demo/README.md).

It is outside the official Dataset V4 benchmark and does not change the
reported results, model selection, locked final-test evaluation, or official
test suite.
