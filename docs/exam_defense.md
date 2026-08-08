# Project Defense — Automotive Part Image-Text Matching

## What I built

This project is a Deep Learning experiment for multimodal classification of automotive-part image–text pairs.

The system receives:

- an image of an automotive part;
- a short text description.

It predicts one of three relations:

- `MATCH` — the image and text describe the same part category;
- `PARTIAL_MATCH` — they describe different categories from the same functional family;
- `MISMATCH` — they describe categories from different functional families.

My main research question is whether using **both image and text information** performs better than using only one modality.

---

## Why I chose this problem

The idea comes from a realistic automotive-parts scenario: product images and text descriptions should correspond correctly.

A system like this could eventually support catalog validation, search assistance, or detection of inconsistent image-description pairs.

This project is an educational research prototype, not a production system.

---

## Deep Learning approach

The visual part uses **ResNet18 pretrained on ImageNet** as a fixed feature extractor.

Each RGB image is resized to `224 x 224`, converted to a PyTorch tensor and normalized. A batch of images has the conceptual tensor shape:

```text
[B, 3, 224, 224]
```

ResNet18 produces a **512-dimensional image feature vector**.

The text is represented with **TF-IDF**. TF-IDF itself is not Deep Learning, but its representation is passed into a trainable neural text branch.

The neural model projects both modalities to learned 128-dimensional representations:

```text
image features: 512 -> 128
text features:  TF-IDF -> 128
```

The two tensors are concatenated:

```text
[B, 128] + [B, 128] -> [B, 256]
```

The final neural classifier produces three logits:

```text
[B, 3]
```

corresponding to `MATCH`, `PARTIAL_MATCH` and `MISMATCH`.

The selected model also uses auxiliary image-category and text-category losses during training.

---

## Training

The downstream neural classifier is trained with:

- `CrossEntropyLoss`;
- Adam optimizer;
- learning rate `0.001`;
- batch size `256`;
- maximum 8 epochs;
- fixed random seed `44`.

During training, PyTorch performs the forward pass, calculates the loss, computes gradients with backpropagation through `loss.backward()`, and updates the trainable parameters with `optimizer.step()`.

The pretrained ResNet18 backbone is frozen and is not updated in this training loop.

---

## Dataset V4

The final dataset contains:

- **9,239 unique images**;
- **50 automotive-part categories**;
- **12 functional families**.

The split is performed at **image level before relation generation**:

| Split | Independent images | Relation rows |
|---|---:|---:|
| Train | 6,463 | 41,742 |
| Validation | 1,388 | 9,030 |
| Locked final test | 1,388 | 9,030 |

One image can participate in several image-text relations, so relation rows are not the same as independent photographs.

The source, provenance, licensing and dataset manifests are documented separately in the repository.

---

## Data leakage prevention

A major part of the project was making sure the reported result could be trusted.

I therefore:

- split independent images before generating image-text relations;
- checked exact image hashes across splits;
- fitted TF-IDF only on the allowed development text;
- kept validation separate from training;
- selected the final model only from validation results;
- locked the final-test relations and their SHA-256;
- prevented normal development code from loading the final test;
- did not tune the model after the final-test result.

This was important because a random split of relation rows could place the same image in both training and test and artificially inflate performance.

---

## Model comparison

I compared four models:

| Model | Validation macro F1 |
|---|---:|
| Text only | 0.3042 |
| Image only | 0.3003 |
| Multimodal | 0.8592 |
| Multimodal + auxiliary tasks | **0.8958** |

The text-only and image-only baselines are near chance for the balanced three-class relation task.

The strong improvement of the multimodal models supports the main research conclusion: **the relation is learned much better when both image and text information are available**.

The final model was frozen as:

```text
multimodal_auxiliary
best epoch: 7
```

before opening the locked final test.

---

## Official locked final test

The final locked test contains:

- 1,388 independent images;
- 9,030 relation rows.

The frozen result is:

```text
Accuracy:  0.9540
Macro F1:  0.9541
Correct:   8,615 / 9,030
```

I do not interpret this as 95.4% real-world automotive performance.

It means that the frozen model achieved this result on the prepared Dataset V4 benchmark under the locked evaluation protocol.

---

## Additional checks

Because the final-test result was very high, I performed additional checks instead of simply accepting the number.

I checked:

- exact split overlap;
- TF-IDF fitting scope;
- saved prediction consistency;
- near-duplicate / visually similar images;
- image-level error concentration.

These checks did not identify an obvious leakage explanation for the locked-test result.

---

## External robustness audit

I also wanted to see how the model behaves outside the prepared benchmark.

For this reason I created a separate external audit using:

- 60 independently sourced Wikimedia Commons images;
- 6 known categories;
- 3 functional families;
- 360 locked image-text relations.

The results were:

| Condition | Accuracy | Macro F1 |
|---|---:|---:|
| External overall | 0.6722 | 0.6725 |
| Clean text | 0.7444 | 0.7458 |
| Natural text | 0.6000 | 0.6001 |

This was an important result.

It showed that the official 95.4% benchmark score is valid for Dataset V4, but the model generalizes substantially worse to a different image source and more natural text.

The external audit therefore confirms that the current system is **not production-ready**.

---

## Practical visualization

After the research experiment, I also built a separate practical demonstration because I wanted to see how the model would look when used interactively rather than only through metrics and tables.

The `practical_demo/` application allows a user to:

1. upload an automotive-part image;
2. enter a short description;
3. run inference;
4. see the predicted relation;
5. inspect the model scores for all three relation classes.

I also included a command-line inference option.

This practical demo is separated from the official benchmark. It does not change the frozen Dataset V4 result and was not used for post-test tuning.

Its purpose is to visualize the complete inference workflow and understand how the research idea could be presented to a real user.

---

## Testing and reproducibility

The final repository passes:

```text
47 passed
```

The automated tests cover dataset integrity, model behavior, result consistency, notebook/report checks, locked-test policy, leakage protection and the external-audit protocol.

I also use manifests and SHA-256 hashes to make important data and result artifacts verifiable.

The normal final-test command is intentionally only a policy check:

```powershell
python -m tools.run_dataset_v4_final_test --check-only
```

and confirms:

```text
Final test relations loaded: False
```

This protects the locked evaluation from accidental reuse.

---

## Previous research

I used previous work to position the project and understand the methods, including:

- ResNet — deep residual learning for image recognition;
- multimodal machine learning research;
- VisualBERT;
- CLIP;
- PyTorch / Torchvision documentation;
- scikit-learn TF-IDF documentation.

I do not claim that my model is state of the art, and I do not directly compare my Dataset V4 score with results from unrelated datasets or tasks.

---

## Main limitations

The most important limitations are:

- the official text is more structured than real customer language;
- all official Dataset V4 images come from one source collection;
- TF-IDF is limited when wording or vocabulary changes;
- ResNet18 is used as a frozen feature extractor;
- the project covers 50 known categories, not a complete automotive catalog;
- the external audit is useful but small;
- the project does not solve unknown-category or open-set recognition;
- a strong benchmark result does not guarantee real-world performance.

---

## What I learned

The most important lesson from this project was that a high Deep Learning metric is not enough.

I learned that the experimental protocol matters just as much as the neural architecture:

- correct data splitting;
- leakage prevention;
- validation discipline;
- frozen model selection;
- locked final testing;
- reproducibility;
- external evaluation;
- honest interpretation of limitations.

The main conclusion I would defend is:

> **On the prepared Dataset V4 benchmark, combining image and text information works substantially better than using either modality alone.**

The equally important limitation is:

> **The project does not prove that the current system will achieve the same performance on arbitrary real-world automotive images and natural customer descriptions.**

---

## Where to inspect the project

For more detail:

- [`../project.ipynb`](../project.ipynb) — main experimental report and visualizations;
- [`methodology.md`](methodology.md) — full methodology;
- [`research_context.md`](research_context.md) — previous research;
- [`provenance.md`](provenance.md) — dataset provenance;
- [`../src/`](../src/) — reusable model/data/training code;
- [`../tests/`](../tests/) — automated tests;
- [`../external_audit/`](../external_audit/) — external robustness audit;
- [`../practical_demo/`](../practical_demo/) — practical interactive visualization.
