# Methodology

## 1. Study design

The task is three-class relation classification between an automotive-part image and a text description: `MATCH`, `MISMATCH`, or `PARTIAL_MATCH`.

The original Dataset V3 experiment compared seven models on a grouped development split. The multimodal model was selected using validation only, frozen, and evaluated once on a physically separated final test after explicit authorization. The unified project preserves those original results and adds one controlled validation-only auxiliary-loss ablation.

## 2. Hypotheses

- **H1:** the complete multimodal training design will outperform the tested unimodal and simple combined baselines.
- **H2:** `PARTIAL_MATCH` will be the most difficult relation class.
- **H3:** performance will vary by automotive-part category.
- **H4, controlled follow-up:** removing the two auxiliary category objectives will reduce relation-classification performance under the same setup.

H4 is retrospective. It was formulated after examining the selected model's training objective and was evaluated only on validation.

## 3. Dataset construction

Dataset V3 contains 640 unique images from eight balanced categories. A deterministic grouped split assigns 480 images to train, 80 to validation, and 80 to final test.

For each image, six relation rows are generated:

- two `MATCH` descriptions from the same category;
- two `PARTIAL_MATCH` descriptions from a different category in the same functional family;
- two `MISMATCH` descriptions from a category in a different functional family.

This produces 2,880 train rows, 480 validation rows, and 480 final-test rows. Every split is balanced with 160 rows per label for validation and test.

Split isolation is verified using image ID, image-group ID, path, SHA-256 image hash, and exact description text. No overlap is permitted across train, validation, and test.

## 4. Representations and models

Images are resized to 48 × 48 RGB. Text uses TF-IDF unigrams and bigrams fitted on train descriptions only.

The compared models are:

- majority baseline;
- text Logistic Regression;
- image-pixel Logistic Regression;
- image + text Logistic Regression;
- neural text MLP;
- image CNN;
- multimodal CNN + text MLP.

The selected multimodal network contains:

- a three-layer convolutional image encoder followed by a 48-dimensional projection;
- a two-layer text encoder producing 32 features;
- a relation head over the concatenated 80-dimensional representation;
- an image-category auxiliary head;
- a text-category auxiliary head.

The complete training objective is:

```text
relation cross-entropy
+ 0.40 × image-category cross-entropy
+ 0.40 × text-category cross-entropy
```

The selected model has 58,579 trainable parameters. Removing only the category heads leaves 57,923 trainable parameters.

## 5. Training protocol

Neural models use Adam with learning rate `0.001` and weight decay `0.0001`, batch size `32`, a maximum of `80` epochs, and early stopping with patience `10`. Image batches receive random horizontal flipping and mild brightness scaling. The selected multimodal initialization seed is `44`.

Training and early stopping read train and validation data only. The active training code rejects locked-test paths and has no final-test evaluation path.

## 6. Model selection and statistics

Selection is based on validation macro F1, with validation accuracy as supporting evidence. Accuracy and macro F1 are reported. Because six rows share an image, uncertainty and paired comparisons are grouped by complete image groups rather than treating all 480 relation rows as independent.

The original study includes grouped bootstrap confidence intervals and grouped paired randomization comparisons.

## 7. Controlled auxiliary-loss ablation

The follow-up removes the image-category head, text-category head, and both auxiliary losses. Everything else is held fixed:

- Dataset V3 train and validation relations;
- image and text representations;
- shared image encoder, text encoder, and relation head;
- optimizer and hyperparameters;
- image augmentation;
- early-stopping rule.

Three predefined seeds are run: `43`, `44`, and `45`. Seed 44 provides a direct initialization control: all 18 tensors shared by the complete and relation-only models are identical before training.

All three relation-only runs achieved 160/480 correct predictions, accuracy `0.3333333`, and macro F1 `0.1666667`. Every run collapsed to a single output class. The complete selected model achieved accuracy `0.7854167` and macro F1 `0.7872843`.

The ablation supports H4 under this fixed setup. It changes the interpretation of H1: the successful result belongs to multimodal fusion **with auxiliary category supervision**, not fusion alone.

## 8. Locked final test

The selected checkpoint and original development notebook were frozen before test access. One explicit authorization permitted one final-test evaluation. The evaluation was completed once and the authorization was consumed.

The frozen result is 354/480 correct predictions, accuracy `0.7375`, and macro F1 `0.7382299830` across 80 independent images. It is preserved for final reporting and error analysis only. It is not used to tune the model, select ablation settings, or choose another checkpoint.

The unified verifier recomputes final metrics from the saved predictions. It does not perform new model inference on final-test images.

## 9. Reproducibility and limitations

The project pins PyTorch `2.10.0`, preserves the selected checkpoint hash, stores all three ablation checkpoints and predictions, and provides automated integrity tests. Original imported evidence and newly generated unified-project artifacts are distinguished in `evidence/lineage.json`.

The conclusions remain limited to one source collection, eight categories, a compact from-scratch model, and this relation-generation protocol. External data and a separately collected test set would be required before practical deployment.
