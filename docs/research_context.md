# Research context and baseline rationale

## Purpose

This project is a controlled educational experiment in multimodal
classification. Its main question is whether combining an automotive-part image
with a short text description predicts their relation better than using either
modality alone.

The project does not propose a new state-of-the-art vision-language
architecture. This note positions the official baselines relative to established
multimodal and vision-language research and explains why intentionally simple
components were used.

## Multimodal learning

Baltrušaitis, Ahuja and Morency describe multimodal machine learning through
problems including representation, translation, alignment, fusion and
co-learning. This project focuses mainly on **fusion**: image and text
representations are produced separately and then combined to predict one
relation label.

The comparison between text-only, image-only and multimodal models is therefore
central. The main experimental claim is not architectural novelty; it is that
the two modalities contain complementary information for this prepared
benchmark.

## Image representation and transfer learning

The image branch uses an ImageNet-pretrained ResNet18 as a fixed feature
extractor. Residual networks were introduced by He et al. to make deep visual
models easier to optimize through residual connections.

Using a pretrained frozen image encoder keeps the experiment manageable for a
course project while providing a strong visual representation. It also isolates
the multimodal question from the much larger problem of training a vision
backbone from scratch.

## Text representation

The official experiment uses TF-IDF unigrams and bigrams followed by a small
neural network. This is intentionally simple and interpretable. Because the
benchmark descriptions clearly name automotive-part categories, lexical
features are a reasonable controlled baseline.

This is also a limitation. The experiment does not test whether a modern
language model can understand unrestricted customer descriptions, synonyms,
misspellings, long descriptions or more ambiguous language.

## Relation to modern vision-language models

Modern vision-language systems use substantially richer cross-modal pretraining.

VisualBERT (Li et al., 2019) applies Transformer layers to jointly model text
and visual-region representations and can learn implicit alignment between the
two modalities.

CLIP (Radford et al., 2021) learns aligned image and text representations with a
contrastive objective on a very large collection of image-text pairs and
supports transfer to downstream visual tasks through natural-language
supervision.

These systems are important reference points, but they are **not official
baselines in this project**. Adding a pretrained vision-language model after
seeing the locked final-test result would change the original experimental
scope and would require a new validation/test protocol.

Therefore the 0.9540 final-test accuracy must not be interpreted as a result
directly comparable with CLIP, VisualBERT or other large pretrained
vision-language systems.

## Why the official baselines are deliberately simple

The experiment keeps the architecture controlled for four reasons:

1. **Clear ablation.** Text-only, image-only and multimodal models can be
   compared under the same data construction and split.
2. **Interpretability.** It is easier to explain what information each branch
   contributes.
3. **Course scope.** The project demonstrates transfer learning, text
   vectorization, multimodal fusion, validation-based model selection and
   reproducible evaluation without depending on a large pretrained
   vision-language system.
4. **Locked-test discipline.** The literature positioning does not alter the frozen model-selection or final-test protocol; no new model is selected from the final-test result.

## What the benchmark result supports

The narrow conclusion supported by the experiment is:

> On this prepared Dataset V4 benchmark, a model using both image and text
> substantially outperforms otherwise comparable one-modality baselines.

The experiment does **not** establish that:

- the model is state of the art in general image-text matching;
- the model understands unrestricted natural language;
- 95.4% accuracy will transfer to customer photographs or unseen part classes;
- the approach is better than CLIP or another pretrained vision-language model;
- the benchmark represents all real automotive-part matching conditions.

## Stronger future comparison

A natural next experiment would define a **new** benchmark or a new untouched
external test set and compare:

- the current ResNet18 + TF-IDF fusion model;
- a pretrained CLIP-style image-text similarity baseline;
- optionally, a pretrained text encoder combined with the same visual features;
- performance on free-form customer text and more difficult external images.

That would answer a different research question and should not reuse the
current locked final test for model selection.

## References

1. K. He, X. Zhang, S. Ren, J. Sun, *Deep Residual Learning for Image
   Recognition*, CVPR 2016. https://arxiv.org/abs/1512.03385
2. T. Baltrušaitis, C. Ahuja, L.-P. Morency, *Multimodal Machine Learning:
   A Survey and Taxonomy*, IEEE TPAMI 2019.
   https://arxiv.org/abs/1705.09406
3. L. H. Li, M. Yatskar, D. Yin, C.-J. Hsieh, K.-W. Chang, *VisualBERT:
   A Simple and Performant Baseline for Vision and Language*, 2019.
   https://arxiv.org/abs/1908.03557
4. A. Radford et al., *Learning Transferable Visual Models From Natural
   Language Supervision*, ICML 2021.
   https://proceedings.mlr.press/v139/radford21a.html
