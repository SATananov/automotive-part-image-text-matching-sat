# Data source and project history

## Image source

The images come from the public Kaggle dataset [50 Types of Car Parts - Image Classification](https://www.kaggle.com/datasets/gpiosenka/car-parts-40-classes), created by G. Piosenka. The dataset page records the licence as Apache-2.0.

For this project, I selected 640 images from eight categories. I did not use the original train/test split. I created a new grouped split for this image-text task.

The source name, category, path, and SHA-256 hash of every selected image are stored in `data/manifests/images.csv`. Because all Dataset V3 images come from this one source, `data/licenses.csv` contains one dataset-level licence record covering the 640 selected images.

## Earlier project work used here

The Dataset V3 images, relation tables, selected model, validation results, and saved final-test results were first produced in:

- repository: `SATananov/automotive-part-image-text-matching`;
- branch: `dataset-v3`;
- result checkpoint: `76f62ef91faa1e544c9074a91e80d86ed71a99fa`;
- later report/check state: `d9ead2a8ef6637a3e86713b98433fc1ca1389158`.

The simpler code layout was also informed by:

- repository: `SATananov/automotive-part-multimodal-classification`;
- commit: `a9717986b2b73d089f29ce3a619bbe9e08494ef6`.

Both earlier repositories use the same Dataset V3 experiment. I do not present them as two independent experiments.

## Work completed in this repository

In this repository I:

- kept one README, one methodology, and one official notebook;
- kept only the Dataset V3 files needed for the final project;
- removed Dataset V2 and duplicate entry points;
- aligned the active licence table with the Dataset V3 source actually used;
- added portable verification and automated tests;
- added a validation-only category-rule diagnostic without reading the final test;
- ran the additional validation-only experiment without the two helper category tasks, using seeds 43, 44, and 45.

The validation diagnostic performs no training and uses only the frozen selected checkpoint and the validation split. The additional experiment was completed after the original model comparison. Neither item changes the selected model or the saved final-test result.
