# Optional Practical Demo

> **Personal experiment - separate from the official Dataset V4 benchmark**

This folder is not part of the official benchmark evaluation of the project.

After completing the research experiment, I created this small practical
demonstration because I wanted to see the idea working in a visual and
usable form.

The purpose is to show, in a simple way, how the image-text relation model
could be used in an everyday automotive-parts workflow.

## What the demo does

The user provides:

- an automotive-part image;
- a short text description.

The model predicts one of three relations:

- `MATCH`
- `PARTIAL_MATCH`
- `MISMATCH`

For example, using the same air-compressor image:

- `air compressor` -> `MATCH`
- `overflow tank` -> `PARTIAL_MATCH`
- `camshaft` -> `MISMATCH`

These examples are only a visual functional demonstration.
They are not a new accuracy evaluation.

## Important separation from the official project

The official Dataset V4 research project and benchmark remain at the
repository root.

This optional demo:

- does not change the reported Dataset V4 benchmark results;
- was not used to select the official benchmark model;
- was not used to tune the official final-test result;
- does not perform new inference on the locked final-test set;
- is not presented as production-ready automotive software.

The separate deployment classifier was built only from Dataset V4
`train + validation` data using the already frozen model configuration.

I created the demo as a personal practical experiment after the research
workflow, mainly to see the project idea working in a visible end-to-end
form.

## Run the visual demo

From the repository root:

~~~powershell
python -m pip install -r practical_demo\requirements.txt
python -m streamlit run practical_demo\app.py
~~~

The browser interface allows an automotive-part image to be uploaded and
a short description to be entered interactively.

## Command-line version

~~~powershell
python -m practical_demo.predict IMAGE_PATH "DESCRIPTION"
~~~

Example:

~~~powershell
python -m practical_demo.predict `
  "data\images\dataset_v4\train\air_compressor\v4_air_compressor_train_0059_2526446c8c41f002.jpg" `
  "The description names the automotive part as overflow tank."
~~~

That example returns `PARTIAL_MATCH` because `air_compressor` and
`overflow_tank` are different categories from the same functional family.

## Scope

This demo is an educational visualization of a possible practical use.
It is separate from the official notebook, benchmark metrics, locked final
test and official project test suite.
