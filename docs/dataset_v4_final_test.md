# Dataset V4 final-test policy

Step 04 is intentionally separated from model development.

The model was selected only from the Step 03 validation experiment. The frozen choice is `multimodal_auxiliary`, with the epoch count taken from the best validation epoch. The locked final test is not used to choose a model, epoch count, learning rate, or other setting.

Before opening the final test, the evaluation code is committed. A policy check can be run with:

```powershell
python -m tools.run_dataset_v4_final_test --check-only
```

The one-time final evaluation is then started explicitly with:

```powershell
python -m tools.run_dataset_v4_final_test --confirm-final-test
```

For the final model, train and validation relations are combined as development data. The selected model is trained for the already-frozen number of epochs. The final test is evaluated once. After that evaluation, no additional tuning or model selection is allowed.

The final run saves only a JSON summary and a CSV with predictions. ResNet feature caches remain under `.cache/` and are not committed.
