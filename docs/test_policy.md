# Final-test policy

The final test is physically separated under `data/locked_test/dataset_v3/`.

The original selected checkpoint was frozen before test access. One authorization permitted exactly one final-test evaluation. That evaluation was completed and the authorization was consumed.

Therefore:

- training reads only train and validation;
- early stopping uses validation only;
- model comparison and selection use validation only;
- auxiliary-loss ablation uses validation only;
- no new checkpoint may be selected from final-test performance;
- the final-test evaluator must not be rerun;
- saved final-test predictions may be used only for integrity verification, final reporting, and error analysis.

`python -m src.evaluate` supports validation only. `python -m src.verify` recomputes final metrics from saved predictions and checks the preserved evidence without performing new model inference.
