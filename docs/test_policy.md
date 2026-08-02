# Final-test rule

The final-test files are stored separately in `data/locked_test/dataset_v3/`.

I followed this order:

1. train the models on the training split;
2. compare them on validation;
3. select and save the best model;
4. use the final test once to report the final result.

The final test is not used for training, early stopping, model selection, or the additional experiment without helper tasks.

The saved final-test predictions are kept so the notebook can show the final metrics, confusion matrix, category results, and mistakes. The notebook and verifier do not run the model again on the final-test images.

Useful commands:

```powershell
python -m src.evaluate          # validation only
python -m src.verify --full-hashes
python -m pytest -q
```
