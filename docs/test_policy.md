# Dataset V4 final-test policy

The official locked final test is stored under:

```text
data/locked_test/dataset_v4/
```

## Order of work

I followed this order:

1. build and audit the Dataset V4 image split;
2. balance the relation construction before training;
3. train and compare four models using train and validation only;
4. select `multimodal_auxiliary` from validation macro F1;
5. freeze its best epoch at 7;
6. commit the final-test evaluation code;
7. run the locked final test once;
8. save predictions and metrics;
9. allow no post-test tuning.

## Frozen development choice

The validation result used for selection is stored in:

```text
results/dataset_v4/step03_validation_summary.json
```

The original one-time final-test run recorded the validation summary with
Windows CRLF line endings. Its historical byte SHA-256, preserved in the
Step 04 final summary, is:

```text
3e7bd14a0aecee3a574fe9e51a22a01d956bf3be10a7260e872dd61598118769
```

Git normalizes tracked text to LF through `.gitattributes`. Therefore the same
frozen JSON content in a clean clone has the repository-canonical SHA-256:

```text
41e88e8ac40f03c38229f52bc13a893bc0cb0a414b6b2087890c655d8ca55cd4
```

The safe `--check-only` verifier uses the repository-canonical LF hash. The
historical Step 04 result is intentionally left unchanged.

Selected model:

```text
multimodal_auxiliary
```

Frozen best epoch:

```text
7
```

## Locked-test identity

The frozen relation-table SHA-256 is:

```text
79bae874c5b365200e7562b1a3a1d8ee28e8ea3fccfab86d4034b89332260ef2
```

## Official final result

The one-time evaluation produced:

- 9,030 relation predictions;
- 1,388 independent images;
- 8,615 correct relation predictions;
- accuracy 0.9540420819490587;
- macro F1 0.9540841368243022.

The saved files are:

```text
results/dataset_v4/step04_final_test_predictions.csv
results/dataset_v4/step04_final_test_summary.json
```

The prediction CSV SHA-256 recorded in the summary is:

```text
587a607c9364c73e0bb9aee282ba950b6e0a604db57e0e5d0c88b5a719bbacd0
```

The final summary file was independently hashed as:

```text
53bfc8985321156712edae1e91a6c026ab273ed24d2d91ffe145de7a67252883
```

## Commands after the final result

The safe policy check is:

```powershell
python -m tools.run_dataset_v4_final_test --check-only
```

The diagnostic audit can be regenerated from saved predictions with:

```powershell
python -m tools.audit_dataset_v4_final_result
```

The official final-test inference has already been performed. The project should **not** run `--confirm-final-test` again and should not use the result for additional tuning or model selection.

## Reporting

The notebook reads the frozen result files for tables and plots. Reading already saved predictions for analysis is not a new model inference.
