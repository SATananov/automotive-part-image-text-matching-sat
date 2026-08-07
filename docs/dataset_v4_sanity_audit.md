# Dataset V4 final-result sanity audit

This audit was added because the locked final-test result is high enough to deserve an extra check before reporting it.

The audit does **not** train a model and does **not** run final-test inference again. It works from the already frozen prediction CSV, the image manifest, and the existing project code.

It checks:

- recomputation of final accuracy and macro F1 from saved predictions;
- consistency between saved predictions and the locked test relations;
- whether category ground truth is passed to the model during final-test inference;
- whether TF-IDF is fitted on development text only;
- exact SHA-256 overlap across train, validation, and test;
- perceptual near-duplicate candidates across splits using pHash and dHash;
- equal-weight metrics over independent test images;
- results by automotive-part category;
- how the original Kaggle train/valid/test folders were redistributed into Dataset V4;
- sensitivity of the frozen result after excluding test images flagged as possible near-duplicates.

Perceptual hashing is a screening method, not proof that two files are duplicates. Candidate pairs should be reviewed visually before making a final methodological decision.

Run:

```powershell
python -m tools.audit_dataset_v4_final_result
```

The audit writes only two diagnostic files:

- `results/dataset_v4/step04_sanity_audit.json`
- `results/dataset_v4/step04_sanity_near_duplicate_candidates.csv`

The existing Step 04 final-test files are never modified.
