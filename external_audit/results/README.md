# External Audit V1 results

This directory intentionally contains no score at protocol-scaffold time.

After the image set and relation table are locked and committed, the one-time
secondary evaluation is run with:

```powershell
python -m external_audit.run --confirm-external-audit
```

The runner will create:

- `external_predictions.csv`
- `external_summary.json`

The result is secondary robustness evidence. It does not replace or modify the
official Dataset V4 locked final-test result.
