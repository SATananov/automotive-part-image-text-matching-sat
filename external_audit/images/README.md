# External Audit V1.1 images

The original 12-category V1 plan was narrowed **before any external inference**
after a source-feasibility review found that generic Wikimedia search results
were too noisy for several categories.

V1.1 uses exactly six categories:

```text
external_audit/images/
- alternator/
- starter/
- headlight/
- taillight/
- spark_plug/
- ignition_coil/
```

Place exactly 10 visually correct, independent images in each folder.

Approved source categories:

- alternator -> Wikimedia Commons `Category:Automobile alternators`
- starter -> Wikimedia Commons `Category:Electric starter motors`
- headlight -> Wikimedia Commons `Category:Automobile headlamps`
- taillight -> Wikimedia Commons `Category:Automobile rear lights`
- spark_plug -> Wikimedia Commons `Category:Spark plugs`
- ignition_coil -> Wikimedia Commons `Category:Ignition coils`

Selection rules:

- select by visual category correctness and reasonable diversity only;
- do not use model predictions to accept or reject an image;
- prefer images where the target part is clear and visually dominant;
- reject diagrams, unrelated objects, people-only images and ambiguous scenes;
- do not create multiple crops of one source image;
- preserve per-image Commons author, source page, license and SHA-256.

Only after 60 images and complete provenance exist:

```powershell
python -m external_audit.prepare --lock
```

Commit and push the lock before any external inference.
