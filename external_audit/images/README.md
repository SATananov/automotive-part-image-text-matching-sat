# External Audit V1 images

Do not place Dataset V4 images here.

The preferred source is newly self-captured photography because it gives the
clearest independence and redistribution story for a public exam repository.
Independently licensed images are also acceptable if their provenance and
redistribution rights are documented.

Create exactly 10 images in each category folder:

```text
external_audit/images/
- alternator/
- starter/
- brake_pad/
- brake_rotor/
- headlight/
- taillight/
- oil_filter/
- oil_pan/
- spark_plug/
- ignition_coil/
- radiator/
- water_pump/
```

Supported file types are JPG/JPEG, PNG and WebP.

Capture guidance:

- use real photographs rather than screenshots from Dataset V4;
- prefer different physical parts, not ten crops of one photograph;
- vary background, lighting, angle and distance naturally;
- keep one dominant automotive part in the image;
- do not select or discard photographs based on model predictions;
- do not run the model while collecting the images;
- preserve the original files until the audit has been locked.

After all 120 images are present, copy `provenance_template.json` to
`provenance.json`, complete its fields, and run:

```powershell
python -m external_audit.prepare --lock
```

Commit the locked manifest, relation table, provenance and lock **before**
running external inference.
