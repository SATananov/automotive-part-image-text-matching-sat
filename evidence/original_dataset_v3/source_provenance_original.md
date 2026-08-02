# Dataset V3 source provenance

## Source

- Kaggle dataset slug: `gpiosenka/car-parts-40-classes`
- Dataset title during acquisition audit:
  `50 Types of Car Parts -Image Classification`
- Declared dataset license during acquisition audit: `Apache-2.0`
- Selected archive root: `car parts 50`

## Selected source classes

| Source class | Project category |
|---|---|
| ALTERNATOR | alternator |
| BRAKE ROTOR | brake_disc |
| BRAKE PAD | brake_pad |
| COIL SPRING | coil_spring |
| HEADLIGHTS | headlight |
| OIL FILTER | oil_filter |
| STARTER | starter |
| TAILLIGHTS | taillight |

## Exclusions

The duplicated legacy `car parts` archive root was not used.
The source model files were not used.
Images with obvious watermarks, annotations, diagrams, severe quality
problems, or weak class specificity were removed during manual curation.

## Independence and split policy

The final pool contains 640 unique SHA-256 values and 640 unique image
groups. Exact, perceptual, cross-category, and current-project overlap
audits were zero before the split was created.

The original source split labels are retained only as provenance metadata.
They do not determine Dataset V3 train, validation, or locked-test
membership.
