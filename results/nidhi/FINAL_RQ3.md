# RQ3: Prosodic information for speech geolocation

## Experimental setup

- Evaluation: speaker-disjoint split
- Seed: 42
- Test utterances: 3,581
- Prosody model: contour-based ProsodyGRU
- Transcript model: ZIPA phone-sequence GRU
- Fusion: unit-vector weighted interpolation
- Fusion weight: 0.85 on transcript predictions
- Weight selection: validation mean-distance error

## Test results

| Model | Recall@1 | Mean error (km) |
|---|---:|---:|
| ZIPA transcript GRU | 17.51% | 441.13 |
| Contour prosody GRU | 11.31% | 475.24 |
| Late fusion | 16.31% | 426.26 |

Fusion reduced mean distance error by 14.87 km
(3.37%) relative to the transcript-only model.

However, Recall@1 decreased by 1.20 percentage points.

## Interpretation

The contour-based prosody model contributes complementary
information for reducing geographic distance error, but
fusion does not improve every evaluation metric.

## Limitations

Only one random seed was run for the contour model.
No multi-seed confidence interval is available.
Fusion weights were chosen on validation, not test data.
The blinded acoustic-annotation comparison is separate
and must not be reported as completed without its key
and reference measurements.
