# RQ2: Recognizer syllable structure

These results use heuristic IPA syllabification, not manually verified syllable boundaries.

| Recognizer | State | Speakers | Closed syllables (%) | 95% speaker-bootstrap CI (%) |
|---|---|---:|---:|---|
| W2V2 | Bihar | 31 | 17.56 | [16.43, 18.94] |
| W2V2 | Chhattisgarh | 264 | 18.37 | [17.81, 18.97] |
| W2V2 | Haryana | 12 | 17.08 | [15.35, 18.42] |
| W2V2 | HimachalPradesh | 77 | 19.98 | [18.96, 20.95] |
| W2V2 | Jharkhand | 440 | 19.67 | [19.20, 20.14] |
| W2V2 | MadhyaPradesh | 522 | 19.53 | [19.15, 19.90] |
| W2V2 | Punjab | 17 | 16.40 | [15.43, 19.74] |
| W2V2 | Rajasthan | 175 | 18.68 | [17.96, 19.40] |
| W2V2 | Uttarakhand | 32 | 18.47 | [16.97, 19.83] |
| ZIPA | Bihar | 31 | 18.16 | [16.66, 19.93] |
| ZIPA | Chhattisgarh | 264 | 18.89 | [18.24, 19.49] |
| ZIPA | Haryana | 12 | 20.01 | [19.00, 21.01] |
| ZIPA | HimachalPradesh | 77 | 21.47 | [19.06, 23.16] |
| ZIPA | Jharkhand | 440 | 20.12 | [19.58, 20.57] |
| ZIPA | MadhyaPradesh | 522 | 20.83 | [20.42, 21.28] |
| ZIPA | Punjab | 17 | 21.52 | [20.40, 23.80] |
| ZIPA | Rajasthan | 175 | 22.43 | [21.76, 23.13] |
| ZIPA | Uttarakhand | 32 | 19.10 | [17.31, 21.16] |

## Limitations
- The syllabifier is heuristic and recognizer-dependent.
- Confidence intervals cluster by speaker.
- Haryana includes only 12 speaker IDs.
- These results are distinct from acoustic consonant-duration measurements.
