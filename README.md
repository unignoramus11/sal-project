# Order, Duration, or Melody?

Probing what phone recognizers use to geolocate Haryanvi and Hindi-belt speech.

Course project for Speech Analysis and Linguistics (Monsoon 2026), IIIT Hyderabad.
Mohit Kumar Singh and Nidhi Vaidya.

PRiSM (Bharadwaj et al., ACL 2026) finds that a transcript probe over predicted phones geolocates
Vaani-Hi speakers better than a representation probe over hidden states (146 km vs 253 km mean error),
and suggests phone order as the reason without testing it. We test three explanations:

- **RQ1 Order.** A 2x2 design of {transcript, representation} x {order-aware, order-free} probes, with layer choice held fixed.
- **RQ2 Duration.** Consonant gemination in Bangru Haryanvi, in model outputs, in Praat closure durations, and across layers.
- **RQ3 Melody.** Whether F0, intensity and voiced-duration contours add to the transcript probe.

## Layout

| Path        | Contents                                                                    |
| ----------- | --------------------------------------------------------------------------- |
| `proposal/` | ACL-format LaTeX source and PDF of the proposal                             |
| `docs/`     | Experiment plan, data and model notes, compute and storage                  |
| `scripts/`  | Standalone utilities (e.g., `vaani_check.py`, a metadata check of Vaani-Hi) |

Datasets, extracted features and checkpoints are not tracked (see `.gitignore`).

## Build

```sh
make proposal   # latexmk, writes proposal/proposal.pdf
make plot       # regenerates proposal/figs/geov_gap.* from PRiSM Tables 4 and 7
```

## Dates

Proposal 9 Oct 2026. Mid evaluation 3 Nov 2026. End evaluation 4 Dec 2026.
