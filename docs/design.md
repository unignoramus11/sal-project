# Study design: decisions and reasons

This file records each methodological decision, why we made it, and where we depart from PRiSM.
PRiSM's own settings are documented in `docs/prism_reproduction.md`. The rule: we reproduce PRiSM
once, faithfully, as an anchor; every other experiment uses the protocol below, and any departure
from PRiSM is justified here and, where it could change a conclusion, run both ways.

## 1. Two datasets, one protocol

- **Vaani-Hi (official)**, `changelinglab/vaani-hi-geo`. Access is gated with manual approval and still
  pending (17 Oct 2026). Used for the anchor reproduction if access arrives.
- **Vaani-Hi (rebuilt)**, `data/vaani_hi_rebuild/`, built from the public Vaani corpus with PRiSM's own
  recipe (`src/sal/data/rebuild.py`, Vaani commit pinned). The study does not depend on the official
  copy. The 7 states with at most 4 districts, which include Haryana and all its neighbours, match
  PRiSM's district selection exactly; 5 states sample districts from a sorted list instead of PRiSM's
  file-system order.

## 2. Speakers: the central confound

Vaani file names carry a speaker code (field 6 of `IISc_VaaniProject_<M>_<State>_<District>_<code>_...`).
On the first 16,800 sampled rows it equals the explicit `speakerID` wherever that exists (100%), is present
on every row, and maps to one gender and one district per code. So every Vaani-Hi utterance has a speaker.

Why it matters: PRiSM splits at random within each pincode. Utterances per speaker are skewed (median 3,
max 309 in the sample), so the same voice, microphone and room appear in train and test. A probe can then
place a test utterance by recognising the speaker or the channel rather than the dialect. This inflates
both probes, and not necessarily equally: the representation probe sees speaker and channel directly,
the transcript probe only through recognition errors. The TP-RP gap, the object of this study, can
therefore change under a speaker-disjoint split.

Evidence (17 Oct, first 68 of 140 shards): under PRiSM's random split, 97% of test utterances have their
speaker in train. Foley et al. (2024), the geolocation work PRiSM follows, hold out whole radio stations
to reduce speaker overlap and to geolocate "the speech itself, and not other channel artifacts".

Decisions:
- **Main protocol: speaker-disjoint splits.** Speakers, not utterances, are assigned to train/val/test,
  stratified by pincode so every pincode appears in every split, targeting 75/10/15 of utterances.
  Pincodes with too few speakers to split are reported, not silently dropped.
- **Anchor protocol: PRiSM's random split**, for the reproduction and as a second column in the main tables.
- **Leakage is measured, not assumed:** share of test utterances whose speaker is in train under the
  random split; and the drop from random to speaker-disjoint for each probe.
- **Speaker-only baseline:** a probe on a speaker-verification embedding (ECAPA-TDNN), which encodes voice
  and channel but is trained not to encode content. Its accuracy under each split bounds how much of GEO-v
  can be solved without dialect information.

## 3. Statistics

- Utterances from one speaker are not independent. Confidence intervals and paired tests use a
  **cluster bootstrap over speakers** (1,000 resamples), not over utterances.
- 5 seeds per configuration; we report mean ± sd across seeds and the bootstrap CI of the seed-averaged
  predictions.
- Metrics: Recall@1 (PRiSM definition: nearest distinct location in the split), mean and **median**
  great-circle error (median is robust to the long tail), per-state Recall@1.
- Control task (Hewitt and Liang, 2019): labels permuted across speakers within the training set; a probe
  whose control accuracy is high is memorising.

## 4. Transcript probe tokenisation

PRiSM joins CTC tokens without a separator and tokenises by character, so `kʰ`, `tʃ` and `kː` are split
into several tokens. This matters for RQ1: shuffling characters destroys phones as well as their order.

- Anchor: character tokens, exactly as PRiSM.
- Main: **phone tokens** from one segmentation rule applied to both recognizers (`src/sal/phones.py`):
  diacritics and modifier letters attach to their base, `tʃ`/`dʒ` are joined, the result is lossless.
  Raw tokens are not comparable across models: wav2vec2 emits eSpeak phonemes (`oː`, `ɔːɹ`), ZIPA emits
  characters (`o` + `ː`). PanPhon's segmenter was rejected because it silently drops symbols it does not
  know (ASCII `g`, precomposed `ẽ`, `ᵻ`).
- TP-Shuffle permutes phones within an utterance (fresh permutation each epoch for training, a fixed
  seeded permutation for evaluation). TP-Bigram uses phone bigram counts. Character and phone variants
  are both reported for the GRU.

## 5. Representation probe

PRiSM runs the frozen encoder online at every step and probes only the last layer. We cache features
(storage plan in `docs/compute.md`), which makes 5 seeds and the order ablations affordable, and we
select layers on validation data only (time-averaged layer sweep), never on test.
The attention head is PRiSM's (learned query over L2-normalised frames); the order-aware variants keep
the trainable parameter count within ±10% of it.

## 6. What we will not claim

- A district comparison is not a dialect comparison (Vaani language labels are self-reported).
- Results on the rebuilt Vaani-Hi are compared with PRiSM's published numbers only as a sanity check;
  the comparison of probes is always made within one dataset and one split.

## 7. Pre-registered predictions for RQ0 (written 18 Oct 2026, before the TP-GRU and RP results)

Why PRiSM's transcript probe can beat its representation probe under a leaky split even though only the
RP hears the voice: the last layer of a CTC phone recognizer is trained to discard speaker identity, and the
RP head L2-normalises frames and pools with one query. The TP, in turn, can identify a speaker without
hearing them, through topic (68% of utterances describe district-specific pictures, several per speaker),
habitual wording, and recognition errors that are consistent for one voice and microphone.

Predictions:
- P1. Under the speaker-disjoint split, TP Recall@1 drops by a large margin, and drops further on test
  utterances about shared (GENERIC) pictures.
- P2. RP on the last layer drops less than TP between splits (it has less speaker information to lose).
- P3. The weighted-layer gain of the RP (PRiSM App. F: +8.8 to +14.0) shrinks under the speaker-disjoint
  split, because lower layers re-admit speaker and channel information. If the gain survives, middle layers
  carry more location information that generalises to new speakers.
- P4. The speaker-embedding probe (ECAPA) is near the majority baseline under the speaker-disjoint split
  (observed: 10.1 vs 10.0).
