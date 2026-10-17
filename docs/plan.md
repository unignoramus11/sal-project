# Project plan (from 17 Oct 2026)

Goal: a study that would hold up as a workshop or conference paper, not only a course report.
Design decisions and their reasons are in `docs/design.md`; PRiSM's exact settings in
`docs/prism_reproduction.md`.

## 1. What exploration changed

Findings from 17 Oct (first 68 of 140 rebuilt shards; final numbers will come from the full build):

- **Every Vaani utterance has a speaker code** (in the file name; equals the explicit speaker ID where
  that exists). The proposal planned only an approximate speaker split from voice clustering.
- **PRiSM's random split leaks speakers almost completely**: 97% of test utterances have their speaker in
  train. Half of all utterances come from speakers with at least 20 recordings, and one speaker supplies
  up to 55% of a location.
- **The geolocation work PRiSM builds on avoids this.** Foley et al. (2024) hold out whole radio stations
  "to reduce the risk of speaker overlap" and aim to geolocate "the speech itself, and not other channel
  artifacts". Dialect-ID work does the same, and reports large drops without it (North Sámi: 60.1% to
  32.9%). No prior work has examined this for Vaani or GEO-v.
- PRiSM's transcript probe splits phones into characters (`kʰ`, `tʃ`, `kː`), so a character shuffle would
  destroy phones as well as order.
- The ZIPA checkpoint is CC BY-NC-SA 4.0 (the proposal said MIT, which is the code licence).
- Vaani-Hi access is still pending; the rebuilt Vaani-Hi (PRiSM recipe, pinned Vaani commit) makes the
  study independent of it.

## 2. Revised framing

The proposal asked why the transcript probe beats the representation probe. The question stays, but it
now has a prior step: how much of GEO-v is dialect, and how much is the speaker?

- **RQ0 (new) Speaker.** How much do TP and RP scores fall from the random split to a speaker-disjoint
  split, and how well does a speaker-only representation (ECAPA) geolocate under each?
- **RQ1 Order.** Order × input-type design with layer choice controlled, under both splits.
- **RQ2 Duration.** Consonant doubling in model output (DPR), closure duration measured from audio,
  layer-wise encoding, attribution.
- **RQ3 Melody.** Prosodic contours alone and fused with the transcript probe.

Every comparison is run under both protocols; the speaker-disjoint one is primary.

## 3. Work plan (compressed 17 Oct: aim to finish the whole study before the 3 Nov mid evaluation)

At the mid evaluation we present what the proposal promised (baselines, first RQ1 result, DPR) and hold
the full analysis for the end evaluation. Three machines run in parallel:

| Machine | Owner | Jobs |
|---|---|---|
| Ada (SLURM) | Nidhi | rebuild checksum check; feature extraction (3 models); layer sweep; all RP runs |
| PC (RTX 4050, 25 W) | Mohit | transcription (2 models); all transcript-probe runs; ECAPA speaker embeddings |
| Mac (M1 Pro) | Mohit | data and split reports; DPR; prosody (Parselmouth); syllables; closure script; statistics; figures |

### Week 1 (17-24 Oct): data, transcripts, baselines
- [x] Rebuilt Vaani-Hi (22,551 utt, 40.3 h); speaker-disjoint split; dataset and leakage report.
- [x] Probe library (PRiSM-faithful TP and RP, order-aware variants with matched capacity), tests.
- [ ] Transcription, W2V2P and ZIPA (PC; first 50 per model identical to PRiSM's runner).
- [ ] Ada: environment, rebuild checksum, feature extraction, layer sweep.
- [ ] TP: GRU (char, phone) x 2 models x 2 protocols x 5 seeds; Shuffle; Unigram; Bigram (PC, Mac overflow).
- [ ] Naive baselines (train centroid, majority location); ECAPA speaker baseline (RQ0).
- [ ] DPR per utterance and state, both recognizers.
- [ ] RP-Attn last layer, 3 models x 2 protocols x 5 seeds (Ada).

### Week 2 (24-31 Oct): main experiments
- [ ] RP sweep on Ada: attn, attn_pe, gru, transformer, chunkshuffle x {last, weighted top layers} x 3 models
      x 2 protocols x 3-5 seeds.
- [ ] Gemination: transcripts of the four Vaani districts, forced alignment (MMS), closure script,
      50 shared hand-measured tokens (both members), mixed-effects model with speaker as random effect.
- [ ] Frame-level geminate probe by layer; integrated gradients on the Haryana test set.
- [ ] Prosody stream and late fusion (RQ3); syllabification and closed-syllable share.
- [ ] MDL (online coding) for the key comparisons; cluster bootstrap over speakers; control tasks.

### 1-2 Nov: mid-evaluation package
Slides with the proposal's promised results; full results tables and figures generated from `results/`.

### After the mid evaluation (to 4 Dec)
Write-up in ACL format, extra robustness checks suggested at the mid evaluation, code and data-build release.

## 4. Compute

| Job | Estimate | Where |
|---|---|---|
| Transcription (2 models) | ~0.5 h each | PC (RTX 4050, 6 GB) |
| Feature extraction (3 models) | 1-2 h each; ~15 GB per cached layer per model | PC (160 GB free) |
| TP probes (~80 runs) | 5-15 min each | PC |
| Prosody, DPR, closure script, syllables | CPU | Mac |
| RP sweep (~300 runs) | 30-60 min each: 150-300 GPU-h | Ada (2-4 days on 4 GPUs); the PC alone would need 1-2 weeks non-stop |

Ada is needed only for the RP sweep, which doubles because of the second split protocol.

## 5. Division of work

- Mohit: data pipeline, transcription, probes, RQ0 and RQ1, feature extraction.
- Nidhi: Ada runs (rebuild check, RP sweep), Praat measurements and the 50-token audit, DPR and
  syllable analysis, prosody (RQ2, RQ3).
- Both: baselines, checking each other's results, the report.

## 6. Risks

| Risk | Plan |
|---|---|
| Speaker-disjoint scores collapse to near chance for all probes | That is itself the main finding; the order and layer analyses still run, with power estimated from the CIs |
| ZIPA needs k2/CUDA | Runs on the PC and Ada only; W2V2P works everywhere |
| Ada queue | RP sweep starts mid-November; 3 seeds instead of 5 for non-anchor variants if late |
| WSL window closed on the PC | Long PC jobs are checkpointed and resumable |
| Vaani-Hi access never arrives | Not needed; if it arrives, run the anchor reproduction on it as well |
