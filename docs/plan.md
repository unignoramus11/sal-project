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

## 3. Work plan

### Phase 0: data (17-19 Oct)
- [x] Rebuild script, tests (splits identical to PRiSM's code), pinned Vaani commit.
- [ ] Finish download; assemble; manifest with counts and checksum.
- [ ] Dataset report: utterances, hours, pincodes, speakers per pincode, utterances per speaker,
      leakage under the random split, language labels, gender. Figure: map of pincodes.
- [ ] Speaker-disjoint split fixed and saved (seed 42); split report.
- [ ] Rebuild rerun on Ada with Nidhi's login: checksum must match (reproducibility).

### Phase 1: transcripts and transcript probes (19-27 Oct)
- [ ] PRiSM inference environment on the PC; transcribe with W2V2P-LV60 and ZIPA-CTC-NS; first 50
      utterances must equal PRiSM's runner output.
- [ ] Training loop (Adam, LinearLR warm-up, clip 1.0, 4 validations per epoch, early stopping on val km,
      best checkpoint); predictions saved for every run.
- [ ] Baselines: train-centroid and nearest-majority-location (naive), per-state prior.
- [ ] TP-GRU, character tokens (PRiSM anchor) and phone tokens; random and speaker-disjoint; 5 seeds.
- [ ] TP-Shuffle (phones), TP-Bigram, TP-Unigram (inventory only).
- [ ] DPR per utterance and per state from both recognizers.
- [ ] Speaker baseline: ECAPA embeddings + MLP, both splits.

### Phase 2: representations (24 Oct-2 Nov)
- [ ] Feature extraction (PC): time-averaged vector for every layer of W2V2P, ZIPA, Whisper-small;
      full frames (fp16) for the last layer.
- [ ] Layer sweep on pooled vectors (validation only); pick 2-3 layers per model; cache their frames.
- [ ] RP-Attn (PRiSM anchor) on the last layer, both splits, on the PC.

### Mid evaluation (3 Nov)
Dataset and leakage report; TP baselines and RP-Attn under both splits; TP-Shuffle/Bigram; speaker
baseline; DPR by state; layer sweep curves.

### Phase 3: main experiments (4-17 Nov)
- [ ] RP sweep on Ada: RP-Attn, +PE, GRU, Transformer, ChunkShuffle × {last, weighted selected layers}
      × 3 models × 5 seeds × 2 splits (SLURM array; scripts and guide in `slurm/`, run by Nidhi).
- [ ] Gemination: download the four Vaani districts' transcribed audio; word alignment with the
      MMS forced aligner (romanised Devanagari); closure-duration script on intervocalic voiceless stops;
      50 tokens hand-measured by both members (agreement, script error); Haryana vs Delhi model with
      speaker as a random effect.
- [ ] Frame-level geminate probe at every layer; integrated gradients on the Haryana test set.

### Phase 4: prosody, depth, statistics (18-28 Nov)
- [ ] Parselmouth F0 (semitones re utterance median), intensity, voicing; prosody BiGRU; late fusion.
- [ ] Syllabification and closed-syllable share.
- [ ] Minimum description length (online coding, Voita and Titov 2020) for the key TP and RP
      comparisons, as a check that accuracy differences are not probe-capacity effects.
- [ ] Cluster bootstrap over speakers for all CIs and paired tests; control tasks.

### Phase 5: writing (28 Nov-4 Dec)
- [ ] ACL-format report; every number generated from `results/`; code and data-build release.

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
