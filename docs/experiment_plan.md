# Experiment plan

Timeline from the proposal (Table 4), expanded into concrete steps.

## 10-16 Oct: data and features
- [x] Accept terms for `ARTPARK-IISc/Vaani` (Mohit; auto-approved).
- [ ] Vaani-Hi access (Mohit requested; manual approval pending). Nidhi to request on her account too.
- [x] Download W2V2P-LV60, ZIPA-CTC-NS (.pth) and Whisper-small checkpoints; clone PRiSM into `external/prism`.
- [x] Check Vaani district folder names and metadata (`scripts/vaani_districts.py`; results in `docs/data.md`).
- [ ] Run `scripts/vaani_check.py`; record total hours, Haryana counts per district, language labels in `docs/data.md`.
- [ ] In PRiSM, set `data.hf_repo=changelinglab/vaani-hi-geo` (its default `shikhar7ssu/vaani-hi-geo` is unreleased).
- [ ] Download Vaani folders `audio/Haryana/{Rohtak,Jhajjar,CharkhiDadri}` and `audio/Delhi/NewDelhi` (92.3 GB; keep the 55.9 h transcribed part).
- [ ] Transcribe Vaani-Hi with W2V2P-LV60 and ZIPA-CTC-NS (greedy CTC); save transcripts as JSONL keyed by metadata_idx.
- [ ] Extract last-layer features (fp16) and time-averaged vectors for every layer (layer sweep input).

## 17-26 Oct: reproduce baselines (O1)
- [ ] TP-GRU and RP-Attn for W2V2P-LV60 and ZIPA-CTC-NS with PRiSM recipes
      (`configs/experiment/probing/geolocation_vaani_{w2v2ph,zipactc,whisper}.yaml`).
- [ ] Target: within 2 Recall@1 points of PRiSM Table 4 (TP 45.7 / 44.7; RP 24.6 / 24.0; Whisper RP 35.0).

## 27 Oct - 2 Nov: first RQ1 result + DPR
- [ ] TP-Shuffle (permute phones within each utterance, train and test) and TP-Bigram.
- [ ] Doubled-phone rate per utterance and per state; map for Haryana vs neighbours.

## 3 Nov: Mid evaluation
Show: reproduced baselines, TP-Shuffle vs TP-GRU, DPR by state.

## 4-17 Nov: order-aware RP + gemination (O2, O3)
- [ ] Layer sweep on pooled vectors -> pick 2-3 layers per model; cache their frame sequences.
- [ ] RP-Attn+PE, RP-GRU, RP-Transformer, RP-ChunkShuffle; each on last layer and weighted selected layers; 5 seeds.
- [ ] Praat: ~200 doubled + ~200 single tokens (Haryana vs New Delhi), closure durations, two-annotator agreement on a shared subset.
- [ ] Frame-level geminate probe at every layer.
- [ ] Start the RP sweep on Ada by mid-November (queue fills late Nov).

## 18-30 Nov: prosody + analysis (O4)
- [ ] Parselmouth F0 (semitones re speaker median), intensity, voiced durations; BiGRU; TP, prosody, fusion.
- [ ] Integrated gradients over the Haryana test set; syllabification and closed-syllable share.
- [ ] Speaker-embedding clusters -> approximately speaker-disjoint split; shuffled-label control; paired bootstrap.

## 4 Dec: End evaluation (O5)
Report, code release, per-state error analysis.
