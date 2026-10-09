# Compute and storage

## Hardware available
- Ada (IIIT Hyderabad HPC, SLURM): nodes with four 11 GB GPUs (GTX 1080 Ti or RTX 2080 Ti),
  128 GB RAM, a few TB node-local scratch. (Specs from the team's ANLP project notes.)
- Mohit's laptop: RTX 4050, 6 GB VRAM.
- Mohit's MacBook (Apple silicon): fine for Praat, Parselmouth, metadata checks, writing.
- Renting a GPU: not needed unless Ada's queue blocks the RP sweep in late November. One 24 GB card
  for a day covers the whole sweep once features are cached.

## What runs where
| Job | Where | Cost |
|---|---|---|
| Phone transcripts, TP probes (~30 runs), TP-Shuffle, DPR | Laptop | ~15 min per TP run |
| Praat, Parselmouth, syllabification | Laptop / Mac CPU | small |
| Feature extraction, 3 models over Vaani-Hi | Ada, 1 GPU per model | < 1 h per model |
| RP sweep: 5 variants x 2 layer settings x 3 models x 5 seeds ≈ 150 runs | Ada | ~75-150 GPU-h; 2-4 days wall-clock on one node |

Everything for the 3 Nov mid evaluation fits on the laptop.

## Storage maths
- Vaani-Hi is 6.61 GB of parquet holding WAV bytes. If the WAVs are 32-bit float: ~29 h audio (~3.9 s/utt).
  If 16-bit: ~57 h (~7.8 s/utt). `scripts/vaani_check.py` reads the WAV header and settles this.
- wav2vec2-large: 25 hidden-state outputs (embedding + 24 layers), 1024-d, 50 frames/s.
  One layer over Vaani-Hi in fp16: ~10.6 GB (29 h) to ~21 GB (57 h). All 25 layers: ~265-525 GB. Do not cache all.
- Whisper-small: 13 outputs, 768-d, 50 frames/s (trim the 30 s padding to real length).
- Plan: time-average every layer into one vector per utterance (~1.3 GB per model), run a cheap layer sweep,
  then cache full frames only for the last layer + 2-3 best layers (~30-60 GB per model).
- The geminate frame probe only needs frames around ~400 tokens, so it is tiny at every layer.
- 128 GB node RAM can hold one model's cached features, so training never waits on disk.

## Ada notes
- Prefer 2080 Ti nodes; 1080 Ti is slow in fp16 (compute in fp32 there, store fp16).
- Node-local scratch may be wiped after a job: copy features to persistent storage before the job ends.
- Hugging Face token needed on the cluster for both gated datasets; download where internet is available.
- 6 GB laptop: RP-Transformer over 1000-frame sequences fits at batch ~8 in fp16.
