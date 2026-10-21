# The official Vaani-Hi (GEO-v) release

`changelinglab/vaani-hi-geo`, commit `264fb053` (8 Jan 2026), downloaded 21 Oct 2026 to `data/vaani_hi_official/`.
Every number below is produced by `scripts/official_audit.py` (results in `results/official/`) unless noted.

## Composition

26,385 rows (train 19,777, val 2,623, test 3,985), 30 pincodes, 20 districts, 10 states, 16 kHz, mean 4.5 s.
It differs from our rebuild (22,551 utterances, 26 pincodes, 9 states): PRiSM's recipe draws 4 districts per state
with `random.sample` over the file system's listing order, which we could not reproduce, so the two samples share
few districts. The release includes four Uttar Pradesh districts (about 6,800 rows) and Chandigarh.

## Findings

1. **Duplicated audio with conflicting labels.** The 26,385 rows hold 21,481 distinct clips; 4,559 clips appear in
   2-4 rows (9,463 rows). Among duplicated clips, 38.4% of the copies carry different pincodes and 33.6% different
   genders; duplicates never cross districts.
   Cause (PRiSM's `src/recipe/geolocation/local/vaanihindi_sample.py`): each district's 4 shards write their first 600
   clips to `WAVS/{state}_{district}_{i}.wav` with `i` restarting at 0 for every shard, so the shards overwrite each
   other's files, while the metadata keeps one row per shard and index; rows with the same index then point to
   whichever file was written last, under the labels of their own shard. The pincode filter (> 450 rows) removes
   some of the rows, which is why most clips appear once or twice rather than four times.
   Consequence: many rows carry the audio of one clip and the labels (pincode, gender) of another clip of the same
   district.
2. **Exact test audio in training.** 28.0% of test rows (and 27.3% of validation rows) have byte-identical audio in
   the training set; in 64.1% of these the training copy has the same pincode. A lookup that only recognises
   identical recordings scores 22.7 Recall@1 against a majority baseline of 7.6.
3. **A misplaced pincode.** Pincode 171202 (Shimla, 2,014 rows, the largest class) is released at 36.94°N
   71.54°E, in Afghanistan, 844 km from its post offices. The release equals the unfiltered mean of the 16 India
   Post offices of that pincode, 3 of which have latitude and longitude swapped. All other pincodes are within
   64 km of the median of their offices. Mean-km metrics on GEO-v are dominated by this point (the majority
   baseline's mean error is 1,384 km).
4. **Speakers.** The release has no speaker column. Clips matched sample-for-sample to our cached Vaani shards
   (1%, 271 rows) all have their speaker in training when they are in val/test. Recovering the source of every
   row is in progress (`scripts/official_sources.py`: metadata-only scan of all 1,181 shards of the 20 districts,
   matched by duration within district).

## Why exact matching to Vaani is limited

The Vaani revision PRiSM sampled from (on or before 22 Dec 2025) was later re-sharded (Chandigarh: 47 shards then,
61 now) and its files are no longer downloadable (HTTP 403 on the old LFS objects). The same clips exist at the
current revision in other shards, so sources are recovered from metadata (district + duration) rather than by
shard and row.

## Plan

- PC: W2V2 transcription of the release with PRiSM's runner, PRiSM's char TP-GRU on the released split (5 seeds),
  scored on the full test set and on test rows whose audio is not in training (`tools/pc_official.sh`).
- A corrected version once sources are recovered: one row per clip, labels from the Vaani source, valid
  coordinates, and speaker-disjoint splits; the same probes on it.
- Representation probes on the release and the corrected version: Ada (Nidhi; docs/ada.md).
