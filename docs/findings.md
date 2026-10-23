# Findings so far (18 Oct 2026)

Every number below comes from `results/` (tables: `results/tables/configs.csv`, `comparisons.csv`,
`results/baselines.csv`, `results/dpr/`, `results/mdl/summary.csv`; regenerate with `scripts/summarise.py`,
`scripts/baselines.py`, `scripts/dpr.py`, `scripts/mdl.py`). Recall@1 (R@1) values are seed means ± sd (5
seeds) unless marked "ens" (seed-ensemble prediction, with a 95% speaker-cluster bootstrap CI in brackets).
TP-GRU and representation-probe results are still running; this file is updated as they land.

## Data

- Rebuilt Vaani-Hi with PRiSM's recipe (Vaani commit 9e6ed4f9): 22,551 utterances, 40.3 h, 26 pincodes,
  9 states, 1,570 speakers (speaker code from the file name).
- PRiSM's random split puts the speaker of 97.9% of test utterances in train; our speaker-disjoint split
  (pincode-stratified) has 0%.
- 68% of utterances describe a district-specific picture (named after the speaker's own district); 32%
  describe shared ("generic") pictures. The generic share varies by state from 12% (Madhya Pradesh) to 58%
  (Chhattisgarh), so the shared-picture subset has its own location prior: its majority location covers
  ~20% of test utterances against ~10% overall. Scores on that subset are therefore reported as lift over
  the majority baseline (R@1) and gain over the best constant location (km), on the same utterances.

## The official GEO-v release (access 21 Oct 2026; details in docs/official_data.md)

- Reproduction: PRiSM's char TP-GRU with our code on the official data and split scores 45.5 +- 1.1 Recall@1
  (5 seeds; PRiSM: 45.7). The probe reproduction holds; our lower rebuild scores come from the data.
- The release duplicates audio: 26,385 rows, 21,481 distinct recordings; 38% of duplicated recordings carry
  different pincodes in different rows (a file-name collision in PRiSM's sampling script). 28.0% of test rows
  have their byte-identical audio in training.
- Pincode 171202 (Shimla, the largest class) is placed in Afghanistan, 844 km off (unfiltered mean of post offices,
  three with latitude and longitude swapped).
- Memorisation solves the benchmark: a nearest-neighbour transcript matcher with no location learning scores
  58.9 Recall@1 on the official test set, phone bigrams 59.6 +- 0.7, and ECAPA speaker embeddings (voice only, no
  words) 55.1 +- 1.4, all above PRiSM's TP-GRU (45.5).
- On a corrected version (deduplicated, consistent labels, fixed coordinates) the matcher still scores 57.5 under
  the released split but 9.6 [6.4, 13.6] under a pseudo-speaker-disjoint split (majority 6.8), and so do the probes:
  phone bigrams 57.3 -> 8.7, ECAPA 54.0 -> 3.4. Speaker overlap, not duplicate audio, carries the scores.
  ZIPA transcripts behave the same: phone bigrams 59.5 +- 1.0 on the released split, 59.8 on the corrected rows,
  9.2 +- 1.3 on the pseudo-speaker split.
- Phone order is speaker memorisation: on the corrected rows, PRiSM's phone BiGRU beats the same probe on
  shuffled phones by +18.6 Recall@1 [15.5, 22.3] when speakers are shared and by -0.3 [-3.5, 2.4] when they are
  not (paired, pseudo-speaker bootstrap; 3 seeds each). Bigram bags over unigram bags: +9.2 to +11.4 with shared
  speakers, +0.2 to +1.8 (all CIs include 0) on new speakers, for both recognizers. Speaker leakage in the release, estimated from calibrated ECAPA
  embeddings, is at least the rebuild's 98%.

## RQ0: speaker leakage

| Reference (test) | random split | speaker-disjoint |
|---|---|---|
| Speaker lookup (speaker ID only) | 98.0 | 10.0 (= majority) |
| Majority location | 10.5 | 10.0 |
| Best constant location, mean km | 601 | 607 |
| ECAPA speaker embedding, R@1 | 41.6 ± 1.6 | 10.1 ± 0.6 |
| ECAPA, mean km (ens) | 169 | 435 |

- Under PRiSM's split the task is almost solvable from the speaker identity alone (98.0 R@1).
- ECAPA, which encodes voice and channel and is trained to ignore content, reaches 41.6 R@1 under the random
  split and falls to the majority level (10.1; lift +0.4 [-6.2, 6.3]) on unseen speakers (prediction P4).
  It still beats the best constant location by 172 km [96, 245] on unseen speakers, so voice and channel
  carry some coarse regional information that transfers across speakers, but not enough to pick a pincode.
- Control task (each speaker labelled with another speaker's pincode, so labels carry speaker identity but no
  dialect): ECAPA reaches 33.2 ± 0.8 R@1 on control labels under the random split (control majority 14.0),
  against 41.6 on real labels (majority 10.5). Most of ECAPA's random-split score is speaker recognition.
  Under the speaker-disjoint split its control score is 8.5, below the control majority (16.2), as expected.
  Transcript bag probes (bigrams, both recognizers, chars and phones) do not recognise speakers by Recall@1:
  control R@1 10.1-16.0 under the random split against a control majority of 14.0 (lifts -4.0 to +1.9). In
  km, with the speaker-disjoint control runs as the no-speaker-information reference, the random split
  lowers the control error by 29-47 km for bigrams (e.g., ZIPA phone bigram 515 -> 470 km) and by 308 km for
  ECAPA (556 -> 248 km). Trained regression probes on transcripts exploit little speaker identity; the same
  holds for PRiSM's BiGRU (W2V2 phones, random split: control R@1 10.6 +- 0.8, 2 seeds, below the control
  majority 14.0; ZIPA 10.2 +- 0.4).
- Transcripts are nonetheless speaker-identifiable. A 1-nearest-neighbour matcher (TF-IDF over character
  1-4-grams, no location learning; scripts/knn_transcripts.py) reaches 39.6 R@1 [37.1, 42.0] on W2V2
  transcripts under the random split (ZIPA 39.0) and 18.2 [14.5, 22.2] under the speaker-disjoint split
  (ZIPA 22.6). For 25% of random-split test utterances the nearest training transcript comes from the same
  speaker (chance 0.3%; 8.6% even for a matcher that already knew the pincode). Speakers repeat recognisable
  content across utterances; a memorising matcher turns that into +21 R@1 of leakage, while the trained
  probes (bigram MLP 27.9, TP-GRU ~21) stay well below it. A TP that fits its training set more tightly
  than ours would move toward the matcher, which is one candidate explanation for PRiSM's 45.7.
  Leakage grows with how many training utterances the test speaker has (W2V2, 1-NN, random split): speaker
  unseen 15.3 R@1 (72 test utterances), 1-5 training utterances 22.1, 6-20 34.1, 21-50 45.0, more than 50
  46.3, with the same-speaker share of nearest neighbours rising from 0 to 6%, 19%, 31% and 32% (ZIPA alike;
  results/tables/knn_by_speaker_count.csv).
- MDL: under the random split the bag probes save 15.7-21.0 kbits over the location prior, against 5.0-7.9
  kbits on unseen speakers (ZIPA phone bigram 21.0 ± 0.3 vs 7.9 ± 1.0): leakage makes location about three
  times cheaper to code.
- Transcript bag probes also drop between splits, by less than ECAPA:

| Probe (W2V2 transcripts) | random split | speaker-disjoint | lift over majority, speaker-disjoint |
|---|---|---|---|
| Phone bigram | 27.9 ± 0.8 | 17.8 ± 1.7 | +7.9 [2.6, 13.9] |
| Char bigram | 26.8 ± 0.5 | 14.3 ± 1.6 | +4.4 [-0.7, 10.2] |
| Phone unigram | 15.5 ± 1.1 | 12.6 ± 0.7 | +3.1 [-1.0, 8.3] |
| Char unigram | 14.5 ± 0.9 | 12.4 ± 1.1 | +2.8 [-1.2, 7.6] |

| Probe (ZIPA transcripts) | random split | speaker-disjoint | lift over majority, speaker-disjoint |
|---|---|---|---|
| Phone bigram | 28.6 ± 0.8 | 21.3 ± 1.0 | +10.6 [6.2, 15.9] |
| Char bigram | 27.9 ± 0.8 | 21.5 ± 0.6 (2 seeds) | +11.5 [6.2, 17.0] |
| Phone unigram | 15.2 ± 0.5 | 12.7 ± 1.0 | +2.9 [-0.9, 7.2] |
| Char unigram | 14.5 ± 0.9 | 12.5 ± 1.3 | +3.2 [-0.7, 7.6] |

- Topic check (P1). The clean test trains and tests on shared-picture utterances only and compares with that
  subset's own baselines (majority R@1 21.3, state accuracy 38.6, best constant 537 km). Speaker-disjoint:
  W2V2 phone bigram R@1 lift +1.1 [-7.3, 8.2], state lift +4.4 [-5.4, 15.5], km gain +94 [7, 186]; W2V2 phone
  unigram +76 km [1, 152]; ZIPA phone bigram R@1 lift +2.6 [-5.1, 9.7], state lift +5.9 [-6.6, 19.1], km gain
  +84 [-14, 188]; ZIPA phone unigram +48 km [-20, 117]; ECAPA +129 km [1, 272]. On all
  utterances the same probes are far above baseline (ZIPA phone bigram: R@1 lift +10.6 [6.2, 15.9], correct
  state 51.0% vs 18.4%, lift +32.7 [21.5, 43.9]). Probes trained on all utterances and scored on the
  shared-picture subset are not a clean test: the transcript reveals that the picture is a shared one, and
  the share of shared pictures differs by state (12-58%), so the prompt type itself moves the location prior.
  Reading: most of the speaker-disjoint signal of the bag probes comes from what speakers talk about; what
  survives without topic is at most a coarse regional signal (transcript km gains of 48-94 km, only the two
  W2V2 ones with CIs above zero; R@1 and state accuracy not significant). The subset has 1,222 test utterances from 119 speakers, so
  the CIs are wide; state accuracy: results/tables/state_level.csv.

## RQ1: inventory, local order, units

- TP-GRU (2 seeds per condition, seeds 44-46 on Ada; PRiSM's probe and settings). Phone BiGRU minus shuffled phone
  BiGRU (paired, speaker bootstrap): random split W2V2 +3.9 R@1 [1.9, 5.9] (-53 km), ZIPA +4.3 [2.5, 6.0] (-76 km);
  new speakers W2V2 -3.5 [-6.5, -1.0] (km -23 [-53, 4], n.s.), ZIPA +0.4 [-1.9, 2.9] (km -43 [-83, -9]; 2 seeds
  each). Order helps the BiGRU under the random split with both recognizers; on new speakers it brings no
  Recall@1 gain with either (W2V2 loses), and only ZIPA's km improves. Whether the random-split gain is speaker
  memorisation (see the nearest-neighbour result in RQ0) is open until seeds 44-46 (Ada) are in.
  MDL agrees (W2V2 phones, 3 speaker orderings; saving over the location prior): random split BiGRU 21.0 +- 0.4
  kbits, shuffled BiGRU 17.2 +- 0.1, bigram 19.4 +- 0.1; new speakers 6.4 +- 1.8, 5.5 +- 1.4, 6.6 +- 1.1. Paired by
  ordering, order saves 3.4-4.2 kbits on the random split but 0.4-1.6 on new speakers, and the BiGRU beats the
  bigram bag by 1.3-2.0 kbits on the random split but not on new speakers (-0.9 to +0.6).
  The BiGRU is below the bigram bag on both splits (W2V2 phone GRU vs phone bigram -5.7 [-7.4, -4.0] random,
  -6.1 [-9.9, -3.1] speaker-disjoint) and far below PRiSM's 45.7 on the official data; see
  docs/prism_reproduction.md (reproduction status).

All on the speaker-disjoint split, paired on the same test utterances (speaker-cluster bootstrap):

- Local order helps: W2V2 phone bigram vs phone unigram +4.8 R@1 [2.6, 6.9], -33 km [-51, -15].
- Phone units vs raw characters: W2V2 phone bigram vs char bigram +3.5 R@1 [2.1, 5.1] (km n.s.); ZIPA phone
  vs char bigram -0.9 R@1 [-3.3, 1.3] but -20 km [-28, -11].
- Recognizer: ZIPA phone bigram vs W2V2 phone bigram +2.7 R@1 [-0.1, 5.8] (p = 0.06), -33 km [-55, -12].
- MDL (online coding of the pincode, speaker-ordered blocks, temperature-calibrated, 3 orderings): every bag
  probe compresses better than the location prior (prior 78.8 kbits; uniform 78.7). Savings over the prior:
  ZIPA phone bigram 7.9 ± 1.0 kbits, ZIPA char bigram 7.2 ± 0.9, W2V2 phone bigram 6.6 ± 1.1, W2V2 char bigram
  6.6 ± 1.1, unigrams 5.0-6.0. Paired by ordering, ZIPA phone bigram saves 1.37 kbits over W2V2 phone bigram
  (1.21-1.52 across the 3 orderings), bigrams save 1.6-2.0 kbits over unigrams, and phone units save 0.71
  kbits over characters for ZIPA but nothing for W2V2 (0.02), which fits ZIPA's richer diacritic output: its
  phone units group modifiers that the character tokenisation splits.

## RQ2: gemination (doubled-phone rate)

- W2V2 transcripts: Haryana has the lowest doubled-phone rate of the 9 states, 1.00% of consonants vs 1.17%
  elsewhere, difference -0.17 pp [-0.32, -0.04], p = 0.012 (speaker bootstrap; Haryana has 12 speakers).
- ZIPA transcripts: no difference, 1.63% vs 1.82%, -0.19 pp [-0.53, 0.19], p = 0.30. ZIPA doubles about 1.6x
  as often overall, and its state ranking differs (Rajasthan highest at 3.6%).
- So doubling in recognizer output depends on the recognizer; it cannot by itself establish a dialect feature.
- Acoustics (Rohtak, Jhajjar, Charkhi Dadri vs New Delhi Vaani audio; MMS alignment of the standard spelling,
  independent of the recognizers; results/gemination/). 39,008 candidate single voiceless stops, 85.5% measured
  (Haryana 83.8%, New Delhi 85.4% in the main contexts). Main analysis: stops between vowels in speech, words
  with >= 20 tokens; mixed models with word fixed effects, following-vowel duration as rate control, speaker
  random intercepts.
  - Voiceless interval (voicing offset to voicing onset, the primary measure): Haryana x1.142 [1.106, 1.178],
    p < 1e-15 (26,033 tokens, 62 words); without the rate control x1.133 [1.098, 1.169]; including stops after
    a deleted schwa x1.141 [1.108, 1.175]. Lee bounds for the small difference in measurement rates: x1.070 to
    x1.118 on the raw difference (x1.096).
  - Closure (voicing offset to burst): x1.256 [1.208, 1.307] on all tokens with a burst; x1.269 [1.215, 1.324]
    on clear bursts only (9,340 tokens).
  - Validity: lexical geminates (अच्छा, बच्चे, मिट्टी, ...; 2,971 tokens) have voiceless intervals x1.59
    [1.51, 1.67] those of single stops in New Delhi, so the measure tracks consonant length. On the log scale,
    Haryana single stops are lengthened by 28% of the geminate-single difference; the geminate-single gap is
    slightly smaller in Haryana (interaction -0.054, p = 0.049).
  - Robustness (scripts/gemination_robustness.py, results/gemination/robustness.txt): every Haryana district
    differs from New Delhi on its own (Charkhi Dadri x1.132 [1.090, 1.176], Jhajjar x1.153 [1.109, 1.197],
    Rohtak x1.140 [1.099, 1.183]); Haryana is longer in 45 of 57 words (sign test p = 1.3e-5, median x1.095);
    speaker-level means (word-centred, 2,287 vs 716 speakers) give x1.140 [1.105, 1.177] without any model;
    per-word effects pooled by random-effects meta-analysis (53 words) give x1.110 [1.081, 1.139], with modest
    between-word variation (tau 0.049 on the log scale, I2 31%).
  - Reading: Haryana speakers produce longer voiceless stops between vowels than New Delhi speakers, by 14%
    (voiceless interval) to 26% (closure), the direction PRiSM's attribution example suggests. It is a
    lengthening, not a categorical doubling. Still to do: the blind manual audit of 50 tokens
    (internal/audit/), and whether the recognizers' doubling tracks the measured length (needs the PC's
    district transcripts).
- Recognizer doubling vs measured length (results/gemination/model_doubling.csv; W2V2 transcribed on the Mac
  with transformers, frame-identical to PRiSM's runner; ZIPA on the PC). Inside each token's window W2V2 writes
  a doubled consonant for 0.7% [0.4, 0.9] of New Delhi and 1.0% [0.8, 1.1] of Haryana single stops, and its
  doubled tokens are longer (Haryana median voiceless interval 125 vs 80 ms; AUC of length for predicting
  doubling 0.73 Haryana, 0.64 New Delhi). ZIPA doubles 1.7% [1.3, 2.2] and 1.9% [1.7, 2.1], unrelated to length
  (AUC 0.46). So the lengthening is gradient and mostly below the threshold at which a recognizer writes a double
  consonant: only W2V2's rare doubling follows it, and ZIPA's does not. (An earlier run counted a vowel's length
  mark cut off at the window start as a doubled consonant; fixed in sal.phones, Vaani-Hi DPR unchanged.)
- A first version of the measurement made burst strength and intensity drop hard filters; it measured only 31%
  of tokens and rejected Haryana tokens more often (75% vs 65%). It is not used (docs/design.md section 8).

## RQ3: prosody

- Pooled utterance-level prosody statistics (12 within-utterance-normalised F0/intensity statistics, MLP):
  below the majority R@1 under both splits (random 5.5 ± 0.7, speaker-disjoint 4.8 ± 0.4) and close to the
  best constant on km (speaker-disjoint +12 km [-31, 53]). Speaker-normalised global prosody statistics
  carry no usable location signal. The sequence model (prosody BiGRU on contours) is queued on the GPU.

## Open
- TP-GRU (char, phone, shuffled phone; both recognizers; both splits; 5 seeds) running on the PC.
- Representation probes, layer sweep and the layer-wise geminate probe: on Ada (Nidhi).
- Gemination acoustics and the blind manual audit of 50 tokens.
