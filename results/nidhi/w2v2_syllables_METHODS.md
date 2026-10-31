# Exploratory sonority-based syllabification

Original input: `outputs/transcripts/w2v2_nidhi.csv`. Rows supplied: 22551; with nuclei: 22551.

IPA normalized with `sal.phones.segment`; nucleus=IPA vowel, coda assigned using rising-sonority maximum legal onset; repeated consonants and length marks are treated as possible codas. Bootstrap resamples **speaker groups**, 1000 times per state. This is a heuristic, not a hand-verified syllable annotation or causal demonstration of dialect differences. Punctuation/word boundaries may be absent from processed recognizer strings.
