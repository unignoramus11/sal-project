"""Inputs for transcript probes.

Two tokenisations:
- "char": PRiSM's CharacterTokenizer. Vocabulary = <pad>, <unk>, <s>, </s>, then the characters of the
  training transcripts in sorted order; unknown characters map to <unk>; transcripts are cut to 512
  characters and an empty transcript becomes "?".
- "phone": phones from `sal.phones.segment` applied to the same string, one rule for both recognizers
  (wav2vec2 tokens are eSpeak phonemes, ZIPA tokens are characters, so raw tokens are not comparable);
  same special tokens, same 512 cap.

Order manipulations:
- shuffle: phones (or characters) permuted within each utterance; a new permutation every training
  step, a fixed seeded permutation per utterance for evaluation.
Bag features (order-free inputs for an MLP): relative frequencies of unigrams, or of unigrams plus
bigrams (bigrams kept if seen at least `min_bigram` times in train).
"""
from __future__ import annotations

from collections import Counter

import numpy as np

from sal.phones import segment
import torch
from torch.nn.utils.rnn import pad_sequence

SPECIALS = ("<pad>", "<unk>", "<s>", "</s>")
PAD, UNK = 0, 1
MAX_LEN = 512


def units(transcript: str, tokens: list[str], mode: str) -> list[str]:
    if mode == "char":
        t = transcript if transcript else "?"
        return list(t[:MAX_LEN])
    if mode == "phone":
        t = segment(transcript) or ["?"]
        return t[:MAX_LEN]
    raise ValueError(mode)


def build_vocab(sequences: list[list[str]]) -> dict[str, int]:
    """PRiSM order: specials, then sorted symbols seen in training data."""
    c = Counter()
    for s in sequences:
        c.update(s)
    vocab = {t: i for i, t in enumerate(SPECIALS)}
    for sym in sorted(c):
        if sym not in vocab:
            vocab[sym] = len(vocab)
    return vocab


def encode(seq: list[str], vocab: dict[str, int]) -> np.ndarray:
    return np.array([vocab.get(s, UNK) for s in seq], dtype=np.int64)


class SequenceBatcher:
    """Pads token-id sequences; optional within-utterance shuffling."""

    def __init__(self, seqs: list[np.ndarray], targets: np.ndarray, shuffle: bool = False, seed: int = 0):
        self.seqs, self.targets, self.shuffle = seqs, torch.as_tensor(targets, dtype=torch.float32), shuffle
        self.rng = np.random.default_rng(seed)
        # fixed evaluation permutations, one per example, independent of batch composition
        self.eval_perm = None
        if shuffle:
            g = np.random.default_rng(seed + 1)
            self.eval_perm = [g.permutation(len(s)) for s in seqs]

    def __call__(self, idx: np.ndarray, train: bool):
        out = []
        for i in idx:
            s = self.seqs[i]
            if self.shuffle and len(s) > 1:
                s = s[self.rng.permutation(len(s))] if train else s[self.eval_perm[i]]
            out.append(torch.from_numpy(s))
        lengths = torch.tensor([len(s) for s in out])
        return pad_sequence(out, batch_first=True, padding_value=PAD), lengths, self.targets[idx]


def bag_features(seqs: list[np.ndarray], vocab_size: int, train_idx: np.ndarray, bigrams: bool,
                 min_bigram: int = 5) -> tuple[np.ndarray, dict]:
    """Relative unigram (and bigram) frequencies per utterance; bigram vocabulary from train only."""
    uni = np.zeros((len(seqs), vocab_size), dtype=np.float32)
    for i, s in enumerate(seqs):
        if len(s):
            np.add.at(uni[i], s, 1.0 / len(s))
    info = {"unigrams": vocab_size, "bigrams": 0}
    if not bigrams:
        return uni, info
    c = Counter()
    for i in train_idx:
        s = seqs[i]
        c.update(zip(s[:-1].tolist(), s[1:].tolist()))
    keep = {bg: j for j, bg in enumerate(sorted(b for b, n in c.items() if n >= min_bigram))}
    bi = np.zeros((len(seqs), len(keep)), dtype=np.float32)
    for i, s in enumerate(seqs):
        pairs = list(zip(s[:-1].tolist(), s[1:].tolist()))
        if pairs:
            for p in pairs:
                j = keep.get(p)
                if j is not None:
                    bi[i, j] += 1.0 / len(pairs)
    info["bigrams"] = len(keep)
    return np.concatenate([uni, bi], axis=1), info


class DenseBatcher:
    def __init__(self, x: np.ndarray, targets: np.ndarray):
        self.x = torch.as_tensor(x)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __call__(self, idx: np.ndarray, train: bool):
        return self.x[idx], torch.ones(len(idx), dtype=torch.long), self.targets[idx]
