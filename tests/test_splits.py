import numpy as np
import pandas as pd
import pytest

from sal.data.splits import leakage, speaker_disjoint_splits, split_report


@pytest.fixture
def corpus():
    """Skewed speakers per pincode, as in Vaani: many light speakers, a few heavy ones."""
    rng = np.random.default_rng(3)
    rows = []
    for g, n_spk in [(1, 40), (2, 12), (3, 5), (4, 3), (5, 90)]:
        for s in range(n_spk):
            n = int(rng.zipf(1.8)) if s else 120  # one heavy speaker per pincode
            n = min(n, 300)
            rows += [{"pincode": g, "speaker": f"g{g}s{s}"}] * n
    # one speaker who recorded in two pincodes (mostly in 1)
    rows += [{"pincode": 1, "speaker": "mover"}] * 7 + [{"pincode": 2, "speaker": "mover"}] * 2
    return pd.DataFrame(rows)


def test_no_speaker_in_two_splits(corpus):
    corpus["split"] = speaker_disjoint_splits(corpus)
    assert corpus.groupby("speaker")["split"].nunique().max() == 1
    assert leakage(corpus, "split") == {"val": 0.0, "test": 0.0}


def test_every_pincode_in_every_split(corpus):
    corpus["split"] = speaker_disjoint_splits(corpus)
    present = corpus.groupby("pincode")["split"].nunique()
    assert (present == 3).all()


def test_shares_close_to_target_for_large_groups(corpus):
    corpus["split"] = speaker_disjoint_splits(corpus)
    rep = split_report(corpus, "split")
    big = rep.loc[[1, 5]]
    assert ((big[("share", "train")] - 0.75).abs() < 0.12).all()


def test_deterministic(corpus):
    a = speaker_disjoint_splits(corpus, seed=7)
    b = speaker_disjoint_splits(corpus, seed=7)
    c = speaker_disjoint_splits(corpus, seed=8)
    assert (a == b).all() and not (a == c).all()


def test_random_split_leaks(corpus):
    rng = np.random.default_rng(0)
    corpus["rand"] = rng.choice(["train", "val", "test"], p=[0.75, 0.1, 0.15], size=len(corpus))
    assert leakage(corpus, "rand")["test"] > 0.5


def test_folds_disjoint_and_balanced(corpus):
    from sal.data.splits import speaker_folds
    corpus["fold"] = speaker_folds(corpus, k=5)
    assert corpus.groupby("speaker")["fold"].nunique().max() == 1
    assert set(corpus["fold"]) == set(range(5))
    for _, g in corpus[corpus["pincode"].isin([1, 5])].groupby("pincode"):
        heaviest = g["speaker"].value_counts(normalize=True).iloc[0]   # a fold holds at least its heaviest speaker
        assert g["fold"].value_counts(normalize=True).max() < max(heaviest, 0.2) + 0.12
