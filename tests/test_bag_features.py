import numpy as np

from sal.probes import tp_data as T


def test_bag_matrix_with_saved_bigrams_matches_training_features():
    g = np.random.default_rng(0)
    seqs = [g.integers(4, 30, g.integers(0, 60)) for _ in range(300)]
    train = np.arange(200)
    x, info = T.bag_features(seqs, 30, train, bigrams=True)
    keep = T.bigram_vocab(seqs, train)
    assert info["bigrams"] == len(keep)
    # what scripts/infer.py does: the bigram list stored in the checkpoint, applied to new utterances
    assert np.array_equal(T.bag_matrix(seqs, 30, [tuple(b) for b in keep]), x)
    assert np.array_equal(T.bag_matrix(seqs, 30, None), T.bag_features(seqs, 30, train, bigrams=False)[0])
