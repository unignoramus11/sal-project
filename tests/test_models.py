import torch

from sal.probes import models as M


def _seq(b=4, t=37, d=64):
    torch.manual_seed(0)
    x = torch.randn(b, t, d)
    lengths = torch.tensor([37, 20, 5, 1])
    return x, lengths


def test_shapes():
    x, lengths = _seq()
    ids = torch.randint(4, 30, (4, 37))
    for m, inp in [(M.TPGRU(30), ids), (M.RPAttn(64), x), (M.RPAttn(64, pe=True), x),
                   (M.RPGRU(64, 32, 32), x), (M.RPTransformer(64, 32), x)]:
        assert m(inp, lengths).shape == (4, 3)
    stacked = torch.randn(4, 3, 37, 64)
    assert M.RPAttn(64, n_layers=3)(stacked, lengths).shape == (4, 3)


def test_padding_does_not_change_output():
    """Values beyond each row's length must not affect any probe (eval mode)."""
    x, lengths = _seq()
    y = x.clone()
    for b, n in enumerate(lengths.tolist()):
        y[b, n:] = 1e3 * torch.randn_like(y[b, n:])
    ids = torch.randint(4, 30, (4, 37))
    ids2 = ids.clone()
    for b, n in enumerate(lengths.tolist()):
        ids2[b, n:] = 7
    for m, a, c in [(M.TPGRU(30), ids, ids2), (M.RPAttn(64), x, y), (M.RPGRU(64, 32, 32), x, y),
                    (M.RPTransformer(64, 32), x, y)]:
        m.eval()
        assert torch.allclose(m(a, lengths), m(c, lengths), atol=1e-5), type(m).__name__


def test_attention_pool_is_order_free():
    """RPAttn is invariant to permuting valid frames; RPGRU is not."""
    x, lengths = _seq()
    perm = torch.randperm(37)
    xp = x.clone()
    xp[0] = x[0, perm]  # row 0 has full length
    a, g = M.RPAttn(64).eval(), M.RPGRU(64, 32, 32).eval()
    assert torch.allclose(a(x, lengths)[0], a(xp, lengths)[0], atol=1e-5)
    assert not torch.allclose(g(x, lengths)[0], g(xp, lengths)[0], atol=1e-4)


def test_shuffles_keep_multisets():
    g = torch.Generator().manual_seed(0)
    ids = torch.arange(40).reshape(4, 10)
    lengths = torch.tensor([10, 6, 1, 3])
    s = M.shuffle_within(ids, lengths, g)
    for b, n in enumerate(lengths.tolist()):
        assert sorted(s[b, :n].tolist()) == sorted(ids[b, :n].tolist())
        assert (s[b, n:] == ids[b, n:]).all()
    x = torch.arange(4 * 25).float().reshape(4, 25, 1)
    lengths = torch.tensor([25, 12, 3, 10])
    c = M.shuffle_chunks(x, lengths, chunk=5, generator=g)
    for b, n in enumerate(lengths.tolist()):
        assert sorted(c[b, :n, 0].tolist()) == sorted(x[b, :n, 0].tolist())
        out = c[b, :n, 0].tolist()
        base = x[b, 0, 0].item()
        for start in range(0, n, 5):                      # every original chunk is a contiguous,
            chunk = [base + v for v in range(start, min(start + 5, n))]   # in-order run in the output
            i = out.index(chunk[0])
            assert out[i:i + len(chunk)] == chunk


def test_prism_tp_size():
    # embedding 30x128 + 2-layer biGRU(256) + LN + Linear(512, 3)
    m = M.TPGRU(30)
    gru = sum(p.numel() for p in m.rnn.parameters())
    assert gru == 2 * (3 * 256 * (128 + 256) + 6 * 256) + 2 * (3 * 256 * (512 + 256) + 6 * 256)


def test_capacity_matching_within_10_percent():
    for dim in (768, 1024, 512):
        for kind in ("gru", "transformer", "chunkshuffle"):
            s = M.match_size(kind, dim)
            assert abs(s["params"] - s["target"]) <= 0.10 * s["target"], (kind, dim, s)
