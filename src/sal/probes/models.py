"""Probe architectures. All return an unnormalised xyz direction of shape (B, 3).

Transcript probes take token ids (B, T) and lengths; representation probes take frame features
(B, T, D) and lengths, or stacked layers (B, L, T, D) through `LayerMix`.

TPGRU and RPAttn reproduce PRiSM's `RNNHead` + `IPAEmbedding` and `AttentionMLPHead` (see
docs/prism_reproduction.md). The other probes are ours, sized to a comparable number of parameters.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def length_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """(B, T) bool, True at valid positions."""
    return torch.arange(max_len, device=lengths.device)[None, :] < lengths[:, None]


def masked_mean(x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    m = length_mask(lengths, x.shape[1]).unsqueeze(-1).to(x.dtype)
    return (x * m).sum(1) / lengths.clamp(min=1).unsqueeze(-1).to(x.dtype)


class BiGRUEncoder(nn.Module):
    """Packed bidirectional GRU returning (B, T, 2H) padded outputs."""

    def __init__(self, input_dim: int, hidden: int, layers: int, dropout: float):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                          dropout=dropout if layers > 1 else 0.0)
        self.out_dim = 2 * hidden

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(x, lengths.cpu().clamp(min=1), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(packed)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=x.shape[1])
        return out


# ---------------------------------------------------------------- transcript probes

class TPGRU(nn.Module):
    """PRiSM transcript probe: embedding(128, dropout .1) -> 2x biGRU(256) -> mean -> LN -> dropout -> 3."""

    def __init__(self, vocab_size: int, emb_dim: int = 128, hidden: int = 256, layers: int = 2,
                 dropout: float = 0.1, pad_id: int = 0):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)
        self.emb_drop = nn.Dropout(dropout)
        self.rnn = BiGRUEncoder(emb_dim, hidden, layers, dropout)
        self.norm = nn.LayerNorm(self.rnn.out_dim)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(self.rnn.out_dim, 3)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        h = self.rnn(self.emb_drop(self.emb(ids)), lengths)
        return self.out(self.drop(self.norm(masked_mean(h, lengths))))


class BagMLP(nn.Module):
    """Order-free probe over a fixed-length count vector (unigram and/or bigram frequencies)."""

    def __init__(self, input_dim: int, hidden: int = 512, dropout: float = 0.1):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
                               nn.Dropout(dropout), nn.Linear(hidden, hidden), nn.GELU(),
                               nn.Dropout(dropout), nn.Linear(hidden, 3))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return self.f(x)


# ---------------------------------------------------------------- representation probes

class LayerMix(nn.Module):
    """Learned softmax-weighted sum over stacked layers (B, L, T, D) -> (B, T, D)."""

    def __init__(self, n_layers: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(n_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("l,bltd->btd", F.softmax(self.w, 0), x)


class AttnMLPHead(nn.Module):
    """PRiSM AttentionMLPHead: learned query, single-head attention over L2-normalised frames."""

    def __init__(self, dim: int, hidden: int | None = None, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=1)
        hidden = hidden or dim // 2
        self.f = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(),
                               nn.Dropout(dropout) if dropout > 0 else nn.Identity(), nn.Linear(hidden, 3))

    def pool(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        h = F.normalize(x, dim=-1, eps=1e-8)
        key_pad = ~length_mask(lengths, x.shape[1])
        q = F.normalize(self.query, dim=-1, eps=1e-8).expand(1, b, -1)
        h = self.attn(query=q, key=h.transpose(0, 1), value=h.transpose(0, 1), key_padding_mask=key_pad)[0]
        return F.normalize(h.squeeze(0), dim=-1, eps=1e-8)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.f(self.pool(x, lengths))


def sinusoidal(t: int, d: int, device) -> torch.Tensor:
    pos = torch.arange(t, device=device, dtype=torch.float32)[:, None]
    i = torch.arange(0, d, 2, device=device, dtype=torch.float32)
    ang = pos / torch.pow(10000.0, i / d)
    pe = torch.zeros(t, d, device=device)
    pe[:, 0::2] = torch.sin(ang)
    pe[:, 1::2] = torch.cos(ang[:, : d // 2])
    return pe


class RPAttn(nn.Module):
    """PRiSM representation probe on (optionally layer-mixed) frames."""

    def __init__(self, dim: int, n_layers: int = 1, pe: bool = False):
        super().__init__()
        self.mix = LayerMix(n_layers) if n_layers > 1 else None
        self.pe = pe
        self.head = AttnMLPHead(dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if self.mix is not None:
            x = self.mix(x)
        if self.pe:
            # scaled so that, after the head's L2 normalisation, position is a weak but present signal
            x = x + sinusoidal(x.shape[1], x.shape[2], x.device) * x.norm(dim=-1, keepdim=True).mean() / math.sqrt(x.shape[2])
        return self.head(x, lengths)


class RPGRU(nn.Module):
    """Order-aware: projection -> biGRU -> attention pooling head (frames interact before pooling)."""

    def __init__(self, dim: int, proj: int, hidden: int, layers: int = 1, n_layers: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.mix = LayerMix(n_layers) if n_layers > 1 else None
        self.proj = nn.Sequential(nn.Linear(dim, proj), nn.LayerNorm(proj))
        self.rnn = BiGRUEncoder(proj, hidden, layers, dropout)
        self.head = AttnMLPHead(self.rnn.out_dim)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if self.mix is not None:
            x = self.mix(x)
        return self.head(self.rnn(self.proj(x), lengths), lengths)


class RPTransformer(nn.Module):
    """Order-aware: projection + sinusoidal PE -> Transformer encoder -> attention pooling head."""

    def __init__(self, dim: int, proj: int, layers: int = 2, heads: int = 4, ff: int | None = None,
                 n_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.mix = LayerMix(n_layers) if n_layers > 1 else None
        self.proj = nn.Linear(dim, proj)
        layer = nn.TransformerEncoderLayer(proj, heads, ff or 2 * proj, dropout, batch_first=True,
                                           norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = AttnMLPHead(proj)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if self.mix is not None:
            x = self.mix(x)
        h = self.proj(x)
        h = h + sinusoidal(h.shape[1], h.shape[2], h.device)
        h = self.enc(h, src_key_padding_mask=~length_mask(lengths, h.shape[1]))
        return self.head(h, lengths)


# ---------------------------------------------------------------- order-destroying transforms

def shuffle_within(ids: torch.Tensor, lengths: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Permute the valid tokens of each row independently (padding stays at the end)."""
    out = ids.clone()
    for b, n in enumerate(lengths.tolist()):
        if n > 1:
            perm = torch.randperm(n, generator=generator, device="cpu").to(ids.device)
            out[b, :n] = ids[b, :n][perm]
    return out


def shuffle_chunks(x: torch.Tensor, lengths: torch.Tensor, chunk: int, generator: torch.Generator) -> torch.Tensor:
    """Permute fixed-size chunks of frames within each row; order inside a chunk is kept.

    Works on (B, T, D) or (B, L, T, D) (same permutation for every layer).
    """
    out = x.clone()
    t_axis = 1 if x.dim() == 3 else 2
    for b, n in enumerate(lengths.tolist()):
        k = math.ceil(n / chunk)
        if k < 2:
            continue
        perm = torch.randperm(k, generator=generator).tolist()
        idx = torch.cat([torch.arange(p * chunk, min((p + 1) * chunk, n)) for p in perm]).to(x.device)
        if t_axis == 1:
            out[b, :n] = x[b, idx]
        else:
            out[b, :, :n] = x[b, :, idx]
    return out


# ---------------------------------------------------------------- capacity matching

def build_rp(kind: str, dim: int, n_layers: int = 1, size: dict | None = None) -> nn.Module:
    """Construct a representation probe. `size` holds proj/hidden for the order-aware probes."""
    size = size or {}
    if kind in ("attn", "chunkshuffle_attn"):
        return RPAttn(dim, n_layers=n_layers)
    if kind == "attn_pe":
        return RPAttn(dim, n_layers=n_layers, pe=True)
    if kind in ("gru", "chunkshuffle"):
        return RPGRU(dim, proj=size["proj"], hidden=size["hidden"], n_layers=n_layers)
    if kind == "transformer":
        return RPTransformer(dim, proj=size["proj"], layers=size.get("layers", 2), n_layers=n_layers)
    raise ValueError(kind)


def match_size(kind: str, dim: int, n_layers: int = 1, tol: float = 0.10) -> dict:
    """Pick proj/hidden so the probe's trainable parameters are within `tol` of RPAttn(dim).

    Searches a fixed grid and returns the closest configuration; raises if none is within tol.
    """
    target = n_params(RPAttn(dim, n_layers=n_layers))
    if kind in ("attn", "attn_pe", "chunkshuffle_attn"):
        return {"params": target, "target": target}
    grid = []
    if kind in ("gru", "chunkshuffle"):
        for proj in range(128, 801, 64):
            for hidden in range(128, 513, 32):
                grid.append({"proj": proj, "hidden": hidden})
    elif kind == "transformer":
        for proj in range(128, 801, 32):          # multiples of 32, divisible by the 4 heads
            for layers in (2, 3):                 # at least 2 layers so frames interact in depth
                grid.append({"proj": proj, "layers": layers})
    best = min(grid, key=lambda s: abs(n_params(build_rp(kind, dim, n_layers, s)) - target))
    p = n_params(build_rp(kind, dim, n_layers, best))
    if abs(p - target) > tol * target:
        raise ValueError(f"{kind}: closest size {p} not within {tol:.0%} of {target}")
    return {**best, "params": p, "target": target}


class VectorMLP(nn.Module):
    """Probe on one fixed vector per utterance (speaker embeddings, time-averaged layers).

    LayerNorm -> Linear(D, hidden) -> GELU -> Dropout -> Linear(hidden, 3). Takes and ignores `lengths`
    so it fits the shared training loop.
    """

    def __init__(self, dim: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.f = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
                               nn.Linear(hidden, 3))

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        return self.f(x)


class ProsodyGRU(nn.Module):
    """Probe on frame-level prosodic contours (B, T, F) at 100 Hz.

    Frames are averaged in groups of `pool` (100 Hz -> 25 Hz), projected, encoded by a 2-layer biGRU, mean
    pooled, normalised and mapped to 3 outputs (same back end as TPGRU).
    """

    def __init__(self, n_feats: int = 4, pool: int = 4, proj: int = 64, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.pool = pool
        self.proj = nn.Sequential(nn.Linear(n_feats, proj), nn.LayerNorm(proj), nn.GELU())
        self.rnn = BiGRUEncoder(proj, hidden, 2, dropout)
        self.norm = nn.LayerNorm(self.rnn.out_dim)
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(self.rnn.out_dim, 3)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        b, t, f = x.shape
        pad = (-t) % self.pool
        if pad:
            x = F.pad(x, (0, 0, 0, pad))
        x = x.reshape(b, -1, self.pool, f).mean(2)
        lengths = torch.div(lengths + self.pool - 1, self.pool, rounding_mode="floor").clamp(min=1)
        h = self.rnn(self.proj(x), lengths)
        return self.out(self.drop(self.norm(masked_mean(h, lengths))))
