import math

import numpy as np
import pytest
import torch

from sal.probes import geo


def rad(lat, lon):
    return torch.tensor([[math.radians(lat), math.radians(lon)]], dtype=torch.float64)


def test_roundtrip():
    t = rad(28.61, 77.21)
    xyz = geo.latlon_to_xyz(*t.unbind(-1))
    lat, lon = geo.xyz_to_latlon(xyz * 3.7)  # any norm
    assert torch.allclose(torch.stack([lat, lon], -1), t)


def test_known_distance():
    # Delhi (28.6139, 77.2090) to Rohtak (28.8955, 76.6066): ~67 km on a 6378.1 km sphere
    delhi, rohtak = rad(28.6139, 77.2090), rad(28.8955, 76.6066)
    pred = geo.latlon_to_xyz(*rohtak.unbind(-1))
    km = geo.error_km(pred, delhi).item()
    # haversine reference with the same radius
    la1, lo1, la2, lo2 = map(math.radians, (28.6139, 77.2090, 28.8955, 76.6066))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    ref = 2 * 6378.1 * math.asin(math.sqrt(h))
    assert km == pytest.approx(ref, rel=1e-6) and 60 < km < 75


def test_loss_is_mean_angle():
    t = torch.cat([rad(28.6, 77.2), rad(26.9, 75.8)])
    p = geo.latlon_to_xyz(*torch.cat([rad(28.9, 76.6), rad(25.4, 81.8)]).unbind(-1))
    loss = geo.angular_loss(p, t)
    assert loss.item() == pytest.approx(geo.error_km(p, t).mean().item() / 6378.1, rel=1e-6)


def test_loss_floor_from_prism_clamp():
    # PRiSM clamps cos to 1 - 1e-7, so a perfect prediction still has loss acos(1 - 1e-7)
    t = rad(28.6, 77.2)
    loss = geo.angular_loss(geo.latlon_to_xyz(*t.unbind(-1)), t).item()
    assert loss == pytest.approx(math.acos(1 - 1e-7), rel=1e-3)


def test_recall_matches_prism_definition():
    rng = np.random.default_rng(0)
    locs = torch.tensor(np.radians(rng.uniform([22, 72], [31, 88], size=(12, 2))))
    idx = torch.tensor(rng.integers(0, 12, size=400))
    t = locs[idx]
    p = geo.latlon_to_xyz(*t.unbind(-1)) + 0.02 * torch.tensor(rng.normal(size=(400, 3)))
    ours = 100 * geo.recall_at_k(p, t, 1).mean().item()
    # PRiSM GeolocationMissRate(k=1).compute(), re-implemented line by line
    G, true_idx = torch.unique(t.round(decimals=6), sorted=True, return_inverse=True, dim=0)
    glat, glng = G.unbind(-1)
    G_cart = torch.stack([glat.cos() * glng.cos(), glat.cos() * glng.sin(), glat.sin()], dim=-1)
    sim = torch.nn.functional.normalize(p, p=2, dim=1) @ torch.nn.functional.normalize(G_cart, p=2, dim=1).T
    _, topk_idx = torch.topk(sim, k=1, dim=1)
    hits = (topk_idx == true_idx.unsqueeze(1)).any(dim=1).float().mean()
    prism = 100.0 * hits.item()
    assert ours == pytest.approx(prism)
    assert 0 < ours <= 100


def test_perfect_prediction():
    t = torch.cat([rad(28.6, 77.2), rad(26.9, 75.8), rad(25.6, 85.1)])
    s = geo.summarise(geo.latlon_to_xyz(*t.unbind(-1)), t)
    assert s["recall1"] == 100 and s["mean_km"] < 1e-3
