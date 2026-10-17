"""Geolocation loss and metrics.

Matches PRiSM (`src/recipe/common/geolocation_loss.py`, `src/metrics/geolocation.py`): models output an
unnormalised xyz direction, targets are (lat, lon) in radians, the loss is the mean great-circle angle,
error is reported with Earth radius 6378.1 km, and Recall@k ranks the distinct target locations of the
evaluated split by cosine similarity to the prediction.
"""
from __future__ import annotations

import numpy as np
import torch

EARTH_RADIUS_KM = 6378.1


def xyz_to_latlon(p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x, y, z = p.unbind(-1)
    lon = torch.atan2(y, x)
    lat = torch.atan2(z, torch.clamp(torch.sqrt(x * x + y * y), min=1e-8))
    return lat, lon


def latlon_to_xyz(lat: torch.Tensor, lon: torch.Tensor) -> torch.Tensor:
    return torch.stack([lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()], dim=-1)


def angular_distance(pred_xyz: torch.Tensor, target_latlon: torch.Tensor) -> torch.Tensor:
    """Great-circle angle (radians) between predicted directions and target (lat, lon) in radians."""
    p_lat, p_lon = xyz_to_latlon(pred_xyz)
    t_lat, t_lon = target_latlon.unbind(-1)
    cos = torch.sin(t_lat) * torch.sin(p_lat) + torch.cos(t_lat) * torch.cos(p_lat) * torch.cos(p_lon - t_lon)
    return torch.acos(torch.clamp(cos, -1.0 + 1e-7, 1.0 - 1e-7))


def angular_loss(pred_xyz: torch.Tensor, target_latlon: torch.Tensor) -> torch.Tensor:
    """PRiSM's loss. The clamp to 1 - 1e-7 gives a floor of ~4.5e-4 rad (~2.9 km) per utterance."""
    return angular_distance(pred_xyz, target_latlon).mean()


def error_km(pred_xyz: torch.Tensor, target_latlon: torch.Tensor) -> torch.Tensor:
    """Per-utterance great-circle error in km (PRiSM clamps to [-1, 1] here, not 1 - 1e-7)."""
    p_lat, p_lon = xyz_to_latlon(pred_xyz)
    t_lat, t_lon = target_latlon.unbind(-1)
    cos = torch.sin(t_lat) * torch.sin(p_lat) + torch.cos(t_lat) * torch.cos(p_lat) * torch.cos(p_lon - t_lon)
    return torch.acos(cos.clamp(-1.0, 1.0)) * EARTH_RADIUS_KM


def recall_at_k(pred_xyz: torch.Tensor, target_latlon: torch.Tensor, k: int = 1,
                candidates: torch.Tensor | None = None) -> torch.Tensor:
    """Per-utterance hit (1.0/0.0): is the true location among the k candidates nearest by cosine?

    Candidates default to the distinct targets of the evaluated set (PRiSM: rounded to 6 decimals).
    """
    dtype = torch.promote_types(pred_xyz.dtype, target_latlon.dtype)
    pred_xyz, target_latlon = pred_xyz.to(dtype), target_latlon.to(dtype)
    t = target_latlon.round(decimals=6)
    if candidates is None:
        candidates = torch.unique(t, dim=0)
    cand = candidates.to(dtype).round(decimals=6)
    cand_xyz = latlon_to_xyz(*cand.unbind(-1))
    sim = torch.nn.functional.normalize(pred_xyz, dim=1) @ torch.nn.functional.normalize(cand_xyz, dim=1).T
    topk = torch.topk(sim, k=min(k, cand.shape[0]), dim=1).indices
    # index of each target within candidates
    match = (t[:, None, :] == cand[None, :, :]).all(-1)
    if not bool(match.any(1).all()):
        raise ValueError("a target location is missing from the candidate set")
    true_idx = match.float().argmax(1)
    return (topk == true_idx[:, None]).any(1).float()


def summarise(pred_xyz: np.ndarray | torch.Tensor, target_latlon: np.ndarray | torch.Tensor) -> dict:
    p = torch.as_tensor(pred_xyz, dtype=torch.float64)
    t = torch.as_tensor(target_latlon, dtype=torch.float64)
    km = error_km(p, t)
    return {
        "recall1": 100 * recall_at_k(p, t, 1).mean().item(),
        "recall5": 100 * recall_at_k(p, t, 5).mean().item(),
        "mean_km": km.mean().item(),
        "median_km": km.median().item(),
        "n": int(p.shape[0]),
    }
