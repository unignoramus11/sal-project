"""Probe training loop.

Reproduces the behaviour of PRiSM's Lightning setup (docs/prism_reproduction.md):
Adam; LinearLR from lr/3 to lr over the first 200 optimiser steps; gradient clipping at 1.0; validation
`val_per_epoch` times per epoch; early stopping on validation mean km counted in validation checks and
not before `min_epochs`; the best-validation checkpoint is restored for the final evaluation.

A `Batcher` turns a list of example indices into (inputs, lengths, targets) tensors, so the same loop
serves transcript and representation probes.

Resuming: with `checkpoint` set, the full training state (model, optimiser, scheduler, early-stopping state,
history, and every random generator: Python's, NumPy's, torch's CPU and CUDA ones, the loop's shuffling generator
and any generator the batcher holds as `rng` or `gen`) is saved after each epoch, and a run that finds its
checkpoint continues from the next epoch, as if it had not stopped. The checkpoint is deleted once the run ends.
`checkpoint_for(name)` gives the path under $SAL_CKPT_DIR, or None when that is unset (the default: no checkpoints).
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd
import torch

from sal.probes import geo


class Batcher(Protocol):
    def __call__(self, idx: np.ndarray, train: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 64
    min_epochs: int = 10
    max_epochs: int = 200
    patience: int = 100            # validation checks
    val_per_epoch: int = 4
    warmup_steps: int = 200
    warmup_start_factor: float = 1 / 3
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    seed: int = 42
    device: str = "cpu"
    eval_batch_size: int = 256
    extra: dict = field(default_factory=dict)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint_for(name: str) -> Path | None:
    """Checkpoint path for run `name` under $SAL_CKPT_DIR, or None if checkpointing is off."""
    d = os.environ.get("SAL_CKPT_DIR")
    return Path(d) / f"{name}.pt" if d else None


def _resume_key(cfg: TrainConfig) -> dict:
    """The settings a checkpoint must share to be resumed. `extra` (e.g. a per-job scratch path to the features) and
    `device` may differ; the run name, which names the checkpoint, already fixes the probe, data and split."""
    return {k: v for k, v in asdict(cfg).items() if k not in ("extra", "device")}


def _batcher_rng(batcher) -> dict:
    out = {}
    for a in ("rng", "gen"):
        g = getattr(batcher, a, None)
        if isinstance(g, np.random.Generator):
            out[a] = g.bit_generator.state
        elif isinstance(g, torch.Generator):
            out[a] = g.get_state()
    return out


def _set_batcher_rng(batcher, states: dict) -> None:
    for a, st in states.items():
        g = getattr(batcher, a)
        if isinstance(g, np.random.Generator):
            g.bit_generator.state = st
        else:
            g.set_state(st)


@torch.no_grad()
def predict(model: torch.nn.Module, batcher: Batcher, idx: np.ndarray, cfg: TrainConfig) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(idx), cfg.eval_batch_size):
        x, lengths, _ = batcher(idx[s:s + cfg.eval_batch_size], train=False)
        out.append(model(x.to(cfg.device), lengths.to(cfg.device)).float().cpu())
    return torch.cat(out).numpy()


def evaluate(pred: np.ndarray, targets: np.ndarray) -> dict:
    return geo.summarise(pred, targets)


def train_probe(
    make_model: Callable[[], torch.nn.Module],
    batcher: Batcher,
    targets: np.ndarray,                      # (N, 2) radians, aligned with example indices
    splits: dict[str, np.ndarray],            # name -> example indices; needs "train" and "val"
    cfg: TrainConfig,
    log: Callable[[str], None] = print,
    checkpoint: Path | None = None,
) -> dict:
    seed_everything(cfg.seed)          # before construction, so initial weights depend on the seed
    device = torch.device(cfg.device)
    model = make_model().to(device)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=cfg.lr,
                           weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=cfg.warmup_start_factor, end_factor=1.0,
                                              total_iters=cfg.warmup_steps)
    tr = splits["train"]
    steps_per_epoch = math.ceil(len(tr) / cfg.batch_size)
    check_at = {round(steps_per_epoch * k / cfg.val_per_epoch) for k in range(1, cfg.val_per_epoch + 1)}
    gen = np.random.default_rng(cfg.seed)
    t_val = targets[splits["val"]]

    best = {"km": float("inf"), "state": None, "epoch": None, "check": None}
    history, bad, n_checks, step = [], 0, 0, 0
    t0 = time.time()
    stop = False
    start = 0
    if checkpoint is not None and checkpoint.exists():
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        if ck["config"] != _resume_key(cfg):
            raise ValueError(f"{checkpoint} was written with a different configuration; delete it to start over")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        gen.bit_generator.state = ck["gen"]
        random.setstate(ck["py"])
        np.random.set_state(ck["np"])
        torch.set_rng_state(ck["torch"])
        if ck["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(ck["cuda"])
        _set_batcher_rng(batcher, ck["batcher"])
        best, history, bad, n_checks, step = ck["best"], ck["history"], ck["bad"], ck["n_checks"], ck["step"]
        t0 -= ck["seconds"]
        start = ck["epoch"] + 1
        log(f"resumed from {checkpoint.name} after epoch {start}")
    epoch = start - 1
    for epoch in range(start, cfg.max_epochs):
        order = tr[gen.permutation(len(tr))]
        model.train()
        run_loss, run_n = 0.0, 0
        for b in range(steps_per_epoch):
            idx = order[b * cfg.batch_size:(b + 1) * cfg.batch_size]
            x, lengths, y = batcher(idx, train=True)
            pred = model(x.to(device), lengths.to(device))
            loss = geo.angular_loss(pred.float(), y.to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            sched.step()
            step += 1
            run_loss += loss.item() * len(idx)
            run_n += len(idx)
            if (b + 1) in check_at:
                n_checks += 1
                p_val = predict(model, batcher, splits["val"], cfg)
                m = evaluate(p_val, t_val)
                history.append({"epoch": epoch, "step": step, "train_loss": run_loss / max(run_n, 1),
                                "val_km": m["mean_km"], "val_recall1": m["recall1"]})
                if m["mean_km"] < best["km"]:
                    best = {"km": m["mean_km"], "state": copy.deepcopy(model.state_dict()),
                            "epoch": epoch, "check": n_checks}
                    bad = 0
                else:
                    bad += 1
                model.train()
                if epoch + 1 >= cfg.min_epochs and bad >= cfg.patience:
                    stop = True
                    break
        log(f"epoch {epoch + 1}: train loss {run_loss / max(run_n, 1):.4f}, "
            f"val km {history[-1]['val_km']:.1f}, best {best['km']:.1f} (check {best['check']})")
        if stop:
            break
        if checkpoint is not None:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            tmp = checkpoint.with_suffix(".tmp")
            torch.save({"config": _resume_key(cfg), "epoch": epoch, "model": model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "gen": gen.bit_generator.state, "py": random.getstate(),
                        "np": np.random.get_state(), "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                        "batcher": _batcher_rng(batcher), "best": best, "history": history, "bad": bad,
                        "n_checks": n_checks, "step": step, "seconds": time.time() - t0}, tmp)
            os.replace(tmp, checkpoint)          # atomic: a kill mid-write leaves the previous checkpoint intact

    model.load_state_dict(best["state"])
    result = {"config": asdict(cfg), "best_epoch": best["epoch"], "best_check": best["check"],
              "epochs_run": epoch + 1, "seconds": time.time() - t0, "history": history,
              "n_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
              "metrics": {}, "predictions": {}}
    for name, idx in splits.items():
        p = predict(model, batcher, idx, cfg)
        result["metrics"][name] = evaluate(p, targets[idx])
        result["predictions"][name] = (idx, p)
    if checkpoint is not None:
        checkpoint.unlink(missing_ok=True)
    return result


def save_run(result: dict, out_dir: Path, meta: pd.DataFrame | None = None,
             keep_splits: tuple[str, ...] = ("val", "test")) -> None:
    """Write metrics.json, history.csv and predictions.parquet (one row per example of `keep_splits`;
    metrics for every split are kept in metrics.json)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for split, (idx, p) in result["predictions"].items():
        if split not in keep_splits:
            continue
        for i, v in zip(idx, p, strict=True):
            rows.append({"split": split, "example": int(i), "x": float(v[0]), "y": float(v[1]), "z": float(v[2])})
    pred = pd.DataFrame(rows)
    if meta is not None:
        pred = pred.join(meta, on="example")
    pred.to_parquet(out_dir / "predictions.parquet", index=False)
    pd.DataFrame(result["history"]).to_csv(out_dir / "history.csv", index=False)
    summary = {k: v for k, v in result.items() if k not in ("history", "predictions")}
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
