"""Sprint 4: saving / loading. Two kinds of file:

  last.pt        full training state (weights + optimizer + AMP scaler + step). For --resume.
  best_model.pt  weights only + model config + metrics. This is what Sprint 5/6 will load.
"""
import os
from pathlib import Path

import torch

from model import PolicyNet


def _atomic_save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)             # a crash mid-save never leaves a corrupt file behind


def save_training_state(path, model, opt, scaler, step, best, config, args) -> None:
    _atomic_save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "scaler": scaler.state_dict(),
        "step": step,
        "best": best,
        "config": config,
        "args": args,
    }, Path(path))


def load_training_state(path, device):
    # our own local file, so full unpickling is fine here
    return torch.load(path, map_location=device, weights_only=False)


def save_weights(path, model, config, metrics) -> None:
    _atomic_save({"model": model.state_dict(), "config": config, "metrics": metrics}, Path(path))


def load_policy(path, device="cpu") -> PolicyNet:
    """Load a weights-only file into a ready-to-use PolicyNet (eval mode)."""
    ck = torch.load(path, map_location=device, weights_only=True)
    model = PolicyNet(**ck["config"]).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model
