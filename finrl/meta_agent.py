# finrl/meta_agent.py
from __future__ import annotations
import numpy as np
from typing import Dict, Callable

class MetaAgent:
    """
    Calls several SB3 models every step and returns a single,
    weighted-average action vector.

    Parameters
    ----------
    models : Dict[str, BaseAlgorithm]
        Keys are names ("a2c", "ppo", …), values are loaded SB3 models.
    weight_fn : Callable[[Dict[str, np.ndarray], int], Dict[str, float]]
        Function that returns per-model weights for the *current* step.
        It receives the dict of raw actions and the env.step counter.
    stock_dim : int
        Length of the action vector (equals env.stock_dim).
    clip : bool, default True
        If True, final actions are clipped to the [-1, +1] range FinRL expects.
    """
    def __init__(
        self,
        models: Dict[str, "BaseAlgorithm"],
        weight_fn: Callable[[Dict[str, np.ndarray], int], Dict[str, float]],
        stock_dim: int,
        clip: bool = True,
    ):
        self.models = models
        self.weight_fn = weight_fn
        self.stock_dim = stock_dim
        self.clip = clip

    # ──────────────────────────────────────────────────────────────────
    # main API used inside the trading loop
    # ──────────────────────────────────────────────────────────────────
    def predict(self, state, step: int) -> np.ndarray:
        # 1. raw deterministic actions from every base model
        raw = {
            name: mdl.predict(state, deterministic=True)[0].astype(np.float32)
            for name, mdl in self.models.items()
        }

        # 2. obtain weights for *this* step
        w = self.weight_fn(raw, step)          # dict must include all keys
        wtot = sum(w.values())
        if not np.isclose(wtot, 1.0):
            w = {k: v / wtot for k, v in w.items()}   # normalise

        # 3. blend
        blended = sum(w[k] * raw[k] for k in self.models)
        if self.clip:
            blended = np.clip(blended, -1.0, 1.0)
        return blended


# ─────────────────────────────────────────────────────────────────────────
# Example weighting rules (import the one you prefer in inference.py)
# ─────────────────────────────────────────────────────────────────────────
def equal_weights(raw, step):
    n = len(raw)
    return {k: 1.0 / n for k in raw}

def fixed_sharpe_weights(sharpes: Dict[str, float]):
    tot = sum(sharpes.values())
    base = {k: v / tot for k, v in sharpes.items()}
    return lambda raw, step: base

def risk_tilted(
    base_weights: Dict[str, float],
    risk_series,                 # pd.Series with index aligned to env dates
    high=0.7,
    low=0.3,
    tilt=0.20,                   # ±20 %
    conservative=("a2c", "ppo"),
    aggressive=("sac", "td3"),
):
    """
    Returns a weight_fn that tilts base weights according to the *current*
    risk score (0–1).
    """
    def _fn(raw, step):
        r = risk_series.iloc[step]
        w = base_weights.copy()
        if r > high:                       # high risk ➜ favour conservative
            for k in conservative: w[k] *= 1 + tilt
            for k in aggressive:   w[k] *= 1 - tilt
        elif r < low:                      # low risk ➜ favour aggressive
            for k in aggressive:   w[k] *= 1 + tilt
            for k in conservative: w[k] *= 1 - tilt
        return w
    return _fn