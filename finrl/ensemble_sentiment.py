"""
ensemble_sentiment.py  Ensemble post-processing helpers
==============================================================

Compute portfolio-level ensemble curves from the individual
account-value columns produced by each trained agent 
(A2C, PPO, SAC, TD3).  
Import this module after we merged the per-agent
equity curves in `inference.py`.

Usage
-----
from ensemble_model_sentiment import add_ensemble_strategies

results = merge_results(...)          # your existing step
results = add_ensemble_strategies(results)   # ➜ adds three ensemble columns
results.to_csv(RESULTS_CSV, index=False)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Sequence, List

# ---------------------------------------------------------------------------
# Expected agent columns in the merged results frame
# ---------------------------------------------------------------------------

DEFAULT_AGENT_COLS: List[str] = [
    "A2C Agent 1",
    "PPO Agent 1",
    "SAC Agent 1",
    "TD3 Agent 1",
]

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _inverse_vol_weights(df: pd.DataFrame) -> np.ndarray:
    """Compute inverse-volatility weights for simple risk parity."""
    vol = df.pct_change().std(ddof=0).replace(0, np.nan)
    inv = 1.0 / vol
    w = inv / inv.sum()
    return w.values


def _best_in_window(df: pd.DataFrame, *, window: int) -> pd.Series:
    """
    At each timestamp choose the account‑value column that delivered the
    highest cumulative return over the last *window* rows.

    This version is fully vectorised and avoids deprecated/removed pandas
    methods such as ``DataFrame.lookup``.  It also copes with rows that
    have insufficient history (fewer than *window* rows) by returning NaN.
    """
    # Cumulative growth over the rolling window
    growth = df / df.shift(window) - 1.0

    # use .iloc for forward‑compatibility and avoid positional warning
    win_growth = growth.rolling(window, min_periods=window).apply(lambda x: x.iloc[-1])
    # pandas >=2.3 deprecates positional x[-1]; iloc is explicit

    # idxmax with all‑NA rows will raise in future; fill with -inf then compute
    idx = win_growth.fillna(-np.inf).idxmax(axis=1, skipna=True)

    # Map those labels to positional indices
    pos = df.columns.get_indexer(idx)

    # Advanced numpy indexing to fetch the winning value per row
    rows = np.arange(len(df))
    best_values = df.to_numpy()[rows, pos]

    return pd.Series(best_values, index=df.index, name="Ensemble Best")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def add_ensemble_strategies(
    results: pd.DataFrame,
    agent_cols: Sequence[str] | None = None,
    *,
    window: int = 24,
) -> pd.DataFrame:
    """
    Append ensemble strategies to *results* and return the extended frame.

    Adds three columns:
    • “Ensemble Mean”  equal-weight average of the agent curves

    • “Ensemble VolW”  inverse-volatility weighted blend of the agent curves

    • “Ensemble Best”  best performer over the past *window* rowss
    """
    if agent_cols is None:
        agent_cols = DEFAULT_AGENT_COLS

    missing = [c for c in agent_cols if c not in results.columns]
    if missing:
        raise KeyError(f"add_ensemble_strategies: missing columns {missing}")

    sub = results[agent_cols]

    # 1️⃣  Equal-weight mean
    results["Ensemble Mean"] = sub.mean(axis=1)

    # 2️⃣  Inverse-volatility weighting
    weights = _inverse_vol_weights(sub)
    results["Ensemble VolW"] = (sub * weights).sum(axis=1)
    results.attrs["vol_weights"] = dict(zip(agent_cols, weights))  # optional metadata

    # 3️⃣  Best-in-window
    results["Ensemble Best"] = _best_in_window(sub, window=window)

    return results


# ---------------------------------------------------------------------------
# Quick self-test (run this file directly)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01", periods=100, freq="h")
    data = np.cumprod(1 + rng.normal(0, 0.001, (100, 4)), axis=0) * 1_000_000
    demo = pd.DataFrame(data, index=idx, columns=DEFAULT_AGENT_COLS)
    demo = add_ensemble_strategies(demo)
    print(demo.tail())

