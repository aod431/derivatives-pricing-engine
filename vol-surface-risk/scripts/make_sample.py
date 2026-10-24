"""Generate data/sample_SPY.csv: a synthetic SPY-like chain priced with Heston.

The true model is known, so tests can check the surface pipeline against it.
Quotes get a realistic microstructure (cent ticks, spreads that widen with
price, noise on the mid) and a handful of deliberately bad rows that the
cleaning step is expected to remove.

    python scripts/make_sample.py
"""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from pricing.heston import HestonParams, heston_price

AS_OF = date(2026, 10, 23)
SPOT, RATE, DIV = 500.0, 0.045, 0.013
HESTON = HestonParams(v0=0.035, kappa=2.0, theta=0.045, xi=0.55, rho=-0.75)
DAYS = [30, 60, 91, 182, 365]
STRIKES = np.arange(375.0, 626.0, 5.0)
SEED = 7


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    for days in DAYS:
        T = days / 365.0
        expiry = (AS_OF + timedelta(days=days)).isoformat()
        for K in STRIKES:
            for opt in ("call", "put"):
                mid = heston_price(SPOT, K, T, RATE, HESTON, opt, DIV)
                mid *= 1.0 + 0.0015 * rng.standard_normal()  # quote noise
                half = 0.5 * (0.02 + 0.015 * mid)
                bid = max(round(mid - half, 2), 0.0)
                ask = round(mid + half, 2)
                atm_weight = np.exp(-0.5 * (np.log(K / SPOT) / 0.12) ** 2)
                oi = int(rng.integers(5, 40) + 4000 * atm_weight * rng.uniform(0.3, 1.0))
                vol = int(rng.integers(0, 20) + 800 * atm_weight * rng.uniform(0.0, 1.0))
                rows.append((expiry, opt, K, bid, ask, vol, oi, (AS_OF - timedelta(days=int(rng.integers(0, 2)))).isoformat()))

    df = pd.DataFrame(rows, columns=["expiry", "option_type", "strike", "bid", "ask", "volume",
                                      "open_interest", "last_trade_date"])

    # Deliberately bad rows (appended; the cleaning step must drop all of them).
    e60 = (AS_OF + timedelta(days=60)).isoformat()
    e182 = (AS_OF + timedelta(days=182)).isoformat()
    e5 = (AS_OF + timedelta(days=5)).isoformat()
    bad = [
        (e60, "call", 505.0, 0.0, 14.50, 10, 500, AS_OF.isoformat()),      # zero bid
        (e60, "put", 495.0, 9.80, 9.20, 30, 800, AS_OF.isoformat()),       # crossed
        (e182, "call", 520.0, 12.0, 40.0, 5, 300, AS_OF.isoformat()),      # very wide spread
        (e182, "put", 480.0, 14.0, 15.0, 0, 400, "2026-09-01"),            # stale last trade
        (e60, "call", 610.0, 0.12, 0.16, 0, 1, AS_OF.isoformat()),         # no open interest
        (e5, "call", 500.0, 4.0, 4.2, 900, 9000, AS_OF.isoformat()),       # expiry too short
    ]
    df = pd.concat([df, pd.DataFrame(bad, columns=df.columns)], ignore_index=True)

    out_dir = Path(__file__).resolve().parents[1] / "data"
    out_dir.mkdir(exist_ok=True)
    df.to_csv(out_dir / "sample_SPY.csv", index=False)
    meta = {
        "symbol": "SPY", "as_of": AS_OF.isoformat(), "spot": SPOT, "rate": RATE, "source": "synthetic",
        "note": "Synthetic Heston chain, not market data.",
        "true_model": {"v0": HESTON.v0, "kappa": HESTON.kappa, "theta": HESTON.theta, "xi": HESTON.xi,
                       "rho": HESTON.rho, "dividend_yield": DIV},
        "deliberately_bad_rows": len(bad), "seed": SEED,
    }
    (out_dir / "sample_SPY.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {len(df)} rows ({len(bad)} deliberately bad)")


if __name__ == "__main__":
    main()
