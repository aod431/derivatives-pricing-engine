"""Save diagnostic figures of the calibrated surface into docs/img/.

    python scripts/plot_surface.py [--symbol SPY] [--live]

By default uses the committed synthetic sample; with --live uses the most recent
real snapshot written by `python -m volrisk.cli build`.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from volrisk.data import load_chain, load_sample, prepare_chain  # noqa: E402
from volrisk.surface import build_surface  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "docs" / "img"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--live", action="store_true", help="use the latest real snapshot instead of the sample")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    surf = build_surface(prepare_chain(load_chain(args.symbol) if args.live else load_sample(args.symbol)))
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(surf.slices)))
    k = np.linspace(-0.35, 0.3, 300)

    # 1. Smiles: market implied vols vs SVI fit.
    fig, ax = plt.subplots(figsize=(8, 5))
    for c, s in zip(colors, surf.slices):
        mkt = surf.market[surf.market.expiry == s.expiry]
        ax.scatter(mkt["k"], mkt["iv"], s=12, color=c, alpha=0.7)
        kk = k[(k > s.k_min - 0.05) & (k < s.k_max + 0.05)]
        from pricing.svi import svi_total_variance
        ax.plot(kk, np.sqrt(svi_total_variance(kk, s.params) / s.T), color=c, label=f"{s.expiry} (T={s.T:.2f})")
    ax.set_xlabel("log-moneyness k = ln(K/F)")
    ax.set_ylabel("implied vol")
    ax.set_title(f"{surf.symbol} smiles: OTM market quotes (dots) and SVI fit (lines) [{surf.source}]")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "smiles.png", dpi=140)
    plt.close(fig)

    # 2. Total variance slices: calendar no-arbitrage means curves never cross.
    from pricing.svi import svi_total_variance
    fig, ax = plt.subplots(figsize=(8, 5))
    for c, s in zip(colors, surf.slices):
        ax.plot(k, svi_total_variance(k, s.params), color=c, label=f"T={s.T:.2f}")
    ax.set_xlabel("log-moneyness k")
    ax.set_ylabel("total variance w = sigma^2 T")
    ax.set_title("Total variance per slice (no crossing = no calendar arbitrage)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "total_variance.png", dpi=140)
    plt.close(fig)

    # 3. Heatmaps of implied and local vol over (moneyness, maturity).
    Ts = np.linspace(surf.maturities[0], surf.maturities[-1], 40)
    ks = np.linspace(-0.25, 0.2, 46)
    iv = np.array([[surf.vol(surf.forward(t) * math.exp(x), t) for x in ks] for t in Ts])
    lv = np.array([[surf.local_vol(surf.forward(t) * math.exp(x), t) for x in ks] for t in Ts])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, z, title in ((axes[0], iv, "Implied vol"), (axes[1], lv, "Dupire local vol")):
        im = ax.pcolormesh(ks, Ts, z, shading="auto", cmap="viridis")
        ax.set_title(title)
        ax.set_xlabel("log-moneyness k")
        fig.colorbar(im, ax=ax)
    axes[0].set_ylabel("maturity T (years)")
    fig.tight_layout()
    fig.savefig(OUT / "heatmaps.png", dpi=140)
    plt.close(fig)

    # 4. 3D implied vol surface.
    X, Y = np.meshgrid(ks, Ts)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, iv, cmap="viridis", linewidth=0, antialiased=True)
    ax.set_xlabel("k")
    ax.set_ylabel("T")
    ax.set_zlabel("implied vol")
    ax.view_init(elev=25, azim=-125)
    fig.tight_layout()
    fig.savefig(OUT / "surface3d.png", dpi=140)
    plt.close(fig)
    print(f"saved figures to {OUT}")


if __name__ == "__main__":
    main()
