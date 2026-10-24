"""Command line: build a surface from live or sample data, or run the offline demo.

    python -m volrisk.cli build --symbol SPY            # fetch from Yahoo, save snapshot, calibrate
    python -m volrisk.cli build --symbol SPY --sample   # same pipeline on the synthetic sample
    python -m volrisk.cli demo                          # offline end-to-end run on the sample
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from volrisk.data import (
    MarketDataError, SnapshotNotFound, fetch_chain_yfinance, load_sample,
    prepare_chain, save_snapshot,
)
from volrisk.risk import PortfolioPricer, Position, risk_report
from volrisk.surface import VolSurface, build_surface


def print_surface(surface: VolSurface, report: dict | None = None) -> None:
    if report:
        print("Quote cleaning:", {k: v for k, v in report.items() if v})
    d = surface.diagnostics()
    print(f"\n{surface.symbol}  as of {surface.as_of}  spot {surface.spot:.2f}  r {surface.rate:.4f}  [{surface.source}]")
    df = pd.DataFrame(d["slices"])
    df["svi_b"] = [s["svi"]["b"] for s in d["slices"]]
    df["svi_rho"] = [s["svi"]["rho"] for s in d["slices"]]
    print(df[["expiry", "T", "forward", "implied_q", "n_quotes", "rmse_vol_pts", "butterfly_free",
              "calendar_ok_to_next", "repaired", "svi_b", "svi_rho"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"arbitrage free: {d['arbitrage_free']}   local vol invalid on grid: {d['local_vol_invalid_pct']:.1f}%")


def cmd_build(args: argparse.Namespace) -> int:
    try:
        if args.sample:
            chain = load_sample(args.symbol)
        else:
            chain = fetch_chain_yfinance(args.symbol, rate=args.rate, max_expiries=args.max_expiries)
            path = save_snapshot(chain)
            print(f"saved snapshot: {path}")
    except (MarketDataError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        if not args.sample:
            print("hint: rerun with --sample to use the offline synthetic chain.", file=sys.stderr)
        return 1
    prepared = prepare_chain(chain)
    print_surface(build_surface(prepared), prepared.report)
    return 0


def cmd_demo(_: argparse.Namespace) -> int:
    try:
        chain = load_sample("SPY")
    except SnapshotNotFound as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    prepared = prepare_chain(chain)
    surface = build_surface(prepared)
    print_surface(surface, prepared.report)

    T = float(surface.maturities[2])
    print(f"\nATM vol at T={T:.3f}: {surface.vol(surface.forward(T), T):.4f}   "
          f"local vol: {surface.local_vol(surface.forward(T), T):.4f}")

    positions = [
        Position("call", 520.0, 0.25, -100), Position("put", 480.0, 0.25, -100),
        Position("call", 500.0, 0.50, 50), Position("put", 450.0, 1.00, 80),
    ]
    pricer = PortfolioPricer(surface, positions)
    print("\nPortfolio (short 3m strangle, long 6m call, long 1y put)")
    print(pricer.table().to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("greeks:", {k: round(v, 4) for k, v in pricer.greeks().items()})
    print("bucketed vega:", {k: round(v, 3) for k, v in pricer.bucketed_vega().items()})
    print("\nP&L grid (rows: spot shift, cols: parallel vol shift)")
    print(pricer.scenario_grid([-0.1, -0.05, 0.0, 0.05, 0.1], [-0.05, 0.0, 0.05]).round(1).to_string())
    rep = risk_report(pricer)
    print(f"\n1-day 99% VaR / ES  full reval: {rep['full_revaluation']['var']:.1f} / {rep['full_revaluation']['es']:.1f}"
          f"   delta-gamma-vega: {rep['delta_gamma_vega']['var']:.1f} / {rep['delta_gamma_vega']['es']:.1f}"
          f"   (VaR error {rep['approx_error_pct_var']:+.1f}%)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="volrisk")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="fetch (or load sample), clean and calibrate a surface")
    b.add_argument("--symbol", default="SPY")
    b.add_argument("--rate", type=float, default=None, help="continuously compounded r; default: ^IRX, else 4.5%%")
    b.add_argument("--max-expiries", type=int, default=8)
    b.add_argument("--sample", action="store_true", help="use the committed synthetic snapshot (offline)")
    b.set_defaults(func=cmd_build)
    d = sub.add_parser("demo", help="offline end-to-end demo on the sample snapshot")
    d.set_defaults(func=cmd_demo)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
