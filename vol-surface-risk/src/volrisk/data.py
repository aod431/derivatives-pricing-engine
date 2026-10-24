"""Option chain acquisition, cleaning and forward extraction.

Pipeline: raw quotes -> quality filters -> per-expiry forward from put-call
parity -> keep only out-of-the-money quotes. Every step is deterministic given
a snapshot, so the whole project runs offline from CSV files.

Snapshot layout (in the data directory):
    <SYMBOL>_<YYYY-MM-DD>.csv / .json    live snapshots written by `cli build`
    sample_<SYMBOL>.csv / .json          committed synthetic snapshot (Heston)
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

QUOTE_COLUMNS = [
    "expiry", "option_type", "strike", "bid", "ask", "volume", "open_interest", "last_trade_date",
]
DEFAULT_RATE = 0.045


class SnapshotNotFound(FileNotFoundError):
    """Raised when no snapshot exists for a symbol and the sample fallback was not requested."""


class MarketDataError(RuntimeError):
    """Raised when live market data cannot be fetched or is unusable."""


def default_data_dir() -> Path:
    """Data directory: $VOLRISK_DATA_DIR, else <project root>/data."""
    env = os.environ.get("VOLRISK_DATA_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[2] / "data"


@dataclass
class OptionChain:
    """Raw option quotes for one underlying at one date."""

    symbol: str
    as_of: date
    spot: float
    rate: float
    quotes: pd.DataFrame
    source: str = "snapshot"  # "live", "snapshot" or "sample"


@dataclass(frozen=True)
class CleaningConfig:
    """Thresholds of the quote filters (defaults suit liquid index/ETF options)."""

    min_days: int = 7            # very short expiries are dominated by microstructure noise
    max_rel_spread: float = 0.50  # (ask - bid) / mid
    min_mid: float = 0.10         # cheap options: one tick is a huge IV move
    min_open_interest: int = 10
    min_volume: int = 0
    max_stale_days: int = 5       # last trade older than this (only if the column is present)
    n_forward_pairs: int = 5      # strikes nearest spot used for the parity forward


@dataclass
class PreparedChain:
    """Cleaned OTM quotes plus per-expiry forwards, ready for surface building."""

    symbol: str
    as_of: date
    spot: float
    rate: float
    forwards: pd.DataFrame   # expiry, T, forward, implied_q, n_pairs
    otm: pd.DataFrame        # expiry, T, strike, option_type, bid, ask, mid, forward, k, volume, open_interest
    report: dict = field(default_factory=dict)
    source: str = "snapshot"


# --------------------------------------------------------------------------- IO


def _normalise_quotes(df: pd.DataFrame, as_of: date) -> pd.DataFrame:
    missing = [c for c in ("expiry", "option_type", "strike", "bid", "ask") if c not in df.columns]
    if missing:
        raise ValueError(f"Quote table is missing columns: {missing}")
    out = df.copy()
    for col in ("volume", "open_interest"):
        out[col] = pd.to_numeric(out.get(col, 0), errors="coerce").fillna(0)
    if "last_trade_date" not in out.columns:
        out["last_trade_date"] = pd.NaT
    out["last_trade_date"] = pd.to_datetime(out["last_trade_date"], errors="coerce", utc=True).dt.tz_localize(None)
    out["expiry"] = pd.to_datetime(out["expiry"]).dt.strftime("%Y-%m-%d")
    out["option_type"] = out["option_type"].str.lower()
    for col in ("strike", "bid", "ask"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["T"] = [(date.fromisoformat(e) - as_of).days / 365.0 for e in out["expiry"]]
    return out[QUOTE_COLUMNS[:7] + ["last_trade_date", "T"]].reset_index(drop=True)


def save_snapshot(chain: OptionChain, data_dir: str | Path | None = None, name: str | None = None) -> Path:
    """Write quotes (CSV) and metadata (JSON); returns the CSV path."""
    directory = Path(data_dir) if data_dir else default_data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    stem = name or f"{chain.symbol}_{chain.as_of.isoformat()}"
    csv_path = directory / f"{stem}.csv"
    chain.quotes.drop(columns=["T"], errors="ignore").to_csv(csv_path, index=False)
    meta = {
        "symbol": chain.symbol,
        "as_of": chain.as_of.isoformat(),
        "spot": chain.spot,
        "rate": chain.rate,
        "source": chain.source,
        "written_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (directory / f"{stem}.json").write_text(json.dumps(meta, indent=2))
    return csv_path


def load_snapshot(csv_path: str | Path, source: str = "snapshot") -> OptionChain:
    """Load a snapshot written by save_snapshot (CSV next to a JSON of the same stem)."""
    csv_path = Path(csv_path)
    meta = json.loads(csv_path.with_suffix(".json").read_text())
    as_of = date.fromisoformat(meta["as_of"])
    quotes = _normalise_quotes(pd.read_csv(csv_path), as_of)
    return OptionChain(meta["symbol"], as_of, float(meta["spot"]), float(meta["rate"]), quotes, source)


def load_sample(symbol: str = "SPY", data_dir: str | Path | None = None) -> OptionChain:
    """The committed synthetic snapshot, regardless of any live snapshots present."""
    directory = Path(data_dir) if data_dir else default_data_dir()
    path = directory / f"sample_{symbol.upper()}.csv"
    if not path.exists():
        raise SnapshotNotFound(f"No sample snapshot at {path}. Generate it with scripts/make_sample.py.")
    return load_snapshot(path, source="sample")


def load_chain(symbol: str, data_dir: str | Path | None = None, allow_sample: bool = False) -> OptionChain:
    """Most recent live snapshot for symbol; the sample only if allow_sample is set.

    Raises:
        SnapshotNotFound: If nothing is available. The sample is never used silently.
    """
    directory = Path(data_dir) if data_dir else default_data_dir()
    symbol = symbol.upper()
    snaps = sorted(directory.glob(f"{symbol}_[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9].csv"))
    if snaps:
        return load_snapshot(snaps[-1])
    sample = directory / f"sample_{symbol}.csv"
    if allow_sample and sample.exists():
        return load_snapshot(sample, source="sample")
    raise SnapshotNotFound(
        f"No snapshot for {symbol} in {directory}. Run `python -m volrisk.cli build --symbol {symbol}` "
        "to fetch one, or pass allow_sample=True / --sample to use the synthetic sample."
    )


# ------------------------------------------------------------------------- live


def fetch_risk_free_rate(default: float = DEFAULT_RATE) -> float:
    """Latest 13-week T-bill yield (^IRX) as a continuously compounded rate; default on failure."""
    try:
        import yfinance as yf

        hist = yf.Ticker("^IRX").history(period="5d")["Close"].dropna()
        if hist.empty:
            return default
        return math.log(1.0 + float(hist.iloc[-1]) / 100.0)
    except Exception:  # network or parsing problems must not break a build
        return default


def _spread_expiries(candidates: list[str], as_of: date, n: int) -> list[str]:
    """Pick up to n expiries spread geometrically in maturity (a dense run of weeklies would
    otherwise cover only the first month and leave the term structure unconstrained)."""
    if len(candidates) <= n:
        return candidates
    days = np.array([(date.fromisoformat(e) - as_of).days for e in candidates], dtype=float)
    targets = np.geomspace(days[0], days[-1], n)
    picked = {int(np.argmin(np.abs(days - t))) for t in targets}
    return [candidates[i] for i in sorted(picked)]


def fetch_chain_yfinance(
    symbol: str,
    rate: float | None = None,
    max_expiries: int = 8,
    min_days: int = 7,
    max_days: int = 400,
) -> OptionChain:
    """Download the option chain from Yahoo Finance.

    Raises:
        MarketDataError: On any failure (no network, delisted symbol, empty chain).
            Nothing silently falls back to stale or synthetic data.
    """
    try:
        import warnings

        import yfinance as yf

        warnings.filterwarnings("ignore", module="yfinance")
        warnings.filterwarnings("ignore", message=".*inplace.*")
        tk = yf.Ticker(symbol)
        expiries = list(tk.options)
        if not expiries:
            raise MarketDataError(f"Yahoo returned no option expiries for {symbol}.")
        spot_hist = tk.history(period="5d")["Close"].dropna()
        if spot_hist.empty:
            raise MarketDataError(f"No recent price history for {symbol}.")
        spot = float(spot_hist.iloc[-1])
        as_of = date.today()

        candidates = [e for e in expiries if min_days <= (date.fromisoformat(e) - as_of).days <= max_days]
        chosen = _spread_expiries(candidates, as_of, max_expiries)
        frames = []
        for exp in chosen:
            oc = tk.option_chain(exp)
            for opt_type, table in (("call", oc.calls), ("put", oc.puts)):
                t = table.rename(columns={"openInterest": "open_interest", "lastTradeDate": "last_trade_date"})
                t = t.assign(expiry=exp, option_type=opt_type)
                frames.append(t[[c for c in QUOTE_COLUMNS if c in t.columns]])
    except MarketDataError:
        raise
    except Exception as exc:
        raise MarketDataError(f"Could not fetch {symbol} from Yahoo Finance: {exc}") from exc

    if not frames:
        raise MarketDataError(f"No expiries between {min_days} and {max_days} days for {symbol}.")
    r = fetch_risk_free_rate() if rate is None else rate
    return OptionChain(symbol.upper(), as_of, spot, r, _normalise_quotes(pd.concat(frames), as_of), "live")


# ---------------------------------------------------------------------- cleaning


def clean_quotes(chain: OptionChain, cfg: CleaningConfig | None = None) -> tuple[pd.DataFrame, dict]:
    """Apply quality filters in order; returns (clean quotes, dropped-count per reason)."""
    cfg = cfg or CleaningConfig()
    df = chain.quotes.copy()
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    as_of_ts = pd.Timestamp(chain.as_of)

    rules = [
        ("invalid_numbers", df[["strike", "bid", "ask"]].isna().any(axis=1) | (df["strike"] <= 0)),
        ("zero_bid", df["bid"] <= 0),
        ("crossed_or_locked", df["ask"] <= df["bid"]),
        ("short_expiry", df["T"] < cfg.min_days / 365.0),
        ("stale_last_trade", (as_of_ts - df["last_trade_date"]).dt.days > cfg.max_stale_days),
        ("low_open_interest", df["open_interest"] < cfg.min_open_interest),
        ("low_volume", df["volume"] < cfg.min_volume),
        ("low_price", df["mid"] < cfg.min_mid),
        ("wide_spread", (df["ask"] - df["bid"]) / df["mid"] > cfg.max_rel_spread),
    ]
    dropped: dict[str, int] = {}
    keep = pd.Series(True, index=df.index)
    for name, bad in rules:
        bad = bad.fillna(False) & keep
        dropped[name] = int(bad.sum())
        keep &= ~bad
    dropped["kept"] = int(keep.sum())
    dropped["raw"] = int(len(df))
    return df[keep].reset_index(drop=True), dropped


def estimate_forwards(clean: pd.DataFrame, spot: float, rate: float, n_pairs: int = 5) -> pd.DataFrame:
    """Per-expiry forward from put-call parity: F = K + (C - P) e^{rT}.

    Uses the n_pairs strikes closest to spot that have both a call and a put,
    and takes the median (robust to one bad pair). Expiries with fewer than two
    pairs are dropped: a forward built from one pair is not trustworthy.
    Also returns the implied carry yield q = r - ln(F / S) / T.
    """
    rows = []
    for expiry, grp in clean.groupby("expiry"):
        T = float(grp["T"].iloc[0])
        calls = grp[grp.option_type == "call"].set_index("strike")["mid"]
        puts = grp[grp.option_type == "put"].set_index("strike")["mid"]
        common = calls.index.intersection(puts.index)
        if len(common) < 2:
            continue
        nearest = sorted(common, key=lambda k: abs(k - spot))[:n_pairs]
        estimates = [k + (calls[k] - puts[k]) * math.exp(rate * T) for k in nearest]
        fwd = float(np.median(estimates))
        rows.append({
            "expiry": expiry, "T": T, "forward": fwd,
            "implied_q": rate - math.log(fwd / spot) / T, "n_pairs": len(nearest),
        })
    return pd.DataFrame(rows, columns=["expiry", "T", "forward", "implied_q", "n_pairs"])


def prepare_chain(chain: OptionChain, cfg: CleaningConfig | None = None) -> PreparedChain:
    """Clean, extract forwards, and keep only OTM quotes (calls above F, puts below).

    OTM options carry almost no early-exercise value and the most vega per unit
    of bid-ask noise, so they give the cleanest implied vols; this is also why
    treating American-style ETF options as European is acceptable here.
    """
    cfg = cfg or CleaningConfig()
    clean, report = clean_quotes(chain, cfg)
    forwards = estimate_forwards(clean, chain.spot, chain.rate, cfg.n_forward_pairs)
    report["expiries_dropped_no_forward"] = int(clean["expiry"].nunique() - len(forwards))

    df = clean.merge(forwards[["expiry", "forward"]], on="expiry", how="inner")
    df["k"] = np.log(df["strike"] / df["forward"])
    is_otm = ((df["option_type"] == "call") & (df["strike"] >= df["forward"])) | (
        (df["option_type"] == "put") & (df["strike"] < df["forward"])
    )
    otm = df[is_otm].sort_values(["T", "strike"]).reset_index(drop=True)
    report["itm_dropped"] = int(len(df) - len(otm))
    report["otm_used"] = int(len(otm))
    cols = ["expiry", "T", "strike", "option_type", "bid", "ask", "mid", "forward", "k", "volume", "open_interest"]
    return PreparedChain(chain.symbol, chain.as_of, chain.spot, chain.rate, forwards, otm[cols], report, chain.source)
