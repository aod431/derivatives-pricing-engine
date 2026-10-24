"""Data layer: cleaning rules, forward extraction, snapshot IO."""

import math
from datetime import date

import pandas as pd
import pytest

from volrisk.data import (
    CleaningConfig, OptionChain, SnapshotNotFound, clean_quotes, estimate_forwards, load_chain,
    prepare_chain, save_snapshot, _normalise_quotes,
)

TRUE_Q = 0.013


def test_deliberately_bad_rows_are_all_dropped(chain, prepared):
    raw = len(chain.quotes)
    kept = prepared.report["kept"]
    assert raw - kept >= 6
    bad = prepared.report
    assert bad["crossed_or_locked"] >= 1 and bad["short_expiry"] >= 1
    assert bad["stale_last_trade"] >= 1 and bad["low_open_interest"] >= 1 and bad["wide_spread"] >= 1


def test_no_expiry_shorter_than_min_days_survives(prepared):
    assert (prepared.otm["T"] >= 7 / 365).all()


def test_every_kept_quote_passes_quality_thresholds(chain):
    clean, _ = clean_quotes(chain)
    cfg = CleaningConfig()
    assert (clean["bid"] > 0).all() and (clean["ask"] > clean["bid"]).all()
    assert (clean["mid"] >= cfg.min_mid).all()
    assert ((clean["ask"] - clean["bid"]) / clean["mid"] <= cfg.max_rel_spread).all()
    assert (clean["open_interest"] >= cfg.min_open_interest).all()


def test_only_otm_quotes_are_kept(prepared):
    df = prepared.otm
    assert ((df.option_type == "call") == (df.strike >= df.forward)).all()


def test_parity_forward_recovers_true_forward(chain, prepared):
    """Heston sample was built with q = 1.3%: the parity forward must be close to the true one."""
    for row in prepared.forwards.itertuples():
        true_fwd = chain.spot * math.exp((chain.rate - TRUE_Q) * row.T)
        assert row.forward == pytest.approx(true_fwd, rel=2e-3)
        assert abs(row.implied_q - TRUE_Q) < 0.003 + 0.0006 / row.T**0.5


def test_forward_needs_two_pairs(chain):
    clean, _ = clean_quotes(chain)
    one_expiry = clean[clean.expiry == clean.expiry.iloc[0]]
    only_calls = one_expiry[one_expiry.option_type == "call"]
    assert estimate_forwards(only_calls, chain.spot, chain.rate).empty


def test_forward_is_robust_to_one_bad_pair(chain):
    clean, _ = clean_quotes(chain)
    e = clean.expiry.unique()[1]
    base = estimate_forwards(clean, chain.spot, chain.rate).set_index("expiry").loc[e, "forward"]
    corrupted = clean.copy()
    idx = corrupted[(corrupted.expiry == e) & (corrupted.option_type == "call")].index[
        (corrupted[(corrupted.expiry == e) & (corrupted.option_type == "call")].strike - chain.spot).abs().argmin()
    ]
    corrupted.loc[idx, "mid"] += 5.0
    got = estimate_forwards(corrupted, chain.spot, chain.rate).set_index("expiry").loc[e, "forward"]
    assert got == pytest.approx(base, abs=0.5)  # median ignores the outlier pair


def test_snapshot_roundtrip(tmp_path, chain):
    save_snapshot(OptionChain("SPY", chain.as_of, chain.spot, chain.rate, chain.quotes, "live"), tmp_path)
    loaded = load_chain("SPY", tmp_path)
    assert loaded.spot == chain.spot and loaded.as_of == chain.as_of
    assert len(loaded.quotes) == len(chain.quotes)


def test_missing_snapshot_raises_and_sample_is_opt_in(tmp_path, chain):
    with pytest.raises(SnapshotNotFound):
        load_chain("SPY", tmp_path)
    # a sample file exists but must not be used silently
    save_snapshot(chain, tmp_path, name="sample_SPY")
    with pytest.raises(SnapshotNotFound):
        load_chain("SPY", tmp_path)
    assert load_chain("SPY", tmp_path, allow_sample=True).source == "sample"


def test_missing_columns_rejected():
    with pytest.raises(ValueError):
        _normalise_quotes(pd.DataFrame({"strike": [1.0]}), date(2026, 1, 1))
