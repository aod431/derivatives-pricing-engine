"""API: contract, validation errors and consistency with the library."""

import pytest
from fastapi.testclient import TestClient

from volrisk.api import SurfaceStore, create_app
from volrisk.data import SnapshotNotFound, prepare_chain
from volrisk.surface import build_surface


@pytest.fixture(scope="module")
def client(surface):
    def loader(symbol: str):
        if symbol != "SPY":
            raise SnapshotNotFound(f"No snapshot for {symbol}")
        return surface

    return TestClient(create_app(SurfaceStore(loader)))


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_surface_endpoint(client):
    r = client.get("/surface/spy")
    assert r.status_code == 200
    body = r.json()
    assert body["symbol"] == "SPY" and body["arbitrage_free"] is True
    assert len(body["slices"]) == 5 and {"a", "b", "rho", "m", "s"} == set(body["slices"][0]["svi"])


def test_unknown_symbol_is_404(client):
    r = client.get("/surface/ZZZZ")
    assert r.status_code == 404 and "No snapshot" in r.json()["detail"]


def test_price_uses_surface_vol(client, surface):
    r = client.post("/price", json={"symbol": "SPY", "option_type": "call", "strike": 500, "maturity": 0.5})
    assert r.status_code == 200
    body = r.json()
    assert body["vol_source"] == "surface"
    assert body["vol"] == pytest.approx(surface.vol(500.0, 0.5))
    assert 0 < body["greeks"]["delta"] < 1 and body["warnings"] == []


def test_price_vol_override_and_shift(client):
    base = {"symbol": "SPY", "option_type": "put", "strike": 480, "maturity": 0.5}
    over = client.post("/price", json={**base, "vol_override": 0.30}).json()
    assert over["vol"] == 0.30 and over["vol_source"] == "override"
    surf = client.post("/price", json=base).json()
    shifted = client.post("/price", json={**base, "vol_shift": 0.01}).json()
    assert shifted["price"] > surf["price"]


def test_price_warns_on_extrapolation(client):
    r = client.post("/price", json={"symbol": "SPY", "option_type": "call", "strike": 500, "maturity": 3.0})
    assert r.status_code == 200 and "beyond" in r.json()["warnings"][0]


@pytest.mark.parametrize("payload", [
    {"symbol": "SPY", "option_type": "straddle", "strike": 500, "maturity": 0.5},
    {"symbol": "SPY", "option_type": "call", "strike": -5, "maturity": 0.5},
    {"symbol": "SPY", "option_type": "call", "strike": 500, "maturity": 0},
    {"symbol": "SPY", "option_type": "call", "strike": 500, "maturity": 0.5, "vol_override": 0},
    {"symbol": "SPY", "option_type": "call", "strike": 500},
])
def test_price_validation_gives_422(client, payload):
    assert client.post("/price", json=payload).status_code == 422


def test_risk_endpoint(client):
    payload = {
        "symbol": "SPY",
        "positions": [
            {"option_type": "call", "strike": 520, "maturity": 0.25, "quantity": -100},
            {"option_type": "put", "strike": 480, "maturity": 0.25, "quantity": -100},
            {"option_type": "call", "strike": 500, "maturity": 0.5, "quantity": 50},
        ],
        "var": {"n_sims": 5000},
    }
    r = client.post("/risk", json=payload)
    assert r.status_code == 200, r.text
    b = r.json()
    assert set(b) >= {"value", "greeks", "bucketed_vega", "scenarios", "var_es", "positions"}
    sc = b["scenarios"]
    assert len(sc["pnl"]) == len(sc["spot_shifts"]) and len(sc["pnl"][0]) == len(sc["vol_shifts"])
    assert abs(sum(b["bucketed_vega"].values()) - b["greeks"]["vega"]) < 1e-2 * abs(b["greeks"]["vega"]) + 1e-6
    assert b["var_es"]["full_revaluation"]["es"] >= b["var_es"]["full_revaluation"]["var"] > 0


@pytest.mark.parametrize("mutation", [
    {"positions": []},
    {"positions": [{"option_type": "call", "strike": 500, "maturity": 0.5, "quantity": 0}]},
    {"var": {"confidence": 1.5}},
    {"var": {"n_sims": 10}},
    {"spot_shifts": [-1.0]},
])
def test_risk_validation_gives_422(client, mutation):
    base = {"symbol": "SPY", "positions": [{"option_type": "call", "strike": 500, "maturity": 0.5, "quantity": 1}]}
    assert client.post("/risk", json={**base, **mutation}).status_code == 422


def test_default_store_uses_sample_only_when_allowed(monkeypatch, tmp_path):
    monkeypatch.setenv("VOLRISK_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("VOLRISK_ALLOW_SAMPLE", raising=False)
    assert TestClient(create_app()).get("/surface/SPY").status_code == 404
