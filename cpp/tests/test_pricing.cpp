// Minimal test runner, no framework. Reference values come from the Python
// library in src/pricing with S=K=100, T=1, r=0.05, sigma=0.2, q=0.02.
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "pricing.hpp"

static int failures = 0;

static void check_close(const char* name, double got, double want, double tol) {
    bool ok = std::fabs(got - want) <= tol;
    std::printf("[%s] %s: got %.8f want %.8f\n", ok ? "ok" : "FAIL", name, got, want);
    if (!ok) ++failures;
}

static void check_true(const char* name, bool cond) {
    std::printf("[%s] %s\n", cond ? "ok" : "FAIL", name);
    if (!cond) ++failures;
}

int main() {
    double S = 100, K = 100, T = 1, r = 0.05, sigma = 0.2, q = 0.02;

    check_close("bs call", pricing::bs_price(S, K, T, r, sigma, "call", q), 9.227005508154036, 1e-10);
    check_close("bs put", pricing::bs_price(S, K, T, r, sigma, "put", q), 6.330080627549918, 1e-10);

    auto g = pricing::bs_greeks(S, K, T, r, sigma, "call", q);
    check_close("delta", g.delta, 0.586851146134764, 1e-10);
    check_close("gamma", g.gamma, 0.018950578755008718, 1e-10);
    check_close("vega", g.vega, 37.901157510017434, 1e-8);

    // put-call parity: C - P = S e^{-qT} - K e^{-rT}
    double parity = pricing::bs_price(S, K, T, r, sigma, "call", q) -
                    pricing::bs_price(S, K, T, r, sigma, "put", q);
    check_close("put-call parity", parity, S * std::exp(-q * T) - K * std::exp(-r * T), 1e-10);

    check_close("binomial european call",
                pricing::binomial_price(S, K, T, r, sigma, "call", q, 500, "european"),
                9.223118455217449, 1e-9);
    check_close("binomial american put",
                pricing::binomial_price(S, K, T, r, sigma, "put", q, 500, "american"),
                6.6588158934667, 1e-9);

    auto mc = pricing::mc_price(S, K, T, r, sigma, "call", q, 1000000, false);
    check_true("mc within 4 standard errors of BS", std::fabs(mc.price - 9.227005508154036) < 4 * mc.std_error);
    auto mca = pricing::mc_price(S, K, T, r, sigma, "call", q, 1000000, true);
    check_true("mc antithetic within 4 standard errors of BS", std::fabs(mca.price - 9.227005508154036) < 4 * mca.std_error);

    bool threw = false;
    try { pricing::bs_price(-1, K, T, r, sigma, "call"); } catch (const std::invalid_argument&) { threw = true; }
    check_true("negative spot throws", threw);

    std::printf("%s\n", failures ? "SOME TESTS FAILED" : "all tests passed");
    return failures ? 1 : 0;
}
