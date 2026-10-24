#include <cstdio>

#include "pricing.hpp"

int main() {
    // same parameters as the Python docs: S=K=100, T=1, r=5%, sigma=20%, q=2%
    double S = 100, K = 100, T = 1, r = 0.05, sigma = 0.2, q = 0.02;

    std::printf("Black-Scholes call: %.6f (python: 9.227006)\n",
                pricing::bs_price(S, K, T, r, sigma, "call", q));
    std::printf("Black-Scholes put : %.6f (python: 6.330081)\n",
                pricing::bs_price(S, K, T, r, sigma, "put", q));

    auto g = pricing::bs_greeks(S, K, T, r, sigma, "call", q);
    std::printf("Call delta %.6f gamma %.6f vega %.6f\n", g.delta, g.gamma, g.vega);

    auto mc = pricing::mc_price(S, K, T, r, sigma, "call", q, 1000000, true);
    std::printf("MC call (1e6, antithetic): %.4f +/- %.4f\n", mc.price, mc.std_error);

    std::printf("Binomial European call (500): %.6f (python: 9.223118)\n",
                pricing::binomial_price(S, K, T, r, sigma, "call", q, 500, "european"));
    std::printf("Binomial American put  (500): %.6f (python: 6.658816)\n",
                pricing::binomial_price(S, K, T, r, sigma, "put", q, 500, "american"));
    return 0;
}
