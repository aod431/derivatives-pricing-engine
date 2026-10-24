#include "pricing.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace pricing {

double binomial_price(double S, double K, double T, double r, double sigma,
                      const std::string& type, double q, int n,
                      const std::string& style) {
    if (S <= 0 || K <= 0 || T <= 0 || sigma <= 0 || n <= 0)
        throw std::invalid_argument("S, K, T, sigma and n_steps must be positive");
    if (type != "call" && type != "put")
        throw std::invalid_argument("option_type must be 'call' or 'put'");
    if (style != "european" && style != "american")
        throw std::invalid_argument("exercise_style must be 'european' or 'american'");

    double dt = T / n;
    double u = std::exp(sigma * std::sqrt(dt));
    double d = 1.0 / u;
    double p = (std::exp((r - q) * dt) - d) / (u - d);
    double disc = std::exp(-r * dt);
    double sign = (type == "call") ? 1.0 : -1.0;
    bool american = (style == "american");

    // node j at step i has spot S * u^j * d^(i-j)
    std::vector<double> v(n + 1);
    for (int j = 0; j <= n; ++j)
        v[j] = std::max(sign * (S * std::pow(u, j) * std::pow(d, n - j) - K), 0.0);

    for (int i = n - 1; i >= 0; --i) {
        for (int j = 0; j <= i; ++j) {
            v[j] = disc * (p * v[j + 1] + (1.0 - p) * v[j]);
            if (american) {
                double spot = S * std::pow(u, j) * std::pow(d, i - j);
                v[j] = std::max(v[j], sign * (spot - K));
            }
        }
    }
    return v[0];
}

}  // namespace pricing
