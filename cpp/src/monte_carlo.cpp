#include "pricing.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

namespace pricing {

McResult mc_price(double S, double K, double T, double r, double sigma,
                  const std::string& type, double q, long n_paths,
                  bool antithetic, unsigned long long seed) {
    if (S <= 0 || K <= 0 || T <= 0 || sigma <= 0 || n_paths <= 0)
        throw std::invalid_argument("S, K, T, sigma and n_paths must be positive");
    if (type != "call" && type != "put")
        throw std::invalid_argument("option_type must be 'call' or 'put'");

    std::mt19937_64 gen(seed);
    std::normal_distribution<double> normal(0.0, 1.0);

    double drift = (r - q - 0.5 * sigma * sigma) * T;
    double vol = sigma * std::sqrt(T);
    double disc = std::exp(-r * T);
    double sign = (type == "call") ? 1.0 : -1.0;

    auto payoff = [&](double z) {
        double ST = S * std::exp(drift + vol * z);
        double p = sign * (ST - K);
        return disc * (p > 0 ? p : 0.0);
    };

    // With antithetic variates each sample is the mean of f(z) and f(-z),
    // so the standard error uses the n_paths pair averages.
    double sum = 0.0, sum_sq = 0.0;
    for (long i = 0; i < n_paths; ++i) {
        double z = normal(gen);
        double x = antithetic ? 0.5 * (payoff(z) + payoff(-z)) : payoff(z);
        sum += x;
        sum_sq += x * x;
    }
    double mean = sum / n_paths;
    double var = (sum_sq - n_paths * mean * mean) / (n_paths - 1);
    return {mean, std::sqrt(var / n_paths)};
}

}  // namespace pricing
