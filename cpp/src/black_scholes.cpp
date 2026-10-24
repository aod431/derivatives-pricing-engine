#include "pricing.hpp"

#include <cmath>
#include <stdexcept>

namespace pricing {

static double norm_cdf(double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); }
static double norm_pdf(double x) { return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI); }

static void check(double S, double K, double T, double sigma, const std::string& type) {
    if (S <= 0 || K <= 0 || T <= 0 || sigma <= 0)
        throw std::invalid_argument("S, K, T and sigma must be positive");
    if (type != "call" && type != "put")
        throw std::invalid_argument("option_type must be 'call' or 'put'");
}

double bs_price(double S, double K, double T, double r, double sigma,
                const std::string& type, double q) {
    check(S, K, T, sigma, type);
    double sq = sigma * std::sqrt(T);
    double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sq;
    double d2 = d1 - sq;
    double fs = S * std::exp(-q * T), fk = K * std::exp(-r * T);
    if (type == "call") return fs * norm_cdf(d1) - fk * norm_cdf(d2);
    return fk * norm_cdf(-d2) - fs * norm_cdf(-d1);
}

Greeks bs_greeks(double S, double K, double T, double r, double sigma,
                 const std::string& type, double q) {
    check(S, K, T, sigma, type);
    double sq = sigma * std::sqrt(T);
    double d1 = (std::log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / sq;
    double eq = std::exp(-q * T);
    Greeks g;
    g.delta = (type == "call") ? eq * norm_cdf(d1) : eq * (norm_cdf(d1) - 1.0);
    g.gamma = eq * norm_pdf(d1) / (S * sq);
    g.vega = S * eq * norm_pdf(d1) * std::sqrt(T);
    return g;
}

}  // namespace pricing
