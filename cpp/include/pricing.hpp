#pragma once
#include <string>

namespace pricing {

struct McResult {
    double price;
    double std_error;
};

struct Greeks {
    double delta, gamma, vega;
};

// option_type: "call" or "put". Throws std::invalid_argument on bad input.
double bs_price(double S, double K, double T, double r, double sigma,
                const std::string& option_type, double q = 0.0);

Greeks bs_greeks(double S, double K, double T, double r, double sigma,
                 const std::string& option_type, double q = 0.0);

McResult mc_price(double S, double K, double T, double r, double sigma,
                  const std::string& option_type, double q = 0.0,
                  long n_paths = 100000, bool antithetic = false,
                  unsigned long long seed = 42);

// exercise_style: "european" or "american"
double binomial_price(double S, double K, double T, double r, double sigma,
                      const std::string& option_type, double q = 0.0,
                      int n_steps = 500,
                      const std::string& exercise_style = "european");

}  // namespace pricing
