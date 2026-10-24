# C++ core

A small C++17 port of three pieces of the Python library: Black-Scholes (price, delta, gamma, vega),
European Monte Carlo (with optional antithetic variates) and the CRR binomial tree (European and American).
It is deliberately basic: plain functions, `std::vector`, no external dependencies.

```
include/pricing.hpp      public API
src/black_scholes.cpp    closed form, normal cdf via std::erfc
src/monte_carlo.cpp      std::mt19937_64, returns price and standard error
src/binomial.cpp         backward induction on one std::vector
src/main.cpp             demo, prints values next to the Python ones
tests/test_pricing.cpp   assert-style runner, reference values taken from the Python library
bindings/pybind.cpp      pybind11 module `cpp_pricing`
benchmark.py             C++ vs Python, speed and accuracy
```

## Build and run

No CMake, just `make` (needs a C++17 compiler; the Python module needs `pybind11` in `../.venv`).

```
make all      # demo + tests
make test     # run the tests (reference values: S=K=100, T=1, r=5%, sigma=20%, q=2%)
make run      # run the demo
make python   # build cpp_pricing*.so next to benchmark.py
make bench    # build the module and run the benchmark
make clean
```

## Benchmark

Measured on one run (Apple laptop, `-O2`, best of 3, Python 3.15, numpy):

| Case | Python | C++ | Speedup |
|---|---|---|---|
| American put, binomial 2000 steps | 41.1 ms | 26.4 ms | 1.6x |
| European call, binomial 2000 steps | 4.0 ms | 0.9 ms | 4.5x |
| Monte Carlo 1e6 paths | 13.8 ms | 29.1 ms | 0.5x (slower) |

Binomial prices agree with Python to 1e-14. Monte Carlo prices differ only by sampling noise
(standard error about 0.014 in both).

Reading the numbers honestly:

- The Python MC is already vectorised numpy, so it runs mostly in C. My scalar loop with `std::normal_distribution`
  is slower. C++ is not automatically faster than numpy.
- The American tree is only modestly faster because the Python version is vectorised per time step, and my C++
  version calls `std::pow` for every node. Precomputing the spot ladder would help.
- The simple European tree shows a bigger gap because the Python version there does little work per step.

## What I would extend next

- Precompute powers of u/d in the tree instead of calling `std::pow` per node.
- Multithreaded MC with independent RNG streams per thread (and Sobol sequences).
- Normal inverse cdf and a Box-Muller or ziggurat sampler to control the RNG cost.
- Heston and barrier pricing, then Greeks by bump-and-reprice with common random numbers.
- Proper test framework (Catch2/GoogleTest) and CMake once the code grows beyond a few files.

## Why C++ shows up in a bank

Pricing engines and risk systems evaluate millions of trades and scenarios, so per-call cost and memory layout matter.
Latency-sensitive code (market making, real-time Greeks) avoids garbage collection pauses and interpreter overhead.
C++ also gives control over floating-point behaviour and data layout, which matters for reproducible numerics.
Many quant libraries are C++ with Python on top, which is the same shape as this folder: a compiled core
exposed to Python through pybind11. Python remains the better tool for research and prototyping.
