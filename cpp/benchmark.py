"""Compare the C++ core against the Python implementations (speed and accuracy).

Build the extension first: make python   (or just: make bench)
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cpp_pricing  # noqa: E402
from pricing.binomial import binomial_price as py_binomial  # noqa: E402
from pricing.black_scholes import black_scholes_price  # noqa: E402
from pricing.monte_carlo import mc_price as py_mc  # noqa: E402

S, K, T, R, SIGMA, Q = 100.0, 100.0, 1.0, 0.05, 0.2, 0.02


def best_of(fn, repeats=3):
    times, result = [], None
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return min(times), result


def row(name, t_py, t_cpp, detail):
    print(f"{name:<28}python {t_py*1e3:9.1f} ms   c++ {t_cpp*1e3:9.1f} ms   speedup {t_py/t_cpp:6.1f}x   {detail}")


bs = black_scholes_price(S, K, T, R, SIGMA, "call", Q)

t_py, p = best_of(lambda: py_binomial(S, K, T, R, SIGMA, "put", Q, 2000, "american"))
t_cpp, c = best_of(lambda: cpp_pricing.binomial_price(S, K, T, R, SIGMA, "put", Q, 2000, "american"))
row("binomial american put 2000", t_py, t_cpp, f"|diff| = {abs(p - c):.2e}")

t_py, p = best_of(lambda: py_binomial(S, K, T, R, SIGMA, "call", Q, 2000, "european"))
t_cpp, c = best_of(lambda: cpp_pricing.binomial_price(S, K, T, R, SIGMA, "call", Q, 2000, "european"))
row("binomial european call 2000", t_py, t_cpp, f"|diff| = {abs(p - c):.2e}")

t_py, p = best_of(lambda: py_mc(S, K, T, R, SIGMA, "call", Q, n_paths=1_000_000))
t_cpp, c = best_of(lambda: cpp_pricing.mc_price(S, K, T, R, SIGMA, "call", Q, 1_000_000))
row("monte carlo 1e6 paths", t_py, t_cpp,
    f"err vs BS: py {abs(p.price - bs):.4f} (se {p.std_error:.4f}), c++ {abs(c.price - bs):.4f} (se {c.std_error:.4f})")

print("\nPython MC is vectorised numpy; the C++ loop is scalar. Different RNGs, so MC prices differ by noise only.")
