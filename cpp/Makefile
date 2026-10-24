CXX      ?= c++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -Iinclude
PY       := ../.venv/bin/python
LIB_SRC  := src/black_scholes.cpp src/monte_carlo.cpp src/binomial.cpp
PYEXT    = $(shell $(PY) -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")

all: build/demo build/test_pricing

build/demo: $(LIB_SRC) src/main.cpp include/pricing.hpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(LIB_SRC) src/main.cpp -o $@

build/test_pricing: $(LIB_SRC) tests/test_pricing.cpp include/pricing.hpp
	@mkdir -p build
	$(CXX) $(CXXFLAGS) $(LIB_SRC) tests/test_pricing.cpp -o $@

test: build/test_pricing
	./build/test_pricing

run: build/demo
	./build/demo

# Python module; the .so lands next to benchmark.py
python: $(LIB_SRC) bindings/pybind.cpp include/pricing.hpp
	$(CXX) $(CXXFLAGS) -shared -fPIC -undefined dynamic_lookup \
		$$($(PY) -m pybind11 --includes) \
		$(LIB_SRC) bindings/pybind.cpp -o cpp_pricing$(PYEXT)

bench: python
	$(PY) benchmark.py

clean:
	rm -rf build cpp_pricing*.so
