#include <pybind11/pybind11.h>

#include "pricing.hpp"

namespace py = pybind11;

PYBIND11_MODULE(cpp_pricing, m) {
    m.doc() = "C++ pricing core (Black-Scholes, Monte Carlo, binomial tree)";

    py::class_<pricing::McResult>(m, "McResult")
        .def_readonly("price", &pricing::McResult::price)
        .def_readonly("std_error", &pricing::McResult::std_error);

    py::class_<pricing::Greeks>(m, "Greeks")
        .def_readonly("delta", &pricing::Greeks::delta)
        .def_readonly("gamma", &pricing::Greeks::gamma)
        .def_readonly("vega", &pricing::Greeks::vega);

    m.def("bs_price", &pricing::bs_price, py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("sigma"), py::arg("option_type") = "call", py::arg("q") = 0.0);
    m.def("bs_greeks", &pricing::bs_greeks, py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("sigma"), py::arg("option_type") = "call", py::arg("q") = 0.0);
    m.def("mc_price", &pricing::mc_price, py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("sigma"), py::arg("option_type") = "call", py::arg("q") = 0.0,
          py::arg("n_paths") = 100000, py::arg("antithetic") = false, py::arg("seed") = 42);
    m.def("binomial_price", &pricing::binomial_price, py::arg("S"), py::arg("K"), py::arg("T"),
          py::arg("r"), py::arg("sigma"), py::arg("option_type") = "call", py::arg("q") = 0.0,
          py::arg("n_steps") = 500, py::arg("exercise_style") = "european");
}
