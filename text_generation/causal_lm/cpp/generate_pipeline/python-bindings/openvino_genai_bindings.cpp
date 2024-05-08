// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdlib>
#include <limits>
#include <functional>

#include "pybind11/pybind11.h"
#include <pybind11/stl.h>

struct GenerationConfig {
    bool do_sample;

};

namespace py = pybind11;


PYBIND11_MODULE(py_continuous_batching, m) {
    py::class_<GenerationConfig>(m, "GenerationConfig")
        .def(py::init<>())
        .def_readwrite("do_sample", &GenerationConfig::do_sample);

}
