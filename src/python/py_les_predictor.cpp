// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "openvino/genai/les_predictor.hpp"
#include "py_utils.hpp"

namespace py = pybind11;

void init_les_predictor(py::module_& m) {
    py::class_<ov::genai::LESPredictor>(m, "LESPredictor", R"(
        Expert predictor for lookahead expert scheduling (LES) of Mixture-of-Experts models.

        One small head per MoE layer predicts, one layer ahead, which routed experts the router of that layer
        selects, from the current and the previous token; it predicts as many experts as the router selects.
        GenAI builds the heads in a separate model and passes it to the device plugin (GPU), which runs them with
        MoE offloading to prefetch the offloaded experts the next MoE layer is likely to use.
        The model results do not change. Pass it to the pipeline as the `les_predictor` property, e.g.
        LLMPipeline(models_path, "GPU", les_predictor=LESPredictor(path)); the property also accepts the path
        to the predictor file (str or pathlib.Path): LLMPipeline(models_path, "GPU", les_predictor=path).
    )")
        .def(py::init([](const std::filesystem::path& path) {
                 return ov::genai::LESPredictor(path);
             }),
             py::arg("path"),
             R"(
                path (os.PathLike): Path to the predictor file in safetensors format, with the tensors
                expert_predictors.<i>.{weight, previous_weight, selection_bias, input_mode_code}.
             )")
        .def("__bool__", [](const ov::genai::LESPredictor& self) {
            return static_cast<bool>(self);
        });
}
