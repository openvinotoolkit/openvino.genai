// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "openvino/genai/expert_predictor.hpp"
#include "py_utils.hpp"

namespace py = pybind11;

void init_expert_predictor(py::module_& m) {
    py::class_<ov::genai::ExpertPredictor>(m, "ExpertPredictor", R"(
        Expert predictor for lookahead expert scheduling (LES) of Mixture-of-Experts models.

        One small head per MoE layer predicts, one layer ahead, which routed experts the router of that layer
        selects. GenAI builds the heads inside the language model and exposes the predicted expert indices as
        model outputs `les_predicted_experts.<layer>`, so that a device plugin can prefetch offloaded experts.
        The model results do not change. Pass it to the pipeline as the `expert_predictor` property, e.g.
        LLMPipeline(models_path, "GPU", expert_predictor=ExpertPredictor(path)).
    )")
        .def(py::init([](const std::filesystem::path& path, size_t top_k, bool use_previous_token) {
                 return ov::genai::ExpertPredictor(path, top_k, use_previous_token);
             }),
             py::arg("path"),
             py::arg("top_k") = 0,
             py::arg("use_previous_token") = true,
             R"(
                path (os.PathLike): Path to the predictor file in safetensors format, with the tensors
                expert_predictors.<i>.{weight, previous_weight, selection_bias, input_mode_code}.
                top_k (int): Experts predicted per MoE layer and token, 0 means as many as the router selects.
                use_previous_token (bool): Use the previous token input of the heads.
             )")
        .def_property_readonly("top_k", &ov::genai::ExpertPredictor::get_top_k)
        .def_property_readonly("use_previous_token", &ov::genai::ExpertPredictor::get_use_previous_token)
        .def("__bool__", [](const ov::genai::ExpertPredictor& self) {
            return static_cast<bool>(self);
        });
}
