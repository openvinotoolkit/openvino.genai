// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "openvino/genai/bagel/pipeline.hpp"
#include "py_utils.hpp"
#include "tokenizer/tokenizers_path.hpp"

namespace py = pybind11;
namespace pyutils = ov::genai::pybind::utils;

using ov::genai::BagelImageGenerationConfig;
using ov::genai::BagelPerfMetrics;
using ov::genai::BagelPipeline;

void init_bagel_pipeline(py::module_& m) {
    py::class_<BagelImageGenerationConfig>(m,
                                           "BagelImageGenerationConfig",
                                           "BAGEL text-to-image / image editing config.")
        .def(py::init<>())
        .def_readwrite("size", &BagelImageGenerationConfig::size, "Text-to-image output side.")
        .def_readwrite("num_inference_steps", &BagelImageGenerationConfig::num_inference_steps)
        .def_readwrite("timestep_shift", &BagelImageGenerationConfig::timestep_shift)
        .def_readwrite("cfg_text_scale", &BagelImageGenerationConfig::cfg_text_scale)
        .def_readwrite("cfg_img_scale", &BagelImageGenerationConfig::cfg_img_scale, "Image editing only.")
        .def_readwrite("cfg_interval",
                       &BagelImageGenerationConfig::cfg_interval,
                       "Defaults to (0.4, 1.0) for text-to-image and (0.0, 1.0) for image editing.")
        .def_readwrite("cfg_renorm_min", &BagelImageGenerationConfig::cfg_renorm_min)
        .def_readwrite("seed", &BagelImageGenerationConfig::seed, "Matches numpy.random.RandomState(seed).randn.");

    py::class_<BagelPerfMetrics>(m, "BagelPerfMetrics", "BAGEL compile and per-stage inference times.")
        .def_readonly("compile_ms",
                      &BagelPerfMetrics::compile_ms,
                      "(name, ms) per compile_model and per NPUW_BAGEL static variant compiled on first use.")
        .def_readonly("infer_ms", &BagelPerfMetrics::infer_ms, "(stage, ms) per call, variant compile excluded.")
        .def_readonly("first_call_ms",
                      &BagelPerfMetrics::first_call_ms,
                      "Sum of first calls of new variants (compile + one inference).");

    py::class_<BagelPipeline>(m, "BagelPipeline", "BAGEL-7B-MoT pipeline: VQA, text-to-image and image editing.")
        .def(py::init([](const std::filesystem::path& models_path,
                         const std::string& device,
                         const std::optional<std::filesystem::path>& tokenizer_path,
                         const py::kwargs& kwargs) {
                 ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
                 return std::make_unique<BagelPipeline>(models_path,
                                                        device,
                                                        tokenizer_path.value_or(models_path),
                                                        pyutils::kwargs_to_any_map(kwargs));
             }),
             py::arg("models_path"),
             py::arg("device"),
             py::arg("tokenizer_path") = py::none(),
             R"(
            BagelPipeline class constructor.
            models_path (os.PathLike): Directory with the BAGEL IRs and bagel_ov_config.json, or an optimum-intel export.
            device (str): Inference device.
            tokenizer_path (os.PathLike | None): Directory with openvino_tokenizer.xml / openvino_detokenizer.xml;
                models_path if None.
            kwargs: Properties applied on top of the default per-model configuration.
        )")
        .def("generate_text",
             &BagelPipeline::generate_text,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("prompt"),
             py::arg("image") = py::none(),
             py::arg("max_new_tokens") = 256,
             py::arg("system_prompt") = "",
             "Answers the prompt, optionally about a u8 RGB image with [1, H, W, 3] or [H, W, 3] layout.")
        .def("generate_image",
             &BagelPipeline::generate_image,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("prompt"),
             py::arg_v("config", BagelImageGenerationConfig{}, "BagelImageGenerationConfig()"),
             "Text-to-image; returns a u8 [1, H, W, 3] RGB tensor.")
        .def("edit_image",
             &BagelPipeline::edit_image,
             py::call_guard<py::gil_scoped_release>(),
             py::arg("image"),
             py::arg("prompt"),
             py::arg_v("config", BagelImageGenerationConfig{}, "BagelImageGenerationConfig()"),
             "Edits a u8 RGB image; returns a u8 [1, H, W, 3] RGB tensor.")
        .def("get_perf_metrics", &BagelPipeline::get_perf_metrics, py::return_value_policy::copy)
        .def("reset_perf_metrics", &BagelPipeline::reset_perf_metrics);
}
