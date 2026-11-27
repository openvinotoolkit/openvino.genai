// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>

#include "openvino/genai/classification/classification_pipeline.hpp"
#include "py_utils.hpp"
#include "tokenizer/tokenizers_path.hpp"

namespace py = pybind11;
using ov::genai::ClassificationActivation;
using ov::genai::ClassificationPipeline;
using ov::genai::ClassificationResult;

namespace pyutils = ov::genai::pybind::utils;

namespace {

const auto classification_pipeline_docstring = R"(
Classifies a prompt with optional images using a vision-language model exported as
`*ForSequenceClassification` (text-classification task).

The scores are computed from the logits of the last prompt token. No text is generated, and no chat template or
special tokens are added: the caller builds the complete model prompt.
)";

const auto classification_config_docstring = R"(
Structure to keep ClassificationPipeline configuration parameters.

Attributes:
    classification_activation (ClassificationActivation, optional):
        Activation applied to the logits. If not set, it is derived from `problem_type` in config.json:
        NONE for `regression`, SIGMOID for `multi_label_classification` or a single label, SOFTMAX otherwise.
)";

const auto classify_docstring = R"(
Classifies a prompt.

:param text: Complete model prompt. Images are referenced by native or zero-based `<ov_genai_image_N>` tags.
:type text: str
:param images: RGB uint8 tensors [H, W, 3] or [1, H, W, 3], one image per tensor.
:type images: list[openvino.Tensor]
:return: Logits and activated scores, both in label ID order.
:rtype: ClassificationResult
)";

}  // namespace

void init_classification_pipeline(py::module_& m) {
    py::enum_<ClassificationActivation>(m, "ClassificationActivation", "Activation applied to the classifier logits")
        .value("SOFTMAX", ClassificationActivation::SOFTMAX, "Single-label classification: scores of an item sum to 1")
        .value("SIGMOID", ClassificationActivation::SIGMOID, "Multi-label classification or a single relevance logit")
        .value("NONE", ClassificationActivation::NONE, "Raw logits");

    py::class_<ClassificationResult>(m, "ClassificationResult", "Result of classifying one item")
        .def(py::init<>())
        .def_readwrite("logits", &ClassificationResult::logits, "Raw logits [1, num_labels] in label ID order")
        .def_readwrite("scores",
                       &ClassificationResult::scores,
                       "Logits after the activation. scores[0][i] is the score of label ID i");

    auto classification_pipeline =
        py::class_<ClassificationPipeline>(m, "ClassificationPipeline", classification_pipeline_docstring);

    py::class_<ClassificationPipeline::Config>(classification_pipeline, "Config", classification_config_docstring)
        .def(py::init<>())
        .def(py::init([](const py::kwargs& kwargs) {
            return ClassificationPipeline::Config(pyutils::kwargs_to_any_map(kwargs));
        }))
        .def_readwrite("classification_activation", &ClassificationPipeline::Config::classification_activation);

    classification_pipeline
        .def(py::init([](const std::filesystem::path& models_path,
                         const std::string& device,
                         const std::optional<ClassificationPipeline::Config>& config,
                         const py::kwargs& kwargs) {
                 ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
                 const ov::AnyMap properties = pyutils::kwargs_to_any_map(kwargs);
                 py::gil_scoped_release rel;
                 if (config.has_value()) {
                     return std::make_unique<ClassificationPipeline>(models_path, device, *config, properties);
                 }
                 return std::make_unique<ClassificationPipeline>(models_path, device, properties);
             }),
             py::arg("models_path"),
             "Directory with the exported model, tokenizer and config.json",
             py::arg("device"),
             "Device to run the model on (e.g., CPU, GPU)",
             py::arg("config") = std::nullopt,
             "Pipeline configuration",
             R"(
            ClassificationPipeline class constructor.
            models_path (os.PathLike): Directory with the exported model, tokenizer and config.json.
            device (str): Device to run the model on (e.g., CPU, GPU).
            config (ClassificationPipeline.Config): Pipeline configuration.
            kwargs: Plugin and configuration properties, e.g. CACHE_DIR=/tmp/, classification_activation=ClassificationActivation.SIGMOID.
            )")
        .def(
            "classify",
            [](ClassificationPipeline& pipe,
               const std::string& text,
               const std::vector<ov::Tensor>& images) -> ClassificationResult {
                py::gil_scoped_release rel;
                return pipe.classify(text, images);
            },
            py::arg("text"),
            py::arg("images") = std::vector<ov::Tensor>{},
            classify_docstring);
}
