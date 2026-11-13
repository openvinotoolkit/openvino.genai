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
Classification pipeline for vision-language models with a sequence-classification head.

Uses models exported with `--task text-classification` as vision, text embeddings and language model
parts. The language model must end in a classification head, not a token-generation head.

The pipeline returns the logits of the last prompt token and applies the configured activation.
It does not generate text, apply a chat template or add extra special tokens.
Model-specific prompt construction and input preparation are the caller's responsibility.
Native image tags and zero-based `<ov_genai_image_N>` tags are accepted.

Results contain owned logits [1, num_labels] and activated scores [1][num_labels], both in label ID order:
`scores[0][i]` belongs to label ID `i`.
)";

const auto classification_config_docstring = R"(
Structure to keep ClassificationPipeline configuration parameters.

Attributes:
    activation (ClassificationActivation, optional):
        Activation applied to the logits. Derived from config.json when not set: NONE for
        `problem_type == regression`, SIGMOID for `multi_label_classification` or a single label,
        SOFTMAX otherwise.
)";

const auto classify_docstring = R"(
Classifies one already-formatted prompt.

:param text: Complete model prompt, including any suffix required by the model. No chat template or extra
    special tokens are added. Images may use native tags or zero-based `<ov_genai_image_N>` tags.
:type text: str
:param images: Optional RGB uint8 tensors of shape [H, W, 3] or [N, H, W, 3].
    The number, order and content of the images must match the model's input requirements.
:type images: list[openvino.Tensor]
:return: Owned logits [1, num_labels] and activated scores [1][num_labels], both in label ID order.
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
        .def_readwrite("logits",
                       &ClassificationResult::logits,
                       "Owned raw logits of shape [1, num_labels] in label ID order, before activation")
        .def_readwrite("scores",
                       &ClassificationResult::scores,
                       "Scores after the activation in label ID order: scores[0][i] belongs to label ID i");

    auto classification_pipeline =
        py::class_<ClassificationPipeline>(m, "ClassificationPipeline", classification_pipeline_docstring);

    py::class_<ClassificationPipeline::Config>(classification_pipeline, "Config", classification_config_docstring)
        .def(py::init<>())
        .def(py::init([](const py::kwargs& kwargs) {
            return ClassificationPipeline::Config(pyutils::kwargs_to_any_map(kwargs));
        }))
        .def_readwrite("activation", &ClassificationPipeline::Config::activation);

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
             "Path to the exported model, tokenizer and config.json",
             py::arg("device"),
             "Device to run the model on (e.g., CPU, GPU)",
             py::arg("config") = std::nullopt,
             "Optional pipeline configuration",
             R"(
            ClassificationPipeline class constructor.
            models_path (os.PathLike): Path to the model directory.
            device (str): Device to run the model on (e.g., CPU, GPU).
            config (ClassificationPipeline.Config): Optional pipeline configuration.
            kwargs: Plugin and/or config properties, e.g. CACHE_DIR=/tmp/, activation=ClassificationActivation.SIGMOID.
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
