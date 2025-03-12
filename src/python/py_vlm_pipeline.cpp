// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include <filesystem>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/functional.h>

#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/genai/visual_language/perf_metrics.hpp"
#include "tokenizers_path.hpp"
#include "py_utils.hpp"

namespace py = pybind11;
namespace pyutils = ov::genai::pybind::utils;


auto vlm_generate_docstring = R"(
    Generates sequences for VLMs.

    :param prompt: input prompt
    :type prompt: str
    The prompt can contain <ov_genai_image_i> with i replaced with
    an actual zero based index to refer to an image. Reference to
    images used in previous prompts isn't implemented.
    A model's native image tag can be used instead of
    <ov_genai_image_i>. These tags are:
    MiniCPM-V-2_6: <image>./</image>
    Qwen2-VL: <|vision_start|><|image_pad|><|vision_end|>
    If the prompt doesn't contain image tags, but images are
    provided, the tags are prepended to the prompt.

    :param images: image or list of images
    :type images: List[ov.Tensor] or ov.Tensor

    :param generation_config: generation_config
    :type generation_config: GenerationConfig or a Dict

    :param streamer: streamer either as a lambda with a boolean returning flag whether generation should be stopped
    :type : Callable[[str], bool], ov.genai.StreamerBase

    :param kwargs: arbitrary keyword arguments with keys corresponding to GenerationConfig fields.
    :type : Dict

    :return: return results in decoded form
    :rtype: VLMDecodedResults
)";

auto vlm_generate_kwargs_docstring = R"(
    Generates sequences for VLMs.

    :param prompt: input prompt
    The prompt can contain <ov_genai_image_i> with i replaced with
    an actual zero based index to refer to an image. Reference to
    images used in previous prompts isn't implemented.
    A model's native image tag can be used instead of
    <ov_genai_image_i>. These tags are:
    MiniCPM-V-2_6: <image>./</image>
    Qwen2-VL: <|vision_start|><|image_pad|><|vision_end|>
    If the prompt doesn't contain image tags, but images are
    provided, the tags are prepended to the prompt.

    :type prompt: str

    :param kwargs: arbitrary keyword arguments with keys corresponding to generate params.

    Expected parameters list:
    image: ov.Tensor - input image,
    images: List[ov.Tensor] - input images,
    generation_config: GenerationConfig,
    streamer: Callable[[str], bool], ov.genai.StreamerBase - streamer either as a lambda with a boolean returning flag whether generation should be stopped

    :return: return results in decoded form
    :rtype: VLMDecodedResults
)";

auto raw_perf_metrics_docstring = R"(
    Structure with VLM specific raw performance metrics for each generation before any statistics are calculated.

    :param prepare_embeddings_durations: Durations of embeddings preparation.
    :type prepare_embeddings_durations: List[MicroSeconds]
)";

auto perf_metrics_docstring = R"(
    Structure with raw performance metrics for each generation before any statistics are calculated.

    :param get_prepare_embeddings_duration: Returns mean and standard deviation of embeddings preparation duration in milliseconds
    :type get_prepare_embeddings_duration: MeanStdPair

    :param vlm_raw_metrics: VLM specific raw metrics
    :type VLMRawPerfMetrics:
)";

auto decoded_results_docstring = R"(
    Structure to store resulting batched text outputs and scores for each batch.
    The first num_return_sequences elements correspond to the first batch element.

    Parameters:
    texts:      vector of resulting sequences.
    scores:     scores for each sequence.
    metrics:    performance metrics with tpot, ttft, etc. of type openvino_genai.VLMPerfMetrics.
)";

py::object call_vlm_generate(
    ov::genai::VLMPipeline& pipe,
    const std::string& prompt,
    const std::vector<ov::Tensor>& images,
    const ov::genai::GenerationConfig& generation_config,
    const pyutils::PyBindStreamerVariant& py_streamer,
    const py::kwargs& kwargs
) {
    auto updated_config = *pyutils::update_config_from_kwargs(generation_config, kwargs);
    ov::genai::StreamerVariant streamer = pyutils::pystreamer_to_streamer(py_streamer);
    ov::genai::VLMDecodedResults res;
    {
        py::gil_scoped_release rel;
        res= pipe.generate(prompt, images, updated_config, streamer);
    }
    return py::cast(res);
}

void init_vlm_pipeline(py::module_& m) {
    py::class_<ov::genai::VLMRawPerfMetrics>(m, "VLMRawPerfMetrics", raw_perf_metrics_docstring)
        .def(py::init<>())
        .def_property_readonly("prepare_embeddings_durations", [](const ov::genai::VLMRawPerfMetrics& rw) {
            return pyutils::get_ms(rw, &ov::genai::VLMRawPerfMetrics::prepare_embeddings_durations);
        });

    py::class_<ov::genai::VLMPerfMetrics, ov::genai::PerfMetrics>(m, "VLMPerfMetrics", perf_metrics_docstring)
        .def(py::init<>())
        .def("get_prepare_embeddings_duration", &ov::genai::VLMPerfMetrics::get_prepare_embeddings_duration)
        .def_readonly("vlm_raw_metrics", &ov::genai::VLMPerfMetrics::vlm_raw_metrics);

    py::class_<ov::genai::VLMDecodedResults, ov::genai::DecodedResults>(m, "VLMDecodedResults", decoded_results_docstring)
        .def(py::init<>())
        .def_property_readonly("texts", [](const ov::genai::VLMDecodedResults &dr) -> py::typing::List<py::str> { return pyutils::handle_utf8(dr.texts); })
        .def_readonly("scores", &ov::genai::VLMDecodedResults::scores)
        .def_readonly("perf_metrics", &ov::genai::VLMDecodedResults::perf_metrics)
        .def("__str__", [](const ov::genai::VLMDecodedResults &dr) -> py::str {
            auto valid_utf8_strings = pyutils::handle_utf8(dr.texts);
            py::str res;
            if (valid_utf8_strings.size() == 1)
                return valid_utf8_strings[0];

            for (size_t i = 0; i < valid_utf8_strings.size() - 1; i++) {
                res += py::str(std::to_string(dr.scores[i])) + py::str(": ") + valid_utf8_strings[i] + py::str("\n");
            }
            res += py::str(std::to_string(dr.scores.back())) + py::str(": ") + valid_utf8_strings[valid_utf8_strings.size() - 1];
            return res;
        });

    py::class_<ov::genai::VLMPipeline>(m, "VLMPipeline", "This class is used for generation with VLMs")
        .def(py::init([](
            const std::filesystem::path& models_path,
            const std::string& device,
            const py::kwargs& kwargs
        ) {
            ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
            return std::make_unique<ov::genai::VLMPipeline>(models_path, device, pyutils::kwargs_to_any_map(kwargs));
        }),
        py::arg("models_path"), "folder with exported model files",
        py::arg("device"), "device on which inference will be done"
        R"(
            VLMPipeline class constructor.
            models_path (os.PathLike): Path to the folder with exported model files.
            device (str): Device to run the model on (e.g., CPU, GPU). Default is 'CPU'.
            kwargs: Device properties
        )")

        .def("start_chat", &ov::genai::VLMPipeline::start_chat, py::arg("system_message") = "")
        .def("finish_chat", &ov::genai::VLMPipeline::finish_chat)
        .def("set_chat_template", &ov::genai::VLMPipeline::set_chat_template, py::arg("chat_template"))
        .def("get_tokenizer", &ov::genai::VLMPipeline::get_tokenizer)
        .def("get_generation_config", &ov::genai::VLMPipeline::get_generation_config, py::return_value_policy::copy)
        .def("set_generation_config", &ov::genai::VLMPipeline::set_generation_config, py::arg("config"))
        .def(
            "generate",
            [](ov::genai::VLMPipeline& pipe,
                const std::string& prompt,
                const std::vector<ov::Tensor>& images,
                const ov::genai::GenerationConfig& generation_config,
                const pyutils::PyBindStreamerVariant& streamer,
                const py::kwargs& kwargs
            ) -> py::typing::Union<ov::genai::VLMDecodedResults> {
                return call_vlm_generate(pipe, prompt, images, generation_config, streamer, kwargs);
            },
            py::arg("prompt"), "Input string",
            py::arg("images"), "Input images",
            py::arg("generation_config"), "generation_config",
            py::arg("streamer") = std::monostate(), "streamer",
            (vlm_generate_docstring + std::string(" \n ")).c_str()
        )
        .def(
            "generate",
            [](ov::genai::VLMPipeline& pipe,
                const std::string& prompt,
                const ov::Tensor& images,
                const ov::genai::GenerationConfig& generation_config,
                const pyutils::PyBindStreamerVariant& streamer,
                const py::kwargs& kwargs
            ) -> py::typing::Union<ov::genai::VLMDecodedResults> {
                return call_vlm_generate(pipe, prompt, {images}, generation_config, streamer, kwargs);
            },
            py::arg("prompt"), "Input string",
            py::arg("images"), "Input images",
            py::arg("generation_config"), "generation_config",
            py::arg("streamer") = std::monostate(), "streamer",
            (vlm_generate_docstring + std::string(" \n ")).c_str()
        )
        .def(
            "generate",
            [](ov::genai::VLMPipeline& pipe,
               const std::string& prompt,
               const py::kwargs& kwargs
            )  -> py::typing::Union<ov::genai::VLMDecodedResults> {
                auto map = pyutils::kwargs_to_any_map(kwargs);
                ov::genai::VLMDecodedResults res;
                {
                    py::gil_scoped_release rel;
                    res = pipe.generate(prompt, map);
                }
                return py::cast(res);
            },
            py::arg("prompt"), "Input string",
            (vlm_generate_kwargs_docstring + std::string(" \n ")).c_str()
        );
}
