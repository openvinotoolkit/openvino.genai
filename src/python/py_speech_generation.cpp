// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/stl_bind.h>

#include "openvino/genai/speech_generation/speech_generation_config.hpp"
#include "openvino/genai/speech_generation/speech_generation_perf_metrics.hpp"
#include "openvino/genai/speech_generation/text2speech_pipeline.hpp"
#include "openvino/runtime/tensor.hpp"
#include "py_utils.hpp"
#include "tokenizer/tokenizers_path.hpp"

namespace py = pybind11;
using ov::genai::GenerationConfig;
using ov::genai::PerfMetrics;
using ov::genai::SpeechGenerationConfig;
using ov::genai::SpeechGenerationPerfMetrics;
using ov::genai::Text2SpeechDecodedResults;
using ov::genai::Text2SpeechPipeline;

namespace pyutils = ov::genai::pybind::utils;

namespace {

auto speech_generation_config_docstring = R"(
    SpeechGenerationConfig
    
    Speech-generation specific parameters:
    :param minlenratio: minimum ratio of output length to input text length; prevents output that's too short.
    :type minlenratio: float

    :param maxlenratio: maximum ratio of output length to input text length; prevents excessively long outputs.
    :type minlenratio: float

    :param threshold: probability threshold for stopping decoding; when output probability exceeds above this, generation will stop.
    :type threshold: float
)";

auto speech_generation_perf_metrics_docstring = R"(
    Structure with raw performance metrics for each generation before any statistics are calculated.

    :param num_generated_samples: Returns a number of generated samples in output
    :type num_generated_samples: int
)";

auto text_to_speech_decoded_results = R"(
    Structure that stores the result from the generate method, including a list of waveform tensors
    sampled at 16 kHz, along with performance metrics

    :param speeches: a list of waveform tensors sampled at 16 kHz
    :type speeches: list

    :param perf_metrics: performance metrics
    :type perf_metrics: SpeechGenerationPerfMetrics
)";

auto text_to_speech_generate_docstring = R"(
    Generates speeches based on input texts

    :param text(s): input text(s) for which to generate speech
    :type text(s): str or list[str]

    :param speaker_embedding optional speaker embedding tensor representing the unique characteristics of a speaker's
                             voice. If not provided for SpeechT5 TSS model, the 7306-th vector from the validation set of the
                             `Matthijs/cmu-arctic-xvectors` dataset is used by default.
    :type speaker_embedding: openvino.Tensor or None

    :param properties: speech generation parameters specified as properties
    :type properties: dict

    :returns: raw audios of the input texts spoken in the specified speaker's voice, with a sample rate of 16 kHz
    :rtype: Text2SpeechDecodedResults
)";

SpeechGenerationConfig update_speech_generation_config_from_kwargs(const SpeechGenerationConfig& config,
                                                                   const py::kwargs& kwargs) {
    if (kwargs.empty())
        return config;

    SpeechGenerationConfig res_config = config;
    if (!kwargs.empty())
        res_config.update_generation_config(pyutils::kwargs_to_any_map(kwargs));

    return res_config;
}

}  // namespace

void init_speech_generation_pipeline(py::module_& m) {
    // Binding for SpeechGenerationConfig
    py::class_<SpeechGenerationConfig, GenerationConfig>(m,
                                                         "SpeechGenerationConfig",
                                                         speech_generation_config_docstring)
        .def(py::init<std::filesystem::path>(), py::arg("json_path"), "path where generation_config.json is stored")
        .def(py::init([](const py::kwargs& kwargs) {
            return update_speech_generation_config_from_kwargs(SpeechGenerationConfig(), kwargs);
        }))
        .def_readwrite("minlenratio", &SpeechGenerationConfig::minlenratio)
        .def_readwrite("maxlenratio", &SpeechGenerationConfig::maxlenratio)
        .def_readwrite("threshold", &SpeechGenerationConfig::threshold)
        .def("update_generation_config", [](ov::genai::SpeechGenerationConfig& config, const py::kwargs& kwargs) {
            config.update_generation_config(pyutils::kwargs_to_any_map(kwargs));
        });

    py::class_<SpeechGenerationPerfMetrics, PerfMetrics>(m,
                                                         "SpeechGenerationPerfMetrics",
                                                         speech_generation_perf_metrics_docstring)
        .def(py::init<>())
        .def_readonly("num_generated_samples", &SpeechGenerationPerfMetrics::num_generated_samples)
        .def_readonly("m_evaluated", &SpeechGenerationPerfMetrics::m_evaluated)
        .def_readonly("generate_duration", &SpeechGenerationPerfMetrics::generate_duration)
        .def_readonly("throughput", &SpeechGenerationPerfMetrics::throughput);

    py::class_<Text2SpeechDecodedResults>(m, "Text2SpeechDecodedResults", text_to_speech_decoded_results)
        .def(py::init<>())
        .def_readonly("speeches", &Text2SpeechDecodedResults::speeches)
        .def_readonly("perf_metrics", &Text2SpeechDecodedResults::perf_metrics);

    py::class_<Text2SpeechPipeline>(m, "Text2SpeechPipeline", "Text-to-speech pipeline")
        .def(
            py::init([](const std::filesystem::path& models_path, const std::string& device, const py::kwargs& kwargs) {
                ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
                return std::make_unique<Text2SpeechPipeline>(models_path, device, pyutils::kwargs_to_any_map(kwargs));
            }),
            py::arg("models_path"),
            "folder with tokenizer, encoder, decoder, postnet and vocoder .xml files",
            py::arg("device"),
            "device on which inference will be done",
            "openvino.properties map",
            R"(
            Text2SpeechPipeline class constructor.
            models_path (os.PathLike): Path to the model file.
            device (str): Device to run the model on (e.g., CPU, GPU).
        )")

        .def(
            "generate",
            [](Text2SpeechPipeline& pipe,
               const std::string& text,
               py::object speaker_embedding,
               const py::kwargs& kwargs) -> py::typing::Union<ov::genai::Text2SpeechDecodedResults> {
                SpeechGenerationConfig base_config = pipe.get_generation_config();

                auto updated_config = update_speech_generation_config_from_kwargs(base_config, kwargs);

                ov::genai::Text2SpeechDecodedResults res;
                {
                    py::gil_scoped_release rel;
                    if (speaker_embedding.is_none()) {
                        res = pipe.generate(text);
                    } else {
                        const ov::Tensor& tensor = speaker_embedding.cast<ov::Tensor>();
                        res = pipe.generate(text, tensor);
                    }
                }
                return py::cast(res);
            },
            py::arg("text"),
            "input text for which to generate speech.",
            py::arg("speaker_embedding") = py::none(),
            "vector representing the unique characteristics of a speaker's voice.",
            (text_to_speech_generate_docstring + std::string(" \n ") + speech_generation_config_docstring).c_str())

        .def(
            "generate",
            [](Text2SpeechPipeline& pipe,
               const std::vector<std::string>& texts,
               py::object speaker_embedding,
               const py::kwargs& kwargs) -> py::typing::Union<ov::genai::Text2SpeechDecodedResults> {
                SpeechGenerationConfig base_config = pipe.get_generation_config();

                auto updated_config = update_speech_generation_config_from_kwargs(base_config, kwargs);

                ov::genai::Text2SpeechDecodedResults res;
                {
                    py::gil_scoped_release rel;
                    if (speaker_embedding.is_none()) {
                        res = pipe.generate(texts);
                    } else {
                        const ov::Tensor& tensor = speaker_embedding.cast<ov::Tensor>();
                        res = pipe.generate(texts, tensor);
                    }
                }
                return py::cast(res);
            },
            py::arg("texts"),
            "a list of input texts for which to generate speeches.",
            py::arg("speaker_embedding") = py::none(),
            "vector representing the unique characteristics of a speaker's voice.",
            (text_to_speech_generate_docstring + std::string(" \n ") + speech_generation_config_docstring).c_str())

        .def("get_generation_config", &Text2SpeechPipeline::get_generation_config, py::return_value_policy::copy)
        .def("set_generation_config", &Text2SpeechPipeline::set_generation_config, py::arg("config"));
}
