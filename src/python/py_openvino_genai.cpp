// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/functional.h>
#include <pybind11/typing.h>

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/text_streamer.hpp"
#include "openvino/genai/version.hpp"

#include "py_utils.hpp"

namespace py = pybind11;
namespace pyutils = ov::genai::pybind::utils;

using ov::genai::CallbackTypeVariant;
using ov::genai::DecodedResults;
using ov::genai::EncodedResults;
using ov::genai::StreamerBase;
using ov::genai::StringInputs;
using ov::genai::StreamingStatus;
using ov::genai::TextStreamer;
using ov::genai::Tokenizer;
using ov::genai::get_version;

void init_lora_adapter(py::module_& m);
void init_perf_metrics(py::module_& m);
void init_chat_history(py::module_& m);
void init_tokenizer(py::module_& m);
void init_streamers(py::module_& m);
void init_generation_config(py::module_& m);

void init_continuous_batching_pipeline(py::module_& m);
void init_llm_pipeline(py::module_& m);
void init_image_generation_pipelines(py::module_& m);
void init_vlm_pipeline(py::module_& m);
void init_whisper_pipeline(py::module_& m);
void init_rag_pipelines(py::module_& m);
void init_speech_generation_pipeline(py::module_& m);

namespace {

auto decoded_results_docstring = R"(
    Structure to store resulting batched text outputs and scores for each batch.
    The first num_return_sequences elements correspond to the first batch element.

    Parameters: 
    texts:      vector of resulting sequences.
    scores:     scores for each sequence.
    metrics:    performance metrics with tpot, ttft, etc. of type ov::genai::PerfMetrics.
    extended_perf_metrics: performance pipeline specifics metrics,
                           applicable for pipelines with implemented extended metrics: SpeculativeDecoding Pipeline.
)";

auto encoded_results_docstring = R"(
    Structure to store resulting batched tokens and scores for each batch sequence.
    The first num_return_sequences elements correspond to the first batch element.
    In the case if results decoded with beam search and random sampling scores contain
    sum of logarithmic probabilities for each token in the sequence. In the case
    of greedy decoding scores are filled with zeros.

    Parameters: 
    tokens: sequence of resulting tokens.
    scores: sum of logarithmic probabilities of all tokens in the sequence.
    metrics: performance metrics with tpot, ttft, etc. of type ov::genai::PerfMetrics.
    extended_perf_metrics: performance pipeline specifics metrics,
                           applicable for pipelines with implemented extended metrics: SpeculativeDecoding Pipeline.
)";

} // namespace


#ifdef Py_GIL_DISABLED
PYBIND11_MODULE(py_openvino_genai, m, py::mod_gil_not_used()) {
#else
PYBIND11_MODULE(py_openvino_genai, m) {
#endif
    m.doc() = "Pybind11 binding for OpenVINO GenAI library";

    m.def("get_version", [] () -> py::str {
        return get_version().buildNumber;
    }, get_version().description);

    init_perf_metrics(m);

    py::class_<DecodedResults>(m, "DecodedResults", decoded_results_docstring)
        .def(py::init<>())
        .def_property_readonly("texts", [](const DecodedResults &dr) -> py::typing::List<py::str> { return pyutils::handle_utf8((std::vector<std::string>)dr); })
        .def_readonly("scores", &DecodedResults::scores)
        .def_readonly("perf_metrics", &DecodedResults::perf_metrics)
        .def_readonly("extended_perf_metrics", &DecodedResults::extended_perf_metrics)
        .def("__str__", [](const DecodedResults &dr) -> py::str {
            auto valid_utf8_strings = pyutils::handle_utf8((std::vector<std::string>)dr);
            py::str res;
            if (valid_utf8_strings.size() == 1)
                return valid_utf8_strings[0];
            
            for (size_t i = 0; i < valid_utf8_strings.size() - 1; i++) {
                res += py::str(std::to_string(dr.scores[i])) + py::str(": ") + valid_utf8_strings[i] + py::str("\n");
            }
            res += py::str(std::to_string(dr.scores.back())) + py::str(": ") + valid_utf8_strings[valid_utf8_strings.size() - 1];
            return res;
        });

    py::class_<EncodedResults>(m, "EncodedResults", encoded_results_docstring)
        .def_readonly("tokens", &EncodedResults::tokens)
        .def_readonly("scores", &EncodedResults::scores)
        .def_readonly("perf_metrics", &EncodedResults::perf_metrics)
        .def_readonly("extended_perf_metrics", &EncodedResults::extended_perf_metrics);

    init_lora_adapter(m);
    init_generation_config(m);
    init_chat_history(m);
    init_tokenizer(m);
    init_streamers(m);

    init_llm_pipeline(m);
    init_continuous_batching_pipeline(m);
    init_image_generation_pipelines(m);
    init_vlm_pipeline(m);
    init_whisper_pipeline(m);
    init_rag_pipelines(m);
    init_speech_generation_pipeline(m);
}
