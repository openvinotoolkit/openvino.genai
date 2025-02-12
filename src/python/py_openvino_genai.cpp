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
void init_tokenizer(py::module_& m);
void init_generation_config(py::module_& m);

void init_continuous_batching_pipeline(py::module_& m);
void init_llm_pipeline(py::module_& m);
void init_image_generation_pipelines(py::module_& m);
void init_vlm_pipeline(py::module_& m);
void init_whisper_pipeline(py::module_& m);

namespace {

auto decoded_results_docstring = R"(
    Structure to store resulting batched text outputs and scores for each batch.
    The first num_return_sequences elements correspond to the first batch element.

    Parameters: 
    texts:      vector of resulting sequences.
    scores:     scores for each sequence.
    metrics:    performance metrics with tpot, ttft, etc. of type ov::genai::PerfMetrics.
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
)";

auto streamer_base_docstring =  R"(
    Base class for streamers. In order to use inherit from from this class and implement put, and methods.
)";

auto text_streamer_docstring =  R"(
TextStreamer is used to decode tokens into text and call a user-defined callback function.

tokenizer: Tokenizer object to decode tokens into text.
callback: User-defined callback function to process the decoded text, callback should return either boolean flag or StreamingStatus.

)";

class ConstructableStreamer: public StreamerBase {
    bool put(int64_t token) override {
        PYBIND11_OVERRIDE(
            bool,  // Return type
            StreamerBase,  // Parent class
            put,  // Name of function in C++ (must match Python name)
            token  // Argument(s)
        );
    }
    StreamingStatus write(int64_t token) override {
        PYBIND11_OVERRIDE(
            StreamingStatus,  // Return type
            StreamerBase,  // Parent class
            write,  // Name of function in C++ (must match Python name)
            token  // Argument(s)
        );
    }
    void end() override {
        PYBIND11_OVERRIDE_PURE(void, StreamerBase, end);
    }
};

} // namespace


PYBIND11_MODULE(py_openvino_genai, m) {
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
        .def_readonly("perf_metrics", &EncodedResults::perf_metrics);

    py::class_<StreamerBase, ConstructableStreamer, std::shared_ptr<StreamerBase>>(m, "StreamerBase", streamer_base_docstring)  // Change the holder form unique_ptr to shared_ptr
        .def(py::init<>())
        .def("put", &StreamerBase::put, "Put is called every time new token is decoded. Returns a bool flag to indicate whether generation should be stopped, if return true generation stops", py::arg("token"))
        .def("write", &StreamerBase::write, "Write is called every time new token is decoded. Returns a StreamingStatus flag to indicate whether generation should be stopped or cancelled", py::arg("token"))
        .def("end", &StreamerBase::end, "End is called at the end of generation. It can be used to flush cache if your own streamer has one");

    py::class_<TextStreamer, std::shared_ptr<TextStreamer>>(m, "TextStreamer", text_streamer_docstring)
        .def(py::init<const Tokenizer&, std::function<CallbackTypeVariant(std::string)>>(), py::arg("tokenizer"), py::arg("callback"))
        .def("write", &TextStreamer::write)
        .def("end", &TextStreamer::end);

    py::enum_<ov::genai::StreamingStatus>(m, "StreamingStatus")
        .value("RUNNING", ov::genai::StreamingStatus::RUNNING)
        .value("CANCEL", ov::genai::StreamingStatus::CANCEL)
        .value("STOP", ov::genai::StreamingStatus::STOP);

    init_tokenizer(m);
    init_lora_adapter(m);
    init_generation_config(m);

    init_continuous_batching_pipeline(m);
    init_llm_pipeline(m);
    init_image_generation_pipelines(m);
    init_vlm_pipeline(m);
    init_whisper_pipeline(m);
}
