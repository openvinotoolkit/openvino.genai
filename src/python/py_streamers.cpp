// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>

#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/text_streamer.hpp"
#include "py_utils.hpp"

namespace py = pybind11;

using ov::genai::CallbackTypeVariant;
using ov::genai::StreamingStatus;
using ov::genai::TextStreamer;
using ov::genai::Tokenizer;

namespace pyutils = ov::genai::pybind::utils;

namespace {

auto streamer_base_docstring =  R"(
    Base class for streamers. In order to use inherit from from this class and implement write and end methods.
)";

auto text_streamer_docstring =  R"(
TextStreamer is used to decode tokens into text and call a user-defined callback function.

tokenizer: Tokenizer object to decode tokens into text.
callback: User-defined callback function to process the decoded text, callback should return either boolean flag or StreamingStatus.
detokenization_params: AnyMap with detokenization parameters, e.g. ov::genai::skip_special_tokens(...)
)";

class ConstructableStreamer: public StreamerBase {
    OPENVINO_SUPPRESS_DEPRECATED_START
    bool put(int64_t token) override {
        PYBIND11_OVERRIDE(
            bool,  // Return type
            StreamerBase,  // Parent class
            put,  // Name of function in C++ (must match Python name)
            token  // Argument(s)
        );
    }
    OPENVINO_SUPPRESS_DEPRECATED_END
    StreamingStatus write(int64_t token) override {
        PYBIND11_OVERRIDE(
            StreamingStatus,  // Return type
            StreamerBase,  // Parent class
            write,  // Name of function in C++ (must match Python name)
            token  // Argument(s)
        );
    }
    StreamingStatus write(const std::vector<int64_t>& token) override {
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

void init_streamers(py::module_& m) {
    py::enum_<ov::genai::StreamingStatus>(m, "StreamingStatus")
        .value("RUNNING", ov::genai::StreamingStatus::RUNNING)
        .value("CANCEL", ov::genai::StreamingStatus::CANCEL)
        .value("STOP", ov::genai::StreamingStatus::STOP);

    auto streamer = py::class_<StreamerBase, ConstructableStreamer, std::shared_ptr<StreamerBase>>(m, "StreamerBase", streamer_base_docstring)  // Change the holder form unique_ptr to shared_ptr
        .def(py::init<>())
        .def("write",
            [](StreamerBase& self, std::variant<int64_t, std::vector<int64_t>> token) {
               if (auto _token = std::get_if<int64_t>(&token)) {
                   return self.write(*_token);
               } else {
                   auto tokens = std::get<std::vector<int64_t>>(token);
                   return self.write(tokens);
               }
            },
            "Write is called every time new token or vector of tokens is decoded. Returns a StreamingStatus flag to indicate whether generation should be stopped or cancelled",
            py::arg("token"))
        .def("end", &StreamerBase::end, "End is called at the end of generation. It can be used to flush cache if your own streamer has one");
    OPENVINO_SUPPRESS_DEPRECATED_START
    streamer.def("put", &StreamerBase::put, "Put is called every time new token is decoded. Returns a bool flag to indicate whether generation should be stopped, if return true generation stops", py::arg("token"));
    OPENVINO_SUPPRESS_DEPRECATED_END

    py::class_<TextStreamer, std::shared_ptr<TextStreamer>, StreamerBase>(m, "TextStreamer", text_streamer_docstring)
        .def(py::init([](const Tokenizer& tokenizer, std::function<CallbackTypeVariant(std::string)> callback, const std::map<std::string, py::object>& detokenization_params) {
            return std::make_shared<TextStreamer>(tokenizer, callback, pyutils::properties_to_any_map(detokenization_params));
        }),
        py::arg("tokenizer"),
        py::arg("callback"),
        py::arg("detokenization_params") = ov::AnyMap({}))

        .def("write",
            [](TextStreamer& self, std::variant<int64_t, std::vector<int64_t>> token) {
                if (auto _token = std::get_if<int64_t>(&token)) {
                    return self.write(*_token);
                } else {
                    auto tokens = std::get<std::vector<int64_t>>(token);
                    return self.write(tokens);
                }
            },
            py::arg("token"))
        .def("end", &TextStreamer::end);
}
