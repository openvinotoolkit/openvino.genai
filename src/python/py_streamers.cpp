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

void init_streamers(py::module_& m) {
    py::enum_<ov::genai::StreamingStatus>(m, "StreamingStatus")
        .value("RUNNING", ov::genai::StreamingStatus::RUNNING)
        .value("CANCEL", ov::genai::StreamingStatus::CANCEL)
        .value("STOP", ov::genai::StreamingStatus::STOP);

    auto streamer = py::class_<StreamerBase, ConstructableStreamer, std::shared_ptr<StreamerBase>>(m, "StreamerBase", streamer_base_docstring)  // Change the holder form unique_ptr to shared_ptr
        .def(py::init<>())
        .def("write", &StreamerBase::write, "Write is called every time new token is decoded. Returns a StreamingStatus flag to indicate whether generation should be stopped or cancelled", py::arg("token"))
        .def("end", &StreamerBase::end, "End is called at the end of generation. It can be used to flush cache if your own streamer has one");
    OPENVINO_SUPPRESS_DEPRECATED_START
    streamer.def("put", &StreamerBase::put, "Put is called every time new token is decoded. Returns a bool flag to indicate whether generation should be stopped, if return true generation stops", py::arg("token"));
    OPENVINO_SUPPRESS_DEPRECATED_END

    py::class_<TextStreamer, std::shared_ptr<TextStreamer>, StreamerBase>(m, "TextStreamer", text_streamer_docstring)
        .def(py::init<const Tokenizer&, std::function<CallbackTypeVariant(std::string)>>(), py::arg("tokenizer"), py::arg("callback"))
        .def("write", &TextStreamer::write, py::arg("token"))
        .def("end", &TextStreamer::end);
}
