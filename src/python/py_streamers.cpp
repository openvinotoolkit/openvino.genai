// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>

#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/text_streamer.hpp"
#include "openvino/genai/parsers.hpp"
#include "py_utils.hpp"
#include "openvino/genai/json_container.hpp"

namespace py = pybind11;

using ov::genai::CallbackTypeVariant;
using ov::genai::StreamingStatus;
using ov::genai::TextStreamer;
using ov::genai::TextParserStreamer;
using ov::genai::IncrementalParser;
using ov::genai::JsonContainer;
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

auto text_parser_streamer_docstring = R"(
Base class for text streamers which works with parsed messages. In order to use inherit from this class and implement write method which takes a dict as input parameter.

tokenizer: Tokenizer object to decode tokens into text.
parsers: vector of IncrementalParser to process the text stream incrementally.
)";

class ConstructableStreamer: public StreamerBase {
    StreamingStatus write(int64_t token) override {
        PYBIND11_OVERRIDE_PURE(
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

class ConstructableTextParserStreamer: public TextParserStreamer {
public:
    using TextParserStreamer::TextParserStreamer;  // inherit base constructors

    StreamingStatus write(JsonContainer& message) override {
        // Since C++ calls function with JsonContainer while python override expects py::dict, 
        // this function is a wrapper to call Python implementation of 'write' with py::dict
        py::gil_scoped_acquire acquire;

        py::dict message_py = pyutils::json_container_to_py_object(message);
        
        // Call python implementation which accepts py::dict instead of JsonContainer
        // And convert back the resulting message back to JsonContainer
        auto res = py::get_override(this, "write")(message_py);
        message = pyutils::py_object_to_json_container(message_py);
        
        return res.cast<StreamingStatus>();
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
        
    py::class_<TextParserStreamer, ConstructableTextParserStreamer, std::shared_ptr<TextParserStreamer>, TextStreamer>(m, "TextParserStreamer", text_parser_streamer_docstring)
        .def(py::init([](const Tokenizer& tokenizer,
                         std::vector<std::shared_ptr<IncrementalParser>> parsers) {
                return std::make_shared<ConstructableTextParserStreamer>(tokenizer, parsers);
            }),
            py::arg("tokenizer"),
            py::arg("parsers") = std::vector<std::shared_ptr<IncrementalParser>>(),
            py::keep_alive<1, 3>())
        
        // If we inherit and implement 'write' in Python and try to call write with text chunks or integer tokens 
        // then Python implementation will be called since python does not have overloads.
        // But for texts we need to check that when we call write with strings/integer tokens they are accumulated and stored correctly in py::dict.
        // Therefore we provide a private method '_write' which is used to call 'write' with correct parameters from C++ side.
        .def("_write", 
            [](TextParserStreamer& self, std::variant<std::vector<int64_t>, std::string> chunk) -> StreamingStatus {
                if (auto _token = std::get_if<std::vector<int64_t>>(&chunk)) {
                    return self.write(*_token);
                } else if (auto _str =  std::get_if<std::string>(&chunk)) {
                    auto res = self.write(*_str);
                    return std::get<StreamingStatus>(res);
                }
                return StreamingStatus::RUNNING;
            },
            py::arg("chunk"), "This is a private method is used to call write with integer tokens or text chunks. Is used for text purposes only.")
        
        .def("get_parsed_message",
            [](TextParserStreamer& self) -> py::dict{
                return pyutils::json_container_to_py_object(self.get_parsed_message());

            }, "Returns the accumulated message.")
        
        .def("reset", &TextParserStreamer::reset, "Resets the internal state of the parser streamer.");
}
