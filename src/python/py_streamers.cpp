// Copyright (C) 2023-2025 Intel Corporation
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
using ov::genai::IncrementalParserBase;
using ov::genai::ParsedMessage;
using ov::genai::Tokenizer;
using ov::genai::JsonContainer;

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

class ConstructableTextParserStreamer: public TextParserStreamer {
public:
    using TextParserStreamer::TextParserStreamer;  // inherit base constructors

    StreamingStatus write(ParsedMessage& message) override {
        py::dict message_py;
        auto json_obj = message.to_json();
        for (auto it = json_obj.begin(); it != json_obj.end(); ++it) {
            message_py[py::cast(it.key())] = py::cast(it.value().get<std::string>());
        }
        
        // call python implementation which accepts py::dict instead of ParsedMessage
        auto res = py::get_override(this, "write")(message_py);
        
        auto msg_anymap = ov::genai::pybind::utils::py_object_to_any_map(message_py);
        message = JsonContainer(msg_anymap);
        
        return res.cast<StreamingStatus>();
    }

    StreamingStatus write(py::dict& message) {
        PYBIND11_OVERRIDE_PURE(
            StreamingStatus,
            TextParserStreamer,
            "write",
            message
        );
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
            py::arg("token"));
        
    // TODO: double check/add more relevant docstrings for TextParserStreamer.
    py::class_<TextParserStreamer, ConstructableTextParserStreamer, std::shared_ptr<TextParserStreamer>, TextStreamer>(m, "TextParserStreamer")
        .def(py::init([](const Tokenizer& tokenizer,
                         std::vector<std::variant<std::shared_ptr<IncrementalParserBase>, std::string>> parsers) {
                return std::make_shared<ConstructableTextParserStreamer>(tokenizer, parsers);
            }),
            py::arg("tokenizer"),
            py::arg("parsers") = std::vector<std::variant<std::shared_ptr<IncrementalParserBase>, std::string>>({}),
            "TextParserStreamer is used to decode tokens into text, parse the text and call user-defined incremental parsers.")
        .def("write",
            [](TextParserStreamer& self, py::dict& message) {
                // Downcast to ConstructableTextParserStreamer if needed
                auto* derived = dynamic_cast<ConstructableTextParserStreamer*>(&self);
                if (!derived) {
                    throw std::runtime_error("write(py::dict&) only available for ConstructableTextParserStreamer");
                }
                return derived->write(message);
            },
            py::arg("message"),
            "Write is called with a ParsedMessage. Returns StreamingStatus.")
        .def("_write",
             py::overload_cast<std::string>(&TextParserStreamer::write),
             py::arg("message"),
             "Write is called with a string message. Returns CallbackTypeVariant. This is a private method.")
        
        .def("get_parsed_message", &TextParserStreamer::get_parsed_message, "Get the current parsed message")

        .def("get_parsers", &TextParserStreamer::get_parsers, "Get the list of parsers");
}
