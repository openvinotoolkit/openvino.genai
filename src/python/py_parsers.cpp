// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/functional.h>

#include "openvino/genai/parsers.hpp"
#include "py_utils.hpp"
#include "openvino/genai/json_container.hpp"

namespace py = pybind11;

using ov::genai::IncrementalParser;
using ov::genai::Parser;
using ov::genai::ReasoningParser;
using ov::genai::ReasoningIncrementalParser;
using ov::genai::DeepSeekR1ReasoningParser;
using ov::genai::DeepSeekR1ReasoningIncrementalParser;
using ov::genai::Phi4ReasoningIncrementalParser;
using ov::genai::Phi4ReasoningParser;
using ov::genai::JsonContainer;
using ov::genai::Llama3JsonToolParser;
using ov::genai::Llama3PythonicToolParser;
using ov::genai::Tokenizer;
using ov::genai::StreamingStatus;

namespace pyutils = ov::genai::pybind::utils;

namespace {

class ConstructableParser: public Parser {
public:
    void parse(JsonContainer& msg) override {
        py::gil_scoped_acquire acquire;
        
        py::function parse_method = py::get_override(static_cast<const Parser*>(this), "parse");
        if (!parse_method) {
            OPENVINO_THROW("parse method not implemented in Python subclass");
        }
        
        // Convert JsonContainer to py::dict
        py::dict py_msg = pyutils::json_container_to_py_object(msg);
        parse_method(py_msg);
        msg = pyutils::py_object_to_json_container(py_msg);
    }
};

// ConstructableIncremental and ConstructableBase are used when python overload is called from C++
// and we need to convert JsonContainer to py::dict and then update back JsonContainer from the py::dict which was modified in Python.
class ConstructableIncrementalParser: public IncrementalParser {
public:
    using IncrementalParser::IncrementalParser;
    std::string parse(
        JsonContainer& msg,
        std::string& delta_text, 
        const std::optional<std::vector<int64_t>>& delta_tokens = std::nullopt
    ) override {
        py::gil_scoped_acquire acquire;
        // Convert JsonContainer to py::dict
        py::dict py_msg = pyutils::json_container_to_py_object(msg);

        py::function parse_method = py::get_override(static_cast<const IncrementalParser*>(this), "parse");
        if (!parse_method) {
            OPENVINO_THROW("parse method not implemented in Python subclass");
        }

        auto res = parse_method(py_msg, delta_text, delta_tokens);
        msg = pyutils::py_object_to_json_container(py_msg);
        
        return res.cast<std::string>();
    }

    void reset() override {
        PYBIND11_OVERLOAD_PURE(
            void,
            IncrementalParser,
            reset,
        );
    }
};

} // namespace

void init_parsers(py::module_& m) {
    py::class_<Parser, ConstructableParser, std::shared_ptr<Parser>>(m, "Parser")
        .def(py::init<>())
        .def("parse",
            [](Parser& self, py::dict& message) {
                auto msg_cpp = pyutils::py_object_to_json_container(message);
                self.parse(msg_cpp);
                py::dict result = pyutils::json_container_to_py_object(msg_cpp);
                message.attr("update")(result);
            },
            py::arg("message"),
            "Parse is called with the full text. Returns a dict with parsed content.");

    py::class_<ReasoningParser, std::shared_ptr<ReasoningParser>, Parser>(m, "ReasoningParser")
        .def(py::init<bool, bool, const std::string&, const std::string&>(),
                py::arg("expect_open_tag") = true,
                py::arg("keep_original_content") = true,
                py::arg("open_tag") = "<think>",
                py::arg("close_tag") = "</think>");

    py::class_<DeepSeekR1ReasoningParser, std::shared_ptr<DeepSeekR1ReasoningParser>, ReasoningParser>(m, "DeepSeekR1ReasoningParser")
        .def(py::init<>());
    
    py::class_<Phi4ReasoningParser, std::shared_ptr<Phi4ReasoningParser>, ReasoningParser>(m, "Phi4ReasoningParser")
        .def(py::init<>());
    
    py::class_<Llama3JsonToolParser, std::shared_ptr<Llama3JsonToolParser>, Parser>(m, "Llama3JsonToolParser")
        .def(py::init<>());

    py::class_<Llama3PythonicToolParser, std::shared_ptr<Llama3PythonicToolParser>, Parser>(m, "Llama3PythonicToolParser")
        .def(py::init<>());
    
    py::class_<IncrementalParser, ConstructableIncrementalParser, std::shared_ptr<IncrementalParser>>(m, "IncrementalParser")
        .def(py::init<>())
        .def("parse", [](IncrementalParser& self,
                         py::dict& message,
                         std::string& delta_text,
                         const std::optional<std::vector<int64_t>>& delta_tokens = std::nullopt) {
            auto msg_cpp = pyutils::py_object_to_json_container(message);
            auto res = self.parse(msg_cpp, delta_text, delta_tokens);
            auto result = pyutils::json_container_to_py_object(msg_cpp);
            message.attr("update")(result);
            return res;
        }, py::arg("message"), py::arg("delta_text"), py::arg("delta_tokens") = std::nullopt,
           "Parse is called every time new text delta is decoded. Returns a string with any additional text to append to the current output.")
        .def("reset", &IncrementalParser::reset, "Reset the internal state of the parser.");
    
    py::class_<ReasoningIncrementalParser, std::shared_ptr<ReasoningIncrementalParser>, IncrementalParser>(m, "ReasoningIncrementalParser")
        .def(py::init<bool, bool, const std::string&, const std::string&>(),
             py::arg("expect_open_tag") = true,
             py::arg("keep_original_content") = true,
             py::arg("open_tag") = "<think>",
             py::arg("close_tag") = "</think>");
    
    py::class_<Phi4ReasoningIncrementalParser, std::shared_ptr<Phi4ReasoningIncrementalParser>, IncrementalParser>(m, "Phi4ReasoningIncrementalParser")
        .def(py::init<>());

    py::class_<DeepSeekR1ReasoningIncrementalParser, std::shared_ptr<DeepSeekR1ReasoningIncrementalParser>, IncrementalParser>(m, "DeepSeekR1ReasoningIncrementalParser")
        .def(py::init<>());
}
