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
using ov::genai::ReasoningIncrementalParser;
using ov::genai::Phi4ReasoningIncrementalParser;
using ov::genai::DeepSeekR1ReasoningIncrementalParser;
using ov::genai::JsonContainer;
using ov::genai::Llama3JsonToolParser;
using ov::genai::Llama3PythonicToolParser;
using ov::genai::Tokenizer;
using ov::genai::StreamingStatus;

namespace pyutils = ov::genai::pybind::utils;

namespace {

// ConstructableIncremental and ConstructableBase are used when python overload is called from C++
// and we need to convert JsonContainer to py::dict and then update back JsonContainer from the py::dict which was modified in Python.
class ConstructableIncrementalParser: public IncrementalParser {
public:
    using IncrementalParser::IncrementalParser;
    std::string parse(
        JsonContainer& msg,
        const std::string& previous_text, 
        std::string& delta_text, 
        const std::optional<std::vector<int64_t>>& previous_tokens = std::nullopt, 
        const std::optional<std::vector<int64_t>>& delta_tokens = std::nullopt
    ) override {
        // Convert JsonContainer to py::dict
        py::dict py_msg = pyutils::json_container_to_py_object(msg);

        py::function parse_method = py::get_override(static_cast<const IncrementalParser*>(this), "parse");
        if (!parse_method) {
            throw std::runtime_error("parse method not implemented in Python subclass");
        }
        
        auto res = parse_method(
            py_msg,
            previous_text,
            delta_text,
            previous_tokens,
            delta_tokens
        );
        
        // iterate throught py_msg and update msg
        auto msg_anymap = pyutils::py_object_to_any_map(py_msg);
        for (const auto& [key, value] : msg_anymap) {
            if (value.is<std::string>()) {
                msg[key] = value.as<std::string>();
            } else if (value.is<ov::AnyMap>()) {
                msg[key] = JsonContainer(value.as<ov::AnyMap>());
            } else {
                OPENVINO_THROW("Unsupported type in JsonContainer update from Python dict");
            }
        }
        return res.cast<std::string>();
    }
};

class ConstructableParser: public Parser {
public:
    void parse(JsonContainer& msg) override {
        py::gil_scoped_acquire acquire;
        
        py::function parse_method = py::get_override(static_cast<const Parser*>(this), "parse");
        if (!parse_method) {
            throw std::runtime_error("parse method not implemented in Python subclass");
        }
        
        // Convert JsonContainer to py::dict
       py::dict py_msg = pyutils::json_container_to_py_object(msg);
       parse_method(py_msg);

       // iterate throught py_msg and update msg
       auto msg_anymap = pyutils::py_object_to_any_map(py_msg);
       for (const auto& [key, value] : msg_anymap) {
           if (value.is<std::string>()) {
               msg[key] = value.as<std::string>();
           } else if (value.is<ov::AnyMap>()) {
               msg[key] = JsonContainer(value.as<ov::AnyMap>());
           } else {
               OPENVINO_THROW("Unsupported type in JsonContainer update from Python dict");
           }
       }
    }
};

} // namespace

// TODO: double check/add more relevant docstrings for parsers.
void init_parsers(py::module_& m) {
    py::class_<IncrementalParser, ConstructableIncrementalParser, std::shared_ptr<IncrementalParser>>(m, "IncrementalParser")
        .def(py::init<>())
        .def("parse", [](IncrementalParser& self,
                         py::dict& msg,
                         std::string& previous_text,
                         std::string& delta_text,
                         const std::optional<std::vector<int64_t>>& previous_tokens = std::nullopt,
                         const std::optional<std::vector<int64_t>>& delta_tokens = std::nullopt) {
            auto msg_cpp = pyutils::py_object_to_json_container(msg);
            auto res = self.parse(msg_cpp, previous_text, delta_text, previous_tokens, delta_tokens);
            auto json_str = msg_cpp.to_json_string();
            
            // TODO: msg = pyutils::json_container_to_py_object(msg_cpp) does not work properly here,
            // since it create a new object instead of updating existing dict.
            py::object json_mod = py::module_::import("json");
            py::dict result = json_mod.attr("loads")(json_str);
            // update msg with result
            for (auto item : result) {
                msg[item.first] = item.second;
            }
            return res;
        }, py::arg("msg"), py::arg("previous_text"), py::arg("delta_text"),
           py::arg("previous_tokens") = std::nullopt, py::arg("delta_tokens") = std::nullopt,
           "Parse is called every time new text delta is decoded. Returns a string with any additional text to append to the current output.");
    
    py::class_<ReasoningIncrementalParser, std::shared_ptr<ReasoningIncrementalParser>, IncrementalParser>(m, "ReasoningIncrementalParser")
        .def(py::init<bool, bool, const std::string&, const std::string&>(),
             py::arg("expect_open_tag") = true,
             py::arg("keep_original_content") = true,
             py::arg("open_tag") = "<think>",
             py::arg("close_tag") = "</think>");
    
    py::class_<Phi4ReasoningIncrementalParser, std::shared_ptr<Phi4ReasoningIncrementalParser>, IncrementalParser>(m, "Phi4ReasoningIncrementalParser")
        .def(py::init<bool>(), py::arg("expect_open_tag") = true);

    py::class_<DeepSeekR1ReasoningIncrementalParser, std::shared_ptr<DeepSeekR1ReasoningIncrementalParser>, IncrementalParser>(m, "DeepSeekR1ReasoningIncrementalParser")
        .def(py::init<bool>(), py::arg("expect_open_tag") = false);

    py::class_<Parser, ConstructableParser, std::shared_ptr<Parser>>(m, "Parser")
        .def(py::init<>())
        .def("parse",
            [](Parser& self, py::dict& msg) {
                auto msg_cpp = pyutils::py_object_to_json_container(msg);
                self.parse(msg_cpp);

                // TODO: msg = pyutils::json_container_to_py_object(msg_cpp) does not work properly here,
                py::object json_mod = py::module_::import("json");
                
                // since it create a new object instead of updating existing dict.
                auto json_str = msg_cpp.to_json_string();
                py::dict result = json_mod.attr("loads")(json_str);
                
                // update msg with result
                for (auto item : result) {
                    msg[item.first] = item.second;
                }
            },
            py::arg("text"),
            "Parse is called with the full text. Returns a dict with parsed content.");

    py::class_<Llama3JsonToolParser, std::shared_ptr<Llama3JsonToolParser>, Parser>(m, "Llama3JsonToolParser")
        .def(py::init<>());

    py::class_<Llama3PythonicToolParser, std::shared_ptr<Llama3PythonicToolParser>, Parser>(m, "Llama3PythonicToolParser")
        .def(py::init<>());
}
