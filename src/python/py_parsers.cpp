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

using ov::genai::ParsedMessage;
using ov::genai::IncrementalParserBase;
using ov::genai::ParserVariant;
using ov::genai::ParserBase;
using ov::genai::Tokenizer;
using ov::genai::StreamingStatus;
using ov::genai::JsonContainer;

namespace pyutils = ov::genai::pybind::utils;

namespace {


class ConstructableIncrementalParserBase: public IncrementalParserBase {
public:
    std::string parse(
        ParsedMessage& msg,
        const std::string& previous_text, 
        std::string& delta_text, 
        const std::optional<std::vector<int64_t>>& previous_tokens = std::nullopt, 
        const std::optional<std::vector<int64_t>>& delta_tokens = std::nullopt
    ) override {
        PYBIND11_OVERRIDE_PURE(
            std::string,  // Return type
            IncrementalParserBase,  // Parent class
            parse,  // Name of function in C++ (must match Python name)
            msg,
            previous_text,
            delta_text,
            previous_tokens,
            delta_tokens
        );
    }
    
    bool is_active() const override {
        PYBIND11_OVERRIDE_PURE(
            bool,  // Return type
            IncrementalParserBase,  // Parent class
            is_active,  // Name of function in C++ (must match Python name)
        );
    }
};

class ConstructableParserBase: public ParserBase {
public:
    ParsedMessage parse(ParsedMessage& text) override {
        PYBIND11_OVERRIDE_PURE(
            ParsedMessage,  // Return type
            ParserBase,  // Parent class
            parse,  // Name of function in C++ (must match Python name)
            text  // Argument(s)
        );
    }
};

} // namespace

// TODO: double check/add more relevant docstrings for parsers.
void init_parsers(py::module_& m) {
    py::class_<IncrementalParserBase, ConstructableIncrementalParserBase, std::shared_ptr<IncrementalParserBase>>(m, "IncrementalParserBase")
        .def(py::init<>())
        .def("parse", [](IncrementalParserBase& self,
                         py::dict& msg,
                         std::string& previous_text,
                         std::string& delta_text,
                         const std::optional<std::vector<int64_t>>& previous_tokens = std::nullopt,
                         const std::optional<std::vector<int64_t>>& delta_tokens = std::nullopt) {
            // TODO: optimize conversion between py::dict and ParsedMessage
            auto msg_anymap = ov::genai::pybind::utils::py_object_to_any_map(msg);
            auto msg_cpp = JsonContainer(msg_anymap);


            auto res = self.parse(msg_cpp, previous_text, delta_text, previous_tokens, delta_tokens);
            msg.clear();
            
            auto json_obj = msg_cpp.to_json();
            for (auto it = json_obj.begin(); it != json_obj.end(); ++it) {
                msg[py::cast(it.key())] = py::cast(it.value());
            }

            return res;
        }, py::arg("msg"), py::arg("previous_text"), py::arg("delta_text"),
           py::arg("previous_tokens") = std::nullopt, py::arg("delta_tokens") = std::nullopt,
           "Parse is called every time new text delta is decoded. Returns a string with any additional text to append to the current output.")
        .def("is_active", &IncrementalParserBase::is_active, "Indicates whether the parser is active and should be used during parsing.");

        
        py::class_<ParserBase, ConstructableParserBase, std::shared_ptr<ParserBase>>(m, "ParserBase")
        .def(py::init<>())
        .def("parse",
            &ParserBase::parse,
            "Parse is called with the full text. Returns a ParsedMessage with parsed content.",
            py::arg("text"));
}
