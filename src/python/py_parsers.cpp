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

using ov::genai::IncrementalParserBase;
using ov::genai::ParserBase;
using ov::genai::ReasoningParser;
using ov::genai::Phi4ReasoningParser;
using ov::genai::DeepSeekR1ReasoningParser;
using ov::genai::JsonContainer;
using ov::genai::Llama32JsonToolParser;
using ov::genai::Llama32PythonicToolParser;
using ov::genai::Tokenizer;
using ov::genai::StreamingStatus;

namespace pyutils = ov::genai::pybind::utils;

namespace {


class ConstructableIncrementalParserBase: public IncrementalParserBase {
public:
    std::string parse(
        JsonContainer& msg,
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
};

class ConstructableParserBase: public ParserBase {
public:
    JsonContainer parse(JsonContainer& text) override {
        PYBIND11_OVERRIDE_PURE(
            JsonContainer,  // Return type
            ParserBase,  // Parent class
            parse,  // Name of function in C++ (must match Python name)
            text  // Argument(s)
        );
    }
};

static py::object json_mod = py::module_::import("json");

// wrapper to enhance calling parser from Python
void call_parser(py::dict& msg, std::function<JsonContainer(JsonContainer&)> func) {
    auto msg_anymap = ov::genai::pybind::utils::py_object_to_any_map(msg);
    auto msg_cpp = JsonContainer(msg_anymap);

    func(msg_cpp);

    auto json_str = msg_cpp.to_json_string();
    py::dict result = json_mod.attr("loads")(json_str);
    
    // update msg with result
    msg.clear();
    for (auto item : result) {
        msg[item.first] = item.second;
    }
}

// wrapper to enhance calling incremental parser from Python
std::string call_incremental_parser(
    IncrementalParserBase& parser,
    py::dict& msg,
    const std::string& previous_text,
    std::string& delta_text,
    const std::optional<std::vector<int64_t>>& previous_tokens,
    const std::optional<std::vector<int64_t>>& delta_tokens,
    std::function<std::string(JsonContainer&, const std::string&, std::string&, const std::optional<std::vector<int64_t>>&,
                               const std::optional<std::vector<int64_t>>&)> func) {
    auto msg_anymap = ov::genai::pybind::utils::py_object_to_any_map(msg);
    auto msg_cpp = JsonContainer(msg_anymap);

    auto res = func(msg_cpp, previous_text, delta_text, previous_tokens, delta_tokens);

    auto json_str = msg_cpp.to_json_string();
    py::dict result = json_mod.attr("loads")(json_str);
    
    // update msg with result
    msg.clear();
    for (auto item : result) {
        msg[item.first] = item.second;
    }
    return res;   
}

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
            // TODO: optimize conversion between py::dict and JsonContainer
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
           "Parse is called every time new text delta is decoded. Returns a string with any additional text to append to the current output.");
    
    py::class_<Phi4ReasoningParser, std::shared_ptr<Phi4ReasoningParser>, IncrementalParserBase>(m, "Phi4ReasoningParser")
        .def(py::init<bool>(), py::arg("expect_open_tag") = false)
        .def("parse",
            [](Phi4ReasoningParser& self,
               py::dict& msg,
               const std::string& previous_text,
               std::string& delta_text,
               const std::optional<std::vector<int64_t>>& previous_tokens = std::nullopt,
               const std::optional<std::vector<int64_t>>& delta_tokens = std::nullopt) {
                return call_incremental_parser(
                    self,
                    msg,
                    previous_text,
                    delta_text,
                    previous_tokens,
                    delta_tokens,
                    [&self](JsonContainer& m, const std::string& prev_text, std::string& delta_t,
                            const std::optional<std::vector<int64_t>>& prev_tokens,
                            const std::optional<std::vector<int64_t>>& delta_toks) {
                        return self.parse(m, prev_text, delta_t, prev_tokens, delta_toks);
                    });
            },
            "Parse is called every time new text delta is decoded. Returns a string with any additional text to append to the current output.",
            py::arg("msg"), py::arg("previous_text"), py::arg("delta_text"),
            py::arg("previous_tokens") = std::nullopt, py::arg("delta_tokens") = std::nullopt);

    py::class_<DeepSeekR1ReasoningParser, std::shared_ptr<DeepSeekR1ReasoningParser>, IncrementalParserBase>(m, "DeepSeekR1ReasoningParser")
        .def(py::init<>())
        .def("parse",
            [](DeepSeekR1ReasoningParser& self,
               py::dict& msg,
               const std::string& previous_text,
               std::string& delta_text,
               const std::optional<std::vector<int64_t>>& previous_tokens = std::nullopt,
               const std::optional<std::vector<int64_t>>& delta_tokens = std::nullopt) {
                return call_incremental_parser(
                    self,
                    msg,
                    previous_text,
                    delta_text,
                    previous_tokens,
                    delta_tokens,
                    [&self](JsonContainer& m, const std::string& prev_text, std::string& delta_t,
                            const std::optional<std::vector<int64_t>>& prev_tokens,
                            const std::optional<std::vector<int64_t>>& delta_toks) {
                        return self.parse(m, prev_text, delta_t, prev_tokens, delta_toks);
                    });
            },
            "Parse is called with the full text. Returns a dict with parsed content.",
            py::arg("msg"), py::arg("previous_text"), py::arg("delta_text"),
            py::arg("previous_tokens") = std::nullopt, py::arg("delta_tokens") = std::nullopt);

    py::class_<ParserBase, ConstructableParserBase, std::shared_ptr<ParserBase>>(m, "ParserBase")
    .def(py::init<>())
    .def("parse",
        [](ParserBase& self, py::dict& msg) {
            return call_parser(msg, [&self](JsonContainer& m) {return self.parse(m);});
        },
        py::arg("text"),
        "Parse is called with the full text. Returns a dict with parsed content.");
    
    py::class_<Llama32JsonToolParser, std::shared_ptr<Llama32JsonToolParser>, ParserBase>(m, "Llama32JsonToolParser")
        .def(py::init<>())
        .def("parse",
            [](Llama32JsonToolParser& self, py::dict& msg) {
                return call_parser(msg, [&self](JsonContainer& m) { return self.parse(m); });
            },
            py::arg("text"),
            "Parse is called with the full text. Returns a dict with parsed content.");

    py::class_<Llama32PythonicToolParser, std::shared_ptr<Llama32PythonicToolParser>, ParserBase>(m, "Llama32PythonicToolParser")
        .def(py::init<>())
        .def("parse",
            [](Llama32PythonicToolParser& self, py::dict& msg) {
                return call_parser(msg, [&self](JsonContainer& m) { return self.parse(m); });
            },
            py::arg("text"),
            "Parse is called with the full text. Returns a dict with parsed content.");
}
