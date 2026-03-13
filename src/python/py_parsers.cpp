// Copyright (C) 2023-2026 Intel Corporation
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

using ov::genai::Parser;
using ov::genai::ReasoningParser;
using ov::genai::DeepSeekR1ReasoningParser;
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
    void parse(JsonContainer& msg,
               const std::optional<std::vector<int64_t>>& tokens = std::nullopt) override {
        py::gil_scoped_acquire acquire;
        
        py::function parse_method = py::get_override(static_cast<const Parser*>(this), "parse");
        if (!parse_method) {
            OPENVINO_THROW("parse method not implemented in Python subclass");
        }
        
        // Convert JsonContainer to py::dict
        py::dict py_msg = pyutils::json_container_to_py_object(msg);
        try {
            parse_method(py_msg, tokens);
        } catch (const py::type_error&) {
            parse_method(py_msg);
        }
        msg = pyutils::py_object_to_json_container(py_msg);
    }

    std::string parseChunk(
        JsonContainer& msg,
        std::string& delta_text,
        const std::optional<std::vector<int64_t>>& delta_tokens = std::nullopt
    ) override {
        py::gil_scoped_acquire acquire;

        py::function parse_chunk_method = py::get_override(static_cast<const Parser*>(this), "parse_chunk");
        if (!parse_chunk_method) {
            parse_chunk_method = py::get_override(static_cast<const Parser*>(this), "parseChunk");
        }
        if (!parse_chunk_method) {
            OPENVINO_THROW("parse_chunk/parseChunk method not implemented in Python subclass");
        }

        py::dict py_msg = pyutils::json_container_to_py_object(msg);
        py::object res;
        try {
            res = parse_chunk_method(py_msg, delta_text, delta_tokens);
        } catch (const py::type_error&) {
            res = parse_chunk_method(py_msg, delta_text);
        }
        msg = pyutils::py_object_to_json_container(py_msg);
        return res.cast<std::string>();
    }

    void reset() override {
        py::gil_scoped_acquire acquire;

        py::function reset_method = py::get_override(static_cast<const Parser*>(this), "reset");
        if (!reset_method) {
            OPENVINO_THROW("reset method not implemented in Python subclass");
        }
        reset_method();
    }
};

class VLLMParserWrapper: public Parser {
// Wraps a Python parser to be used as a Parser.
// from vllm.entrypoints.openai.tool_parsers.*

// vLLM's Python object has implemented methods 'extract_tool_calls' and 'extract_reasoning'.
// This wrapper will call those methods and convert the results back to JsonContainer so that
// vLLM parsers can be used out of the box in Python.

public:
    std::vector<std::function<JsonContainer(const std::string&)>> m_parsers;
    VLLMParserWrapper(py::object py_parser) {
        // Check that has tool calling method
        if (py::hasattr(py_parser, "extract_tool_calls")) {
            m_parsers.push_back(
                [py_parser](const std::string& content) -> JsonContainer {
                    py::object parsed = py_parser.attr("extract_tool_calls")(content, py::none());
                    if (py::hasattr(parsed, "model_dump_json")) {
                        return JsonContainer::from_json_string(
                            parsed.attr("model_dump_json")().cast<std::string>());
                    } else if (py::hasattr(parsed, "json")) {
                        // json() method is deprecated but still supported in old versions
                        return JsonContainer::from_json_string(
                            parsed.attr("json")().cast<std::string>());
                    } else {
                        OPENVINO_THROW("Parsed object does not have model_dump_json() or json() method");
                    }
                }
            );
        }
        // Check that has reasoning extraction method
        if (py::hasattr(py_parser, "extract_reasoning")) {
            m_parsers.push_back(
                [py_parser](const std::string& content) -> JsonContainer {
                    py::object parsed = py_parser.attr("extract_reasoning")(content, py::none());
                    if (py::isinstance<py::tuple>(parsed) && py::len(parsed) == 2) {
                        auto msg_str_1 = parsed.attr("__getitem__")(0).cast<std::string>();
                        auto msg_str_2 = parsed.attr("__getitem__")(1).cast<std::string>();
                        JsonContainer new_message;
                        new_message["reasoning"] = msg_str_1;
                        new_message["content"] = msg_str_2;
                        return new_message;
                    } else {
                        OPENVINO_THROW("Parsed object is not a tuple of length 2");
                    }
                }
            );
        }
        OPENVINO_ASSERT(!m_parsers.empty(), "Provided vLLM parser does not have supported parsing methods: 'extract_tool_calls' or 'extract_reasoning'");
    }

    void parse(JsonContainer& message,
               const std::optional<std::vector<int64_t>>& /*tokens*/ = std::nullopt) override {
        py::gil_scoped_acquire acquire;

        JsonContainer new_message;
        auto content_str = message["content"].as_string();
        if (!content_str.has_value()) {
            OPENVINO_THROW("Message does not contain 'content' string field");
        }

        for (const auto& parser_func : m_parsers) {
            // Call each implemented parser from vLLM parser python object.
            auto result = parser_func(content_str.value());
            new_message.concatenate(result);
        }
        message = new_message;
    }

    std::string parseChunk(JsonContainer& /*delta_message*/,
                           std::string& delta_text,
                           const std::optional<std::vector<int64_t>>& /*delta_tokens*/ = std::nullopt) override {
        return delta_text;
    }

    void reset() override {}
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
            "Parse is called with the full text. Returns a dict with parsed content.")
        .def("parse_chunk",
            [](Parser& self,
               py::dict& delta_message,
               std::string& delta_text,
               const std::optional<std::vector<int64_t>>& delta_tokens = std::nullopt) {
                auto msg_cpp = pyutils::py_object_to_json_container(delta_message);
                auto res = self.parseChunk(msg_cpp, delta_text, delta_tokens);
                auto result = pyutils::json_container_to_py_object(msg_cpp);
                delta_message.attr("update")(result);
                return res;
            },
            py::arg("delta_message"), py::arg("delta_text"), py::arg("delta_tokens") = std::nullopt,
            "Parse is called for every decoded text delta. Returns text to append to output.")
        .def("reset", &Parser::reset, "Reset the internal state of the parser.");

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

    // Backward-compatible aliases for legacy incremental parser names.
    m.attr("IncrementalParser") = m.attr("Parser");
    m.attr("ReasoningIncrementalParser") = m.attr("ReasoningParser");
    m.attr("Phi4ReasoningIncrementalParser") = m.attr("Phi4ReasoningParser");
    m.attr("DeepSeekR1ReasoningIncrementalParser") = m.attr("DeepSeekR1ReasoningParser");

    py::class_<VLLMParserWrapper, std::shared_ptr<VLLMParserWrapper>, Parser>(m, "VLLMParserWrapper")
        .def(py::init<py::object>(), py::arg("py_parser"), "Wraps a vLLM parser to be used out of the box in Python.");
}
