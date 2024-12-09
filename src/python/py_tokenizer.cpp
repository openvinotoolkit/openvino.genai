// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/functional.h>

#include "tokenizers_path.hpp"

#include "py_utils.hpp"

namespace py = pybind11;
namespace pyutils = ov::genai::pybind::utils;

using ov::genai::ChatHistory;
using ov::genai::TokenizedInputs;
using ov::genai::Tokenizer;

void init_tokenizer(py::module_& m) {
    py::class_<TokenizedInputs>(m, "TokenizedInputs")
        .def(py::init<ov::Tensor, ov::Tensor>(), py::arg("input_ids"), py::arg("attention_mask"))
        .def_readwrite("input_ids", &TokenizedInputs::input_ids)
        .def_readwrite("attention_mask", &TokenizedInputs::attention_mask);

    py::class_<ov::genai::Tokenizer>(m, "Tokenizer",
        R"(openvino_genai.Tokenizer object is used to initialize Tokenizer
           if it's located in a different path than the main model.)")

        .def(py::init([](const std::filesystem::path& tokenizer_path, const std::map<std::string, py::object>& properties, const py::kwargs& kwargs) {
            ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
            auto kwargs_properties = pyutils::kwargs_to_any_map(kwargs);
            if (properties.size()) {
                PyErr_WarnEx(PyExc_DeprecationWarning, 
                         "'properties' parameters is deprecated, please use kwargs to pass config properties instead.", 
                         1);
                auto map_properties = pyutils::properties_to_any_map(properties);
                kwargs_properties.insert(map_properties.begin(), map_properties.end());
            }

            return std::make_unique<ov::genai::Tokenizer>(tokenizer_path, kwargs_properties);
        }), py::arg("tokenizer_path"), py::arg("properties") = ov::AnyMap({}))

        .def("encode", [](Tokenizer& tok, std::vector<std::string>& prompts, bool add_special_tokens) {
                ov::AnyMap tokenization_params;
                tokenization_params[ov::genai::add_special_tokens.name()] = add_special_tokens;
                return tok.encode(prompts, tokenization_params);
            },
            py::arg("prompts"),
            py::arg("add_special_tokens") = true,
            R"(Encodes a list of prompts into tokenized inputs.)")

        .def("encode", [](Tokenizer& tok, const std::string prompt, bool add_special_tokens) {
                ov::AnyMap tokenization_params;
                tokenization_params[ov::genai::add_special_tokens.name()] = add_special_tokens;
                return tok.encode(prompt, tokenization_params);
            },
            py::arg("prompt"), py::arg("add_special_tokens") = true,
            R"(Encodes a single prompt into tokenized input.)")

        .def(
            "decode",
            [](Tokenizer& tok, std::vector<int64_t>& tokens, bool skip_special_tokens) -> py::str {
                ov::AnyMap detokenization_params;
                detokenization_params[ov::genai::skip_special_tokens.name()] = skip_special_tokens;
                return pyutils::handle_utf8(tok.decode(tokens, detokenization_params));
            },
            py::arg("tokens"), py::arg("skip_special_tokens") = true,
            R"(Decode a sequence into a string prompt.)"
        )

        .def(
            "decode",
            [](Tokenizer& tok, ov::Tensor& tokens, bool skip_special_tokens) -> py::typing::List<py::str> {
                ov::AnyMap detokenization_params;
                detokenization_params[ov::genai::skip_special_tokens.name()] = skip_special_tokens;
                return pyutils::handle_utf8(tok.decode(tokens, detokenization_params));
            },
            py::arg("tokens"), py::arg("skip_special_tokens") = true,
            R"(Decode tensor into a list of string prompts.)")

        .def(
            "decode",
            [](Tokenizer& tok, std::vector<std::vector<int64_t>>& tokens, bool skip_special_tokens) -> py::typing::List<py::str> {
                ov::AnyMap detokenization_params;
                detokenization_params[ov::genai::skip_special_tokens.name()] = skip_special_tokens;
                return pyutils::handle_utf8(tok.decode(tokens, detokenization_params));
            },
            py::arg("tokens"), py::arg("skip_special_tokens") = true,
            R"(Decode a batch of tokens into a list of string prompt.)")

        .def("apply_chat_template", [](Tokenizer& tok,
                                        ChatHistory history,
                                        bool add_generation_prompt,
                                        const std::string& chat_template) {
            return tok.apply_chat_template(history, add_generation_prompt, chat_template);
        },
            py::arg("history"),
            py::arg("add_generation_prompt"),
            py::arg("chat_template") = "",
            R"(Embeds input prompts with special tags for a chat scenario.)")

        .def(
            "set_chat_template", &Tokenizer::set_chat_template,
            py::arg("chat_template"), "The new template to override with.",
            "Override a chat_template read from tokenizer_config.json."
        )

        .def("get_pad_token_id", &Tokenizer::get_pad_token_id)
        .def("get_bos_token_id", &Tokenizer::get_bos_token_id)
        .def("get_eos_token_id", &Tokenizer::get_eos_token_id)
        .def("get_pad_token", &Tokenizer::get_pad_token)
        .def("get_bos_token", &Tokenizer::get_bos_token)
        .def("get_eos_token", &Tokenizer::get_eos_token);
}
