// Copyright (C) 2023-2025 Intel Corporation
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
using ov::genai::PaddingMode;
using PadModesVariant = std::variant<std::string, PaddingMode>;

namespace {

PaddingMode get_pad_mode(const PadModesVariant& pad_mode) {
    PaddingMode result;
    std::visit(pyutils::overloaded {
    [&](PaddingMode padding_mode) {
        result = padding_mode;
    },
    [&](std::basic_string<char> pad_mode_str) {
        if (pad_mode_str == "truncate") {
            result = PaddingMode::TRUNCATE;
        } else if (pad_mode_str == "longest") {
            result = PaddingMode::LONGEST;
        } else if (pad_mode_str == "max_length") {
            result = PaddingMode::MAX_LENGTH;
        } else if (pad_mode_str == "do_not_pad") {
            result = PaddingMode::DO_NOT_PAD;
        } else {
            OPENVINO_THROW("Unknown padding mode: " + pad_mode_str + ".\nAllowed values are \"longest\", \"max_length\", \"do_not_pad\", \"truncate\"");
        }
    }}, pad_mode);
    return result;
}

}
void init_tokenizer(py::module_& m) {
    py::enum_<PaddingMode>(m, "PaddingMode", "") // TODO: write docsting
        .value("TRUNCATE", PaddingMode::TRUNCATE)
        .value("LONGEST", PaddingMode::LONGEST)
        .value("MAX_LENGTH", PaddingMode::MAX_LENGTH)
        .value("DO_NOT_PAD", PaddingMode::DO_NOT_PAD);

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

        .def("encode", [](Tokenizer& tok, std::vector<std::string>& prompts, 
                          bool add_special_tokens, 
                          size_t max_length,
                          PadModesVariant padding_mode) {
                ov::AnyMap tokenization_params;
                tokenization_params[ov::genai::add_special_tokens.name()] = add_special_tokens;
                tokenization_params[ov::genai::max_length.name()] = max_length;
                tokenization_params[ov::genai::padding_mode.name()] = get_pad_mode(padding_mode);
                return tok.encode(prompts, tokenization_params);
            },
            py::arg("prompts"),
            py::arg("add_special_tokens") = true,
            py::arg("max_length") = std::numeric_limits<int>::max(),
            py::arg("padding_mode") = "truncate",
            R"(Encodes a list of prompts into tokenized inputs.)")

        .def("encode", [](Tokenizer& tok, const std::string prompt, 
                          bool add_special_tokens, 
                          size_t max_length,
                          PadModesVariant padding_mode) {
                ov::AnyMap tokenization_params;
                tokenization_params[ov::genai::add_special_tokens.name()] = add_special_tokens;
                tokenization_params[ov::genai::max_length.name()] = max_length;
                auto res = get_pad_mode(padding_mode);
                tokenization_params[ov::genai::padding_mode.name()] = res;
                return tok.encode(prompt, tokenization_params);
            },
            py::arg("prompt"), py::arg("add_special_tokens") = true, 
            py::arg("max_length") = std::numeric_limits<int>::max(),
            py::arg("padding_mode") = "truncate",
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

        .def_property(
            "chat_template",
            &Tokenizer::get_chat_template,
            &Tokenizer::set_chat_template
        )

        .def("get_pad_token_id", &Tokenizer::get_pad_token_id)
        .def("get_bos_token_id", &Tokenizer::get_bos_token_id)
        .def("get_eos_token_id", &Tokenizer::get_eos_token_id)
        .def("get_pad_token", &Tokenizer::get_pad_token)
        .def("get_bos_token", &Tokenizer::get_bos_token)
        .def("get_eos_token", &Tokenizer::get_eos_token);
}
