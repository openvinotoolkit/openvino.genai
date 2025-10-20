// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/functional.h>

#include "tokenizer/tokenizers_path.hpp"

#include "py_utils.hpp"

namespace {

constexpr char class_docstring[] = R"(
    The class is used to encode prompts and decode resulting tokens

    Chat template is initialized from sources in the following order
    overriding the previous value:
    1. chat_template entry from tokenizer_config.json
    2. chat_template entry from processor_config.json
    3. chat_template entry from chat_template.json
    4. chat_template entry from rt_info section of openvino.Model
    5. If the template is known to be not supported by GenAI, it's
        replaced with a simplified supported version.
)";

constexpr char common_encode_docstring[] =R"(
 'add_special_tokens' - whether to add special tokens like BOS, EOS, PAD. Default is True.
 'pad_to_max_length' - whether to pad the sequence to the maximum length. Default is False.
 'max_length' - maximum length of the sequence. If None (default), the value will be taken from the IR (where default value from original HF/GGUF model is stored).
 'padding_side' - side to pad the sequence, can be 'left' or 'right'. If None (default), the value will be taken from the IR (where default value from original HF/GGUF model is stored).
Returns:
 TokenizedInputs object containing input_ids and attention_mask tensors.
)";

auto encode_list_docstring = (
R"(Encodes a list of prompts into tokenized inputs.
Args:
 'prompts' - list of prompts to encode)"
+ std::string(common_encode_docstring)
);

auto encode_single_prompt_docstring = (
R"(Encodes a single prompt into tokenized input.
Args:
 'prompt' - prompt to encode)"
+ std::string(common_encode_docstring)
);

auto encode_list_of_pairs_docstring = (
R"(Encodes a list of prompts into tokenized inputs. The number of strings must be the same, or one of the inputs can contain one string.
In the latter case, the single-string input will be broadcast into the shape of the other input, which is more efficient than repeating the string in pairs.)
Args:
 'prompts_1' - list of prompts to encode
 'prompts_2' - list of prompts to encode)"
+ std::string(common_encode_docstring)
);

auto encode_list_of_lists_docstring =
(
R"(Encodes a list of paired prompts into tokenized inputs. Input format is same as for HF paired input [[prompt_1, prompt_2], ...].
Args:
 'prompts' - list of prompts to encode\n)"
+ std::string(common_encode_docstring)
);

}  // namespace

namespace py = pybind11;
namespace pyutils = ov::genai::pybind::utils;

using ov::genai::ChatHistory;
using ov::genai::JsonContainer;
using ov::genai::TokenizedInputs;
using ov::genai::Tokenizer;

void init_tokenizer(py::module_& m) {
    py::class_<TokenizedInputs>(m, "TokenizedInputs")
        .def(py::init<ov::Tensor, ov::Tensor>(), py::arg("input_ids"), py::arg("attention_mask"))
        .def_readwrite("input_ids", &TokenizedInputs::input_ids)
        .def_readwrite("attention_mask", &TokenizedInputs::attention_mask);

    py::class_<ov::genai::Tokenizer>(m, "Tokenizer", class_docstring)

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

        .def(py::init([](const std::string& tokenizer_model, const ov::Tensor& tokenizer_weights,
                         const std::string& detokenizer_model, const ov::Tensor& detokenizer_weights,
                         const py::kwargs& kwargs) {
            ScopedVar env_manager(pyutils::ov_tokenizers_module_path());
            auto kwargs_properties = pyutils::kwargs_to_any_map(kwargs);

            return std::make_unique<ov::genai::Tokenizer>(tokenizer_model, tokenizer_weights, detokenizer_model, detokenizer_weights, kwargs_properties);
        }), py::arg("tokenizer_model"), py::arg("tokenizer_weights"), py::arg("detokenizer_model"), py::arg("detokenizer_weights"))
        
        .def("encode", [](Tokenizer& tok, std::vector<std::string>& prompts, 
                          bool add_special_tokens, 
                          bool pad_to_max_length,
                          std::optional<size_t> max_length,
                          std::optional<std::string> padding_side) {
                ov::AnyMap tokenization_params;
                tokenization_params[ov::genai::add_special_tokens.name()] = add_special_tokens;
                tokenization_params[ov::genai::pad_to_max_length.name()] = pad_to_max_length;

                if (max_length.has_value()) {
                    tokenization_params[ov::genai::max_length.name()] = *max_length;
                }
                if (padding_side.has_value()) {
                    tokenization_params[ov::genai::padding_side.name()] = *padding_side;
                }
                return tok.encode(prompts, tokenization_params);
            },
            py::arg("prompts"),
            py::arg("add_special_tokens") = true,
            py::arg("pad_to_max_length") = false,
            py::arg("max_length") = std::nullopt,
            py::arg("padding_side") = std::nullopt,
            encode_list_docstring.c_str())

        .def("encode", [](Tokenizer& tok, const std::string prompt, 
                          bool add_special_tokens, 
                          bool pad_to_max_length,
                          std::optional<size_t> max_length,
                          std::optional<std::string> padding_side
                        ) {
                ov::AnyMap tokenization_params;
                tokenization_params[ov::genai::add_special_tokens.name()] = add_special_tokens;
                tokenization_params[ov::genai::pad_to_max_length.name()] = pad_to_max_length;
                if (max_length.has_value()) {
                    tokenization_params[ov::genai::max_length.name()] = *max_length;
                }
                if (padding_side.has_value()) {
                    tokenization_params[ov::genai::padding_side.name()] = *padding_side;
                }
                return tok.encode(prompt, tokenization_params);
            },
            py::arg("prompt"), 
            py::arg("add_special_tokens") = true, 
            py::arg("pad_to_max_length") = false,
            py::arg("max_length") = std::nullopt,
            py::arg("padding_side") = std::nullopt,
            encode_single_prompt_docstring.c_str())

            .def("encode", [](Tokenizer& tok, 
                std::vector<std::string>& prompts_1, 
                std::vector<std::string>& prompts_2,
                bool add_special_tokens, 
                bool pad_to_max_length,
                std::optional<size_t> max_length,
                std::optional<std::string> padding_side) {
                ov::AnyMap tokenization_params;
                tokenization_params[ov::genai::add_special_tokens.name()] = add_special_tokens;
                tokenization_params[ov::genai::pad_to_max_length.name()] = pad_to_max_length;

                if (max_length.has_value()) {
                    tokenization_params[ov::genai::max_length.name()] = *max_length;
                }
                if (padding_side.has_value()) {
                    tokenization_params[ov::genai::padding_side.name()] = *padding_side;
                }
                return tok.encode(prompts_1, prompts_2, tokenization_params);
            },
            py::arg("prompts_1"),
            py::arg("prompts_2"),
            py::arg("add_special_tokens") = true,
            py::arg("pad_to_max_length") = false,
            py::arg("max_length") = std::nullopt,
            py::arg("padding_side") = std::nullopt,
            encode_list_of_pairs_docstring.c_str())
            
            .def("encode", [](Tokenizer& tok, py::list& prompts, 
                            bool add_special_tokens, 
                            bool pad_to_max_length,
                            std::optional<size_t> max_length,
                            std::optional<std::string> padding_side) {
                ov::AnyMap tokenization_params;
                tokenization_params[ov::genai::add_special_tokens.name()] = add_special_tokens;
                tokenization_params[ov::genai::pad_to_max_length.name()] = pad_to_max_length;
                
                if (max_length.has_value()) {
                    tokenization_params[ov::genai::max_length.name()] = *max_length;
                }
                if (padding_side.has_value()) {
                    tokenization_params[ov::genai::padding_side.name()] = *padding_side;
                }

                // Convert py::list to std::vector<std::string>
                std::vector<std::pair<std::string, std::string>> prompts_vector;
                for (auto item : prompts) {
                    if (!py::isinstance<py::list>(item) || py::len(item) != 2) {
                        throw std::runtime_error("Expected a list of lists with sizes 2. E.g. [[\"What is the capital of GB?\", \"London in the capital of GB\"], ...]");
                    } 

                    prompts_vector.push_back(py::cast<std::pair<std::string, std::string>>(item));
                }
                return tok.encode(prompts_vector, tokenization_params);
            },
            py::arg("prompts"),
            py::arg("add_special_tokens") = true,
            py::arg("pad_to_max_length") = false,
            py::arg("max_length") = std::nullopt,
            py::arg("padding_side") = std::nullopt,
            encode_list_of_lists_docstring.c_str()
        )

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
                                        const std::variant<ChatHistory, std::vector<py::dict>>& history,
                                        bool add_generation_prompt,
                                        const std::string& chat_template,
                                        const std::optional<std::vector<py::dict>>& tools,
                                        const std::optional<py::dict>& extra_context) {
            ChatHistory chat_history;
            std::visit(pyutils::overloaded {
                [&](ChatHistory chat_history_obj) {
                    chat_history = chat_history_obj;
                },
                [&](const std::vector<py::dict>& list_of_dicts) {
                    chat_history = ChatHistory(pyutils::py_object_to_json_container(py::cast(list_of_dicts)));
                }
            }, history);

            std::optional<JsonContainer> tools_jc;
            if (tools.has_value()) {
                tools_jc = pyutils::py_object_to_json_container(py::cast(tools.value()));
            }

            std::optional<JsonContainer> extra_context_jc;
            if (extra_context.has_value()) {
                extra_context_jc = pyutils::py_object_to_json_container(extra_context.value());
            }

            return tok.apply_chat_template(chat_history, add_generation_prompt, chat_template, tools_jc, extra_context_jc);
        },
            py::arg("history"),
            py::arg("add_generation_prompt"),
            py::arg("chat_template") = "",
            py::arg("tools") = py::none(),
            py::arg("extra_context") = py::none(),
            R"(Applies a chat template to format chat history into a prompt string.)")

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

        .def("get_original_chat_template", &Tokenizer::get_original_chat_template)
        .def("get_pad_token_id", &Tokenizer::get_pad_token_id)
        .def("get_bos_token_id", &Tokenizer::get_bos_token_id)
        .def("get_eos_token_id", &Tokenizer::get_eos_token_id)
        .def("get_pad_token", &Tokenizer::get_pad_token)
        .def("get_bos_token", &Tokenizer::get_bos_token)
        .def("get_eos_token", &Tokenizer::get_eos_token)
        .def("get_vocab",
            [](Tokenizer& tok) {
                const auto vocab = tok.get_vocab();
                py::dict result;
                for (const auto& [key, value] : vocab) {
                    py::bytes key_bytes(key);  // Use bytes for keys to avoid UTF-8 encoding issues
                    result[key_bytes] = value;
                }
                return result;
            },
             R"(Returns the vocabulary as a Python dictionary with bytes keys and integer values. 
             Bytes are used for keys because not all vocabulary entries might be valid UTF-8 strings.)"
        )
        .def("get_vocab_vector", &Tokenizer::get_vocab_vector, 
             R"(Returns the vocabulary as list of strings, where position of a string represents token ID.)"
        )
        .def("supports_paired_input", &Tokenizer::supports_paired_input, 
             R"(Returns true if the tokenizer supports paired input, false otherwise.)"
        );
}
