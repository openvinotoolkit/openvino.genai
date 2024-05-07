// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "llm_pipeline.hpp"

namespace py = pybind11;
using namespace ov;

std::string call_with_config(ov::LLMPipeline& pipeline, const std::string& text, const py::kwargs& kwargs) {
    // Create a new GenerationConfig instance and initialize from kwargs
    ov::GenerationConfig config = pipeline.get_generation_config();
    if (kwargs.contains("max_new_tokens")) config.max_new_tokens = kwargs["max_new_tokens"].cast<size_t>();
    if (kwargs.contains("max_length")) config.max_length = kwargs["max_length"].cast<size_t>();
    if (kwargs.contains("ignore_eos")) config.ignore_eos = kwargs["ignore_eos"].cast<bool>();
    if (kwargs.contains("eos_token")) config.eos_token = kwargs["eos_token"].cast<std::string>();
    if (kwargs.contains("num_groups")) config.num_groups = kwargs["num_groups"].cast<size_t>();
    if (kwargs.contains("group_size")) config.group_size = kwargs["group_size"].cast<size_t>();
    if (kwargs.contains("diversity_penalty")) config.diversity_penalty = kwargs["diversity_penalty"].cast<float>();
    if (kwargs.contains("repetition_penalty")) config.repetition_penalty = kwargs["repetition_penalty"].cast<float>();
    if (kwargs.contains("length_penalty")) config.length_penalty = kwargs["length_penalty"].cast<float>();
    
    if (kwargs.contains("no_repeat_ngram_size")) config.no_repeat_ngram_size = kwargs["no_repeat_ngram_size"].cast<size_t>();
    if (kwargs.contains("temperature")) config.temperature = kwargs["temperature"].cast<float>();
    if (kwargs.contains("top_k")) config.top_k = kwargs["top_k"].cast<size_t>();
    if (kwargs.contains("top_p")) config.top_p = kwargs["top_p"].cast<float>();
    if (kwargs.contains("do_sample")) config.do_sample = kwargs["do_sample"].cast<bool>();
    if (kwargs.contains("bos_token_id")) config.bos_token_id = kwargs["bos_token_id"].cast<int64_t>();
    if (kwargs.contains("eos_token_id")) config.eos_token_id = kwargs["eos_token_id"].cast<int64_t>();
    if (kwargs.contains("pad_token_id")) config.pad_token_id = kwargs["pad_token_id"].cast<int64_t>();
    if (kwargs.contains("draft_model")) config.draft_model = kwargs["draft_model"].cast<std::variant<std::string, ov::CompiledModel, ov::InferRequest>>();

    // Call the LLMPipeline with the constructed GenerationConfig
    return pipeline(text, config);
}

PYBIND11_MODULE(py_generate_pipeline, m) {
    m.doc() = "Pybind11 binding for LLM Pipeline";


    py::class_<LLMPipeline>(m, "LLMPipeline")
        .def(py::init<std::string&, std::string&, std::string&, std::string, const ov::AnyMap&>(),
             py::arg("model_path"), py::arg("tokenizer_path"), py::arg("detokenizer_path"),
             py::arg("device") = "CPU", py::arg("plugin_config") = ov::AnyMap{})
        .def(py::init<std::string&, std::string, const ov::AnyMap&>(),
             py::arg("path"), py::arg("device") = "CPU", py::arg("plugin_config") = ov::AnyMap{})
         .def("__call__", py::overload_cast<std::string>(&ov::LLMPipeline::operator()), "Process single text input")
        .def("__call__", py::overload_cast<std::string, ov::GenerationConfig>(&ov::LLMPipeline::operator()), "Process text input with specific generation config")
        .def("__call__", py::overload_cast<std::vector<std::string>, ov::GenerationConfig>(&ov::LLMPipeline::operator()), "Process multiple text inputs with generation config")
        .def("__call__", &call_with_config)
        .def("generate", (EncodedResults (LLMPipeline::*)(ov::Tensor, ov::Tensor, GenerationConfig)) &LLMPipeline::generate)
        .def("generate", (EncodedResults (LLMPipeline::*)(ov::Tensor, ov::Tensor)) &LLMPipeline::generate)
        // Bind other methods similarly
        .def("get_tokenizer", &LLMPipeline::get_tokenizer)
        .def("start_chat", &ov::LLMPipeline::start_chat)
        .def("finish_chat", &ov::LLMPipeline::finish_chat)
        .def("reset_state", &ov::LLMPipeline::reset_state)
        .def("get_generation_config", &ov::LLMPipeline::get_generation_config, py::return_value_policy::copy)
        .def("set_generation_config", &ov::LLMPipeline::set_generation_config)
        .def("apply_chat_template", &LLMPipeline::apply_chat_template);

     // Binding for Tokenizer
    py::class_<ov::Tokenizer>(m, "Tokenizer")
        .def(py::init<>())
        .def(py::init<std::string&, std::string>(), py::arg("tokenizers_path"), py::arg("device") = "CPU")
        .def("encode", py::overload_cast<std::string>(&ov::Tokenizer::encode), "Encode a single prompt")
        .def("encode", py::overload_cast<std::vector<std::string>>(&ov::Tokenizer::encode), "Encode multiple prompts")
        .def("decode", py::overload_cast<std::vector<int64_t>>(&ov::Tokenizer::decode), "Decode a list of tokens")
        .def("decode", py::overload_cast<ov::Tensor>(&ov::Tokenizer::decode), "Decode a tensor of tokens")
        .def("decode", py::overload_cast<std::vector<std::vector<int64_t>>>(&ov::Tokenizer::decode), "Decode multiple lines of tokens");

    py::class_<ov::GenerationConfig>(m, "GenerationConfig")
        .def(py::init<>())
        .def(py::init<std::string>())
        .def_readwrite("max_new_tokens", &ov::GenerationConfig::max_new_tokens)
        .def_readwrite("max_length", &ov::GenerationConfig::max_length)
        .def_readwrite("ignore_eos", &ov::GenerationConfig::ignore_eos)
        .def_readwrite("eos_token", &ov::GenerationConfig::eos_token)
        .def_readwrite("num_groups", &ov::GenerationConfig::num_groups)
        .def_readwrite("group_size", &ov::GenerationConfig::group_size)
        .def_readwrite("diversity_penalty", &ov::GenerationConfig::diversity_penalty)
        .def_readwrite("repetition_penalty", &ov::GenerationConfig::repetition_penalty)
        .def_readwrite("length_penalty", &ov::GenerationConfig::length_penalty)
        .def_readwrite("no_repeat_ngram_size", &ov::GenerationConfig::no_repeat_ngram_size)
        .def_readwrite("temperature", &ov::GenerationConfig::temperature)
        .def_readwrite("top_k", &ov::GenerationConfig::top_k)
        .def_readwrite("top_p", &ov::GenerationConfig::top_p)
        .def_readwrite("do_sample", &ov::GenerationConfig::do_sample)
        .def_readwrite("bos_token_id", &ov::GenerationConfig::bos_token_id)
        .def_readwrite("eos_token_id", &ov::GenerationConfig::eos_token_id)
        .def_readwrite("pad_token_id", &ov::GenerationConfig::pad_token_id)
        .def_readwrite("draft_model", &ov::GenerationConfig::draft_model);


    py::class_<ov::DecodedResults>(m, "DecodedResults")
        .def(py::init<>())
        .def_readwrite("texts", &ov::DecodedResults::texts)
        .def_readwrite("scores", &ov::DecodedResults::scores);

    py::class_<ov::EncodedResults>(m, "EncodedResults")
        .def(py::init<>())
        .def_readwrite("tokens", &ov::EncodedResults::tokens)
        .def_readwrite("scores", &ov::EncodedResults::scores);

}
