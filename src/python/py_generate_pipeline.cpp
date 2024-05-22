// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "openvino/genai/llm_pipeline.hpp"

namespace py = pybind11;
using namespace ov;

void str_to_stop_criteria(ov::GenerationConfig& config, const std::string& stop_criteria_str){
    if (stop_criteria_str == "early") config.stop_criteria = StopCriteria::early;
    else if (stop_criteria_str == "never") config.stop_criteria =  StopCriteria::never;
    else if (stop_criteria_str == "heuristic") config.stop_criteria =  StopCriteria::heuristic;
    else OPENVINO_THROW(stop_criteria_str + " is incorrect value of stop_criteria. "
                       "Allowed values are: \"early\", \"never\", \"heuristic\". ");
}

std::string stop_criteria_to_str(const ov::GenerationConfig& config) {
    switch (config.stop_criteria) {
        case ov::StopCriteria::early: return "early";
        case ov::StopCriteria::heuristic: return "heuristic";
        case ov::StopCriteria::never: return "never";
        default: throw std::runtime_error("Incorrect stop_criteria");
    }
}

void update_config_from_kwargs(ov::GenerationConfig& config, const py::kwargs& kwargs) {
    if (kwargs.contains("max_new_tokens")) config.max_new_tokens = kwargs["max_new_tokens"].cast<size_t>();
    if (kwargs.contains("max_length")) config.max_length = kwargs["max_length"].cast<size_t>();
    if (kwargs.contains("ignore_eos")) config.ignore_eos = kwargs["ignore_eos"].cast<bool>();
    if (kwargs.contains("num_beam_groups")) config.num_beam_groups = kwargs["num_beam_groups"].cast<size_t>();
    if (kwargs.contains("num_beams")) config.num_beams = kwargs["num_beams"].cast<size_t>();
    if (kwargs.contains("diversity_penalty")) config.diversity_penalty = kwargs["diversity_penalty"].cast<float>();
    if (kwargs.contains("length_penalty")) config.length_penalty = kwargs["length_penalty"].cast<float>();
    if (kwargs.contains("num_return_sequences")) config.num_return_sequences = kwargs["num_return_sequences"].cast<size_t>();
    if (kwargs.contains("no_repeat_ngram_size")) config.no_repeat_ngram_size = kwargs["no_repeat_ngram_size"].cast<size_t>();
    if (kwargs.contains("stop_criteria")) str_to_stop_criteria(config, kwargs["stop_criteria"].cast<std::string>());
    if (kwargs.contains("temperature")) config.temperature = kwargs["temperature"].cast<float>();
    if (kwargs.contains("top_p")) config.top_p = kwargs["top_p"].cast<float>();
    if (kwargs.contains("top_k")) config.top_k = kwargs["top_k"].cast<size_t>();
    if (kwargs.contains("do_sample")) config.do_sample = kwargs["do_sample"].cast<bool>();
    if (kwargs.contains("repetition_penalty")) config.repetition_penalty = kwargs["repetition_penalty"].cast<float>();
    if (kwargs.contains("pad_token_id")) config.pad_token_id = kwargs["pad_token_id"].cast<int64_t>();
    if (kwargs.contains("bos_token_id")) config.bos_token_id = kwargs["bos_token_id"].cast<int64_t>();
    if (kwargs.contains("eos_token_id")) config.eos_token_id = kwargs["eos_token_id"].cast<int64_t>();
    if (kwargs.contains("eos_token")) config.eos_token = kwargs["eos_token"].cast<std::string>();
    if (kwargs.contains("bos_token")) config.bos_token = kwargs["bos_token"].cast<std::string>();
}

// operator() and generate methods are identical, operator() is just an alias for generate
std::string call_with_kwargs(ov::LLMPipeline& pipeline, const std::string& text, const py::kwargs& kwargs) {
    // Create a new GenerationConfig instance and initialize from kwargs
    ov::GenerationConfig config = pipeline.get_generation_config();
    update_config_from_kwargs(config, kwargs);
    return pipeline(text, config);
}

std::string call_with_config(ov::LLMPipeline& pipe, const std::string& text, const ov::GenerationConfig& config) {
    std::shared_ptr<StreamerBase> streamer;
    return pipe(text, config);
}

std::string ov_tokenizers_module_path() {
    py::module_ m = py::module_::import("openvino_tokenizers");
    py::list path_list = m.attr("__path__");
    return std::string(py::str(path_list[0])) + "/lib";
}

PYBIND11_MODULE(py_generate_pipeline, m) {
    m.doc() = "Pybind11 binding for LLM Pipeline";

    py::class_<LLMPipeline>(m, "LLMPipeline")
        .def(py::init<const std::string, const ov::Tokenizer&, const std::string, const ov::AnyMap&, const std::string&>(), 
             py::arg("model_path"), py::arg("tokenizer"), py::arg("device") = "CPU", 
             py::arg("plugin_config") = ov::AnyMap{}, py::arg("ov_tokenizers_path") = ov_tokenizers_module_path())
        .def(py::init<std::string&, std::string, const ov::AnyMap&, const std::string>(),
             py::arg("path"), py::arg("device") = "CPU", py::arg("plugin_config") = ov::AnyMap{}, py::arg("ov_tokenizers_path") = ov_tokenizers_module_path())
        .def("__call__", py::overload_cast<ov::LLMPipeline&, const std::string&, const py::kwargs&>(&call_with_kwargs))
        .def("__call__", py::overload_cast<ov::LLMPipeline&, const std::string&, const ov::GenerationConfig&>(&call_with_config))
        .def("generate", py::overload_cast<ov::LLMPipeline&, const std::string&, const py::kwargs&>(&call_with_kwargs))
        .def("generate", py::overload_cast<ov::LLMPipeline&, const std::string&, const ov::GenerationConfig&>(&call_with_config))
        
        // todo: if input_ids is a ov::Tensor/numpy tensor
        // todo: implement calling generate/operator() with StreamerBase or lambda streamer
        // signature to be implemented:
        // EncodedResults generate(ov::Tensor input_ids, 
        //                 std::optional<ov::Tensor> attention_mask, 
        //                 OptionalGenerationConfig generation_config=nullopt,
        //                 OptionalStreamerVariant streamer=nullopt);
        

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
        .def(py::init<std::string&, const std::string&, const std::string&>(), 
             py::arg("tokenizers_path"), 
             py::arg("device") = "CPU",
             py::arg("ov_tokenizers_path") = py::str(ov_tokenizers_module_path()))

        // todo: implement encode/decode when for numpy inputs and outputs
        .def("encode", py::overload_cast<const std::string>(&ov::Tokenizer::encode), "Encode a single prompt")
        // TODO: common.h(1106...) template argument deduction/substitution failed:
        // .def("encode", py::overload_cast<std::vector<std::string>&>(&ov::Tokenizer::encode), "Encode multiple prompts")
        .def("decode", py::overload_cast<std::vector<int64_t>>(&ov::Tokenizer::decode), "Decode a list of tokens")
        .def("decode", py::overload_cast<ov::Tensor>(&ov::Tokenizer::decode), "Decode a tensor of tokens")
        .def("decode", py::overload_cast<std::vector<std::vector<int64_t>>>(&ov::Tokenizer::decode), "Decode multiple lines of tokens");

     // Binding for GenerationConfig
    py::class_<ov::GenerationConfig>(m, "GenerationConfig")
        .def(py::init<>())
        .def(py::init<std::string>())
        .def_readwrite("max_new_tokens", &ov::GenerationConfig::max_new_tokens)
        .def_readwrite("max_length", &ov::GenerationConfig::max_length)
        .def_readwrite("ignore_eos", &ov::GenerationConfig::ignore_eos)
        .def_readwrite("num_beam_groups", &ov::GenerationConfig::num_beam_groups)
        .def_readwrite("num_beams", &ov::GenerationConfig::num_beams)
        .def_readwrite("diversity_penalty", &ov::GenerationConfig::diversity_penalty)
        .def_readwrite("length_penalty", &ov::GenerationConfig::length_penalty)
        .def_readwrite("num_return_sequences", &ov::GenerationConfig::num_return_sequences)
        .def_readwrite("no_repeat_ngram_size", &ov::GenerationConfig::no_repeat_ngram_size)
        .def_property("stop_criteria", &stop_criteria_to_str, &str_to_stop_criteria)
        .def_readwrite("temperature", &ov::GenerationConfig::temperature)
        .def_readwrite("top_p", &ov::GenerationConfig::top_p)
        .def_readwrite("top_k", &ov::GenerationConfig::top_k)
        .def_readwrite("do_sample", &ov::GenerationConfig::do_sample)
        .def_readwrite("repetition_penalty", &ov::GenerationConfig::repetition_penalty)
        .def_readwrite("pad_token_id", &ov::GenerationConfig::pad_token_id)
        .def_readwrite("bos_token_id", &ov::GenerationConfig::bos_token_id)
        .def_readwrite("eos_token_id", &ov::GenerationConfig::eos_token_id)
        .def_readwrite("eos_token", &ov::GenerationConfig::eos_token)
        .def_readwrite("bos_token", &ov::GenerationConfig::bos_token);

    py::class_<ov::DecodedResults>(m, "DecodedResults")
        .def(py::init<>())
        .def_readwrite("texts", &ov::DecodedResults::texts)
        .def_readwrite("scores", &ov::DecodedResults::scores);

    py::class_<ov::EncodedResults>(m, "EncodedResults")
        .def(py::init<>())
        .def_readwrite("tokens", &ov::EncodedResults::tokens)
        .def_readwrite("scores", &ov::EncodedResults::scores);

}
