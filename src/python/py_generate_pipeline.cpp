// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include "openvino/genai/llm_pipeline.hpp"

namespace py = pybind11;
using ov::genai::LLMPipeline;
using ov::genai::Tokenizer;
using ov::genai::GenerationConfig;
using ov::genai::EncodedResults;
using ov::genai::DecodedResults;
using ov::genai::StopCriteria;
using ov::genai::StreamerBase;

void str_to_stop_criteria(GenerationConfig& config, const std::string& stop_criteria_str){
    if (stop_criteria_str == "early") config.stop_criteria = StopCriteria::early;
    else if (stop_criteria_str == "never") config.stop_criteria =  StopCriteria::never;
    else if (stop_criteria_str == "heuristic") config.stop_criteria =  StopCriteria::heuristic;
    else OPENVINO_THROW(stop_criteria_str + " is incorrect value of stop_criteria. "
                       "Allowed values are: \"early\", \"never\", \"heuristic\". ");
}

std::string stop_criteria_to_str(const GenerationConfig& config) {
    switch (config.stop_criteria) {
        case StopCriteria::early: return "early";
        case StopCriteria::heuristic: return "heuristic";
        case StopCriteria::never: return "never";
        default: throw std::runtime_error("Incorrect stop_criteria");
    }
}

void update_config_from_kwargs(GenerationConfig& config, const py::kwargs& kwargs) {
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
std::string call_with_kwargs(LLMPipeline& pipeline, const std::string& text, const py::kwargs& kwargs) {
    // Create a new GenerationConfig instance and initialize from kwargs
    GenerationConfig config = pipeline.get_generation_config();
    update_config_from_kwargs(config, kwargs);
    return pipeline(text, config);
}

std::string call_with_config(LLMPipeline& pipe, const std::string& text, const GenerationConfig& config) {
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
        .def(py::init<const std::string, const Tokenizer&, const std::string, const ov::AnyMap&>(), 
             py::arg("model_path"), py::arg("tokenizer"), py::arg("device") = "CPU", 
             py::arg("plugin_config") = ov::AnyMap{})
        .def(py::init<std::string&, std::string, const ov::AnyMap&, const std::string>(),
             py::arg("path"), py::arg("device") = "CPU", py::arg("plugin_config") = ov::AnyMap{}, py::arg("ov_tokenizers_path") = ov_tokenizers_module_path())
        .def("__call__", py::overload_cast<LLMPipeline&, const std::string&, const py::kwargs&>(&call_with_kwargs))
        .def("__call__", py::overload_cast<LLMPipeline&, const std::string&, const GenerationConfig&>(&call_with_config))
        .def("generate", py::overload_cast<LLMPipeline&, const std::string&, const py::kwargs&>(&call_with_kwargs))
        .def("generate", py::overload_cast<LLMPipeline&, const std::string&, const GenerationConfig&>(&call_with_config))
        
        // todo: if input_ids is a ov::Tensor/numpy tensor
        // todo: implement calling generate/operator() with StreamerBase or lambda streamer
        // signature to be implemented:
        // EncodedResults generate(ov::Tensor input_ids, 
        //                 std::optional<ov::Tensor> attention_mask, 
        //                 OptionalGenerationConfig generation_config=nullopt,
        //                 OptionalStreamerVariant streamer=nullopt);
        

        .def("get_tokenizer", &LLMPipeline::get_tokenizer)
        .def("start_chat", &LLMPipeline::start_chat)
        .def("finish_chat", &LLMPipeline::finish_chat)
        .def("reset_state", &LLMPipeline::reset_state)
        .def("get_generation_config", &LLMPipeline::get_generation_config, py::return_value_policy::copy)
        .def("set_generation_config", &LLMPipeline::set_generation_config)
        .def("apply_chat_template", &LLMPipeline::apply_chat_template);

     // Binding for Tokenizer
    py::class_<Tokenizer>(m, "Tokenizer")
        .def(py::init<>())
        .def(py::init<std::string&, const std::string&, const std::string&>(), 
             py::arg("tokenizers_path"), 
             py::arg("device") = "CPU",
             py::arg("ov_tokenizers_path") = py::str(ov_tokenizers_module_path()))

        // todo: implement encode/decode when for numpy inputs and outputs
        .def("encode", py::overload_cast<const std::string>(&Tokenizer::encode), "Encode a single prompt")
        // TODO: common.h(1106...) template argument deduction/substitution failed:
        // .def("encode", py::overload_cast<std::vector<std::string>&>(&Tokenizer::encode), "Encode multiple prompts")
        .def("decode", py::overload_cast<std::vector<int64_t>>(&Tokenizer::decode), "Decode a list of tokens")
        .def("decode", py::overload_cast<ov::Tensor>(&Tokenizer::decode), "Decode a tensor of tokens")
        .def("decode", py::overload_cast<std::vector<std::vector<int64_t>>>(&Tokenizer::decode), "Decode multiple lines of tokens");

     // Binding for GenerationConfig
    py::class_<GenerationConfig>(m, "GenerationConfig")
        .def(py::init<>())
        .def(py::init<std::string>())
        .def_readwrite("max_new_tokens", &GenerationConfig::max_new_tokens)
        .def_readwrite("max_length", &GenerationConfig::max_length)
        .def_readwrite("ignore_eos", &GenerationConfig::ignore_eos)
        .def_readwrite("num_beam_groups", &GenerationConfig::num_beam_groups)
        .def_readwrite("num_beams", &GenerationConfig::num_beams)
        .def_readwrite("diversity_penalty", &GenerationConfig::diversity_penalty)
        .def_readwrite("length_penalty", &GenerationConfig::length_penalty)
        .def_readwrite("num_return_sequences", &GenerationConfig::num_return_sequences)
        .def_readwrite("no_repeat_ngram_size", &GenerationConfig::no_repeat_ngram_size)
        .def_property("stop_criteria", &stop_criteria_to_str, &str_to_stop_criteria)
        .def_readwrite("temperature", &GenerationConfig::temperature)
        .def_readwrite("top_p", &GenerationConfig::top_p)
        .def_readwrite("top_k", &GenerationConfig::top_k)
        .def_readwrite("do_sample", &GenerationConfig::do_sample)
        .def_readwrite("repetition_penalty", &GenerationConfig::repetition_penalty)
        .def_readwrite("pad_token_id", &GenerationConfig::pad_token_id)
        .def_readwrite("bos_token_id", &GenerationConfig::bos_token_id)
        .def_readwrite("eos_token_id", &GenerationConfig::eos_token_id)
        .def_readwrite("eos_token", &GenerationConfig::eos_token)
        .def_readwrite("bos_token", &GenerationConfig::bos_token);

    py::class_<DecodedResults>(m, "DecodedResults")
        .def(py::init<>())
        .def_readwrite("texts", &DecodedResults::texts)
        .def_readwrite("scores", &DecodedResults::scores);

    py::class_<EncodedResults>(m, "EncodedResults")
        .def(py::init<>())
        .def_readwrite("tokens", &EncodedResults::tokens)
        .def_readwrite("scores", &EncodedResults::scores);

}
