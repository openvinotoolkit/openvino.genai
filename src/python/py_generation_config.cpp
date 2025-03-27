// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <filesystem>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl/filesystem.h>
#include <pybind11/functional.h>

#include "py_utils.hpp"

namespace py = pybind11;
namespace pyutils = ov::genai::pybind::utils;

using ov::genai::StopCriteria;
using ov::genai::GenerationConfig;

namespace {

auto stop_criteria_docstring =  R"(
    StopCriteria controls the stopping condition for grouped beam search.

    The following values are possible:
        "openvino_genai.StopCriteria.EARLY" stops as soon as there are `num_beams` complete candidates.
        "openvino_genai.StopCriteria.HEURISTIC" stops when is it unlikely to find better candidates.
        "openvino_genai.StopCriteria.NEVER" stops when there cannot be better candidates.
)";

} // namespace

char generation_config_docstring[] = R"(
    Structure to keep generation config parameters. For a selected method of decoding, only parameters from that group
    and generic parameters are used. For example, if do_sample is set to true, then only generic parameters and random sampling parameters will
    be used while greedy and beam search parameters will not affect decoding at all.

    Parameters:
    max_length:    the maximum length the generated tokens can have. Corresponds to the length of the input prompt +
                   max_new_tokens. Its effect is overridden by `max_new_tokens`, if also set.
    max_new_tokens: the maximum numbers of tokens to generate, excluding the number of tokens in the prompt. max_new_tokens has priority over max_length.
    min_new_tokens: set 0 probability for eos_token_id for the first eos_token_id generated tokens.
    ignore_eos:    if set to true, then generation will not stop even if <eos> token is met.
    eos_token_id:  token_id of <eos> (end of sentence)
    stop_strings: a set of strings that will cause pipeline to stop generating further tokens.
    include_stop_str_in_output: if set to true stop string that matched generation will be included in generation output (default: false)
    stop_token_ids: a set of tokens that will cause pipeline to stop generating further tokens.
    echo:           if set to true, the model will echo the prompt in the output.
    logprobs:       number of top logprobs computed for each position, if set to 0, logprobs are not computed and value 0.0 is returned.
                    Currently only single top logprob can be returned, so any logprobs > 1 is treated as logprobs == 1. (default: 0).
    apply_chat_template: whether to apply chat_template for non-chat scenarios

    repetition_penalty: the parameter for repetition penalty. 1.0 means no penalty.
    presence_penalty: reduces absolute log prob if the token was generated at least once.
    frequency_penalty: reduces absolute log prob as many times as the token was generated.

    Beam search specific parameters:
    num_beams:         number of beams for beam search. 1 disables beam search.
    num_beam_groups:   number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
    diversity_penalty: value is subtracted from a beam's score if it generates the same token as any beam from other group at a particular time.
    length_penalty:    exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
        the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
        likelihood of the sequence (i.e. negative), length_penalty > 0.0 promotes longer sequences, while
        length_penalty < 0.0 encourages shorter sequences.
    num_return_sequences: the number of sequences to return for grouped beam search decoding.
    no_repeat_ngram_size: if set to int > 0, all ngrams of that size can only occur once.
    stop_criteria:        controls the stopping condition for grouped beam search. It accepts the following values:
        "openvino_genai.StopCriteria.EARLY", where the generation stops as soon as there are `num_beams` complete candidates;
        "openvino_genai.StopCriteria.HEURISTIC" is applied and the generation stops when is it very unlikely to find better candidates;
        "openvino_genai.StopCriteria.NEVER", where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm).

    Random sampling parameters:
    temperature:        the value used to modulate token probabilities for random sampling.
    top_p:              if set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
    top_k:              the number of highest probability vocabulary tokens to keep for top-k-filtering.
    do_sample:          whether or not to use multinomial random sampling that add up to `top_p` or higher are kept.
    num_return_sequences: the number of sequences to generate from a single prompt.
)";

void init_generation_config(py::module_& m) {
    // Binding for StopCriteria
    py::enum_<StopCriteria>(m, "StopCriteria", stop_criteria_docstring)
        .value("EARLY", StopCriteria::EARLY)
        .value("HEURISTIC", StopCriteria::HEURISTIC)
        .value("NEVER", StopCriteria::NEVER);

     // Binding for GenerationConfig
    py::class_<GenerationConfig>(m, "GenerationConfig", generation_config_docstring)
        .def(py::init<std::filesystem::path>(), py::arg("json_path"), "path where generation_config.json is stored")
        .def(py::init([](py::kwargs kwargs) { return *pyutils::update_config_from_kwargs(GenerationConfig(), kwargs); }))
        .def_readwrite("max_new_tokens", &GenerationConfig::max_new_tokens)
        .def_readwrite("max_length", &GenerationConfig::max_length)
        .def_readwrite("ignore_eos", &GenerationConfig::ignore_eos)
        .def_readwrite("min_new_tokens", &GenerationConfig::min_new_tokens)
        .def_readwrite("num_beam_groups", &GenerationConfig::num_beam_groups)
        .def_readwrite("num_beams", &GenerationConfig::num_beams)
        .def_readwrite("diversity_penalty", &GenerationConfig::diversity_penalty)
        .def_readwrite("length_penalty", &GenerationConfig::length_penalty)
        .def_readwrite("num_return_sequences", &GenerationConfig::num_return_sequences)
        .def_readwrite("no_repeat_ngram_size", &GenerationConfig::no_repeat_ngram_size)
        .def_readwrite("stop_criteria", &GenerationConfig::stop_criteria)
        .def_readwrite("temperature", &GenerationConfig::temperature)
        .def_readwrite("top_p", &GenerationConfig::top_p)
        .def_readwrite("top_k", &GenerationConfig::top_k)
        .def_readwrite("do_sample", &GenerationConfig::do_sample)
        .def_readwrite("repetition_penalty", &GenerationConfig::repetition_penalty)
        .def_readwrite("eos_token_id", &GenerationConfig::eos_token_id)
        .def_readwrite("presence_penalty", &GenerationConfig::presence_penalty)
        .def_readwrite("frequency_penalty", &GenerationConfig::frequency_penalty)
        .def_readwrite("rng_seed", &GenerationConfig::rng_seed)
        .def_readwrite("stop_strings", &GenerationConfig::stop_strings)
        .def_readwrite("echo", &GenerationConfig::echo)
        .def_readwrite("logprobs", &GenerationConfig::logprobs)
        .def_readwrite("assistant_confidence_threshold", &GenerationConfig::assistant_confidence_threshold)
        .def_readwrite("num_assistant_tokens", &GenerationConfig::num_assistant_tokens)
        .def_readwrite("max_ngram_size", &GenerationConfig::max_ngram_size)
        .def_readwrite("include_stop_str_in_output", &GenerationConfig::include_stop_str_in_output)
        .def_readwrite("stop_token_ids", &GenerationConfig::stop_token_ids)
        .def_readwrite("adapters", &GenerationConfig::adapters)
        .def_readwrite("apply_chat_template", &GenerationConfig::apply_chat_template)
        .def("set_eos_token_id", &GenerationConfig::set_eos_token_id, py::arg("tokenizer_eos_token_id"))
        .def("is_beam_search", &GenerationConfig::is_beam_search)
        .def("is_greedy_decoding", &GenerationConfig::is_greedy_decoding)
        .def("is_multinomial", &GenerationConfig::is_multinomial)
        .def("is_assisting_generation", &GenerationConfig::is_assisting_generation)
        .def("is_prompt_lookup", &GenerationConfig::is_prompt_lookup)
        .def("validate", &GenerationConfig::validate)
        .def("update_generation_config", [](
            ov::genai::GenerationConfig& config,
            const py::kwargs& kwargs) {
            config.update_generation_config(pyutils::kwargs_to_any_map(kwargs));
        });
   }
