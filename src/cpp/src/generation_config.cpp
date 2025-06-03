// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <limits>

#include <nlohmann/json.hpp>
#include <openvino/runtime/core.hpp>
#include "openvino/genai/generation_config.hpp"
#include "json_utils.hpp"
#include "utils.hpp"


namespace ov {
namespace genai {

ov::Property<size_t> rng_seed{"rng_seed"};

GenerationConfig::GenerationConfig(const std::filesystem::path& json_path) {
    using utils::read_json_param;

    std::ifstream f(json_path);
    OPENVINO_ASSERT(f.is_open(), "Failed to open '", json_path, "' with generation config");

    nlohmann::json data = nlohmann::json::parse(f);

    read_json_param(data, "eos_token_id", eos_token_id);
    read_json_param(data, "max_new_tokens", max_new_tokens);
    read_json_param(data, "max_length", max_length);
    // note that ignore_eos is not present in HF GenerationConfig
    read_json_param(data, "ignore_eos", ignore_eos);
    read_json_param(data, "min_new_tokens", min_new_tokens);
    read_json_param(data, "stop_strings", stop_strings);
    // note that include_stop_str_in_output is not present in HF GenerationConfig
    read_json_param(data, "include_stop_str_in_output", include_stop_str_in_output);
    // note that stop_token_ids is not present in HF GenerationConfig, but some generation_config.json define
    // multiple eos_token_id (e.g. https://huggingface.co/OpenGVLab/InternVL2-4B/blob/main/generation_config.json)
    // so, we need to read them as 'stop_token_ids'
    std::vector<int64_t> ordered_stop_token_ids;
    read_json_param(data, "eos_token_id", ordered_stop_token_ids);

    if (!ordered_stop_token_ids.empty()) {
        for (int64_t stop_token_id : ordered_stop_token_ids)
            stop_token_ids.insert(stop_token_id);

        if (eos_token_id == -1) {
            eos_token_id = ordered_stop_token_ids[0];
        }
    }

    // note that echo is not present in HF GenerationConfig
    read_json_param(data, "echo", echo);
    // note that logprobs is not present in HF GenerationConfig
    read_json_param(data, "logprobs", logprobs);

    // penalties
    read_json_param(data, "repetition_penalty", repetition_penalty);
    // note that frequency_penalty is not present in HF GenerationConfig
    read_json_param(data, "frequency_penalty", frequency_penalty);
    // note that presence_penalty is not present in HF GenerationConfig
    read_json_param(data, "presence_penalty", presence_penalty);

    // beam search
    read_json_param(data, "num_beam_groups", num_beam_groups);
    read_json_param(data, "num_beams", num_beams);
    read_json_param(data, "diversity_penalty", diversity_penalty);
    read_json_param(data, "length_penalty", length_penalty);
    read_json_param(data, "num_return_sequences", num_return_sequences);
    read_json_param(data, "no_repeat_ngram_size", no_repeat_ngram_size);

    if (data.contains("early_stopping")) {
        auto field_type = data["early_stopping"].type();
        if (field_type == nlohmann::json::value_t::string && data["early_stopping"] == "never") {
            stop_criteria = StopCriteria::NEVER;
        } else if (field_type == nlohmann::json::value_t::boolean && data["early_stopping"] == true) {
            stop_criteria = StopCriteria::EARLY;
        } else if (field_type == nlohmann::json::value_t::boolean && data["early_stopping"] == false) {
            stop_criteria = StopCriteria::HEURISTIC;
        }
    }

    // multinomial
    read_json_param(data, "do_sample", do_sample);
    read_json_param(data, "temperature", temperature);
    read_json_param(data, "top_p", top_p);
    read_json_param(data, "top_k", top_k);

    // assistant generation
    read_json_param(data, "assistant_confidence_threshold", assistant_confidence_threshold);
    read_json_param(data, "num_assistant_tokens", num_assistant_tokens);
    read_json_param(data, "max_ngram_size", max_ngram_size);

    // append EOS to stop_token_ids
    if (eos_token_id != -1)
        set_eos_token_id(eos_token_id);
}

void GenerationConfig::set_eos_token_id(size_t tokenizer_eos_token_id) {
    eos_token_id = tokenizer_eos_token_id;
    stop_token_ids.insert(eos_token_id);
}

void GenerationConfig::update_generation_config(const ov::AnyMap& properties) {
    using utils::read_anymap_param;

    // stop conditions
    read_anymap_param(properties, "eos_token_id", eos_token_id);
    read_anymap_param(properties, "max_new_tokens", max_new_tokens);
    read_anymap_param(properties, "max_length", max_length);
    read_anymap_param(properties, "ignore_eos", ignore_eos);
    read_anymap_param(properties, "min_new_tokens", min_new_tokens);
    read_anymap_param(properties, "stop_strings", stop_strings);
    read_anymap_param(properties, "include_stop_str_in_output", include_stop_str_in_output);
    read_anymap_param(properties, "stop_token_ids", stop_token_ids);
    if (eos_token_id > 0) {
        set_eos_token_id(eos_token_id);
    }

    // generic
    read_anymap_param(properties, "echo", echo);
    read_anymap_param(properties, "logprobs", logprobs);
    read_anymap_param(properties, "num_return_sequences", num_return_sequences);
    read_anymap_param(properties, "adapters", adapters);
    read_anymap_param(properties, "apply_chat_template", apply_chat_template);

    // penalties
    read_anymap_param(properties, "frequency_penalty", frequency_penalty);
    read_anymap_param(properties, "presence_penalty", presence_penalty);
    read_anymap_param(properties, "repetition_penalty", repetition_penalty);

    // beam search
    read_anymap_param(properties, "num_beam_groups", num_beam_groups);
    read_anymap_param(properties, "num_beams", num_beams);
    read_anymap_param(properties, "diversity_penalty", diversity_penalty);
    read_anymap_param(properties, "length_penalty", length_penalty);
    read_anymap_param(properties, "stop_criteria", stop_criteria);
    read_anymap_param(properties, "no_repeat_ngram_size", no_repeat_ngram_size);

    // multinomial
    read_anymap_param(properties, "do_sample", do_sample);
    read_anymap_param(properties, "temperature", temperature);
    read_anymap_param(properties, "top_p", top_p);
    read_anymap_param(properties, "top_k", top_k);
    // TODO: add support of 'generator' property similar to Image generation
    read_anymap_param(properties, "rng_seed", rng_seed);

    // assistant generation
    read_anymap_param(properties, "assistant_confidence_threshold", assistant_confidence_threshold);
    read_anymap_param(properties, "num_assistant_tokens", num_assistant_tokens);
    read_anymap_param(properties, "max_ngram_size", max_ngram_size);
}

size_t GenerationConfig::get_max_new_tokens(size_t prompt_length) const {
    // max_new_tokens has priority over max_length, only if max_new_tokens was not specified use max_length
    if (max_new_tokens != SIZE_MAX) {
        return max_new_tokens;
    } else {
        return max_length - prompt_length;
    }
}

bool GenerationConfig::is_greedy_decoding() const {
    return !do_sample && !is_beam_search();
}

bool GenerationConfig::is_beam_search() const {
    return num_beams > 1;
}

bool GenerationConfig::is_multinomial() const {
    return do_sample;
}

bool GenerationConfig::is_speculative_decoding() const {
    return is_assisting_generation();
}

bool GenerationConfig::is_assisting_generation() const {
    return assistant_confidence_threshold > 0 || num_assistant_tokens > 0;
}

bool GenerationConfig::is_structured_output_generation() const {
    return structured_output_config.has_value();
}

bool GenerationConfig::is_prompt_lookup() const {
    return max_ngram_size > 0 && num_assistant_tokens > 0;
}

void GenerationConfig::validate() const {
    OPENVINO_ASSERT(num_return_sequences > 0, "num_return_sequences must be greater than 0");

    // Stop conditions

    OPENVINO_ASSERT(eos_token_id == -1 || stop_token_ids.find(eos_token_id) != stop_token_ids.end(),
        "'stop_token_ids' must contain 'eos_token_id'. Please, call 'set_eos_token_id' with 'eos_token_id' value");

    auto stop_token_ids_it = std::find_if(stop_token_ids.begin(), stop_token_ids.end(), [] (int64_t stop_token_id) -> bool {
        return stop_token_id < 0;
    });
    OPENVINO_ASSERT(stop_token_ids_it == stop_token_ids.end(), "'stop_token_ids' must be non-negative, but it contains a value ", *stop_token_ids_it);

    OPENVINO_ASSERT(!ignore_eos || max_new_tokens != SIZE_MAX || max_length != SIZE_MAX,
                    "ignore_eos is true, in this case either 'max_new_tokens', or 'max_length' should be defined.");

    OPENVINO_ASSERT(eos_token_id != -1 || !stop_token_ids.empty() || !stop_strings.empty() || max_new_tokens != SIZE_MAX || max_length != SIZE_MAX,
                    "Either 'eos_token_id', or 'stop_token_ids', or 'stop_strings', or 'max_new_tokens', or 'max_length' should be defined.");

    OPENVINO_ASSERT(max_new_tokens > 0 || (max_new_tokens == 0 && echo), "'max_new_tokens' must be greater than 0, if `echo` is set, 0 is also accepted");
    OPENVINO_ASSERT(min_new_tokens <= max_new_tokens, "min_new_tokens must be less or equal max_new_tokens");

    // Sampling strategies

    OPENVINO_ASSERT(num_return_sequences == 1 || (is_multinomial() || is_beam_search()), 
        "'num_return_sequences' can be more than 1 only in case of beam search or multinomial sampling, but got ", num_return_sequences);

    // generic penalties, but not supported by beam search currently
    if (!is_beam_search()) {
        OPENVINO_ASSERT(frequency_penalty >= -2.0f && frequency_penalty <= 2.0f, "'frequence_penalty' penalty must be within [-2.0; 2.0], but got ", frequency_penalty);
        OPENVINO_ASSERT(presence_penalty >= -2.0f && presence_penalty <= 2.0f, "'presence_penalty' penalty must be within [-2.0; 2.0], but got ", presence_penalty);
        OPENVINO_ASSERT(repetition_penalty > 0.0f, "'repetition_penalty' must be a strictly positive float, but got ", repetition_penalty);
    } else {
        OPENVINO_ASSERT(frequency_penalty == 0.0f, "'frequency_penalty' is not currently supported by beam search and should be 0.0f, but got ", frequency_penalty);
        OPENVINO_ASSERT(presence_penalty == 0.0f, "'presence_penalty' is not currently supported by beam search and should be 0.0f, but got ", presence_penalty);
        OPENVINO_ASSERT(repetition_penalty == 1.0f, "'repetition_penalty' is not currently supported by beam search and should be 1.0f, but got ", repetition_penalty);
    }

    if (is_multinomial()) {
        OPENVINO_ASSERT(top_p > 0 && top_p <= 1.0f, "When 'do_sample' is true, top_p must be a positive float > 0.0 and <= 1.0, but got ", top_p);
        OPENVINO_ASSERT(temperature > 0, "When 'do_sample' is true, temperature must be a strictly positive float, but got ", temperature);
    } else {
        // parameters requiring multinomial
        // OPENVINO_ASSERT(top_k == std::numeric_limits<size_t>::max(), "When 'do_sample' is false, top_k must be max of size_t, but got ", top_k);
        // OPENVINO_ASSERT(top_p == 1.0f, "When 'do_sample' is false, top_p must be 1.0f, but got ", top_p);
        // OPENVINO_ASSERT(temperature == 1.0f, "When 'do_sample' is false, temperature must be a 1.0f, but got ", temperature);
    }

    if (is_beam_search()) {
        OPENVINO_ASSERT(num_beams % num_beam_groups == 0, "'num_beams' (", num_beams, ") should be divisible by 'num_beam_groups' (", num_beam_groups, ")");
        OPENVINO_ASSERT(num_beams >= num_return_sequences, "'num_beams' (", num_beams, ") must be greater equal than 'num_return_sequences' (", num_return_sequences, ")");

        OPENVINO_ASSERT(!do_sample,
                        "Beam search with sampling is not supported yet. "
                        "Please either set do_sample=false to use beam search "
                        "or set num_beams=1 if you with to use multinomial sampling.");

        OPENVINO_ASSERT(no_repeat_ngram_size > 0, "'no_repeat_ngram_size' must be positive");
        if (num_beam_groups > 1) {
            OPENVINO_ASSERT(diversity_penalty != 0.0f, "For grouped beam search 'diversity_penalty' should not be zero, otherwise it fallbacks to non-grouped beam search");
        } else {
            OPENVINO_ASSERT(diversity_penalty == 0.0f, "For beam search 'diversity_penalty' is applicable only when grouped beam search is used, but got 'num_beam_groups' == 1");
        }
    } else {
        // parameters requiring beam search
        // OPENVINO_ASSERT(num_beam_groups == 1, "'num_beam_groups' is supported by beam search only and should be 1 otherwise, but got ", num_beam_groups);
        // OPENVINO_ASSERT(no_repeat_ngram_size == std::numeric_limits<size_t>::max(), "'no_repeat_ngram_size' is supported only by beam search, otherwise should be set to max of size_t, but got ", no_repeat_ngram_size);
        // OPENVINO_ASSERT(diversity_penalty == 0.0f, "'diversity_penalty' is set to ", diversity_penalty, " (default is 0.0f), which is supported only by beam search sampling");
        // OPENVINO_ASSERT(length_penalty == 1.0f, "'length_penalty' is set to ", length_penalty, " (default is 1.0f), which is supported only by beam search sampling");
    }

    // assistant generation

    if (is_assisting_generation()) {
        OPENVINO_ASSERT(!is_beam_search() && num_return_sequences == 1, "Beam search and parallel sampling are not compatible with assistant generation");
        OPENVINO_ASSERT(assistant_confidence_threshold == 0.0f || num_assistant_tokens == 0, "Parameters `assistant_confidence_threshold` and `num_assistant_tokens` are mutually exclusive in `GenerationConfig`");
    }

    if (num_assistant_tokens == 0) {
        OPENVINO_ASSERT(max_ngram_size == 0, "'max_ngram_size' should be set to default value 0 when prompt lookup is disabled");
    }

    if(is_structured_output_generation()) {
        #ifndef ENABLE_XGRAMMAR
            OPENVINO_THROW("Structured output is not supported in this build. Please, recompile with -DENABLE_XGRAMMAR=ON");
        #else
           (*structured_output_config).validate();
        #endif
    }
}

void StructuredOutputConfig::validate() const {
    OPENVINO_ASSERT(
        (json_schema.has_value() + regex.has_value() + choices.has_value() + grammar.has_value()) == 1,
        "Only one of json, regex, choices or grammar shoud be set in StructuredOutputConfig, but got: ",
        (json_schema.has_value() ? "json=" + *json_schema +", " : ""),
        (regex.has_value() ? "regex=" + *regex + ", " : ""),
        (choices.has_value() ? "choices, " : ""),
        (grammar.has_value() ? "grammar=" + *grammar : "")
    );
}

GenerationConfig beam_search() {
    GenerationConfig beam_search_config;
    beam_search_config.num_beams = 4;
    beam_search_config.num_return_sequences = 3;
    beam_search_config.num_beam_groups = 2;
    beam_search_config.max_new_tokens = 100;
    beam_search_config.diversity_penalty = 2.0f;
    return beam_search_config;
}

GenerationConfig greedy() {
    GenerationConfig greedy_config;
    greedy_config.max_new_tokens = 30;
    return greedy_config;
}

GenerationConfig multinomial() {
    GenerationConfig multinomial_config;
    multinomial_config.do_sample = true;
    multinomial_config.temperature = 0.9f;
    multinomial_config.top_p = 0.9f;
    multinomial_config.top_k = 20;
    multinomial_config.num_return_sequences = 3;
    multinomial_config.presence_penalty = 0.01f;
    multinomial_config.frequency_penalty = 0.1f;
    multinomial_config.min_new_tokens = 15;
    multinomial_config.max_new_tokens = 30;
    return multinomial_config;
}

}  // namespace genai
}  // namespace ov
