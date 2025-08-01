// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <limits>
#include <variant>
#include <string>

#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/scheduler_config.hpp"
#include "openvino/genai/lora_adapter.hpp"

namespace ov {
namespace genai {

/**
 * @brief controls the stopping condition for grouped beam search. The following values are possible:
 *        "EARLY" stops as soon as there are `num_beams` complete candidates.
          "HEURISTIC" stops when is it unlikely to find better candidates.
          "NEVER" stops when there cannot be better candidates.
 */
enum class StopCriteria { EARLY, HEURISTIC, NEVER };


/**
 * @brief StructuralTagItem is used to define a structural tag with its properties.
 * @param begin the string that marks the beginning of the structural tag.
 * @param schema JSON schema that defines the structure of the tag.
 * @param end the string that marks the end of the structural tag.
 */
struct OPENVINO_GENAI_EXPORTS StructuralTagItem {
    StructuralTagItem() = default;
    StructuralTagItem(const ov::AnyMap& properties);
    void update_config(const ov::AnyMap& properties);
    std::string to_string() const;

    std::string begin;
    std::string schema;
    std::string end;
};

/**
 * @brief Configures structured output generation by combining regular sampling with structural tags.
 *
 * When the model generates a trigger string, it switches to structured output mode and produces output
 * based on the defined structural tags. Afterward, regular sampling resumes.
 *
 * Example:
 *   - Trigger "<func=" activates tags with begin "<func=sum>" or "<func=multiply>".
 *
 * Note:
 *   - Simple triggers like "<" may activate structured output unexpectedly if present in regular text.
 *   - Very specific or long triggers may be difficult for the model to generate, so structured output may not be triggered.
 *
 * @param structural_tags List of StructuralTagItem objects defining structural tags.
 * @param triggers List of strings that trigger structured output generation. Triggers may match the beginning or part of a tag's begin string.
 */
struct OPENVINO_GENAI_EXPORTS StructuralTagsConfig {
public:
    StructuralTagsConfig() = default;
    StructuralTagsConfig(const ov::AnyMap& properties);
    void update_config(const ov::AnyMap& properties);
    std::string to_string() const;

    std::vector<StructuralTagItem> structural_tags;
    std::vector<std::string> triggers;
};


/* 
* Structured output parameters:
* @param json_schema if set, the output will be a JSON string constrained by the specified json_schema.
* @param regex if set, the output will be constrained by specified regex.
* @param grammar if set, the output will be constrained by specified EBNF grammar.
* @param structural_tags_config if set, the output could contain substrings constrained by the specified structural tags.
* @param backend if set, the structured output generation will use specified backend, currently only "xgrammar" is supported.
* 
* If several parameters are set, e.g. json_schema and regex, then an error will be thrown when validating the configuration.
*/
class OPENVINO_GENAI_EXPORTS StructuredOutputConfig {
public:
    /* 
    * @brief Constructor that initializes the structured output configuration with properties.
    * @param properties A map of properties to initialize the structured output configuration.
    * 
    * Example: StructuredOutputConfig config({{ov::genai::json_schema(json_schema_str)}});
    */
    StructuredOutputConfig(const ov::AnyMap& properties);
    StructuredOutputConfig() = default;
    std::optional<std::string> json_schema;
    std::optional<std::string> regex;
    std::optional<std::string> grammar;
    std::optional<StructuralTagsConfig> structural_tags_config;
    std::optional<std::string> backend;
    void validate() const;
    void update_config(const ov::AnyMap& properties);
};

/**
 * @brief Structure to keep generation config parameters. For a selected method of decoding, only parameters from that group
 * and generic parameters are used. For example, if do_sample is set to true, then only generic parameters and random sampling parameters will
 * be used while greedy and beam search parameters will not affect decoding at all.
 *
 * Generic parameters:
 * @param max_length the maximum length the generated tokens can have. Corresponds to the length of the input prompt +
 *        `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
 * @param max_new_tokens the maximum numbers of tokens to generate, excluding the number of tokens in the prompt. max_new_tokens has priority over max_length.
 * @param ignore_eos if set to true, then generation will not stop even if <eos> token is met.
 * @param eos_token_id token_id of <eos> (end of sentence)
 * @param min_new_tokens set 0 probability for eos_token_id for the first eos_token_id generated tokens.
 *
 * @param stop_strings A set of strings that will cause pipeline to stop generating further tokens.
 * @param include_stop_str_in_output if set to true stop string that matched generation will be included in generation output (default: false)
 * @param stop_token_ids A set of tokens that will cause pipeline to stop generating further tokens.
 * @param echo if set to true, output will include user prompt (default: false).
 * @param logprobs number of top logprobs computed for each position, if set to 0, logprobs are not computed and value 0.0 is returned.
 *                 Currently only single top logprob can be returned, so any logprobs > 1 is treated as logprobs == 1. (default: 0).
 *
 * @param repetition_penalty the parameter for repetition penalty. 1.0 means no penalty.
 * @param presence_penalty reduces absolute log prob if the token was generated at least once.
 * @param frequency_penalty reduces absolute log prob as many times as the token was generated.
 *
 * Beam search specific parameters:
 * @param num_beams number of beams for beam search. 1 disables beam search.
 * @param num_beam_groups number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams.
 * @param diversity_penalty this value is subtracted from a beam's score if it generates the same token as any beam from other group at a
 *        particular time. See https://arxiv.org/pdf/1909.05858.
 * @param length_penalty exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to
 *        the sequence length, which in turn is used to divide the score of the sequence. Since the score is the log
 *        likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer sequences, while
 *        `length_penalty` < 0.0 encourages shorter sequences.
 * @param num_return_sequences the number of sequences to return for grouped beam search decoding per batch element. num_return_sequences must be less or equal to num_beams.
 * @param no_repeat_ngram_size if set to int > 0, all ngrams of that size can only occur once.
 * @param stop_criteria controls the stopping condition for grouped beam search. It accepts the following values:
 *        "EARLY", where the generation stops as soon as there are `num_beams` complete candidates; "HEURISTIC", where an
 *        "HEURISTIC" is applied and the generation stops when is it very unlikely to find better candidates;
 *        "NEVER", where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm).
 *
 * Random (or multinomial) sampling parameters:
 * @param do_sample whether or not to use multinomial random sampling that add up to `top_p` or higher are kept.
 * @param temperature the value used to modulate token probabilities for random sampling.
 * @param top_p - if set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
 * @param top_k the number of highest probability vocabulary tokens to keep for top-k-filtering.
 * @param rng_seed initializes random generator.
 *
 * Assisting generation parameters:
 * @param assistant_confidence_threshold the lower token probability of candidate to be validated by main model in case of dynamic strategy candidates number update.
 * @param num_assistant_tokens the defined candidates number to be generated by draft model/prompt lookup in case of static strategy candidates number update.
 * @param max_ngram_size is maximum ngram to use when looking for matches in the prompt.
 *
 * @param structured_output_config if set, the output will be a string constrained by the specified json_schema, regex, or EBNF grammar.
 * 
 * @param apply_chat_template whether or not to apply chat_template for non-chat scenarios
 */

class OPENVINO_GENAI_EXPORTS GenerationConfig {

public:
    GenerationConfig() = default;
    explicit GenerationConfig(const std::filesystem::path& json_path);

    // Generic
    size_t max_new_tokens = SIZE_MAX;
    size_t max_length = SIZE_MAX;
    bool ignore_eos = false;
    size_t min_new_tokens = 0;
    bool echo = false;
    size_t logprobs = 0;

    // EOS special token
    int64_t eos_token_id = -1;
    std::set<std::string> stop_strings;
    // Default setting in vLLM (and OpenAI API) is not to include stop string in the output
    bool include_stop_str_in_output = false;
    std::set<int64_t> stop_token_ids;

    // penalties (not used in beam search)
    float repetition_penalty = 1.0f;
    float presence_penalty = 0.0;
    float frequency_penalty = 0.0f;

    // Beam search specific
    size_t num_beam_groups = 1;
    size_t num_beams = 1;
    float diversity_penalty = 0.0f;
    float length_penalty = 1.0f;
    size_t num_return_sequences = 1;
    size_t no_repeat_ngram_size = std::numeric_limits<size_t>::max();
    StopCriteria stop_criteria = StopCriteria::HEURISTIC;

    // Multinomial
    float temperature = 1.0f;
    float top_p = 1.0f;
    size_t top_k = std::numeric_limits<size_t>::max();
    bool do_sample = false;
    size_t rng_seed = 0;

    // Assisting generation parameters
    float assistant_confidence_threshold = 0.f;
    size_t num_assistant_tokens = 0;
    size_t max_ngram_size = 0;

    // Structured output parameters
    std::optional<StructuredOutputConfig> structured_output_config;

    std::optional<AdapterConfig> adapters;

    // set to true if chat template should be applied for non-chat scenarios, set to false otherwise
    bool apply_chat_template = true;


    /** @brief sets eos_token_id to tokenizer_eos_token_id if eos_token_id is less than 0.
     * Otherwise verifies eos_token_id == tokenizer_eos_token_id.
     */
    void set_eos_token_id(size_t tokenizer_eos_token_id);
    size_t get_max_new_tokens(size_t prompt_length = 0) const;

    bool is_greedy_decoding() const;
    bool is_beam_search() const;
    bool is_multinomial() const;
    bool is_assisting_generation() const;
    bool is_prompt_lookup() const;
    bool is_structured_output_generation() const;

    OPENVINO_DEPRECATED("Please, use `is_assisting_generation()` instead of `is_speculative_decoding()`. This method will be removed in 2026.0.0 release")
    bool is_speculative_decoding() const;

    void update_generation_config(const ov::AnyMap& properties);

    template <typename... Properties>
    util::EnableIfAllStringAny<void, Properties...> update_generation_config(Properties&&... properties) {
        return update_generation_config(AnyMap{std::forward<Properties>(properties)...});
    }

    /// @brief checks that are no conflicting parameters, e.g. do_sample=true and num_beams > 1.
    /// @throws Exception if config is invalid.
    void validate() const;
};

/*
 * utils that allow to use generate and operator() in the following way:
 * pipe.generate(input_ids, ov::genai::max_new_tokens(200), ov::genai::temperature(1.0f),...)
 * pipe(text, ov::genai::max_new_tokens(200), ov::genai::temperature(1.0f),...)
*/
static constexpr ov::Property<size_t> max_new_tokens{"max_new_tokens"};
static constexpr ov::Property<size_t> max_length{"max_length"};
static constexpr ov::Property<bool> ignore_eos{"ignore_eos"};
static constexpr ov::Property<size_t> min_new_tokens{"min_new_tokens"};
static constexpr ov::Property<std::set<std::string>> stop_strings{"stop_strings"};
static constexpr ov::Property<bool> include_stop_str_in_output{"include_stop_str_in_output"};
static constexpr ov::Property<std::set<int64_t>> stop_token_ids{"stop_token_ids"};

static constexpr ov::Property<size_t> num_beam_groups{"num_beam_groups"};
static constexpr ov::Property<size_t> num_beams{"num_beams"};
static constexpr ov::Property<float> diversity_penalty{"diversity_penalty"};
static constexpr ov::Property<float> length_penalty{"length_penalty"};
static constexpr ov::Property<size_t> num_return_sequences{"num_return_sequences"};
static constexpr ov::Property<size_t> no_repeat_ngram_size{"no_repeat_ngram_size"};
static constexpr ov::Property<StopCriteria> stop_criteria{"stop_criteria"};

static constexpr ov::Property<float> temperature{"temperature"};
static constexpr ov::Property<float> top_p{"top_p"};
static constexpr ov::Property<size_t> top_k{"top_k"};
static constexpr ov::Property<bool> do_sample{"do_sample"};
static constexpr ov::Property<float> repetition_penalty{"repetition_penalty"};
static constexpr ov::Property<int64_t> eos_token_id{"eos_token_id"};
static constexpr ov::Property<float> presence_penalty{"presence_penalty"};
static constexpr ov::Property<float> frequency_penalty{"frequency_penalty"};
extern OPENVINO_GENAI_EXPORTS ov::Property<size_t> rng_seed;

static constexpr ov::Property<float> assistant_confidence_threshold{"assistant_confidence_threshold"};
static constexpr ov::Property<size_t> num_assistant_tokens{"num_assistant_tokens"};
static constexpr ov::Property<size_t> max_ngram_size{"max_ngram_size"};

static constexpr ov::Property<StructuredOutputConfig> structured_output_config{"structured_output_config"};
static constexpr ov::Property<std::string> regex{"regex"};
static constexpr ov::Property<std::string> json_schema{"json_schema"};
static constexpr ov::Property<std::string> grammar{"grammar"};
static constexpr ov::Property<std::string> backend{"backend"};

static constexpr ov::Property<bool> apply_chat_template{"apply_chat_template"};

// Predefined Configs

OPENVINO_DEPRECATED("Please, use individual parameters instead of predefined configs. This method will be removed in 2026.0.0 release")
OPENVINO_GENAI_EXPORTS GenerationConfig beam_search();
OPENVINO_DEPRECATED("Please, use individual parameters instead of predefined configs. This method will be removed in 2026.0.0 release")
OPENVINO_GENAI_EXPORTS GenerationConfig greedy();
OPENVINO_DEPRECATED("Please, use individual parameters instead of predefined configs. This method will be removed in 2026.0.0 release")
OPENVINO_GENAI_EXPORTS GenerationConfig multinomial();

}  // namespace genai
}  // namespace ov
