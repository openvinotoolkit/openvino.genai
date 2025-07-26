// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief This is a header file for OpenVINO GenAI C API, which is a C wrapper for  ov::genai::GenerationConfig class.
 *
 * @file generation_config_c.h
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "openvino/c/ov_common.h"
#include "openvino/genai/c/visibility.h"

/**
 * @brief controls the stopping condition for grouped beam search. The following values are possible:
 *        "EARLY" stops as soon as there are `num_beams` complete candidates.
          "HEURISTIC" stops when is it unlikely to find better candidates.
          "NEVER" stops when there cannot be better candidates.
 */
typedef enum { EARLY, HEURISTIC, NEVER } StopCriteria;

/**
 * @struct ov_genai_generation_config
 * @brief type define ov_genai_generation_config from ov_genai_generation_config_opaque
 */
typedef struct ov_genai_generation_config_opaque ov_genai_generation_config;

/**
 * @brief Create ov_genai_generation_config.
 * @param config A pointer to the newly created ov_genai_generation_config.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_generation_config_create(ov_genai_generation_config** config);

/**
 * @brief Create ov_genai_generation_config from JSON file.
 * @param json_path Path to a .json file containing the generation configuration to load.
 * @param config A pointer to the newly created ov_genai_generation_config.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_generation_config_create_from_json(const char* json_path,
                                                                                 ov_genai_generation_config** config);

/**
 * @brief Release the memory allocated by ov_genai_generation_config.
 * @param handle A pointer to the ov_genai_generation_config to free memory.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS void ov_genai_generation_config_free(ov_genai_generation_config* handle);

/**
 * @brief Set the maximum number of tokens to generate, excluding the number of tokens in the prompt. max_new_tokens
 * has priority over max_length.
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @param value The maximum number of tokens to generate.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_generation_config_set_max_new_tokens(ov_genai_generation_config* handle,
                                                                                   const size_t value);

/**
 * @brief Set the maximum length the generated tokens can have. Corresponds to the length of the input prompt +
 * `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @param value The maximum length the generated tokens can have.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_generation_config_set_max_length(ov_genai_generation_config* config,
                                                                               const size_t value);

/**
 * @brief Set whether or not to ignore <eos> token
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @param value If set to true, then generation will not stop even if <eos> token is met.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_generation_config_set_ignore_eos(ov_genai_generation_config* config,
                                                                               const bool value);

/**
 * @brief Set the minimum number of tokens to generate.
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @param value The minimum number of tokens to generate.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_generation_config_set_min_new_tokens(ov_genai_generation_config* config,
                                                                                   const size_t value);

/**
 * @brief Set whether or not to include user prompt in the output.
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @param value If set to true, output will include user prompt.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_generation_config_set_echo(ov_genai_generation_config* config,
                                                                         const bool value);

/**
 * @brief Set the number of top logprobs computed for each position,
          if set to 0, logprobs are not computed and value 0.0 is returned.
          Currently only single top logprob can be returned, so any logprobs > 1 is treated as logprobs
 == 1.(default: 0).
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @param value The number of top logprobs computed for each position.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_generation_config_set_logprobs(ov_genai_generation_config* config,
                                                                             const size_t value);

/**
 * @brief Set the set of strings that will cause pipeline to stop generating further tokens.
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @param strings An array of strings.
 * @param count The number of strings in the array.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_generation_config_set_stop_strings(ov_genai_generation_config* config,
                                                                                 const char** strings,
                                                                                 const size_t count);
/**
 * @brief Set whether or not to include stop string that matched generation in the output.
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @param value If set to true stop string that matched generation will be included in generation output (default:
 * false).
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_generation_config_set_include_stop_str_in_output(ov_genai_generation_config* config, const bool value);

/**
 * @brief Set the set of tokens that will cause pipeline to stop generating further tokens.
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @param token_ids An array of token ids.
 * @param token_ids_num The number of token ids in the array.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_generation_config_set_stop_token_ids(ov_genai_generation_config* config,
                                                                                   const int64_t* token_ids,
                                                                                   const size_t token_ids_num);

/**
 * @brief Set the number of groups to divide `num_beams` into in order to ensure diversity among different groups of
 * beams.
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @param value The number of beam groups.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_generation_config_set_num_beam_groups(ov_genai_generation_config* config,
                                                                                    const size_t value);

/**
 * @brief Set the number of beams for beam search. 1 disables beam search.
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @param value The number of beams for beam search.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_generation_config_set_num_beams(ov_genai_generation_config* config,
                                                                              const size_t value);

/**
 * @brief Set the diversity penalty, this value is subtracted from a beam's score if it generates the same token as
 * any beam from other group at a particular time.
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @param value The parameter for diversity penalty.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_generation_config_set_diversity_penalty(ov_genai_generation_config* config, const float value);

/**
 * @brief Set the length penalty, exponential penalty to the length that is used with beam-based generation. It is
 * applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence. Since
 * the score is the log likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer
 * sequences, while `length_penalty` < 0.0 encourages shorter sequences.
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @param value The exponential penalty.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_generation_config_set_length_penalty(ov_genai_generation_config* config,
                                                                                   const float value);

/**
 * @brief Set the number of sequences to return for grouped beam search decoding per batch element.
 * num_return_sequences must be less or equal to num_beams.
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @param value The number of sequences to return.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_generation_config_set_num_return_sequences(ov_genai_generation_config* config, const size_t value);

/**
 * @brief Set the no_repeat_ngram_size
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @param value If set to int > 0, all ngrams of that size can only occur once.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_generation_config_set_no_repeat_ngram_size(ov_genai_generation_config* config, const size_t value);

/**
 * @brief Set the stopping condition for grouped beam search. It accepts the following values:
 * "EARLY", where the generation stops as soon as there are `num_beams` complete candidates; "HEURISTIC", where an
 * "HEURISTIC" is applied when it is unlikely to find better candidates;
 * "NEVER", where the generation stops when there cannot be better candidates.
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @param value The stopping condition for grouped beam search.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_generation_config_set_stop_criteria(ov_genai_generation_config* config,
                                                                                  const StopCriteria value);

/**
 * @brief Set the temperature value used to modulate token probabilities for random sampling.
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @param value The value of temperature.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_generation_config_set_temperature(ov_genai_generation_config* config,
                                                                                const float value);

/**
 * @brief Set the top_p value. If set to float < 1, only the smallest set of most probable tokens with probabilities
 * that add up to top_p or higher are kept for generation.
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @param value The value of top_p.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_generation_config_set_top_p(ov_genai_generation_config* config,
                                                                          const float value);

/**
 * @brief Set the top_k value. The number of highest probability vocabulary tokens to keep for top-k-filtering.
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @param value The value of top_k.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_generation_config_set_top_k(ov_genai_generation_config* config,
                                                                          const size_t value);

/**
 * @brief Set whether or not to use multinomial random sampling that add up to `top_p` or higher are kept.
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @param value If set to true, multinomial random sampling will be used.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_generation_config_set_do_sample(ov_genai_generation_config* config,
                                                                              const bool value);

/**
 * @brief Set the parameter for repetition penalty. 1.0 means no penalty.
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @param value The value of parameter for repetition penalty.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_generation_config_set_repetition_penalty(ov_genai_generation_config* config, const float value);

/**
 * @brief Set the presence penalty, which reduces absolute log prob if the
 * tokeov_genai_generation_config_set_presence_penaltyn was generated at least once.
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @param value The value of parameter for presence penalty.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_generation_config_set_presence_penalty(ov_genai_generation_config* config,
                                                                                     const float value);

/**
 * @brief Set the frequency penalty, which reduces absolute log prob as many times as the token was generated.
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @param value The value of parameter for frequency penalty.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_generation_config_set_frequency_penalty(ov_genai_generation_config* config, const float value);

/**
 * @brief Set the seed to initialize random number generator.
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @param value The value of seed for random number generator.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_generation_config_set_rng_seed(ov_genai_generation_config* config,
                                                                             const size_t value);

/**
 * @brief Set the lower token probability of candidate to be validated by main model in case of dynamic strategy
 * candidates number update.
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @param value The lower token probability of candidate to be validated by main model.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_generation_config_set_assistant_confidence_threshold(ov_genai_generation_config* config, const float value);
/**
 * @brief Set the defined candidates number to be generated by draft model/prompt lookup in case of static strategy
 * candidates number update.
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @param value The number of assistant tokens.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_generation_config_set_num_assistant_tokens(ov_genai_generation_config* config, const size_t value);

/**
 * @brief Set the maximum ngram to use when looking for matches in the prompt.
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @param value The maximum ngram size.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_generation_config_set_max_ngram_size(ov_genai_generation_config* config,
                                                                                   const size_t value);

/**
 * @brief Set the token_id of <eos> (end of sentence)
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @param id The eos token id.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_generation_config_set_eos_token_id(ov_genai_generation_config* config,
                                                                                 const int64_t id);

/**
 * @brief Get the maximum number of tokens to generate, excluding the number of tokens in the prompt.
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @param The maximum number of tokens to generate.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e
ov_genai_generation_config_get_max_new_tokens(const ov_genai_generation_config* config, size_t* max_new_tokens);

/**
 * @brief Checks that are no conflicting parameters, e.g. do_sample=true and num_beams > 1.
 * @param handle A pointer to the ov_genai_generation_config instance.
 * @return ov_status_e A status code, return OK(0) if successful.
 */
OPENVINO_GENAI_C_EXPORTS ov_status_e ov_genai_generation_config_validate(ov_genai_generation_config* config);
