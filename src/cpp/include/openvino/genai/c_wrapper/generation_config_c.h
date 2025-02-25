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
#include <stdint.h>

#include "../visibility.hpp"

#ifdef __cplusplus
OPENVINO_EXTERN_C {
#endif

#include "stdio.h"

    /**
     * @brief controls the stopping condition for grouped beam search. The following values are possible:
     *        "EARLY" stops as soon as there are `num_beams` complete candidates.
              "HEURISTIC" stops when is it unlikely to find better candidates.
              "NEVER" stops when there cannot be better candidates.
     */
    typedef enum { EARLY, HEURISTIC, NEVER } StopCriteria;

    /**
     * @struct GenerationConfigHandle
     * @brief type define GenerationConfigHandle from OpaqueGenerationConfig
     */
    typedef struct OpaqueGenerationConfig GenerationConfigHandle;

    /**
     * @brief Create GenerationConfigHandle.
     */
    OPENVINO_GENAI_EXPORTS GenerationConfigHandle* CreateGenerationConfig();

    /**
     * @brief Create GenerationConfigHandle from JSON file.
     * @param json_path A path to the JSON file with generation config.
     */
    OPENVINO_GENAI_EXPORTS GenerationConfigHandle* CreateGenerationConfigFromJson(const char* json_path);

    /**
     * @brief Release the memory allocated by GenerationConfigHandle.
     * @param handle A pointer to the GenerationConfigHandle to free memory.
     */
    OPENVINO_GENAI_EXPORTS void DestroyGenerationConfig(GenerationConfigHandle * handle);

    /**
     * @brief Set the maximum number of tokens to generate, excluding the number of tokens in the prompt. max_new_tokens
     * has priority over max_length.
     * @param handle A pointer to the GenerationConfigHandle.
     * @param value The maximum number of tokens to generate.
     */
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetMaxNewTokens(GenerationConfigHandle * handle, size_t value);

    /**
     * @brief Set the maximum length the generated tokens can have. Corresponds to the length of the input prompt +
     * `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.
     * @param handle A pointer to the GenerationConfigHandle.
     * @param value The maximum length the generated tokens can have.
     */
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetMaxLength(GenerationConfigHandle * config, size_t value);

    /**
     * @brief Set whether or not to ignore <eos> token
     * @param handle A pointer to the GenerationConfigHandle.
     * @param value If set to true, then generation will not stop even if <eos> token is met.
     */
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetIgnoreEOS(GenerationConfigHandle * config, bool value);

    /**
     * @brief Set the minimum number of tokens to generate.
     * @param handle A pointer to the GenerationConfigHandle.
     * @param value The minimum number of tokens to generate.
     */
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetMinNewTokens(GenerationConfigHandle * config, size_t value);

    /**
     * @brief Set whether or not to include user prompt in the output.
     * @param handle A pointer to the GenerationConfigHandle.
     * @param value If set to true, output will include user prompt.
     */
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetEcho(GenerationConfigHandle * config, bool value);

    /**
     * @brief Set the number of top logprobs computed for each position,
              if set to 0, logprobs are not computed and value 0.0 is returned.
              Currently only single top logprob can be returned, so any logprobs > 1 is treated as logprobs
     == 1.(default: 0).
     * @param handle A pointer to the GenerationConfigHandle.
     * @param value The number of top logprobs computed for each position.
     */
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetLogProbs(GenerationConfigHandle * config, size_t value);

    /**
     * @brief Set the set of strings that will cause pipeline to stop generating further tokens.
     * @param handle A pointer to the GenerationConfigHandle.
     * @param strings An array of strings
     * @param count The number of strings in the array
     */
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetStopStrings(GenerationConfigHandle * config,
                                                               const char* strings[],
                                                               size_t count);
    /**
     * @brief Set whether or not to include stop string that matched generation in the output.
     * @param handle A pointer to the GenerationConfigHandle.
     * @param value If set to true stop string that matched generation will be included in generation output (default:
     * false).
     */
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetIncludeStopStrInOutput(GenerationConfigHandle * config, bool value);

    /**
     * @brief Set the set of tokens that will cause pipeline to stop generating further tokens.
     * @param handle A pointer to the GenerationConfigHandle.
     * @param token_ids An array of token ids
     * @param token_ids_num The number of token ids in the array
     */
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetStopTokenIds(GenerationConfigHandle * config,
                                                                int64_t * token_ids,
                                                                size_t token_ids_num);

    /**
     * @brief Set the number of groups to divide `num_beams` into in order to ensure diversity among different groups of
     * beams.
     * @param handle A pointer to the GenerationConfigHandle.
     * @param value The number of beam groups.
     */
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetNumBeamGroups(GenerationConfigHandle * config, size_t value);

    /**
     * @brief Set the number of beams for beam search. 1 disables beam search.
     * @param handle A pointer to the GenerationConfigHandle.
     * @param value The number of beams for beam search.
     */
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetNumBeams(GenerationConfigHandle * config, size_t value);

    /**
     * @brief Set the diversity penalty, this value is subtracted from a beam's score if it generates the same token as
     * any beam from other group at a particular time.
     * @param handle A pointer to the GenerationConfigHandle.
     * @param value The parameter for diversity penalty
     */
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetDiversityPenalty(GenerationConfigHandle * config, float value);

    /**
     * @brief Set the length penalty, exponential penalty to the length that is used with beam-based generation. It is
     * applied as an exponent to the sequence length, which in turn is used to divide the score of the sequence. Since
     * the score is the log likelihood of the sequence (i.e. negative), `length_penalty` > 0.0 promotes longer
     * sequences, while `length_penalty` < 0.0 encourages shorter sequences.
     * @param handle A pointer to the GenerationConfigHandle.
     * @param value The exponential penalty.
     */
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetLengthPenalty(GenerationConfigHandle * config, float value);

    /**
     * @brief Set the number of sequences to return for grouped beam search decoding per batch element.
     * num_return_sequences must be less or equal to num_beams.
     * @param handle A pointer to the GenerationConfigHandle.
     * @param value The number of sequences to return.
     */
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetNumReturnSequences(GenerationConfigHandle * config, size_t value);

    /**
     * @brief Set the no_repeat_ngram_size
     * @param handle A pointer to the GenerationConfigHandle.
     * @param value If set to int > 0, all ngrams of that size can only occur once.
     */
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetNoRepeatNgramSize(GenerationConfigHandle * config, size_t value);

    /**
     * @brief Set the stopping condition for grouped beam search. It accepts the following values:
     * "EARLY", where the generation stops as soon as there are `num_beams` complete candidates; "HEURISTIC", where an
     * "HEURISTIC" is applied when it is unlikely to find better candidates;
     * "NEVER", where the generation stops when there cannot be better candidates.
     * @param handle A pointer to the GenerationConfigHandle.
     * @param value The stopping condition for grouped beam search.
     */
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetStopCriteria(GenerationConfigHandle * config, StopCriteria value);

    /**
     * @brief Set the temperature value used to modulate token probabilities for random sampling.
     * @param handle A pointer to the GenerationConfigHandle.
     * @param value The value of temperature
     */
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetTemperature(GenerationConfigHandle * config, float value);

    /**
     * @brief Set the top_p value. If set to float < 1, only the smallest set of most probable tokens with probabilities
     * that add up to top_p or higher are kept for generation.
     * @param handle A pointer to the GenerationConfigHandle.
     * @param value The value of top_p
     */
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetTopP(GenerationConfigHandle * config, float value);

    /**
     * @brief Set the top_k value. The number of highest probability vocabulary tokens to keep for top-k-filtering.
     * @param handle A pointer to the GenerationConfigHandle.
     * @param value The value of top_k
     */
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetTopK(GenerationConfigHandle * config, size_t value);

    /**
     * @brief Set whether or not to use multinomial random sampling that add up to `top_p` or higher are kept.
     * @param handle A pointer to the GenerationConfigHandle.
     * @param value If set to true, multinomial random sampling will be used.
     */
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetDoSample(GenerationConfigHandle * config, bool value);

    /**
     * @brief Set the parameter for repetition penalty. 1.0 means no penalty.
     * @param handle A pointer to the GenerationConfigHandle.
     * @param value The value of parameter for repetition penalty
     */
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetRepetitionPenalty(GenerationConfigHandle * config, float value);

    /**
     * @brief Set the presence penalty, which reduces absolute log prob if the token was generated at least once.
     * @param handle A pointer to the GenerationConfigHandle.
     * @param value The value of parameter for presence penalty
     */
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetPresencePenalty(GenerationConfigHandle * config, float value);

    /**
     * @brief Set the frequency penalty, which reduces absolute log prob as many times as the token was generated.
     * @param handle A pointer to the GenerationConfigHandle.
     * @param value The value of parameter for frequency penalty
     */
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetFrequencyPenalty(GenerationConfigHandle * config, float value);

    /**
     * @brief Set the seed to initialize random number generator.
     * @param handle A pointer to the GenerationConfigHandle.
     * @param value The value of seed for random number generator
     */
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetRngSeed(GenerationConfigHandle * config, size_t value);

    /**
     * @brief Set the lower token probability of candidate to be validated by main model in case of dynamic strategy
     * candidates number update.
     * @param handle A pointer to the GenerationConfigHandle.
     * @param value The lower token probability of candidate to be validated by main model.
     */
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetAssistantConfidenceThreshold(GenerationConfigHandle * config,
                                                                                float value);
    /**
     * @brief Set the defined candidates number to be generated by draft model/prompt lookup in case of static strategy
     * candidates number update.
     * @param handle A pointer to the GenerationConfigHandle.
     * @param value The number of assistant tokens.
     */
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetNumAssistantTokens(GenerationConfigHandle * config, size_t value);

    /**
     * @brief Set the maximum ngram to use when looking for matches in the prompt.
     * @param handle A pointer to the GenerationConfigHandle.
     * @param value The maximum ngram size.
     */
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetMaxNgramSize(GenerationConfigHandle * config, size_t value);

    /**
     * @brief Set the token_id of <eos> (end of sentence)
     * @param handle A pointer to the GenerationConfigHandle.
     * @param id The eos token id.
     */
    OPENVINO_GENAI_EXPORTS void GenerationConfigSetEOSTokenID(GenerationConfigHandle * config, int64_t id);

    /**
     * @brief Get the maximum number of tokens to generate, excluding the number of tokens in the prompt.
     * @param handle A pointer to the GenerationConfigHandle.
     * @return The maximum number of tokens to generate.
     */
    OPENVINO_GENAI_EXPORTS size_t GenerationConfigGetMaxNewTokens(GenerationConfigHandle * config);

    /**
     * @brief Determine whether greedy decoding is enabled.
     * @param handle A pointer to the GenerationConfigHandle.
     * @return A boolean indicating if greedy decoding is enabled.
     */
    OPENVINO_GENAI_EXPORTS bool GenerationConfigIsGreedyDecoding(GenerationConfigHandle * config);

    /**
     * @brief Determine whether beam search is enabled.
     * @param handle A pointer to the GenerationConfigHandle.
     * @return A boolean indicating if beam search is enabled.
     */
    OPENVINO_GENAI_EXPORTS bool GenerationConfigIsBeamSearch(GenerationConfigHandle * config);

    /**
     * @brief Determine whether multinomial random sampling is enabled.
     * @param handle A pointer to the GenerationConfigHandle.
     * @return A boolean indicating if multinomial random sampling is enabled.
     */
    OPENVINO_GENAI_EXPORTS bool GenerationConfigIsMultinomial(GenerationConfigHandle * config);

    /**
     * @brief Determine whether assisting generation is enabled.
     * @param handle A pointer to the GenerationConfigHandle.
     * @return A boolean indicating if assisting generation is enabled.
     */
    OPENVINO_GENAI_EXPORTS bool GenerationConfigIsAssistingGeneration(GenerationConfigHandle * config);

    /**
     * @brief Determine whether prompt lookup is enabled.
     * @param handle A pointer to the GenerationConfigHandle.
     * @return A boolean indicating if prompt lookup is enabled.
     */
    OPENVINO_GENAI_EXPORTS bool GenerationConfigIsPromptLookup(GenerationConfigHandle * config);

    /**
     * @brief Checks that are no conflicting parameters, e.g. do_sample=true and num_beams > 1.
     * @param handle A pointer to the GenerationConfigHandle.
     */
    OPENVINO_GENAI_EXPORTS void GenerationConfigValidate(GenerationConfigHandle * config);

#ifdef __cplusplus
}
#endif