// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>
#include <variant>
#include <string>

#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/genai/tokenizer.hpp"

namespace ov {
namespace genai {

/**
 * @brief controls the stopping condition for grouped beam search. The following values are possible:
 *        "early" stops as soon as there are `num_beams` complete candidates.
          "heuristic" stops when is it unlikely to find better candidates.
          "never" stops when there cannot be better candidates.
 */
enum class StopCriteria { early, heuristic, never };

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
 * @param pad_token_id token_id of <pad> (padding)
 * @param bos_token_id token_id of <bos> (beggining of sentence)
 * @param eos_token_id token_id of <eos> (end of sentence)
 * @param bos_token <bos> token string representation
 * @param eos_token <eos> token string representation
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
 * @param num_return_sequences the number of sequences to return for grouped beam search decoding.
 * @param no_repeat_ngram_size if set to int > 0, all ngrams of that size can only occur once.
 * @param stop_criteria controls the stopping condition for grouped beam search. It accepts the following values: 
 *        "early", where the generation stops as soon as there are `num_beams` complete candidates; "heuristic", where an 
 *        heuristic is applied and the generation stops when is it very unlikely to find better candidates;
 *        "never", where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm).
 * 
 * Random sampling parameters:
 * @param temperature the value used to modulate token probabilities for random sampling.
 * @param top_p - if set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
 * @param top_k the number of highest probability vocabulary tokens to keep for top-k-filtering.
 * @param do_sample whether or not to use multinomial random sampling that add up to `top_p` or higher are kept.
 * @param repetition_penalty the parameter for repetition penalty. 1.0 means no penalty.
 */
class OPENVINO_GENAI_EXPORTS GenerationConfig {
public:
    GenerationConfig() = default;
    explicit GenerationConfig(std::string json_path);

    // Generic
    size_t max_new_tokens = SIZE_MAX;
    size_t max_length = SIZE_MAX;
    bool ignore_eos = false;

    // Beam search specific
    size_t num_beam_groups = 1;
    size_t num_beams = 1;
    float diversity_penalty = 1.0f;
    float length_penalty = 1.0f;
    size_t num_return_sequences = 1;
    size_t no_repeat_ngram_size = std::numeric_limits<size_t>::max();
    StopCriteria stop_criteria = StopCriteria::heuristic;
    
    // Multinomial
    float temperature = 1.0f;
    float top_p = 1.0f;
    size_t top_k = 50;
    bool do_sample = false;
    float repetition_penalty = 1.0f;

    // special tokens
    int64_t pad_token_id = 0;
    int64_t bos_token_id = 1;
    int64_t eos_token_id = 2;
    
    // used for chat scenario
    std::string bos_token = "<s>";
    std::string eos_token = "</s>";

    size_t get_max_new_tokens(size_t prompt_length = 0) const;
    bool is_greedy_decoding() const;
    bool is_beam_search() const;
    bool is_multinomial() const;
    static GenerationConfig anymap_to_generation_config(const ov::AnyMap& config_map = {});
};

}  // namespace genai
}  // namespace ov
