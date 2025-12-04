// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching_for_prompt_lookup.hpp"

#include "logger.hpp"

namespace ov::genai {

std::map<uint64_t, ContinuousBatchingPipeline::ContinuousBatchingForPromptLookupImpl::SequenceLen>
ContinuousBatchingPipeline::ContinuousBatchingForPromptLookupImpl::get_generated_request_len() {
    std::map<uint64_t, ContinuousBatchingPipeline::ContinuousBatchingForPromptLookupImpl::SequenceLen> result;
    for (const auto& request : m_requests) {
        const auto request_id = request->get_request_id();
        auto validation_len = request->get_num_tokens_to_validate();
        auto generated_len = request->get_num_processed_tokens() - request->get_prompt_len() + 1;
        result.insert({ request_id, { generated_len, validation_len } });
    }
    return result;
}

TokenIds ContinuousBatchingPipeline::ContinuousBatchingForPromptLookupImpl::generate_candidates(const TokenIds& input_ids, size_t num_pred_tokens, size_t max_ngram_size) {
    if (num_pred_tokens == 0) {
        return std::vector<int64_t>{};
    }

    const size_t input_length = input_ids.size();
    const int32_t adjusted_ngram_size = static_cast<int32_t>(std::min(max_ngram_size, input_length));
    for (int32_t ngram_size = adjusted_ngram_size; ngram_size > 0; ngram_size--) {
        // extract last ngram_size tokens as search ngram
        std::vector<int64_t> ngram = std::vector<int64_t>{input_ids.cend() - ngram_size, input_ids.cend()};

        // find ngram match in input_ids
        size_t ngram_i = 0;
        for (int32_t input_i = 0; input_i < static_cast<int32_t>(input_length) - ngram_size; input_i++) {
            if (!std::equal(ngram.begin() + ngram_i, ngram.end(), input_ids.begin() + input_i)) {
                ngram_i = 0;
                continue;
            }

            ngram_i++;

            if (ngram_i < ngram_size) {
                continue;
            }

            // match found with the end at input_i
            size_t avaliable_num_pred = std::min(input_length - (input_i + 1), num_pred_tokens);

            // return candidates with length of avaliable_num_pred
            return std::vector<int64_t>{input_ids.cbegin() + input_i + 1,
                                        input_ids.cbegin() + input_i + 1 + avaliable_num_pred};
        }
    }

    return std::vector<int64_t>{};
}

void ContinuousBatchingPipeline::ContinuousBatchingForPromptLookupImpl::generate_candidates() {
    for (auto& request : m_requests) {
        const auto prompt = request->get_prompt_ids();

        size_t max_validation_len = 0;
        for (auto& running_sequence : request->get_running_sequences()) {
            const auto generated_tokens = running_sequence->get_generated_ids();
            if (generated_tokens.empty()) {
                continue;
            }
            TokenIds full_input_ids = prompt;
            full_input_ids.insert(full_input_ids.end(), generated_tokens.begin(), generated_tokens.end());

            size_t min_num_assistant_tokens = 0;
            const auto sampling_params = request->get_sampling_parameters();
            {
                const auto generated_len = running_sequence->get_generated_len();
                const auto left_generated_len = request->get_max_new_tokens() - generated_len - 1;
                min_num_assistant_tokens = std::min(sampling_params.num_assistant_tokens, left_generated_len);
            }
            TokenIds candidates = generate_candidates(full_input_ids, min_num_assistant_tokens, sampling_params.max_ngram_size);

            // Padding to candidate tokens,
            // Avoid shape checking and increasing the amount of computation when the shape changes.
            if (candidates.size() < sampling_params.num_assistant_tokens) {
                OPENVINO_ASSERT(!full_input_ids.empty(), "full_input_ids should not be empty");
                int token_sz = static_cast<int>(candidates.size());
                for (int ci = 0; ci < static_cast<int>(sampling_params.num_assistant_tokens) - token_sz; ci++) {
                    // Padding with -1
                    candidates.push_back(-1);
                }
            }

            auto debug_print_token_id = [&](const TokenIds& print_ids, const std::string& prefix, const size_t& last_num) {
                std::stringstream ss;
                ss << prefix << " = ";
                size_t start_id = (print_ids.size() > last_num) ? (print_ids.size() - last_num) : 0;
                for (size_t id = start_id; id < print_ids.size(); id++) {
                    ss << print_ids[id] << "[" << m_tokenizer.decode(TokenIds{print_ids[id]}) << "],";
                }
                return ss.str();
            };
            GENAI_DEBUG(debug_print_token_id(generated_tokens, "generated_tokens", sampling_params.num_assistant_tokens + 1));
            GENAI_DEBUG(debug_print_token_id(candidates, "candidates", candidates.size()));

            if (!candidates.empty()) {
                for (const auto& candidate : candidates) {
                    running_sequence->append_token(candidate, 0);
                }
                max_validation_len = std::max(max_validation_len, candidates.size());
            }
        }
        request->set_num_validated_tokens(max_validation_len);
    }
}

bool ContinuousBatchingPipeline::ContinuousBatchingForPromptLookupImpl::is_requests_empty() {
    return m_requests.empty();
}

size_t ContinuousBatchingPipeline::ContinuousBatchingForPromptLookupImpl::get_processed_tokens_per_iteration() {
    return m_batch_size;
}
}  // namespace ov::genai
