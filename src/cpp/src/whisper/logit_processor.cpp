// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/openvino.hpp>

#include "openvino/genai/whisper_generation_config.hpp"
#include "sampling/sampler.hpp"

namespace ov {
namespace genai {

void do_suppress_tokens(ov::Tensor& logits, const size_t batch_idx, const std::vector<int64_t>& suppress_tokens) {
    OPENVINO_ASSERT(logits.get_shape()[0] >= batch_idx, "logits batch size doesn't match the batch number");

    size_t vocab_size = logits.get_shape().back();
    size_t batch_offset = batch_idx * logits.get_shape()[1] * vocab_size;
    size_t sequence_offset = (logits.get_shape()[1] - 1) * vocab_size;
    float* logits_data = logits.data<float>() + batch_offset + sequence_offset;

    for (auto supress_token : suppress_tokens) {
        logits_data[supress_token] = -std::numeric_limits<float>::infinity();
    }
}

void process_whisper_timestamp_logits(ov::Tensor& logits,
                                      const size_t batch_idx,
                                      const ov::genai::WhisperGenerationConfig& config,
                                      const std::vector<int64_t>& generated_tokens,
                                      bool initial_step = false) {
    const size_t batch_size = logits.get_shape().at(0);

    size_t vocab_size = logits.get_shape().back();
    size_t batch_offset = batch_idx * logits.get_shape()[1] * vocab_size;
    size_t sequence_offset = (logits.get_shape()[1] - 1) * vocab_size;
    float* logits_data = logits.data<float>() + batch_offset + sequence_offset;

    // suppress<|notimestamps|>
    logits_data[config.no_timestamps_token_id] = -std::numeric_limits<float>::infinity();

    size_t timestamp_begin = config.no_timestamps_token_id + 1;

    // timestamps have to appear in pairs, except directly before eos_token; mask logits accordingly
    size_t generated_length = generated_tokens.size();
    bool last_was_timestamp = generated_length >= 1 && generated_tokens[generated_length - 1] >= timestamp_begin;
    bool penultimate_was_timestamp = generated_length < 2 || generated_tokens[generated_length - 2] >= timestamp_begin;

    if (last_was_timestamp) {
        if (penultimate_was_timestamp) {
            // has to be non-timestamp
            for (size_t i = timestamp_begin; i < vocab_size; i++) {
                logits_data[i] = -std::numeric_limits<float>::infinity();
            }
        } else {
            // cannot be normal text token
            for (size_t i = 0; i < config.eos_token_id; i++) {
                logits_data[i] = -std::numeric_limits<float>::infinity();
            }
        }
    }

    // filter generated timestamps
    std::vector<int64_t> timestamps;
    for (const auto token : generated_tokens) {
        if (token >= timestamp_begin) {
            timestamps.push_back(token);
        }
    }

    if (timestamps.size() > 0) {
        size_t timestamp_last;
        // `timestamps` shouldn't decrease; forbid timestamp tokens smaller than the last
        // The following lines of code are copied from: https://github.com/openai/whisper/pull/914/files#r1137085090
        if (last_was_timestamp && !penultimate_was_timestamp) {
            timestamp_last = timestamps.back();
        } else {
            // Avoid to emit <|0.00|> again
            timestamp_last = timestamps.back() + 1;
        }

        for (size_t i = timestamp_begin; i < timestamp_last; i++) {
            logits_data[i] = -std::numeric_limits<float>::infinity();
        }
    }

    // apply the `max_initial_timestamp` option
    if (initial_step) {
        for (size_t i = 0; i < timestamp_begin; i++) {
            logits_data[i] = -std::numeric_limits<float>::infinity();
        }

        size_t last_allowed = timestamp_begin + config.max_initial_timestamp_index;
        for (size_t i = last_allowed + 1; i < vocab_size; i++) {
            logits_data[i] = -std::numeric_limits<float>::infinity();
        }
    }

    auto tokens = ov::genai::log_softmax(logits, 0);
    float timestamp_exp_prov_sum = 0;

    for (size_t i = timestamp_begin; i < vocab_size; i++) {
        timestamp_exp_prov_sum += std::exp(tokens[i].m_log_prob);
    }
    float timestamp_logprob = std::log(timestamp_exp_prov_sum);

    auto max_logprob_token = std::max_element(tokens.begin(), tokens.end(), [](const Token& left, const Token& right) {
        return left.m_log_prob < right.m_log_prob;
    });

    if (timestamp_logprob > max_logprob_token->m_log_prob) {
        for (size_t i = 0; i < timestamp_begin; i++) {
            logits_data[i] = -std::numeric_limits<float>::infinity();
        }
    }
}

}  // namespace genai
}  // namespace ov
