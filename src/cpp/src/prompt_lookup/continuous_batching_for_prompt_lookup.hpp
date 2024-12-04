// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/continuous_batching_pipeline.hpp"

#include "continuous_batching_impl.hpp"

namespace ov::genai {
class ContinuousBatchingPipeline::ContinuousBatchingForPromptLookupImpl : public ContinuousBatchingPipeline::ContinuousBatchingImpl {
public:
    ContinuousBatchingForPromptLookupImpl() = default;

    ContinuousBatchingForPromptLookupImpl(
        const std::filesystem::path& models_path,
        const Tokenizer& tokenizer,
        const SchedulerConfig& scheduler_config,
        const std::string& device,
        const ov::AnyMap& properties,
        size_t max_ngram_size) :
    ContinuousBatchingImpl{ models_path,
                            tokenizer,
                            scheduler_config,
                            device,
                            properties,
                            true } {
        m_max_ngram_size = max_ngram_size;
    };

    ContinuousBatchingForPromptLookupImpl(
        const std::filesystem::path& models_path,
        const SchedulerConfig& scheduler_config,
        const std::string& device,
        const ov::AnyMap& properties,
        size_t max_ngram_size,
        const ov::AnyMap& tokenizer_properties = {}) :
    ContinuousBatchingImpl{ models_path,
                            Tokenizer(models_path, tokenizer_properties),
                            scheduler_config,
                            device,
                            properties,
                            true } {
        m_max_ngram_size = max_ngram_size;
    };
                            
    void generate_candidates();

    // { generated_len, validation_len }
    using SequenceLen = std::pair<uint64_t, uint64_t>;
    std::map<uint64_t, SequenceLen> get_generated_request_len();

protected:
    TokenIds generate_candidates(const TokenIds& input_ids, size_t num_pred_tokens);

    size_t m_max_ngram_size = 3;
};
}