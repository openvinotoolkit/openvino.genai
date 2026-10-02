// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <map>

#include "fast_draft_strategy.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

// MTP strategy: draft consumes main last_hidden_state plus shifted token embeddings.
class ContinuousBatchingPipeline::MtpDecodingImpl : public ContinuousBatchingPipeline::SpeculativeDecodingImpl {
public:
    template <class Impl>
    friend std::vector<EncodedGenerationResult> generate_common(
        Impl*,
        const std::vector<ov::Tensor>&,
        const std::vector<GenerationConfig>&,
        const StreamerVariant&,
        std::optional<std::vector<ov::Tensor>>,
        std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>,
        std::optional<std::vector<ov::Tensor>>,
        const std::optional<std::vector<std::unordered_map<std::string, ov::Tensor>>>&,
        GenerateStrategy&);

    MtpDecodingImpl(const ov::genai::ModelDesc& main_model_desc,
                    const ov::genai::ModelDesc& draft_model_desc,
                    const std::shared_ptr<InputsEmbedder>& inputs_embedder);

    std::vector<EncodedGenerationResult>
    generate(const std::vector<ov::Tensor>& input_ids,
             const std::vector<GenerationConfig>& sampling_params,
             const StreamerVariant& streamer,
             const std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>& position_ids = std::nullopt,
             const std::optional<std::vector<ov::Tensor>>& prompt_ids = std::nullopt,
             const std::optional<std::vector<std::unordered_map<std::string, ov::Tensor>>>& lm_extra_inputs_list = std::nullopt) override;

    GenerationHandle add_request(uint64_t request_id,
                                 const ov::Tensor& input_ids,
                                 const ov::genai::GenerationConfig& sampling_params,
                                 std::optional<ov::Tensor> prompt_ids = std::nullopt,
                                 std::optional<std::unordered_map<std::string, ov::Tensor>> lm_extra_inputs = std::nullopt) override;

    GenerationHandle add_request(uint64_t request_id,
                                 const std::string& prompt,
                                 const ov::genai::GenerationConfig& sampling_params) override;

protected:
    MtpDecodingImpl() = default;
    void enable_mtp_hidden_state_pairing();
    void align_request_pair_processed_prefix(uint64_t request_id) override;
};

class ContinuousBatchingPipeline::Gemma4MtpDecodingImpl : public ContinuousBatchingPipeline::SpeculativeDecodingImpl {
public:
    template <class Impl>
    friend std::vector<EncodedGenerationResult> generate_common(
        Impl*, const std::vector<ov::Tensor>&, const std::vector<GenerationConfig>&,
        const StreamerVariant&,
        std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>,
        std::optional<std::vector<ov::Tensor>>,
        const std::optional<std::vector<std::unordered_map<std::string, ov::Tensor>>>&,
        GenerateStrategy&);

    Gemma4MtpDecodingImpl(const ModelDesc& main_model_desc,
                          const ModelDesc& draft_model_desc,
                          const std::shared_ptr<InputsEmbedder>& inputs_embedder);

    GenerationHandle add_request(uint64_t request_id, const ov::Tensor& input_embeds,
                                 const GenerationConfig& config,
                                 std::optional<ov::Tensor> prompt_ids = std::nullopt,
                                 std::optional<std::unordered_map<std::string, ov::Tensor>> lm_extra_inputs = std::nullopt) override;
    GenerationHandle add_request(uint64_t request_id, const std::string& prompt,
                                 const GenerationConfig& config) override;
    std::vector<EncodedGenerationResult> generate(
        const std::vector<ov::Tensor>& input_embeds, const std::vector<GenerationConfig>& configs,
        const StreamerVariant& streamer,
        const std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>& position_ids = std::nullopt,
        const std::optional<std::vector<ov::Tensor>>& prompt_ids = std::nullopt,
        const std::optional<std::vector<std::unordered_map<std::string, ov::Tensor>>>& lm_extra_inputs = std::nullopt) override;
    void step() override;

protected:
    void drop_requests();
    bool is_requests_empty();
    std::vector<SequenceGroup::Ptr> get_awaiting_requests();

private:
    struct RequestState {
        ov::Tensor hidden_state;
        size_t prompt_length = 0;
        size_t generated_length = 0;
    };

    void append_main_outputs(const GeneratedRequests& generated);
    std::vector<int64_t> draft_tokens(uint64_t request_id, const GeneratedSequence& sequence,
                                      const GenerationConfig& config);

    std::map<uint64_t, RequestState> m_requests;
    std::map<uint64_t, GenerationConfig> m_request_configs;
    ov::InferRequest m_draft_request;
    std::array<std::string, 4> m_draft_cache_names;
    std::array<size_t, 2> m_main_cache_layers;
    EmbeddingsModel::Ptr m_embedding;
    size_t m_default_num_assistant_tokens;
    size_t m_max_num_batched_tokens;
};
}  // namespace ov::genai
