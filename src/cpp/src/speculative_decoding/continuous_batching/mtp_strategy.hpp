// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fast_draft_strategy.hpp"
#include "visual_language/inputs_embedder.hpp"

namespace ov::genai {

/**
 * @brief Multi-Token Prediction (MTP) speculative-decoding strategy.
 *
 * Drafts speculative tokens with a single-layer MTP module that consumes the main model's
 * `last_hidden_state` plus the embedding of the main-predicted token. Reuses the EAGLE3 hidden-state
 * pairing machinery; distinct from EAGLE3 in that:
 *  - the main VLM `language_model` already outputs `last_hidden_state` (no hidden-state graft on main),
 *  - the MTP draft has no lm_head, so a tied-weight lm_head is grafted onto it at load,
 *  - both main and draft consume `inputs_embeds` from a shared `EmbeddingsModel`.
 */
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
             const std::optional<std::vector<ov::Tensor>>& token_type_ids = std::nullopt,
             const std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>& position_ids = std::nullopt,
             const std::optional<std::vector<ov::Tensor>>& prompt_ids = std::nullopt,
             const std::optional<std::vector<std::unordered_map<std::string, ov::Tensor>>>& lm_extra_inputs_list = std::nullopt) override;

    GenerationHandle add_request(uint64_t request_id,
                                 const ov::Tensor& input_ids,
                                 const ov::genai::GenerationConfig& sampling_params,
                                 std::optional<ov::Tensor> token_type_ids = std::nullopt,
                                 std::optional<ov::Tensor> prompt_ids = std::nullopt,
                                 std::optional<std::unordered_map<std::string, ov::Tensor>> lm_extra_inputs = std::nullopt) override;

    GenerationHandle add_request(uint64_t request_id,
                                 const std::string& prompt,
                                 const ov::genai::GenerationConfig& sampling_params) override;

protected:
    void enable_mtp_hidden_state_pairing();
};
}  // namespace ov::genai
