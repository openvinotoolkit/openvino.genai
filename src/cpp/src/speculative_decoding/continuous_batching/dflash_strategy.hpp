// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "fast_draft_strategy.hpp"
#include "speculative_decoding/continuous_batching/dflash_strategy_utils.hpp"
#include "speculative_decoding/dflash_model_transforms.hpp"

namespace ov::genai {

class ContinuousBatchingPipeline::DFlashDecodingImpl : public ContinuousBatchingPipeline::SpeculativeDecodingImpl {
public:
    DFlashDecodingImpl(const ov::genai::ModelDesc& main_model_desc,
                       const ov::genai::ModelDesc& draft_model_desc,
                       const ov::genai::utils::dflash::DFlashRTInfo& rt_info);

    GenerationHandle add_request(uint64_t request_id,
                                 const ov::Tensor& input_ids,
                                 const ov::genai::GenerationConfig& sampling_params,
                                 std::optional<ov::Tensor> token_type_ids = std::nullopt,
                                 std::optional<ov::Tensor> prompt_ids = std::nullopt,
                                 std::optional<std::unordered_map<std::string, ov::Tensor>> lm_extra_inputs = std::nullopt) override;

    GenerationHandle add_request(uint64_t request_id,
                                 const std::string& prompt,
                                 const ov::genai::GenerationConfig& sampling_params) override;

    bool has_non_finished_requests() override;
    void step() override;

    std::vector<EncodedGenerationResult>
    generate(const std::vector<ov::Tensor>& input_ids,
             const std::vector<GenerationConfig>& sampling_params,
             const StreamerVariant& streamer,
             const std::optional<std::vector<ov::Tensor>>& token_type_ids = std::nullopt,
             const std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>& position_ids = std::nullopt,
             const std::optional<std::vector<ov::Tensor>>& prompt_ids = std::nullopt,
             const std::optional<std::vector<std::unordered_map<std::string, ov::Tensor>>>& lm_extra_inputs_list = std::nullopt) override;

private:
    class DFlashCBDraftRunner;

    struct RequestState {
        dflash_cb::HiddenDeltaBuffer pending_hidden_deltas;
        std::vector<int64_t> generated_tokens;
        size_t prompt_len = 0;
        size_t generated_before_draft = 0;
        size_t draft_generated = 0;
        std::optional<uint64_t> target_la_checkpoint_sequence_id;
        bool finished = false;
        GenerationConfig generation_config;
    };

    GenerationConfig make_draft_generation_config(const GenerationConfig& config) const;
    static void append_pending_hidden_delta(RequestState& state, const ov::Tensor& hidden_delta, bool copy_data);
    static bool has_pending_hidden_delta(const RequestState& state);
    static ov::Tensor materialize_pending_hidden_delta(const RequestState& state);
    static void clear_pending_hidden_delta(RequestState& state);
    void validate_hidden_prefix_length(const RequestState& state) const;
    bool has_active_request_state() const;
    void drop_finished_request_states();
    void update_draft_states_from_main(const GeneratedRequests& main_generated_requests);
    void drop_requests();
    ov::genai::RawPerfMetrics collect_draft_raw_metrics();

    std::shared_ptr<DFlashCBDraftRunner> m_draft;
    ov::genai::utils::dflash::DFlashRTInfo m_rt_info;
    std::map<uint64_t, RequestState> m_request_states;
};

}  // namespace ov::genai
