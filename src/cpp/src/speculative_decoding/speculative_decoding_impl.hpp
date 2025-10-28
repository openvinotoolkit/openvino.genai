// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "continuous_batching/pipeline_impl.hpp"
#include "speculative_decoding/continuous_batching_for_speculative_decoding_impl.hpp"
#include "speculative_decoding/speculative_decoding_metrics.hpp"
#include "openvino/genai/speculative_decoding/perf_metrics.hpp"
#include "utils.hpp"

namespace ov::genai {

class ContinuousBatchingPipeline::SpeculativeDecodingImpl : public ContinuousBatchingPipeline::IContinuousBatchingPipeline {
protected:
    std::shared_ptr<ContinuousBatchingForSpeculativeDecodingImpl> m_main_pipeline, m_draft_pipeline;
    // Metrics
    SpeculativeDecodingMetrics m_sd_metrics;
    ov::genai::SDPerModelsPerfMetrics m_perf_metrics;

    // Mutex protecting access to m_draft_generations, so add_request and step methods can be called from different threads
    std::mutex m_draft_generations_mutex;
    std::map<uint64_t, GenerationHandle> m_draft_generations;

    void drop_requests();
    bool is_requests_empty();
    std::vector<SequenceGroup::Ptr> get_awaiting_requests();
    
public:
    SpeculativeDecodingImpl(const ov::genai::ModelDesc& main_model_desc, const ov::genai::ModelDesc& draft_model_desc);

    GenerationHandle add_request(uint64_t request_id,
                                 const ov::Tensor& input_ids,
                                 const ov::genai::GenerationConfig& sampling_params,
                                 std::optional<ov::Tensor> token_type_ids = std::nullopt) override;
    GenerationHandle add_request(uint64_t request_id,
                                 const std::string& prompt,
                                 const ov::genai::GenerationConfig& sampling_params) override;

    bool has_non_finished_requests() override;

    void step() override;

    std::vector<EncodedGenerationResult>
    generate(const std::vector<ov::Tensor>& input_ids,
             const std::vector<GenerationConfig>& sampling_params,
             const StreamerVariant& streamer,
             std::optional<std::vector<ov::Tensor>> token_type_ids = std::nullopt) override;

    SpeculativeDecodingMetrics get_speculative_decoding_metrics();
};

}
