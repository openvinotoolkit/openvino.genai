// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/continuous_batching_pipeline.hpp"
#include "continuous_batching_impl.hpp"

namespace ov::genai {

class SpeculativeDecodingMetrics {
    using AcceptanceRate = std::vector<float>;
    // { request_id, acceptance_rate }
    std::map<int64_t, AcceptanceRate> m_acceptance_rate;

public:
    float get_avg_acceptance_rate(int64_t request_id = 1) {
        float avg_acceptance_rate = 0.f;
        if (request_id != -1) {
            size_t total_iteration_cnt = 0;
            for (const auto& acceptance_rate : m_acceptance_rate) {
                avg_acceptance_rate += std::accumulate(acceptance_rate.second.begin(), acceptance_rate.second.end(), 0);
                total_iteration_cnt += acceptance_rate.second.size();
            }
            avg_acceptance_rate /= total_iteration_cnt;
        } else {
            OPENVINO_ASSERT(m_acceptance_rate.count(request_id));
            const auto& acceptance_rate = m_acceptance_rate[request_id];
            avg_acceptance_rate = std::accumulate(acceptance_rate.begin(), acceptance_rate.end(), 0);
            avg_acceptance_rate /= acceptance_rate.size();
        }
        return avg_acceptance_rate;
    }

    void update_acceptance_rate(int64_t request_id, float acceptance_rate) {
        if (m_acceptance_rate.count(request_id)) {
            m_acceptance_rate[request_id].push_back(acceptance_rate);
        } else {
            m_acceptance_rate.insert({{ request_id, std::vector<float>{acceptance_rate} }});
        }
    }

    size_t get_iteration_number(int64_t request_id) {
        OPENVINO_ASSERT(m_acceptance_rate.count(request_id));
        return m_acceptance_rate[request_id].size();
    }

};

class ContinuousBatchingPipeline::SpeculativeDecodingImpl : public ContinuousBatchingPipeline::ImplInterface {
protected:
    std::shared_ptr<ContinuousBatchingImpl> m_main_pipeline, m_draft_pipeline;
    // left generation length per request {request_id, len}
    std::map<int64_t, size_t> m_left_gen_len;
    SpeculativeDecodingMetrics m_sd_metrics;
    
public:
    SpeculativeDecodingImpl(const std::string& main_models_path,
                            const std::string& draft_models_path,
                            const SchedulerConfig& scheduler_config,
                            const std::string& device,
                            const ov::AnyMap& plugin_config,
                            const ov::AnyMap& tokenizer_config = {});

    GenerationHandle add_request(uint64_t request_id,
                                 const ov::Tensor& input_ids,
                                 ov::genai::GenerationConfig sampling_params) override;
    GenerationHandle add_request(uint64_t request_id,
                                 const std::string& prompt,
                                 ov::genai::GenerationConfig sampling_params) override;

    bool has_non_finished_requests() override;

    void step() override;

    std::vector<EncodedGenerationResult>
    generate(const std::vector<ov::Tensor>& input_ids,
             const std::vector<GenerationConfig>& sampling_params,
             const StreamerVariant& streamer) override;

    SpeculativeDecodingMetrics get_speculative_decoding_metrics() {
        return SpeculativeDecodingMetrics();
    };
};

}