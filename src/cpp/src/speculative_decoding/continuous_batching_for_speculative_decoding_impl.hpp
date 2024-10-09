// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/continuous_batching_pipeline.hpp"

#include "continuous_batching_impl.hpp"

namespace ov::genai {
class ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl : public ContinuousBatchingPipeline::ContinuousBatchingImpl {
    bool m_is_validation_mode_enabled = false;

public:
    ContinuousBatchingForSpeculativeDecodingImpl(ov::Core& core,
                                                 const std::shared_ptr<ov::Model>& model,
                                                 const Tokenizer& tokenizer,
                                                 const DeviceConfig& device_config,
                                                 const SchedulerConfig& scheduler_config,
                                                 const std::string& device,
                                                 const ov::AnyMap& plugin_config,
                                                 bool is_validation_mode_enabled);
    
    void multistep();

    void finish_request(int64_t request_id = -1);

    struct GeneratedSequence {
        uint64_t request_id = 0, sequence_id = 0;
        std::vector<int64_t> token_ids;
        std::vector<float> log_probs;

        GeneratedSequence(uint64_t req_id, uint64_t seq_id, const  std::vector<int64_t>& generated_token_ids, const std::vector<float>& generated_log_probs) :
            request_id(req_id),
            sequence_id(seq_id),
            token_ids(generated_token_ids),
            log_probs(generated_log_probs) {};
    };

    struct UpdateSeqResult {
        size_t to_insert, to_remove;
        UpdateSeqResult(size_t _to_insert = 0, size_t _to_remove = 0) : to_insert(_to_insert), to_remove(_to_remove) {};
    };

    std::vector<GeneratedSequence> get_generated_sequences();
    UpdateSeqResult update_generated_sequence(const GeneratedSequence& new_sequence);

};
}