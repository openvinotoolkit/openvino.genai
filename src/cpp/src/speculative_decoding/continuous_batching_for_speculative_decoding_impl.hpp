// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/continuous_batching_pipeline.hpp"

#include "continuous_batching_impl.hpp"

namespace ov::genai {
class ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl : public ContinuousBatchingPipeline::ContinuousBatchingImpl {


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

    void align_all_sequence_len_in_request();

    struct GeneratedSequence {
        std::vector<int64_t> token_ids;
        std::vector<float> log_probs;

        GeneratedSequence(const std::vector<int64_t>& generated_token_ids,
                          const std::vector<float>& generated_log_probs) :
            token_ids(generated_token_ids),
            log_probs(generated_log_probs) {};
    };

    struct UpdateRequestResult {
        size_t inserted_tokens_cnt, removed_tokens_cnt;

        UpdateRequestResult(size_t to_insert = 0, size_t to_remove = 0) :
            inserted_tokens_cnt(to_insert),
            removed_tokens_cnt(to_remove) {};
    };

    using GeneratedSequences = std::map<uint64_t, GeneratedSequence>;
    using GeneratedRequests = std::map<uint64_t, GeneratedSequences>;

    GeneratedRequests get_generated_requests();
    UpdateRequestResult update_request(uint64_t request_id, const GeneratedSequences& candidates, bool is_update_sampler);

protected:
std::pair<size_t, size_t>
get_prefix_len(const std::vector<Sequence::Ptr>& running_sequences,
               const ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::GeneratedSequences& candidates);

// init request in case it was not started to be generated
size_t init_request(SequenceGroup::Ptr request, const GeneratedSequences& candidates);
};
}