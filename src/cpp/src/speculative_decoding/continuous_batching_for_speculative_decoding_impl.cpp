// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "continuous_batching_for_speculative_decoding_impl.hpp"

namespace ov::genai {
ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::ContinuousBatchingForSpeculativeDecodingImpl(
    ov::Core& core,
    const std::shared_ptr<ov::Model>& model,
    const Tokenizer& tokenizer,
    const DeviceConfig& device_config,
    const SchedulerConfig& scheduler_config,
    const std::string& device,
    const ov::AnyMap& plugin_config,
    bool is_validation_mode_enabled) {
    m_tokenizer = tokenizer;
    m_is_validation_mode_enabled = is_validation_mode_enabled;
    init(model, scheduler_config, plugin_config,  device_config, core);
}

void ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::finish_request(int64_t request_id) {
    if (request_id == -1) {
        while (!m_requests.empty()) {
            const auto& request = *m_requests.rbegin();
            for (const auto& sequence : request->get_sequences()) {
                m_scheduler->free_sequence(sequence->get_id());
            }
            m_sampler->clear_request_info(request->get_request_id());
            m_requests.pop_back();
        }
    } else {
        for (size_t i = 0; i < m_requests.size(); ++i) {
            auto& request = m_requests[i];
            if (request->get_request_id() != request_id) {
                continue;
            }
            for (const auto& sequence : request->get_sequences()) {
                m_scheduler->free_sequence(sequence->get_id());
            }
            m_sampler->clear_request_info(request->get_request_id());
            m_requests.erase(m_requests.begin() + i);
            break;
        }
    }
}

void ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::align_generated_sequence_len() {
    for (auto& request : m_requests) {
        auto rs = request->get_running_sequences();
        if (rs.size() == 1) {
            continue;
        }
        auto num_gen_tokens = request->get_num_processed_tokens() + request->get_num_tokens_to_validate() - request->get_prompt_len() + 1;
        for (auto& sequence : rs) {
            auto updated_len = sequence->get_generated_len();
            OPENVINO_ASSERT(updated_len >= num_gen_tokens);
            auto a = updated_len - num_gen_tokens;
            if (a > 0) {
                auto b = 0;
            }
            sequence->remove_last_tokens(a);
            // remove from logit proc
        }
        m_sampler->update_logit_processor_gen_len(request->get_request_id(), num_gen_tokens);
    }
}


ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::GeneratedRequests
ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::get_generated_requests() {
    _pull_awaiting_requests();

    ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::GeneratedRequests result;
    for (const auto& request : m_requests) {
        const auto& request_id = request->get_request_id();
        if (!result.count(request_id)) {
            result.insert({request_id, {{}} });
        }
        auto& generated_request = result[request_id];
        for (const auto& sequence : request->get_running_sequences()) {
            const auto& sequence_id = sequence->get_grouped_id();
            OPENVINO_ASSERT(!generated_request.count(sequence_id));
            generated_request.insert({{sequence_id, { sequence->get_generated_ids(), sequence->get_generated_log_probs() } }});
        }
    }
    return result;
}

ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::UpdateRequestResult
ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::update_request(uint64_t request_id,
                                                                                         const ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::GeneratedSequences& candidates) {
    _pull_awaiting_requests();

    ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::UpdateRequestResult result{0, 0};
    for (auto& request : m_requests) {
        if (request_id != request->get_request_id()) {
            continue;
        }
        size_t min_processed_tokens = std::numeric_limits<size_t>::max(),
               min_candidate_len = std::numeric_limits<size_t>::max(),
               prompt_len = request->get_prompt_len();
        // size_t num_proccessed_tokens 
        for (auto& running_sequence : request->get_running_sequences()) {
            const auto& sequence_id = running_sequence->get_grouped_id();
            if (candidates.count(sequence_id) == 0) {
                continue;
            }
            OPENVINO_ASSERT(candidates.count(sequence_id));

            const auto& candidate_sequence = candidates.at(sequence_id);

            const std::vector<int64_t>& candidate_token_ids = candidate_sequence.token_ids,
                                        running_token_ids = running_sequence->get_generated_ids();

            const size_t candidate_sequence_gen_len = candidate_token_ids.size(),
                         running_sequence_gen_len = running_sequence->get_generated_len();
            
            size_t min_generated_len = std::min(candidate_sequence_gen_len, running_sequence_gen_len);

            for (size_t i = 0; i < min_generated_len; ++i) {
                if (candidate_token_ids[i] != running_token_ids[i]) {
                    min_generated_len = i;
                    break;
                }
            }

            min_processed_tokens = std::min(min_generated_len, min_processed_tokens);
            min_candidate_len = std::min(candidate_sequence_gen_len, min_candidate_len);
        }
        // remove extra tokens
        if (request->get_num_processed_tokens() >= min_processed_tokens + prompt_len) {
            result.removed_tokens_cnt = request->get_num_processed_tokens() - (min_processed_tokens + prompt_len) + 1;
            auto updated_num_processed_tokens = request->get_num_processed_tokens() - result.removed_tokens_cnt;
            request->update_processed_tokens_num(updated_num_processed_tokens);
            
            for (auto& running_sequence : request->get_running_sequences()) {
                if (candidates.count(running_sequence->get_grouped_id()) == 0) {
                    continue;
                }
                auto running_sequence_gen_len = running_sequence->get_generated_len();
                OPENVINO_ASSERT(running_sequence_gen_len >= min_processed_tokens);
                const auto running_token_ids = running_sequence->get_generated_ids();
                size_t updated_seq_lenght = running_sequence_gen_len - min_processed_tokens;
                for (size_t i = min_processed_tokens; i < running_sequence_gen_len; ++i) {
                    m_sampler->remove_token_from_logit_processor(request_id, running_token_ids[i]);
                }
                running_sequence->remove_last_tokens(updated_seq_lenght);
                m_sampler->update_logit_processor_gen_len(request_id, min_processed_tokens);
            }   
        }
        // add tokens from candidates
        if (min_candidate_len > min_processed_tokens) {
            result.inserted_tokens_cnt = min_candidate_len - min_processed_tokens;
            
            for (auto& running_sequence : request->get_running_sequences()) {
                if (candidates.count(running_sequence->get_grouped_id()) == 0) {
                    continue;
                }
                const auto& candidate_sequence = candidates.at(running_sequence->get_grouped_id());
                const std::vector<int64_t>& candidate_token_ids = candidate_sequence.token_ids;
                const std::vector<float>& candidate_token_log_probs = candidate_sequence.log_probs;

                for (size_t i = min_processed_tokens; i < min_candidate_len; ++i) {
                    running_sequence->append_token(candidate_token_ids[i], candidate_token_log_probs[i]);
                    m_sampler->insert_token_to_logit_processor(request_id, candidate_token_ids[i]);
                }
            }
            request->set_num_validated_tokens(result.inserted_tokens_cnt);
            // if (min_processed_tokens > 0)
            //     m_sampler->update_logit_processor_gen_len(request_id, min_processed_tokens + result.inserted_tokens_cnt);
        }
    }

    return result;
}

void ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::multistep() {
    static ManualTimer multistep_timer("multistep()");
    multistep_timer.start();

    for (const auto& request : m_requests) {
        request->pause_generation(false);
    }

    size_t iteration_number = 0;
    // cycle to generate several tokens per one iteration for speculative decoding case
    bool to_generate = true;
    while (to_generate) {
        iteration_number += 1;
        step();

        to_generate = false;
        for (auto& request : m_requests) {
            const auto& sampling_params = request->get_sampling_parameters();
            if (!sampling_params.is_speculative_decoding()) {
                to_generate = false;
                break;
            }
            if (sampling_params.num_assistant_tokens_schedule == NumAssistatantTokensScheduleType::CONSTANT &&
                sampling_params.num_assistant_tokens <= iteration_number) {
                request->pause_generation(true);
            }
            to_generate |= request->can_generate_tokens();
        }
    }

    multistep_timer.end();
}
}