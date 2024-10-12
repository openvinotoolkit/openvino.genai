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

std::vector<ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::GeneratedSequence>
ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::get_generated_sequences() {
    _pull_awaiting_requests();
    std::vector<ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::GeneratedSequence> result;
    for (const auto& request : m_requests) {
        const auto request_id = request->get_request_id();
        for (const auto& sequence : request->get_sequences()) {
            auto generated_ids = sequence->get_generated_ids();
            auto log_probs = sequence->get_generated_log_probs();
            result.emplace_back(request_id, sequence->get_grouped_id(), generated_ids, log_probs);
        }
    }
    return result;
}

ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::UpdateSeqResult
ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::update_generated_sequence(
    const ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::GeneratedSequence& candidate_sequence) {
    _pull_awaiting_requests();

    bool is_empty_generated_tokens = false;
    for (auto& request : m_requests) {
        if (candidate_sequence.request_id == request->get_request_id()) {
            bool is_seq_exists = false;
            // todo: iefode: multiseq
            size_t to_remove_tokens = 0, to_insert_tokens = 0;
            for (auto& sequence : request->get_sequences()) {
                if (candidate_sequence.sequence_id == sequence->get_grouped_id()) {
                    is_seq_exists = true;
                    auto present_ids = sequence->get_generated_ids();
                    const auto& candidate_ids = candidate_sequence.token_ids;

                    // remove extra tokens from sequence
                    {
                        auto token_idx = std::min(present_ids.size(), candidate_ids.size());
                        if (token_idx) {
                            while (token_idx-- > 0) {
                                if (present_ids[token_idx] == candidate_ids[token_idx]) {
                                    break;
                                }
                            }
                            to_remove_tokens = present_ids.size() - (token_idx + 1);
                            if (to_remove_tokens > 0) {
                                const auto gen_ids_before = sequence->get_generated_ids();
                                sequence->remove_last_tokens(to_remove_tokens);
                                present_ids = sequence->get_generated_ids();
                                const size_t gen_len_before = gen_ids_before.size(),
                                            gen_len_after = present_ids.size();
                                if (gen_len_after == 0) {
                                    is_empty_generated_tokens = true;
                                }
                                OPENVINO_ASSERT(gen_len_after < gen_len_before);
                                for (size_t i = gen_len_after; i < gen_len_before; ++i) {
                                    m_sampler->update_logit_processor(request->get_request_id(), gen_ids_before[i]);
                                }
                            }
                        }
                    }
                    // insert new tokens to sequence
                    {
                        OPENVINO_ASSERT(candidate_ids.size() >= present_ids.size());
                        const auto& candidate_log_probs = candidate_sequence.log_probs;
                        const size_t start_id = std::min(present_ids.size(), candidate_ids.size()),
                                        stop_id = std::max(present_ids.size(), candidate_ids.size());
                        to_insert_tokens = stop_id - start_id;
                        for (size_t i = start_id; i < stop_id; ++i) {
                            sequence->append_token(candidate_ids[i],  i < candidate_log_probs.size() ? candidate_log_probs[i] : 0.f);
                        }
                    }
                }
                // break;
            }
            if (!is_seq_exists) {
                Sequence::Ptr new_sequence(new Sequence(candidate_sequence.sequence_id));
                const auto& generated_tokens = candidate_sequence.token_ids;
                const auto& generated_log_probs = candidate_sequence.log_probs;
                for (size_t i = 0; i < generated_tokens.size(); ++i) {
                    new_sequence->append_token(generated_tokens[i], generated_log_probs[i]);
                }
                request->add_sequence(new_sequence);
            }
            if (!is_empty_generated_tokens) {
                if (to_remove_tokens > 0) {
                    auto num_processed_tokens = request->get_num_processed_tokens();
                    request->update_processed_tokens_num(num_processed_tokens - to_remove_tokens);
                }
                // to validate tokens/extend kv-cache before generation
                request->set_num_validated_tokens(to_insert_tokens);
            } else if (to_remove_tokens > 0) {
                request->update_processed_tokens_num(request->get_prompt_len());
            }
            return ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::UpdateSeqResult(to_insert_tokens, to_remove_tokens);
        }
    }
    return {0, 0};
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
    std::cout << "iteration_cnt:" << iteration_number << std::endl;

    multistep_timer.end();
}
}