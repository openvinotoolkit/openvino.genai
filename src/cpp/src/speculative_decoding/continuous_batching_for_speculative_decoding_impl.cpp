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

void ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::align_all_sequence_len_in_request() {
    for (auto& request : m_requests) {
        auto running_sequences = request->get_running_sequences();
        // do not neeed in case of 1 sequence
        if (running_sequences.size() == 1) {
            continue;
        }
        auto& logit_processor = m_sampler->get_logit_processor(request->get_request_id());
        auto min_generated_len = request->get_num_processed_tokens() + request->get_num_tokens_to_validate() - request->get_prompt_len() + 1;
        for (auto& sequence : running_sequences) {
            auto generated_tokens = sequence->get_generated_ids();
            auto generated_len = generated_tokens.size();
            OPENVINO_ASSERT(generated_len >= min_generated_len);
            for (size_t i = min_generated_len; i < generated_len; ++i) {
                logit_processor.decrease_generated_token_occurance(generated_tokens[i]);
            }
            sequence->remove_last_tokens(generated_len - min_generated_len);
        }
        logit_processor.update_generated_len(min_generated_len);
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

// { min_len_of_prefix, min_length_of_candidate }
std::pair<size_t, size_t>
ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::get_prefix_len(
    const std::vector<Sequence::Ptr>& running_sequences,
    const ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::GeneratedSequences& candidates) {
    size_t min_generated_tokens = std::numeric_limits<size_t>::max(),
           min_candidate_len = std::numeric_limits<size_t>::max();
    for (const auto& running_sequence : running_sequences) {
        const auto& sequence_id = running_sequence->get_grouped_id();
        OPENVINO_ASSERT(candidates.count(sequence_id));

        const auto& candidate_sequence = candidates.at(sequence_id);

        const std::vector<int64_t>& candidate_token_ids = candidate_sequence.token_ids,
                                    running_token_ids = running_sequence->get_generated_ids();

        const size_t candidate_sequence_gen_len = candidate_token_ids.size(),
                     running_sequence_gen_len = running_sequence->get_generated_len();
        
        // to find the len of prefix
        size_t sequence_prefix_len = std::min(candidate_sequence_gen_len, running_sequence_gen_len);
        for (size_t i = 0; i < sequence_prefix_len; ++i) {
            if (candidate_token_ids[i] != running_token_ids[i]) {
                sequence_prefix_len = i;
                break;
            }
        }

        min_generated_tokens = std::min(sequence_prefix_len, min_generated_tokens);
        min_candidate_len = std::min(candidate_sequence_gen_len, min_candidate_len);
    }
    return { min_generated_tokens, min_candidate_len };
}

size_t
remove_tokens_from_sequence(Sequence::Ptr& sequence,
                            size_t min_generated_tokens,
                            LogitProcessor& logit_proccessor) {
    const auto generated_token_ids = sequence->get_generated_ids(); 
    const auto sequence_generated_len = generated_token_ids.size();
    OPENVINO_ASSERT(sequence_generated_len >= min_generated_tokens);

    size_t removed_token_cnt = sequence_generated_len - min_generated_tokens;
    for (size_t i = min_generated_tokens; i < sequence_generated_len; ++i) {
        logit_proccessor.decrease_generated_token_occurance(generated_token_ids[i]);
    }
    sequence->remove_last_tokens(removed_token_cnt);
    return (sequence_generated_len - min_generated_tokens);
}

size_t
insert_tokens_to_sequence(Sequence::Ptr& sequence,
                          const std::vector<int64_t>& token_ids,
                          const std::vector<float>& token_log_probs,
                          LogitProcessor& logit_proccessor,
                          bool is_update_sampler) {
    size_t generated_len = sequence->get_generated_len(), candidate_len = token_ids.size();
    OPENVINO_ASSERT(generated_len <= candidate_len);
    for (size_t i = generated_len; i < candidate_len; ++i) {
        sequence->append_token(token_ids[i], token_log_probs[i]);
        if (is_update_sampler) {
            logit_proccessor.register_new_generated_token(token_ids[i]);
        }
    }
    return (candidate_len - generated_len);
}

size_t
ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::init_request(
    SequenceGroup::Ptr request,
    const ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::GeneratedSequences& candidates) {
    size_t min_candidate_len = std::numeric_limits<size_t>::max();
    for (const auto& candidate_sequence : candidates) {
        min_candidate_len = std::min(candidate_sequence.second.token_ids.size(), min_candidate_len);
    }
    for (const auto& candidate_sequence : candidates) {
        Sequence::Ptr sequence;
        if (candidate_sequence.first == 0) {
            auto running_sequences = request->get_running_sequences();
            OPENVINO_ASSERT(!running_sequences.empty());
            sequence = request->get_running_sequences()[0];
        } else {
            sequence = Sequence::Ptr(new Sequence(candidate_sequence.first));
            sequence->set_status(ov::genai::SequenceStatus::RUNNING);
            request->add_sequence(sequence);
        }
        auto token_ids = candidate_sequence.second.token_ids;
        auto log_probs = candidate_sequence.second.log_probs;
        token_ids.resize(min_candidate_len);
        log_probs.resize(min_candidate_len);

        for (size_t i = 0; i < min_candidate_len; ++i) {
            sequence->append_token(token_ids[i], log_probs[i]);
        }
    }
    return min_candidate_len;
}

ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::UpdateRequestResult
ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::update_request(uint64_t request_id,
                                                                                         const ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::GeneratedSequences& candidates,
                                                                                         bool is_update_sampler) {
    _pull_awaiting_requests();

    ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::UpdateRequestResult result{0, 0};
    for (auto& request : m_requests) {
        if (request_id != request->get_request_id()) {
            continue;
        }

        std::vector<Sequence::Ptr> running_sequences = request->get_running_sequences();
        size_t min_generated_tokens, min_candidate_len;
        if (request->get_context_len() == 0) {
            result.inserted_tokens_cnt = init_request(request, candidates);
            min_generated_tokens = result.inserted_tokens_cnt;
            m_sampler->create_logit_processor(request_id, request->get_sampling_parameters(), request->get_prompt_ids());
            running_sequences = request->get_running_sequences();
        } else {
            auto& logit_processor = m_sampler->get_logit_processor(request_id);
            std::tie(min_generated_tokens, min_candidate_len) = get_prefix_len(running_sequences, candidates);

            for (auto& running_sequence : running_sequences) {
                OPENVINO_ASSERT(candidates.count(running_sequence->get_grouped_id()));

                result.removed_tokens_cnt = remove_tokens_from_sequence(running_sequence, min_generated_tokens, logit_processor);

                auto candidate_sequence = candidates.at(running_sequence->get_grouped_id());
                std::vector<int64_t> candidate_token_ids = candidate_sequence.token_ids;
                std::vector<float> candidate_token_log_probs = candidate_sequence.log_probs;
                candidate_token_ids.resize(min_candidate_len);
                candidate_token_log_probs.resize(min_candidate_len);
                result.inserted_tokens_cnt = insert_tokens_to_sequence(running_sequence, candidate_token_ids, candidate_token_log_probs, logit_processor, is_update_sampler);
            }
            if (is_update_sampler) {
                logit_processor.update_generated_len(min_candidate_len);
            }
        }

        // remove extra tokens
        const size_t num_processed_tokens = request->get_num_processed_tokens(),
                     prompt_len = request->get_prompt_len(),
                     updated_context_len = min_candidate_len + prompt_len;
        if (num_processed_tokens > 0)
            request->update_processed_tokens_num(num_processed_tokens - result.removed_tokens_cnt);
        request->set_num_validated_tokens(result.inserted_tokens_cnt);
        break;
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