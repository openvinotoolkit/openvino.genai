// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "pipeline_impl.hpp"
#include <numeric>

namespace ov::genai {
ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::ContinuousBatchingForSpeculativeDecodingImpl(
    const std::shared_ptr<ov::Model>& model,
    const Tokenizer& tokenizer,
    const GenerationConfig& generation_config,
    const SchedulerConfig& scheduler_config,
    const std::string& device,
    const ov::AnyMap& plugin_config,
    bool is_validation_mode_enabled) {
    m_tokenizer = tokenizer;
    m_generation_config = generation_config;
    if (m_generation_config.assistant_confidence_threshold == 0.f) {
        if (m_generation_config.num_assistant_tokens == 0) {
            m_generation_config.num_assistant_tokens = default_num_assistant_tokens;
        }
    }
    m_is_validation_mode_enabled = is_validation_mode_enabled;
    initialize_pipeline(model, scheduler_config, device, plugin_config);
}

ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::ContinuousBatchingForSpeculativeDecodingImpl(
    const std::shared_ptr<ov::Model>& model,
    std::shared_ptr<InputsEmbedder> inputs_embedder,
    const Tokenizer& tokenizer,
    const GenerationConfig& generation_config,
    const SchedulerConfig& scheduler_config,
    const std::string& device,
    const ov::AnyMap& plugin_config,
    bool is_validation_mode_enabled)
    : ContinuousBatchingForSpeculativeDecodingImpl(model,
                                                   tokenizer,
                                                   generation_config,
                                                   scheduler_config,
                                                   device,
                                                   plugin_config,
                                                   is_validation_mode_enabled) {
    m_inputs_embedder = inputs_embedder;
    // Note: set_inputs_embedder also sets the embedding model internally.
    m_model_runner->set_inputs_embedder(inputs_embedder);
    m_model_input_type = ModelInputType::EMBEDDINGS;
}

void
ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::finish_request(SequenceGroup::Ptr request) {
    for (const auto& sequence: request->get_sequences()) {
        if (m_scheduler->has_block_table(sequence->get_id())) {
            m_scheduler->free_sequence(sequence->get_id());
        }
    }
    m_sampler->clear_request_info(request->get_request_id());
    request->set_generation_status(GenerationStatus::STOP);
}

void ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::finish_request(int64_t request_id) {
    auto it = m_requests.begin();
    while (it != m_requests.end()) {
        auto& request = *it;
        if (request->get_request_id() != request_id && request_id != -1) {
            it++;
            continue;
        }
        finish_request(request);
        m_requests.erase(it);
        it = request_id == -1 ? m_requests.begin() : m_requests.end();
    }
    if (request_id == -1) {
        OPENVINO_ASSERT(m_requests.empty());
    }
}

GeneratedRequests
ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::get_generated_requests() {
    GeneratedRequests result;
    for (const auto& request : m_requests) {
        const auto& request_id = request->get_request_id();
        if (!result.count(request_id)) {
            result.insert({request_id, {{}} });
        }
        auto& generated_request = result[request_id];
        for (const auto& sequence : request->get_running_sequences()) {
            const auto& sequence_id = sequence->get_grouped_id();
            OPENVINO_ASSERT(!generated_request.count(sequence_id));
            generated_request.insert({{sequence_id, { sequence->get_generated_ids(), sequence->get_generated_log_probs(), sequence->get_hidden_state(), sequence->get_tree_metadata() } }});
        }
    }
    return result;
}

ov::Tensor select_rows_by_indices(const ov::Tensor& tensor, const std::vector<size_t>& indices) {
    OPENVINO_ASSERT(!indices.empty(), "Indices vector is empty");

    const auto& input_shape = tensor.get_shape();
    OPENVINO_ASSERT(input_shape.size() >= 1, "Tensor must have at least 1 dimension");

    // Optimization: Check if indices form a contiguous range [0, 1, 2, ..., n-1]
    // This allows zero-copy slice for common truncation scenarios
    bool is_contiguous = true;
    for (size_t i = 0; i < indices.size(); ++i) {
        if (indices[i] != i) {
            is_contiguous = false;
            break;
        }
    }

    // Fast path: contiguous indices - use zero-copy ROI slice
    if (is_contiguous) {
        OPENVINO_ASSERT(indices.size() <= input_shape[0],
                       "Contiguous index range [0..", indices.size(),
                       ") exceeds tensor dimension 0 size (", input_shape[0], ")");
        auto [start, end] = ov::genai::utils::make_roi(input_shape, 0, 0, indices.size());
        return ov::Tensor(tensor, start, end);
    }

    ov::Shape output_shape = input_shape;
    output_shape[0] = indices.size();
    ov::Tensor result(tensor.get_element_type(), output_shape);

    for (size_t i = 0; i < indices.size(); ++i) {
        const size_t src_index = indices[i];
        OPENVINO_ASSERT(src_index < input_shape[0],
                       "Index ", src_index, " is out of bounds for tensor dimension 0 (size ", input_shape[0], ")");

        auto [src_start, src_end] = ov::genai::utils::make_roi(input_shape, 0, src_index, src_index + 1);
        auto [dst_start, dst_end] = ov::genai::utils::make_roi(output_shape, 0, i, i + 1);

        ov::Tensor src_slice(tensor, src_start, src_end);
        ov::Tensor dst_slice(result, dst_start, dst_end);
        src_slice.copy_to(dst_slice);
    }

    return result;
}
// { min_len_of_prefix, min_length_of_candidate }
std::pair<size_t, size_t>
get_prefix_len(
    const std::vector<Sequence::Ptr>& running_sequences,
    const GeneratedSequences& candidates,
    bool need_adjust = false) {
    size_t min_generated_tokens = std::numeric_limits<size_t>::max(),
           min_candidate_len = std::numeric_limits<size_t>::max();
    for (const auto& running_sequence : running_sequences) {
        const auto& sequence_id = running_sequence->get_grouped_id();
        if (!candidates.count(sequence_id)) {
            continue;
        }

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
        // adjust the len of prefix in tree mode
        // handle situation of token match but position mismatch
        if (need_adjust) {
            auto& tree_metadata = running_sequence->get_tree_metadata();
            const auto& position_ids = tree_metadata.tree_position_ids;
            const auto prev_generated_len = running_sequence->get_generated_len() - position_ids.size();
            auto prefix_len_from_last_generated = sequence_prefix_len - prev_generated_len;
            for (size_t i = 0; i < prefix_len_from_last_generated; ++i) {
                if (position_ids[i] != i) {
                    sequence_prefix_len = prev_generated_len + i;
                    break;
                }
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


// `is_init_all_sequences_in_request` is flag to enable initialization of all sequences in case of `num_return_sequences > 1`.
// Only first sequence from all will be initialized if flag is set to `false` state.
// This approach helped to process prompt once in speculative decoding multisequence case.
size_t
init_request(
    SequenceGroup::Ptr request,
    const GeneratedSequences& candidates,
    LogitProcessor& logit_processor,
    bool is_update_logit_processor,
    bool is_init_all_sequences_in_request = false) {
    OPENVINO_ASSERT(request->get_sampling_parameters().is_assisting_generation(),
                    "Speculative decoding should have initialized options `assistant_confidence_threshold` xor `num_assistant_tokens` in `GenerationConfig`.");
    if (candidates.begin()->second.token_ids.empty() && !is_init_all_sequences_in_request) {
        return 0;
    }
    size_t min_candidate_len = std::numeric_limits<size_t>::max();
    if (is_init_all_sequences_in_request) {
        for (const auto& candidate_sequence : candidates) {
            min_candidate_len = std::min(candidate_sequence.second.token_ids.size(), min_candidate_len);
        }
    } else {
        // place only one token to first sequence in case of multisequence generation.
        // Left sequences in request will be initialized in sampler and validated after (only one token).
        min_candidate_len = request->get_sampling_parameters().num_return_sequences == 1 ? candidates.begin()->second.token_ids.size() : 1;
    }
    for (const auto& candidate_sequence : candidates) {
        Sequence::Ptr sequence;
        if (is_init_all_sequences_in_request && candidate_sequence.first > 0) {
            sequence = Sequence::create(candidate_sequence.first);
            sequence->set_status(ov::genai::SequenceStatus::RUNNING);
            request->add_sequence(sequence);
        } else {
            auto running_sequences = request->get_running_sequences();
            OPENVINO_ASSERT(!running_sequences.empty());
            sequence = request->get_running_sequences()[0];
        }
        auto token_ids = candidate_sequence.second.token_ids;
        auto log_probs = candidate_sequence.second.log_probs;
        token_ids.resize(min_candidate_len);
        log_probs.resize(min_candidate_len);

        for (size_t i = 0; i < min_candidate_len; ++i) {
            sequence->append_token(token_ids[i], log_probs[i]);
            if (is_update_logit_processor) {
                logit_processor.register_new_generated_token(token_ids[i]);
                logit_processor.update_generated_len(sequence->get_generated_len());
            }
        }

        if (!is_init_all_sequences_in_request) {
            break;
        }
    }
    return min_candidate_len;
}

UpdateRequestResult 
ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::init_request_by_candidate(
    uint64_t request_id,
    const GeneratedSequences& candidates) {
    for (auto& request : m_requests) {
        if (request->get_request_id() != request_id) {
            continue;
        }
        
        UpdateRequestResult result;
        m_sampler->create_logit_processor(request_id, request->get_sampling_parameters(), request->get_prompt_ids());
        auto& logit_processor = m_sampler->get_logit_processor(request_id);
        result.inserted_tokens_cnt = init_request(request, candidates, logit_processor, true, true);
        request->set_num_validated_tokens(result.inserted_tokens_cnt);
        return result;
    }
    return {0, 0};
}

UpdateRequestResult
ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::update_request(uint64_t request_id,
                                                                                         const GeneratedSequences& candidates,
                                                                                         bool is_update_logit_processor) {
    UpdateRequestResult result{0, 0};
    for (auto& request : m_requests) {
        if (request_id != request->get_request_id()) {
            continue;
        }

        std::vector<Sequence::Ptr> running_sequences = request->get_running_sequences();
        OPENVINO_ASSERT(running_sequences.size() > 0);
        size_t min_generated_tokens, min_candidate_len;
        int num_tokens_needs_kv_update = -1;
        if (running_sequences.front()->get_generated_len() == 0 && !request->get_num_tokens_to_validate()) {
            m_sampler->create_logit_processor(request_id, request->get_sampling_parameters(), request->get_prompt_ids());
            auto& logit_processor = m_sampler->get_logit_processor(request_id);
            result.inserted_tokens_cnt = init_request(request, candidates, logit_processor, is_update_logit_processor);
            min_generated_tokens = result.inserted_tokens_cnt;
            running_sequences = request->get_running_sequences();
            min_candidate_len = result.inserted_tokens_cnt;
            if (eagle_mode_enabled && !m_is_validation_mode_enabled) {
                m_model_runner->set_initial_hidden_state(request_id,
                                                     candidates.begin()->second.hidden_states);
            }
        } else {
            // update existing sequences by the candidates
            auto& logit_processor = m_sampler->get_logit_processor(request_id);
            // in tree search case, we only consider the prefix with the same token ids and same positions as valid prefix
            auto need_prefix_adjustment = request->get_sampling_parameters().is_tree_search() && !m_is_validation_mode_enabled;
            std::tie(min_generated_tokens, min_candidate_len) = get_prefix_len(running_sequences, candidates, need_prefix_adjustment);
            OPENVINO_ASSERT(running_sequences.size() == 1);
            auto running_sequence = running_sequences[0];
            {
                if (!candidates.count(running_sequence->get_grouped_id())) {
                    continue;
                }

                result.removed_tokens_cnt = remove_tokens_from_sequence(running_sequence, min_generated_tokens, logit_processor);
                auto candidate_sequence = candidates.at(running_sequence->get_grouped_id());
                std::vector<int64_t> candidate_token_ids = candidate_sequence.token_ids;
                std::vector<float> candidate_token_log_probs = candidate_sequence.log_probs;
                candidate_token_ids.resize(min_candidate_len);
                candidate_token_log_probs.resize(min_candidate_len);
                result.inserted_tokens_cnt = insert_tokens_to_sequence(running_sequence, candidate_token_ids, candidate_token_log_probs, logit_processor, is_update_logit_processor);
                // handle hidden states for eagle mode
                if (eagle_mode_enabled && !m_is_validation_mode_enabled && result.inserted_tokens_cnt > 0) { // update hidden states for draft model
                    const auto& hidden_state = candidate_sequence.hidden_states;
                    const size_t current_len = hidden_state.get_shape()[0];

                    std::vector<size_t> indices;
                    if (!request->get_sampling_parameters().is_tree_search()) {
                        OPENVINO_ASSERT(result.removed_tokens_cnt < current_len,
                                       "Cannot remove ", result.removed_tokens_cnt,
                                       " tokens from hidden state with length ", current_len);
                        const size_t keep_len = current_len - result.removed_tokens_cnt;
                        indices.resize(keep_len);
                        std::iota(indices.begin(), indices.end(), 0);
                    } else {
                        indices = candidate_sequence.tree_metadata.validated_indices;
                        auto total_candidates_count = candidate_sequence.tree_metadata.tree_position_ids.size();
                        result.removed_tokens_cnt = total_candidates_count - indices.size() + 1;
                    }

                    OPENVINO_ASSERT(!indices.empty(), "indices cannot be empty for hidden state selection");

                    // Select hidden states based on computed indices
                    ov::Tensor selected_hidden_state = select_rows_by_indices(hidden_state, indices);
                    OPENVINO_ASSERT(selected_hidden_state.get_shape()[0] >= 1,
                                   "Unexpected hidden state shape from the main model.");
                    m_model_runner->set_initial_hidden_state(request_id, selected_hidden_state);

                    // Calculate the number of tokens that need KV cache re-generation in draft model
                    // Safe cast: we know indices is not empty (asserted above)
                    num_tokens_needs_kv_update = static_cast<int>(indices.size()) - 1;
                }
                // update hidden states for main model in validation mode
                if (eagle_mode_enabled && m_is_validation_mode_enabled && result.inserted_tokens_cnt > 0) {
                    // retrieve eagle metadata
                    auto& tree_metadata = candidate_sequence.tree_metadata;
                    tree_metadata.validated_indices = running_sequence->get_tree_metadata().validated_indices;
                    running_sequence->set_tree_metadata(tree_metadata);
                }
            }
            // we should update a logit processor just for draft model to generate the same tokens
            // logit processors of main model will be updated in sampler while validation mode
            if (is_update_logit_processor) {
                logit_processor.update_generated_len(min_candidate_len);
            }
        }

        // update request context information to provide correct scheduling phase
        const size_t num_processed_tokens = request->get_num_processed_tokens(),
                     prompt_len = request->get_prompt_len(),
                     updated_context_len = min_candidate_len + prompt_len,
                     max_new_tokens = request->get_max_new_tokens();
        size_t generated_len = request->get_context_len() >= request->get_prompt_len() ? request->get_context_len() - request->get_prompt_len() + 1 : 0;

        // Update processed tokens count based on KV cache update requirements
        if (generated_len > 0) {
            if (num_tokens_needs_kv_update > 0) {
                // rewind to stable KV position accounting for tree structure
                request->update_processed_tokens_num(num_processed_tokens - result.removed_tokens_cnt + 1 - num_tokens_needs_kv_update);
            } else if (result.removed_tokens_cnt > 0) {
                // Fast draft or sequential EAGLE mode: rewind to last accepted token
                request->update_processed_tokens_num(num_processed_tokens - result.removed_tokens_cnt + 1);
            }
        }

        // Update validated token count for next generation phase
        if (num_tokens_needs_kv_update >= 0) {
            // Generation stage: use KV update count
            request->set_num_validated_tokens(num_tokens_needs_kv_update);
        } else if (result.inserted_tokens_cnt > 0 && result.removed_tokens_cnt == 0) {
            // Validation stage or all inserted tokens were accepted
            request->set_num_validated_tokens(result.inserted_tokens_cnt);
        }
        // to pause `draft_model` generation in case of `generated_len >= max_new_tokens - 1` to generate last token by `main_model`
        // Start `draft_model` generation after the first `main_model` generation is finished. There are two scenarios:
        // 1. When `main_model` generates a new token, in which case `draft_model` naturally starts its generation.
        // 2. When `main_model` does not generate a new token, which usually happens when processing a portion of prompt (we can
        //    slice prompt into chunks when dynamic_split_fuse is enabled),
        //    in this case, `draft_model` can also begin processing the same portion of prompt.
        if (!m_is_validation_mode_enabled) {
            bool pause_gen_status = false;
            generated_len -= result.removed_tokens_cnt;
            generated_len +=  eagle_mode_enabled ? 1 : result.inserted_tokens_cnt;
            if (generated_len >= max_new_tokens - 1 || generated_len != 0 && result.inserted_tokens_cnt == 0) {
                pause_gen_status = true;
            }
            request->pause_generation(pause_gen_status);
        }
        break;
    }
    return result;
}

bool ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::is_requests_empty() {
    return m_requests.empty();
}

size_t ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::get_processed_tokens_per_iteration() {
    return m_batch_size;
}

void
ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::pull_awaiting_requests(bool is_pause_request) {
    std::lock_guard<std::mutex> lock{m_awaiting_requests_mutex};
    if (is_pause_request) {
        for (auto& awaiting_request : m_awaiting_requests) {
            awaiting_request->pause_generation(true);
        }
    }
    m_requests.insert(m_requests.end(), m_awaiting_requests.begin(), m_awaiting_requests.end());
    m_awaiting_requests.clear();
}

void ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::multistep() {
    bool to_generate = true;
    size_t generated_tokens_cnt = 0;

    // cycle to generate several tokens per one iteration for speculative decoding case
    while (to_generate) {
        generated_tokens_cnt++;

        const auto step_start = std::chrono::steady_clock::now();
        step();
        const auto step_end = std::chrono::steady_clock::now();
        const auto generation_duration = PerfMetrics::get_microsec(step_end - step_start);

        const auto num_generated_tokens = get_processed_tokens_per_iteration();
        auto pipeline_metrics = get_metrics();
        if (num_generated_tokens > 0) {
            raw_perf_metrics.m_durations.emplace_back(generation_duration);
            raw_perf_metrics.m_inference_durations[0] += MicroSeconds(pipeline_metrics.inference_duration);
            raw_perf_metrics.m_batch_sizes.emplace_back(num_generated_tokens);
        }

        if (eagle_mode_enabled)
            m_model_runner->enable_hidden_state_import(false);
        to_generate = false;
        for (auto& request : m_requests) {
            const auto& sampling_params = request->get_sampling_parameters();
            if (!sampling_params.is_assisting_generation()) {
                // generate only one token in case of non speculative decoding
                request->pause_generation(true);
            } else if (!sampling_params.is_tree_search() && request->get_num_processed_tokens() >= request->get_prompt_len() &&
                (request->get_num_processed_tokens() - request->get_prompt_len() + 1) >= request->get_max_new_tokens() - 1) {
                request->pause_generation(true);
            } else if (request->get_num_processed_tokens() == 0 && sampling_params.num_return_sequences > 1) {
                request->pause_generation(true);
            } else if (sampling_params.num_assistant_tokens <= generated_tokens_cnt && sampling_params.assistant_confidence_threshold == 0.f) {
                request->pause_generation(true);
            } else if (request->get_max_new_tokens() == 0) {
                request->pause_generation(true);
            } else if (request->get_num_processed_tokens() == request->get_prompt_len()) {
                request->pause_generation(true);
            } else if (!sampling_params.is_tree_search() && is_stop_token_id_hit_in_sequence_group(request, sampling_params.stop_token_ids)) {
                // in branching tree mode, we ignore the stop token, as we may have several candidates at the same tree layer
                request->pause_generation(true);
            } else if (sampling_params.is_tree_search() && sampling_params.tree_depth <= generated_tokens_cnt) {
                // ensure a stable tree structure
                request->pause_generation(true);
            }
            to_generate |= request->can_generate_tokens();
        }
    }
    if (eagle_mode_enabled)
        m_model_runner->enable_hidden_state_import(true);
}

void ContinuousBatchingPipeline::ContinuousBatchingForEagle3DecodingImpl::collect_block_update_info(
    const GeneratedRequests& main_generated_requests,
    std::vector<int32_t>& block_update_indices,
    std::vector<int32_t>& block_update_begins) const {
    block_update_indices.clear();
    block_update_begins.clear();
    block_update_begins.resize(main_generated_requests.size() + 1, 0);

    std::map<size_t, std::map<size_t, size_t>> remap_indices;
    size_t total_indices_to_remap = 0;
    size_t i = 0;
    for (auto& request_entry : main_generated_requests) {
        const auto request_id = request_entry.first;
        const auto& sequences = request_entry.second;

        auto seq_group_it = std::find_if(m_requests.begin(), m_requests.end(),
                                         [request_id](const SequenceGroup::Ptr& sg) { return sg->get_request_id() == request_id; });
        if (seq_group_it == m_requests.end()) {
            block_update_begins[i + 1] = static_cast<int32_t>(total_indices_to_remap);
            i++;
            continue;
        }

        OPENVINO_ASSERT(sequences.size() == 1);
        for (const auto& seq_entry : sequences) {
            const auto& validated_indices = seq_entry.second.tree_metadata.validated_indices;
            if (validated_indices.empty()) {
                continue;
            }

            const size_t prev_num_processed_tokens =
                (*seq_group_it)->get_num_processed_tokens() - validated_indices.size();
            size_t index = prev_num_processed_tokens;

            for (const auto& idx : validated_indices) {
                const size_t src_idx = prev_num_processed_tokens + idx;
                if (src_idx == index) {
                    ++index;
                    continue;
                }

                remap_indices[i].emplace(src_idx, index);  // src idx, dst idx
                ++index;
            }

            const auto remap_it = remap_indices.find(i);
            if (remap_it != remap_indices.end() && !remap_it->second.empty()) {
                total_indices_to_remap += remap_it->second.size();
            }
        }
        block_update_begins[i + 1] = static_cast<int32_t>(total_indices_to_remap);
        i++;
    }

    if (!remap_indices.empty()) {
        block_update_indices.resize(2 * total_indices_to_remap);  // each remap index has src and dst idx
        size_t filled_count = 0;
        for (const auto& [_, indices_map] : remap_indices) {
            for (const auto& [src_idx, dst_idx] : indices_map) {
                const size_t tensor_offset = 2 * filled_count;
                block_update_indices[tensor_offset] = static_cast<int32_t>(src_idx);
                block_update_indices[tensor_offset + 1] = static_cast<int32_t>(dst_idx);
                ++filled_count;
            }
        }
    }
}
}
