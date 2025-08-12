// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include "utils.hpp"
#include "lm_encoding.hpp"
#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/streamer_base.hpp"

namespace {

/**
 * Set position ids tensor data for next token inference based on provided attention mask
 * Supports multi batch
 * Supports sparse attention_mask
 */
void update_position_ids(ov::Tensor&& position_ids, const ov::Tensor&& attention_mask) {
    const size_t batch_size = attention_mask.get_shape().at(0);
    const size_t sequence_length = attention_mask.get_shape().at(1);
    position_ids.set_shape({batch_size, 1});

    for (size_t batch = 0; batch < batch_size; batch++) {
        auto mask_start = attention_mask.data<int64_t>() + batch * sequence_length;
        position_ids.data<int64_t>()[batch] = std::accumulate(mask_start, mask_start + sequence_length - 1, 0);
    }
}

void update_3d_position_ids(ov::Tensor&& position_ids, const ov::Tensor& attention_mask, const int64_t rope_delta) {
    const size_t batch_size = attention_mask.get_shape().at(0);
    const size_t sequence_length = attention_mask.get_shape().at(1);
    const size_t thw_dim_size = 3;

    position_ids.set_shape({thw_dim_size, batch_size, 1});
    int64_t* position_ids_data = position_ids.data<int64_t>();

    int64_t pos_id = static_cast<int64_t>(sequence_length) - 1 + rope_delta;

    for (size_t batch = 0; batch < batch_size; batch++) {
        for (size_t dim = 0; dim < thw_dim_size; ++dim) {
            position_ids_data[dim * batch_size + batch] = pos_id;
        }
    }
}

void update_attention_mask_with_beams(ov::Tensor&& attention_mask, std::vector<int32_t> next_beams) {
    ov::Tensor original_mask{ov::element::i64, attention_mask.get_shape()};
    ov::Shape original_shape = original_mask.get_shape();
    attention_mask.copy_to(original_mask);

    ov::Shape new_shape{next_beams.size(), original_mask.get_shape().at(1) + 1};
    attention_mask.set_shape(new_shape);

    for (size_t beam_id = 0; beam_id < next_beams.size(); beam_id++) {
        const size_t original_prompt_offset = next_beams.at(beam_id) * original_shape.at(1);
        const size_t result_prompt_offset = beam_id * new_shape.at(1);

        int64_t* dest = attention_mask.data<int64_t>() + result_prompt_offset;
        const int64_t* src = original_mask.data<int64_t>() + original_prompt_offset;

        std::memcpy(dest, src, original_shape.at(1) * sizeof(int64_t));
        attention_mask.data<int64_t>()[result_prompt_offset + new_shape.at(1) - 1] = 1;
    }
}
}

namespace ov {
namespace genai {

ov::genai::utils::GenerationFinishInfo get_lm_encoded_results(
    ov::InferRequest& m_llm,
    const ov::Tensor& input_ids,
    const ov::Tensor& attention_mask,
    const std::shared_ptr<StreamerBase>& streamer_ptr,
    Sampler& sampler,
    std::vector<SequenceGroup::Ptr> sequence_groups,
    std::optional<ov::Tensor> position_ids,
    std::optional<ov::Tensor> token_type_ids,
    utils::KVCacheState& kv_cache_state,
    EmbeddingsModel::Ptr m_embedding,
    std::optional<int64_t> rope_delta,
    const size_t max_kv_cache_size
) {
    std::vector<GenerationHandle> generations;
    for (SequenceGroup::Ptr sequence_group : sequence_groups) {
        generations.push_back(std::make_shared<GenerationHandleImpl>(sequence_group->get_generation_stream(), sequence_group->get_sampling_parameters()));
    }

    auto active_sequence_groups{sequence_groups};

    auto stream_generated_tokens = [&streamer_ptr, &generations, &active_sequence_groups]() {
        GenerationHandle& handle = generations.at(0);
        if (streamer_ptr && handle->can_read()) {
            std::unordered_map<uint64_t, GenerationOutput> generation_outputs = handle->read();
            OPENVINO_ASSERT(generation_outputs.size() <= 1);
            if (!generation_outputs.empty()) {
                auto streaming_status = streamer_ptr->write(generation_outputs.begin()->second.generated_ids);
                if (streaming_status != ov::genai::StreamingStatus::RUNNING) {
                    streaming_status == ov::genai::StreamingStatus::CANCEL ? handle->cancel() : handle->stop();
                }
            }
        }
    };

    auto free_non_running_requests = [&streamer_ptr, &generations, &active_sequence_groups, &max_kv_cache_size]() {
        for (auto& sg : active_sequence_groups) {
            for (auto& seq : sg->get_sequences()) {
                if (sg->get_prompt_len() + seq->get_generated_len() - 1 == max_kv_cache_size) {
                    seq->set_status(SequenceStatus::OUT_OF_MEMORY);
                }
            }
        }
        auto removed_it = std::remove_if(active_sequence_groups.begin(), active_sequence_groups.end(),
            [](SequenceGroup::Ptr sg) -> bool {
                return sg->has_finished() || sg->handle_stopped() || sg->handle_cancelled();
            });
        active_sequence_groups.erase(removed_it, active_sequence_groups.end());
    };

    ov::Shape prompts_shape = input_ids.get_shape();
    const size_t batch_size = prompts_shape[0];

    // Initialize results and performance metrics.

    ov::genai::utils::GenerationFinishInfo finish_info;
    auto& raw_perf_counters = finish_info.results.perf_metrics.raw_metrics;
    raw_perf_counters.m_inference_durations = {{ MicroSeconds(0.0f) }};

    // Initialize inputs

    if (m_embedding) {
        m_llm.set_tensor("inputs_embeds", input_ids);
        if (token_type_ids.has_value())
            m_llm.set_tensor("token_type_ids", *token_type_ids);
    } else {
        kv_cache_state.add_inputs(input_ids);
        m_llm.set_tensor("input_ids", input_ids);
    }
    m_llm.set_tensor("attention_mask", attention_mask);
    if (position_ids.has_value())
        m_llm.set_tensor("position_ids", *position_ids);

    ov::Tensor beam_idx = ov::Tensor(ov::element::i32, {batch_size});
    std::fill_n(beam_idx.data<int32_t>(), batch_size, 0);
    m_llm.set_tensor("beam_idx", beam_idx);

    // "Prompt" phase

    const auto infer_start = std::chrono::steady_clock::now();
    m_llm.infer();
    const auto infer_end = std::chrono::steady_clock::now();
    const auto infer_ms = PerfMetrics::get_microsec(infer_end - infer_start);
    raw_perf_counters.m_inference_durations[0] += MicroSeconds(infer_ms);
    raw_perf_counters.m_token_infer_durations.emplace_back(infer_ms);
    raw_perf_counters.m_new_token_times.emplace_back(infer_end);
    raw_perf_counters.m_batch_sizes.emplace_back(batch_size);

    auto logits = m_llm.get_tensor("logits");

    int64_t output_sequence_len = logits.get_shape().at(1);
    for (auto& sequence_group : sequence_groups) {
        sequence_group->schedule_tokens(sequence_group->get_prompt_len());
        sequence_group->set_output_seq_len(output_sequence_len);
    }

    std::map<size_t, size_t> beam_offets;
    for (size_t i = 0; i < sequence_groups.size(); i++)
        beam_offets.insert({sequence_groups.at(i)->get_request_id(), i});

    SamplerOutput sampler_output = sampler.sample(sequence_groups, logits);
    free_non_running_requests(); // handle sampler output

    // "Generation" phase

    while (!active_sequence_groups.empty()) {
        size_t total_num_tokens = 0;

        for (auto& sequence_group : active_sequence_groups) {
            sequence_group->schedule_tokens(1);
            // compute aggregated values
            size_t num_sequences = sequence_group->num_running_seqs();
            total_num_tokens += sequence_group->get_num_scheduled_tokens() * num_sequences;
        }

        ov::Tensor new_input_ids(ov::element::i64, {total_num_tokens, 1});
        int64_t * input_ids_data = new_input_ids.data<int64_t>();

        std::vector<int32_t> next_beams;
        size_t current_batch_size = 0;

        for (auto& sequence_group : active_sequence_groups) {
            std::vector<Sequence::Ptr> running_sequences = sequence_group->get_running_sequences();
            size_t num_running_sequences = running_sequences.size();
            size_t num_scheduled_tokens = sequence_group->get_num_scheduled_tokens();
            size_t group_position_id = sequence_group->get_num_processed_tokens();

            std::map<size_t, int32_t> beam_idxs = sampler.get_beam_idxs(sequence_group);

            for (size_t seq_id = 0; seq_id < num_running_sequences; ++seq_id) {
                Sequence::CPtr sequence = running_sequences[seq_id];

                for (size_t token_id = 0, position_id = group_position_id; token_id < num_scheduled_tokens; ++token_id, ++position_id) {
                    // compute token for current sequence
                    input_ids_data[token_id] = position_id < sequence_group->get_prompt_len() ?
                        sequence_group->get_prompt_ids()[position_id] :
                        sequence->get_generated_ids()[position_id - sequence_group->get_prompt_len()];
                }

                // apply strides to shift to a next sequence
                input_ids_data += num_scheduled_tokens;

                // for different sequences iteration of beams started from 0, but we collect it to one input_ids
                next_beams.push_back(beam_idxs[sequence->get_id()] + beam_offets.at(sequence_group->get_request_id()));
            }

            current_batch_size += num_running_sequences;
        }

        for (size_t i = 0; i < active_sequence_groups.size(); i++) {
            beam_offets[active_sequence_groups.at(i)->get_request_id()] = i == 0 ? 0 : (active_sequence_groups.at(i - 1)->num_running_seqs() + beam_offets[i - 1]);
        }

        if (m_embedding) {
            constexpr bool return_remote_tensor = true;
            CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
            EmbeddingsRequest& req = embeddings_request_guard.get();
            const ov::Tensor& embed_prompt_tensor = m_embedding->infer(req, new_input_ids, return_remote_tensor);
            m_llm.set_tensor("inputs_embeds", embed_prompt_tensor);
            if (token_type_ids.has_value()) {
                ov::Tensor new_token_type_ids(ov::element::i64, {total_num_tokens, 1});
                int64_t* token_type_data = new_token_type_ids.data<int64_t>();
                std::fill(token_type_data, token_type_data + total_num_tokens, 0);
                m_llm.set_tensor("token_type_ids", new_token_type_ids);
            }
        } else {
            m_llm.set_tensor("input_ids", new_input_ids);
        }

        // we don't need to keep state for non chat mode and for beam_search in chat mode
        // in case of beam_search in chat mode, kv cache contains info about longest generated result among all sequences
        // last answer will be removed from kv_cache and will be included to the prompt on the next step
        if (new_input_ids.get_size() == 1)
            kv_cache_state.add_inputs(new_input_ids);

        update_attention_mask_with_beams(m_llm.get_tensor("attention_mask"), next_beams);

        if (position_ids.has_value()) {
            if (position_ids->get_shape().size() == 3 && rope_delta.has_value()) {
                update_3d_position_ids(m_llm.get_tensor("position_ids"), m_llm.get_tensor("attention_mask"), rope_delta.value());
            } else {
                update_position_ids(m_llm.get_tensor("position_ids"), m_llm.get_tensor("attention_mask"));
            }
        }

        m_llm.set_tensor("beam_idx", ov::Tensor{ov::element::i32, {total_num_tokens}, next_beams.data()});

        const auto infer_start = std::chrono::steady_clock::now();
        m_llm.start_async();

        stream_generated_tokens();
        free_non_running_requests(); // to handle streaming response

        m_llm.wait();

        const auto infer_end = std::chrono::steady_clock::now();
        const auto infer_ms = PerfMetrics::get_microsec(infer_end - infer_start);
        raw_perf_counters.m_inference_durations[0] += MicroSeconds(infer_ms);
        raw_perf_counters.m_token_infer_durations.emplace_back(infer_ms);
        raw_perf_counters.m_new_token_times.emplace_back(infer_end);
        raw_perf_counters.m_batch_sizes.emplace_back(current_batch_size);

        sampler_output = sampler.sample(active_sequence_groups, m_llm.get_tensor("logits"));
        free_non_running_requests(); // handle sampler output
    }

    stream_generated_tokens();
    if (streamer_ptr) { // push streamer's cache
        streamer_ptr->end();
    }

    for (auto& sequence_group : sequence_groups) {
        auto sampling_params = sequence_group->get_sampling_parameters();
        const auto& sequences = sequence_group->get_finished_sequences();
        size_t num_outputs = std::min(sequence_group->get_sampling_parameters().num_return_sequences, sequences.size());
        finish_info.streaming_finish_status = sequence_group->get_generation_stream()->get_status();

        for (size_t seq_id = 0; seq_id < num_outputs; ++seq_id) {
            const auto & sequence = sequences[seq_id];
            const float score = sampling_params.is_beam_search() ? sequence->get_beam_search_score(sampling_params) : sequence->get_cumulative_log_prob();

            finish_info.results.tokens.push_back(sequence->get_generated_ids());
            finish_info.results.scores.push_back(score);
        }
    }

    finish_info.streaming_finish_status = sequence_groups[0]->get_generation_stream()->get_status();

    for (SequenceGroup::Ptr sequence_group : sequence_groups)
        sampler.clear_request_info(sequence_group->get_request_id());

    return finish_info;
}


TokenizedInputs get_chat_encoded_input(const ov::Tensor& new_chat_tokens, utils::KVCacheState& kv_cache_state) {
    TokenizedInputs encoded_input;
    size_t kv_cache_len = kv_cache_state.get_state().size();
    if (kv_cache_len == 0) {
        encoded_input.input_ids = new_chat_tokens;
        ov::Tensor new_attention_mask(ov::element::i64, new_chat_tokens.get_shape());
        std::fill_n(new_attention_mask.data<int64_t>(), new_chat_tokens.get_shape()[1], 1);
        encoded_input.attention_mask = new_attention_mask;
    } else {
        ov::Tensor new_tensor = ov::Tensor(new_chat_tokens.get_element_type(),
                                            {1, new_chat_tokens.get_shape().at(1) - kv_cache_len},
                                            new_chat_tokens.data<int64_t>() + kv_cache_len);

        ov::Tensor new_attention_mask(ov::element::i64, new_tensor.get_shape());
        std::fill_n(new_attention_mask.data<int64_t>(), new_tensor.get_shape()[1], 1);

        encoded_input.input_ids = ov::Tensor(new_chat_tokens.get_element_type(),
                                             {1, new_chat_tokens.get_shape().at(1) - kv_cache_len});
        new_tensor.copy_to(encoded_input.input_ids);

        encoded_input.attention_mask = new_attention_mask;
    }

    return encoded_input;
}


void align_kv_cache_and_history(const ov::Tensor& new_chat_tokens, utils::KVCacheState& kv_cache_state) {
    // KV cache in model already contains prompts and answers from previous iterations.
    // So only new prompt wrapped into chat template to be sent into model. Tokenizer always returns
    // token_ids = {<bos token>, ...<valuable tokens>}. So if tokenizer applies only to the new prompt,
    // <bos token> will be inserted on every iteration.
    // So actual pipeline calculates input_ids for whole chat history + for whole chat history without the new prompt
    // and takes only the difference between them.
    // Also some symbols combinations can be encoded by the tokenizer in different ways.
    // So let's check it out, find the same part of tokenized history and templated one, and use that part on the next step.

    std::vector<int64_t>& state = kv_cache_state.get_state();

    if (state.empty())
        return;

    size_t first_diverse_tokens_idx = ov::genai::utils::get_first_history_difference(new_chat_tokens, state);
    // in the case of beam_search the longest answer is in the kv cache, but the best one is needed
    // so generated tokens were not added to KVCacheState and num_tokens_to_trim was set to the size of the generated serquence
    kv_cache_state.num_tokens_to_trim += state.size() - first_diverse_tokens_idx;
    state.resize(first_diverse_tokens_idx);
    kv_cache_state.reset_mem_state = state.empty();
}

}  // namespace genai
}  // namespace ov
