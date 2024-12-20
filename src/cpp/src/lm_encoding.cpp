// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <regex>
#include <vector>

#include "lm_encoding.hpp"
#include "openvino/genai/perf_metrics.hpp"

#include "debug_utils.hpp"

#include "utils.hpp"

namespace ov {
namespace genai {

void update_position_ids(ov::Tensor&& position_ids, const ov::Tensor&& attention_mask) {
    const size_t batch_size = attention_mask.get_shape().at(0);
    const size_t sequence_length = attention_mask.get_shape().at(1);
    position_ids.set_shape({batch_size, 1});

    for (size_t batch = 0; batch < batch_size; batch++) {
        int64_t* mask_start = attention_mask.data<int64_t>() + batch * sequence_length;
        position_ids.data<int64_t>()[batch] = std::accumulate(mask_start, mask_start + sequence_length - 1, 0);
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


std::pair<EncodedResults, int32_t> get_lm_encoded_results(
    ov::InferRequest& m_llm,
    const ov::Tensor& input_ids,
    const ov::Tensor& attention_mask,
    const std::shared_ptr<StreamerBase>& streamer_ptr,
    Sampler& sampler,
    std::vector<SequenceGroup::Ptr> sequence_groups,
    std::optional<ov::Tensor> position_ids,
    std::optional<EmbeddingsModel> m_embedding,
    std::optional<int32_t> selected_beam_idx
) {
    std::vector<GenerationHandle> generations;
    for (SequenceGroup::Ptr sequence_group : sequence_groups) {
        generations.push_back(std::make_shared<GenerationHandleImpl>(sequence_group->get_generation_stream(), sequence_group->get_sampling_parameters()));
    }

    auto active_sequence_groups{sequence_groups};

    auto stream_generated_tokens = [&streamer_ptr, &generations, &active_sequence_groups]() {
        GenerationHandle& handle = generations.at(0);
        if (streamer_ptr && handle->can_read()) {
            std::unordered_map<uint64_t, GenerationOutput> token = handle->back();
            for (const auto& gen_token : token.begin()->second.generated_ids) {
                if (streamer_ptr->put(gen_token)) {
                    handle->drop();
                    break;
                }
            }
        }

        // free non running requests
        auto removed_it = std::remove_if(active_sequence_groups.begin(), active_sequence_groups.end(),
            [](SequenceGroup::Ptr sg) -> bool {
                return sg->has_finished() || sg->out_of_memory() || sg->handle_dropped();
            });
        active_sequence_groups.erase(removed_it, active_sequence_groups.end());
    };

    ov::Shape prompts_shape = input_ids.get_shape();
    const size_t batch_size = prompts_shape[0];

    // Initialize results and performance metrics.

    EncodedResults results;
    auto& raw_perf_counters = results.perf_metrics.raw_metrics;
    raw_perf_counters.m_inference_durations = {{ MicroSeconds(0.0f) }};

    // Initialize inputs
    m_llm.set_tensor(m_embedding.has_value() ? "inputs_embeds" : "input_ids", input_ids);
    m_llm.set_tensor("attention_mask", attention_mask);
    if (position_ids.has_value())
        m_llm.set_tensor("position_ids", *position_ids);

    ov::Tensor beam_idx = ov::Tensor(ov::element::i32, {batch_size});
    std::fill_n(beam_idx.data<int32_t>(), batch_size, selected_beam_idx.has_value() ? *selected_beam_idx : 0);
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

    int64_t sequence_len = logits.get_shape().at(1);
    for (auto& sequence_group : sequence_groups) {
        sequence_group->update_processed_tokens_num(sequence_group->get_prompt_len() - sequence_len);
        sequence_group->schedule_tokens(sequence_len);
    }

    std::map<size_t, size_t> beam_offets;
    for (size_t i = 0; i < sequence_groups.size(); i++)
        beam_offets.insert({sequence_groups.at(i)->get_request_id(), i});

    SamplerOutput sampler_output = sampler.sample(sequence_groups, logits);
    stream_generated_tokens();

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

                // for different sequences iteration of beams started from 0, but we collect it to one input_ids#
                next_beams.push_back(beam_idxs[sequence->get_id()] + beam_offets.at(sequence_group->get_request_id()));
            }
        }

        for (size_t i = 0; i < sequence_groups.size(); i++) {
            beam_offets[sequence_groups.at(i)->get_request_id()] = i == 0 ? 0 : (sequence_groups.at(i - 1)->num_running_seqs() + beam_offets[i - 1]);
        }

        if (m_embedding.has_value()) {
            const ov::Tensor& embed_prompt_tensor = (*m_embedding).infer(new_input_ids);
            m_llm.set_tensor("inputs_embeds", embed_prompt_tensor);
        } else {
            m_llm.set_tensor("input_ids", new_input_ids);
        }

        update_attention_mask_with_beams(m_llm.get_tensor("attention_mask"), next_beams);

        if (position_ids.has_value()) {
            update_position_ids(m_llm.get_tensor("position_ids"), m_llm.get_tensor("attention_mask"));
        }

        m_llm.set_tensor("beam_idx", ov::Tensor{ov::element::i32, {total_num_tokens}, next_beams.data()});

        const auto infer_start = std::chrono::steady_clock::now();
        m_llm.infer();
        const auto infer_end = std::chrono::steady_clock::now();
        const auto infer_ms = PerfMetrics::get_microsec(infer_end - infer_start);
        raw_perf_counters.m_inference_durations[0] += MicroSeconds(infer_ms);
        raw_perf_counters.m_token_infer_durations.emplace_back(infer_ms);
        raw_perf_counters.m_new_token_times.emplace_back(infer_end);
        raw_perf_counters.m_batch_sizes.emplace_back(batch_size);

        sampler_output = sampler.sample(active_sequence_groups, m_llm.get_tensor("logits"));
        stream_generated_tokens();
    }

    if (streamer_ptr) { // push streamer's cache
        streamer_ptr->end();
    }

    // Collect results

    size_t next_selected_beam = 0;
    for (size_t i = 0; i < sequence_groups.size(); i++) {
        auto request = sequence_groups[i];
        std::vector<GenerationOutput> generation_outputs;
        auto sampling_params = request->get_sampling_parameters();
        const auto& sequences = request->get_finished_sequences();
        size_t num_outputs = std::min(request->get_sampling_parameters().num_return_sequences, sequences.size());

        for (size_t seq_id = 0; seq_id < num_outputs; ++seq_id) {
            const auto & sequence = sequences[seq_id];
            const float score = sampling_params.is_beam_search() ? sequence->get_beam_search_score(sampling_params) : sequence->get_cumulative_log_probs();

            results.tokens.push_back(sequence->get_generated_ids());
            results.scores.push_back(score);
        }
        // next_selected_beam = sampler.last_selected_beam(request);
    }

    for (SequenceGroup::Ptr sequence_group : sequence_groups)
        sampler.clear_request_info(sequence_group->get_request_id());

    return {results, next_selected_beam};
}

}  // namespace genai
}  // namespace ov