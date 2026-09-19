// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "decoder.hpp"

#include <algorithm>
#include <cstring>
#include <map>

#include "openvino/genai/generation_handle.hpp"
#include "sequence_group.hpp"
#include "utils.hpp"

namespace ov::genai {

Qwen3ASRDecoder::Qwen3ASRDecoder(const std::filesystem::path& models_path,
                                 const std::string& device,
                                 const ov::AnyMap& properties) {
    ov::Core core = utils::singleton_core();
    ov::CompiledModel compiled_model =
        core.compile_model(models_path / "openvino_decoder_model.xml", device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "qwen3-asr decoder model");
    m_request = compiled_model.create_infer_request();
}

void Qwen3ASRDecoder::set_seed(size_t seed) {
    m_sampler.set_seed(seed);
}

EncodedResults Qwen3ASRDecoder::generate(const ov::Tensor& input_ids,
                                         const ov::Tensor& encoder_hidden_states,
                                         const ASRGenerationConfig& config,
                                         RawPerfMetrics& raw_metrics,
                                         ASRRawPerfMetrics& asr_raw_metrics,
                                         const std::shared_ptr<StreamerBase>& streamer_ptr) {
    const ov::Shape prompts_shape = input_ids.get_shape();
    const size_t batch_size = prompts_shape[0];
    OPENVINO_ASSERT(batch_size == 1 || !streamer_ptr, "Streaming is only supported with batch_size == 1");

    // Reset decoder state for fresh generation
    m_request.reset_state();

    std::vector<SequenceGroup::Ptr> sequence_groups;
    sequence_groups.reserve(batch_size);
    const int64_t* input_ids_data = input_ids.data<const int64_t>();
    const size_t prompt_len = prompts_shape[1];
    for (size_t batch = 0; batch < batch_size; ++batch) {
        std::vector<int64_t> prompt_tokens(input_ids_data + batch * prompt_len,
                                           input_ids_data + (batch + 1) * prompt_len);
        auto seq_group = std::make_shared<SequenceGroup>(batch, prompt_tokens, config);
        sequence_groups.push_back(std::move(seq_group));
    }

    // Streaming handle (only for batch_size == 1)
    std::shared_ptr<GenerationHandleImpl> handle;
    if (streamer_ptr) {
        handle = std::make_shared<GenerationHandleImpl>(sequence_groups[0]->get_generation_stream(),
                                                        sequence_groups[0]->get_sampling_parameters());
    }

    auto stream_generated_tokens = [&]() {
        if (!streamer_ptr || !handle || !handle->can_read()) {
            return;
        }
        std::unordered_map<uint64_t, GenerationOutput> token = handle->read();
        auto streaming_status = streamer_ptr->write(token.begin()->second.generated_ids);
        if (streaming_status == StreamingStatus::CANCEL) {
            handle->cancel();
        } else if (streaming_status == StreamingStatus::STOP) {
            handle->stop();
        }
    };

    ov::Tensor current_encoder_hidden_states = encoder_hidden_states;
    m_request.set_tensor("encoder_hidden_states", current_encoder_hidden_states);

    ov::Tensor beam_idx = ov::Tensor(ov::element::i32, {batch_size});
    std::fill_n(beam_idx.data<int32_t>(), batch_size, 0);
    m_request.set_tensor("beam_idx", beam_idx);

    // Prefill: run decoder with full prompt
    m_request.set_tensor("input_ids", input_ids);
    const auto infer_start = std::chrono::steady_clock::now();
    m_request.infer();
    const auto infer_end = std::chrono::steady_clock::now();
    const auto infer_ms = PerfMetrics::get_microsec(infer_end - infer_start);
    raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);
    raw_metrics.m_token_infer_durations.emplace_back(infer_ms);
    raw_metrics.m_new_token_times.emplace_back(infer_end);
    raw_metrics.m_batch_sizes.emplace_back(batch_size);
    asr_raw_metrics.decode_inference_durations.emplace_back(infer_ms);

    ov::Tensor logits = m_request.get_tensor("logits");
    const int64_t output_sequence_len = logits.get_shape().at(1);

    // Schedule prompt tokens and sample
    for (auto& seq_group : sequence_groups) {
        seq_group->schedule_tokens(seq_group->get_prompt_len());
        seq_group->set_output_seq_len(output_sequence_len);
    }

    // Beam offsets: maps request_id -> starting position in flattened batch
    std::map<size_t, size_t> beam_offsets;
    for (size_t i = 0; i < sequence_groups.size(); ++i) {
        beam_offsets.insert({sequence_groups[i]->get_request_id(), i});
    }

    const auto sample_start = std::chrono::steady_clock::now();
    m_sampler.sample(sequence_groups, logits);
    raw_metrics.m_sampling_durations.emplace_back(
        PerfMetrics::get_microsec(std::chrono::steady_clock::now() - sample_start));
    stream_generated_tokens();

    // Track active (not yet finished) sequence groups
    auto active_sequence_groups = sequence_groups;

    auto free_finished_requests = [&active_sequence_groups]() {
        auto removed_it =
            std::remove_if(active_sequence_groups.begin(),
                           active_sequence_groups.end(),
                           [](const SequenceGroup::Ptr& sg) {
                               return sg->has_finished() || sg->handle_stopped() || sg->handle_cancelled();
                           });
        active_sequence_groups.erase(removed_it, active_sequence_groups.end());
    };

    free_finished_requests();

    // Generation loop
    while (!active_sequence_groups.empty()) {
        size_t total_num_tokens = 0;
        for (auto& seq_group : active_sequence_groups) {
            seq_group->schedule_tokens(1);
            total_num_tokens += seq_group->get_num_scheduled_tokens() * seq_group->num_running_seqs();
        }

        ov::Tensor new_input_ids(ov::element::i64, {total_num_tokens, 1});
        int64_t* input_ids_data = new_input_ids.data<int64_t>();
        std::vector<int32_t> next_beams;

        for (auto& seq_group : active_sequence_groups) {
            std::vector<Sequence::Ptr> running_sequences = seq_group->get_running_sequences();
            const size_t num_scheduled_tokens = seq_group->get_num_scheduled_tokens();
            const size_t group_position_id = seq_group->get_num_processed_tokens();

            std::map<size_t, int32_t> beam_idxs = m_sampler.get_beam_idxs(seq_group);

            for (size_t seq_id = 0; seq_id < running_sequences.size(); ++seq_id) {
                Sequence::CPtr sequence = running_sequences[seq_id];

                for (size_t token_id = 0, position_id = group_position_id; token_id < num_scheduled_tokens;
                     ++token_id, ++position_id) {
                    input_ids_data[token_id] =
                        position_id < seq_group->get_prompt_len()
                            ? seq_group->get_prompt_ids()[position_id]
                            : sequence->get_generated_ids()[position_id - seq_group->get_prompt_len()];
                }

                input_ids_data += num_scheduled_tokens;
                next_beams.push_back(beam_idxs[sequence->get_id()] + beam_offsets.at(seq_group->get_request_id()));
            }
        }

        // Update beam offsets for next iteration
        for (size_t i = 0; i < active_sequence_groups.size(); ++i) {
            beam_offsets[active_sequence_groups[i]->get_request_id()] =
                i == 0 ? 0
                       : (active_sequence_groups[i - 1]->num_running_seqs() +
                          beam_offsets[active_sequence_groups[i - 1]->get_request_id()]);
        }

        m_request.set_tensor("input_ids", new_input_ids);
        m_request.set_tensor("beam_idx", ov::Tensor{ov::element::i32, {total_num_tokens}, next_beams.data()});
        // for beam search investigate encoder batches reordering based on next_beams
        const auto infer_start = std::chrono::steady_clock::now();
        m_request.infer();
        const auto infer_end = std::chrono::steady_clock::now();
        const auto infer_ms = PerfMetrics::get_microsec(infer_end - infer_start);
        raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);
        raw_metrics.m_token_infer_durations.emplace_back(infer_ms);
        raw_metrics.m_new_token_times.emplace_back(infer_end);
        raw_metrics.m_batch_sizes.emplace_back(total_num_tokens);
        asr_raw_metrics.decode_inference_durations.emplace_back(infer_ms);

        logits = m_request.get_tensor("logits");

        const auto sample_start = std::chrono::steady_clock::now();
        m_sampler.sample(active_sequence_groups, logits);
        raw_metrics.m_sampling_durations.emplace_back(
            PerfMetrics::get_microsec(std::chrono::steady_clock::now() - sample_start));
        stream_generated_tokens();
        free_finished_requests();
    }

    // Flush streamer cache
    stream_generated_tokens();

    // Collect results
    EncodedResults results;
    for (size_t b = 0; b < batch_size; ++b) {
        const auto& sequences = sequence_groups[b]->get_finished_sequences();
        OPENVINO_ASSERT(!sequences.empty(), "No finished sequences for batch element ", b);

        const auto& sampling_params = sequence_groups[b]->get_sampling_parameters();
        const auto& sequence = sequences[0];
        const float score = sampling_params.is_beam_search() ? sequence->get_beam_search_score(sampling_params)
                                                             : sequence->get_cumulative_log_prob();

        results.tokens.push_back(sequence->get_generated_ids());
        results.scores.push_back(score);

        m_sampler.clear_request_info(sequence_groups[b]->get_request_id());
    }
    return results;
}

}  // namespace ov::genai
