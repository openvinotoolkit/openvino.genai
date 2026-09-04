// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "whisper.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <openvino/openvino.hpp>
#include <thread>

#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/whisper_generation_config.hpp"
#include "openvino/genai/whisper_pipeline.hpp"
#include "sampling/sampler.hpp"
#include "utils.hpp"
#include "whisper/config.hpp"
#include "whisper/context_tokens.hpp"
#include "whisper/feature_extractor.hpp"
#include "whisper/logit_processor.hpp"
#include "whisper/models.hpp"
#include "whisper/models/decoder.hpp"
#include "whisper/timestamps.hpp"
#include "whisper/whisper_utils.hpp"
#include "whisper/word_level_timestamps.hpp"

using ov::genai::MicroSeconds;

namespace {

// Converts the last-position logits for one batch row to log-probabilities.
void logits_row_to_logprobs(ov::Tensor& logits, size_t batch_idx) {
    const ov::Shape shape = logits.get_shape();
    OPENVINO_ASSERT(shape.size() == 3);
    OPENVINO_ASSERT(batch_idx < shape[0], "Logits batch size doesn't match the number of beams");
    OPENVINO_ASSERT(shape[1] > 0);
    OPENVINO_ASSERT(shape[2] > 0);

    const size_t vocab_size = shape[2];
    const size_t batch_offset = batch_idx * shape[1] * vocab_size;
    const size_t sequence_offset = (shape[1] - 1) * vocab_size;
    float* const row = logits.data<float>() + batch_offset + sequence_offset;

    float max_logit = *std::max_element(row, row + vocab_size);
    double sum = 0.0;
    for (size_t i = 0; i < vocab_size; ++i) {
        sum += std::exp(static_cast<double>(row[i]) - max_logit);
    }
    float log_sum = static_cast<float>(std::log(sum)) + max_logit;
    for (size_t i = 0; i < vocab_size; ++i) {
        row[i] -= log_sum;
    }
}

// row_sequences[i] provides timestamp history for physical_rows[i]; prefill does not need it.
void process_whisper_logits(ov::Tensor logits,
                            const ov::genai::WhisperGenerationConfig& config,
                            const bool return_timestamps,
                            const bool initial_step,
                            const std::vector<size_t>& physical_rows,
                            const std::vector<ov::genai::Sequence::Ptr>& row_sequences) {
    // Normalize before masking so beam scores remain comparable across rows, matching HF beam search.
    const bool is_beam_search = config.is_beam_search();

    if (return_timestamps && !initial_step) {
        OPENVINO_ASSERT(row_sequences.size() == physical_rows.size(),
                        "Internal error: Whisper timestamp processing needs one sequence per physical row; got ",
                        row_sequences.size(),
                        " sequences for ",
                        physical_rows.size(),
                        " rows.");
    }

    for (size_t i = 0; i < physical_rows.size(); i++) {
        const size_t batch = physical_rows[i];
        if (is_beam_search) {
            logits_row_to_logprobs(logits, batch);
        }

        if (initial_step) {
            ov::genai::do_suppress_tokens(logits, batch, config.begin_suppress_tokens);
        }

        ov::genai::do_suppress_tokens(logits, batch, config.suppress_tokens);

        if (return_timestamps) {
            if (initial_step) {
                ov::genai::process_whisper_timestamp_logits(logits, batch, config, {}, initial_step);
            } else {
                ov::genai::process_whisper_timestamp_logits(logits,
                                                            batch,
                                                            config,
                                                            row_sequences[i]->get_generated_ids(),
                                                            initial_step);
            }
        }
    }
}

void record_decode_inference_metrics(ov::genai::RawPerfMetrics& raw_metrics,
                                     ov::genai::WhisperRawPerfMetrics& whisper_raw_metrics,
                                     const std::chrono::steady_clock::time_point infer_start,
                                     const std::chrono::steady_clock::time_point infer_end) {
    const auto infer_ms = ov::genai::PerfMetrics::get_microsec(infer_end - infer_start);
    raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);
    raw_metrics.m_token_infer_durations.emplace_back(infer_ms);
    raw_metrics.m_new_token_times.emplace_back(infer_end);
    whisper_raw_metrics.decode_inference_durations.emplace_back(infer_ms);
}

void init_whisper_perf_metrics(ov::genai::WhisperPerfMetrics& perf_metrics) {
    perf_metrics.num_input_tokens = 0;
    perf_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
    perf_metrics.whisper_raw_metrics.word_level_timestamps_processing_durations = {{MicroSeconds(0.0f)}};
}

ov::Tensor build_prefill_input_ids(const std::shared_ptr<ov::genai::WhisperDecoder>& decoder,
                                   const std::vector<ov::genai::SequenceGroup::Ptr>& sequence_groups) {
    OPENVINO_ASSERT(!sequence_groups.empty(), "Whisper prefill requires at least one prompt.");
    const size_t num_prompts = sequence_groups.size();
    const size_t prompt_len = sequence_groups.front()->get_prompt_ids().size();
    OPENVINO_ASSERT(prompt_len != 0, "Whisper prefill requires a non-empty prompt.");

    for (size_t request = 0; request < num_prompts; request++) {
        const auto& prompt_ids = sequence_groups[request]->get_prompt_ids();
        OPENVINO_ASSERT(prompt_ids.size() == prompt_len,
                        "Batched Whisper generation requires an identical prompt length for every input. Input ",
                        request,
                        " has ",
                        prompt_ids.size(),
                        " prompt tokens, expected ",
                        prompt_len,
                        ".");
    }

    ov::Tensor input_ids = decoder->create_host_tensor(ov::element::i64, {num_prompts, prompt_len});
    auto* input_ids_data = input_ids.data<int64_t>();
    for (size_t request = 0; request < num_prompts; request++) {
        const auto& prompt_ids = sequence_groups[request]->get_prompt_ids();
        std::copy(prompt_ids.begin(), prompt_ids.end(), input_ids_data + request * prompt_len);
    }
    return input_ids;
}

ov::Tensor make_identity_beam_idx(const std::shared_ptr<ov::genai::WhisperDecoder>& decoder, const size_t width) {
    ov::Tensor beam_idx = decoder->create_host_tensor(ov::element::i32, {width});
    std::iota(beam_idx.data<int32_t>(), beam_idx.data<int32_t>() + width, 0);
    return beam_idx;
}

// live_sequences[i] maps to live_physical_rows[i].
void fill_next_input_ids(ov::Tensor& next_input_ids,
                         const std::vector<ov::genai::SequenceGroup::Ptr>& active_groups,
                         const std::vector<ov::genai::Sequence::Ptr>& live_sequences,
                         const std::vector<size_t>& live_physical_rows,
                         const bool retain_physical_rows,
                         const size_t physical_width,
                         const int64_t filler_token) {
    auto* data = next_input_ids.data<int64_t>();
    if (retain_physical_rows) {
        std::fill_n(data, physical_width, filler_token);
    }

    for (size_t i = 0; i < live_sequences.size(); i++) {
        // Fixed rows use one group per row; packed scalar rows share one group.
        const auto& sequence_group = retain_physical_rows ? active_groups[i] : active_groups.front();
        const auto& sequence = live_sequences[i];
        const size_t num_processed_tokens = sequence_group->get_num_processed_tokens();
        const size_t request_prompt_len = sequence_group->get_prompt_len();
        data[live_physical_rows[i]] =
            num_processed_tokens < request_prompt_len
                ? sequence_group->get_prompt_ids()[num_processed_tokens]
                : sequence->get_generated_ids()[num_processed_tokens - request_prompt_len];
    }
}

// Returns a view for a contiguous prefix; otherwise gathers the requested rows.
ov::Tensor pack_live_logits(const ov::Tensor& logits, const std::vector<size_t>& live_rows) {
    const auto& shape = logits.get_shape();

    bool is_contiguous_prefix = true;
    for (size_t i = 0; i < live_rows.size(); i++) {
        if (live_rows[i] != i) {
            is_contiguous_prefix = false;
            break;
        }
    }

    if (is_contiguous_prefix) {
        if (live_rows.size() == shape.at(0)) {
            return logits;
        }

        return ov::genai::utils::make_tensor_slice(logits, 0, 0, live_rows.size());
    }

    const size_t row_size = shape.at(1) * shape.at(2);
    ov::Tensor packed(ov::element::f32, ov::Shape{live_rows.size(), shape.at(1), shape.at(2)});

    const auto* source = logits.data<const float>();
    auto* target = packed.data<float>();
    for (size_t i = 0; i < live_rows.size(); i++) {
        std::copy_n(source + live_rows[i] * row_size, row_size, target + i * row_size);
    }

    return packed;
}

// Timestamp streaming occurs after segment extraction.
void stream_generated_tokens(const std::shared_ptr<ov::genai::StreamerBase>& streamer_ptr,
                             const std::shared_ptr<ov::genai::GenerationHandleImpl>& handle,
                             const bool return_timestamps) {
    if (return_timestamps || !streamer_ptr || !handle->can_read()) {
        return;
    }

    std::unordered_map<uint64_t, ov::genai::GenerationOutput> token = handle->read();

    auto streaming_status = streamer_ptr->write(token.begin()->second.generated_ids);
    if (streaming_status == ov::genai::StreamingStatus::CANCEL) {
        handle->cancel();
    } else if (streaming_status == ov::genai::StreamingStatus::STOP) {
        handle->stop();
    } else if (streaming_status == ov::genai::StreamingStatus::TOOL_CALL_STOP) {
        handle->stop(ov::genai::GenerationFinishReason::TOOL_CALL);
    }
}

// Resets decoder state on exit and settles in-flight inference during unwinding.
struct DecoderRoundResetGuard {
    std::shared_ptr<ov::genai::WhisperDecoder> decoder;
    bool in_flight = false;
    bool finished = false;

    explicit DecoderRoundResetGuard(std::shared_ptr<ov::genai::WhisperDecoder> decoder) : decoder(std::move(decoder)) {}
    DecoderRoundResetGuard(const DecoderRoundResetGuard&) = delete;
    DecoderRoundResetGuard& operator=(const DecoderRoundResetGuard&) = delete;

    void mark_started() {
        in_flight = true;
    }
    void mark_waited() {
        in_flight = false;
    }

    // Set finished only after reset succeeds.
    void finish() {
        decoder->reset_state();
        finished = true;
    }

    ~DecoderRoundResetGuard() {
        if (finished) {
            return;
        }
        try {
            if (in_flight) {
                decoder->wait();
            }
        } catch (...) {
        }
        try {
            decoder->reset_state();
        } catch (...) {
        }
    }
};

struct DecoderRoundConfig {
    bool retain_physical_rows = false;
    bool return_timestamps = false;
    std::shared_ptr<ov::genai::StreamerBase> streamer = nullptr;
};

// Distinct-audio rows remain fixed because beam_idx does not gather cross-attention state;
// scalar rows may follow beam compaction.
void run_decoder_generation_loop(const std::shared_ptr<ov::genai::WhisperDecoder>& decoder,
                                 const ov::Tensor& encoder_hidden_state,
                                 ov::genai::Sampler& sampler,
                                 const std::vector<ov::genai::SequenceGroup::Ptr>& sequence_groups,
                                 ov::Tensor& beam_idx_tensor,
                                 ov::Tensor& fixed_next_input_ids,
                                 DecoderRoundResetGuard& reset_guard,
                                 const DecoderRoundConfig& round_config,
                                 const ov::genai::WhisperGenerationConfig& config,
                                 const std::shared_ptr<ov::genai::GenerationHandleImpl>& handle,
                                 ov::genai::WhisperPerfMetrics& perf_metrics) {
    const bool retain_physical_rows = round_config.retain_physical_rows;
    const bool return_timestamps = round_config.return_timestamps;
    ov::genai::RawPerfMetrics& raw_metrics = perf_metrics.raw_metrics;
    ov::genai::WhisperRawPerfMetrics& whisper_raw_metrics = perf_metrics.whisper_raw_metrics;

    // Reused across steps; all row vectors remain aligned.
    std::vector<ov::genai::SequenceGroup::Ptr> active_groups;
    std::vector<ov::genai::Sequence::Ptr> live_sequences;
    std::vector<size_t> live_physical_rows;
    std::vector<int32_t> packed_beam_idx;
    const size_t capacity = std::max<size_t>(sequence_groups.size(), config.num_beams);
    active_groups.reserve(capacity);
    live_sequences.reserve(capacity);
    live_physical_rows.reserve(capacity);
    packed_beam_idx.reserve(capacity);

    while (true) {
        active_groups.clear();
        live_sequences.clear();
        live_physical_rows.clear();
        packed_beam_idx.clear();
        size_t physical_width = 0;

        if (!retain_physical_rows) {
            OPENVINO_ASSERT(sequence_groups.size() == 1,
                            "Internal error: packed Whisper decoding expects exactly one sequence group, got ",
                            sequence_groups.size(),
                            ".");
            const auto& sequence_group = sequence_groups.front();
            if (sequence_group->has_finished() || sequence_group->handle_stopped() ||
                sequence_group->handle_cancelled()) {
                break;
            }
            sequence_group->schedule_tokens(1);
            const std::vector<ov::genai::Sequence::Ptr> running_sequences = sequence_group->get_running_sequences();
            OPENVINO_ASSERT(sequence_group->get_num_scheduled_tokens() == 1,
                            "Internal error: Whisper generation schedules exactly one token per step.");
            const std::map<size_t, int32_t> beam_idxs = sampler.get_beam_idxs(sequence_group);
            active_groups.push_back(sequence_group);
            for (size_t row = 0; row < running_sequences.size(); row++) {
                live_sequences.push_back(running_sequences[row]);
                live_physical_rows.push_back(row);
                packed_beam_idx.push_back(beam_idxs.at(running_sequences[row]->get_id()));
            }
            physical_width = running_sequences.size();
        } else {
            physical_width = sequence_groups.size();
            for (size_t request = 0; request < sequence_groups.size(); request++) {
                const auto& sequence_group = sequence_groups[request];
                if (sequence_group->has_finished() || sequence_group->handle_stopped() ||
                    sequence_group->handle_cancelled()) {
                    continue;
                }
                sequence_group->schedule_tokens(1);
                const auto running_sequences = sequence_group->get_running_sequences();
                OPENVINO_ASSERT(running_sequences.size() == 1,
                                "Internal error: batched Whisper generation expects exactly one running sequence per "
                                "request, got ",
                                running_sequences.size(),
                                ".");
                active_groups.push_back(sequence_group);
                live_sequences.push_back(running_sequences.front());
                live_physical_rows.push_back(request);
            }
        }

        if (active_groups.empty()) {
            break;
        }

        ov::Tensor next_input_ids =
            retain_physical_rows ? fixed_next_input_ids : ov::Tensor(ov::element::i64, {physical_width, 1});
        fill_next_input_ids(next_input_ids,
                            active_groups,
                            live_sequences,
                            live_physical_rows,
                            retain_physical_rows,
                            physical_width,
                            config.decoder_start_token_id);

        // Fixed rows keep the prefill identity mapping; scalar rows follow the sampler.
        if (!retain_physical_rows) {
            OPENVINO_ASSERT(packed_beam_idx.size() == physical_width,
                            "Internal error: beam-index row count ",
                            packed_beam_idx.size(),
                            " must equal the decoder physical width ",
                            physical_width,
                            ".");
            if (beam_idx_tensor.get_shape().at(0) != packed_beam_idx.size()) {
                beam_idx_tensor.set_shape({packed_beam_idx.size()});
            }
            std::copy_n(packed_beam_idx.data(), packed_beam_idx.size(), beam_idx_tensor.data<int32_t>());
        }

        const auto infer_start = std::chrono::steady_clock::now();
        reset_guard.mark_started();
        decoder->start_async(encoder_hidden_state, next_input_ids, beam_idx_tensor);

        // Preserve callback/inference overlap; the guard settles the request if streaming throws.
        stream_generated_tokens(round_config.streamer, handle, return_timestamps);

        auto logits = decoder->wait();
        reset_guard.mark_waited();

        const auto infer_end = std::chrono::steady_clock::now();
        record_decode_inference_metrics(raw_metrics, whisper_raw_metrics, infer_start, infer_end);
        raw_metrics.m_batch_sizes.emplace_back(live_physical_rows.size());

        process_whisper_logits(logits,
                               config,
                               return_timestamps,
                               /*initial_step=*/false,
                               live_physical_rows,
                               live_sequences);

        OPENVINO_ASSERT(logits.get_shape().at(0) == physical_width,
                        "Internal error: decoder logits row count ",
                        logits.get_shape().at(0),
                        " does not match the physical width ",
                        physical_width,
                        ".");

        // Sampler expects one logits row per active sequence.
        const ov::Tensor sampled_logits = pack_live_logits(logits, live_physical_rows);

        const auto sample_start = std::chrono::steady_clock::now();
        sampler.sample(active_groups, sampled_logits);
        raw_metrics.m_sampling_durations.emplace_back(
            ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - sample_start));
    }
}

struct DecoderRoundResult {
    std::vector<std::vector<int64_t>> tokens;
    std::vector<bool> cancelled;
};

// Request IDs are reused, so sampler state must be cleared on every exit.
struct SamplerRequestCleanup {
    ov::genai::Sampler& sampler;
    std::vector<uint64_t> request_ids;

    explicit SamplerRequestCleanup(ov::genai::Sampler& sampler) : sampler(sampler) {}
    SamplerRequestCleanup(const SamplerRequestCleanup&) = delete;
    SamplerRequestCleanup& operator=(const SamplerRequestCleanup&) = delete;

    ~SamplerRequestCleanup() {
        for (const uint64_t request_id : request_ids) {
            sampler.clear_request_info(request_id);
        }
    }
};

DecoderRoundResult run_decoder_round(std::shared_ptr<ov::genai::WhisperDecoder> decoder,
                                     const std::vector<ov::genai::SequenceGroup::Ptr>& sequence_groups,
                                     const ov::Tensor& encoder_hidden_state,
                                     ov::genai::Sampler& sampler,
                                     const ov::genai::WhisperGenerationConfig& config,
                                     const DecoderRoundConfig& round_config,
                                     ov::genai::WhisperPerfMetrics& perf_metrics) {
    OPENVINO_ASSERT(!sequence_groups.empty(), "Whisper decoder round requires at least one sequence group.");

    OPENVINO_ASSERT(!round_config.streamer || sequence_groups.size() == 1,
                    "Internal error: Whisper streaming requires exactly one sequence group.");

    ov::genai::RawPerfMetrics& raw_metrics = perf_metrics.raw_metrics;
    ov::genai::WhisperRawPerfMetrics& whisper_raw_metrics = perf_metrics.whisper_raw_metrics;

    const size_t prefill_width = sequence_groups.size();

    // Streaming is scalar-only, so the handle belongs to the single group.
    std::shared_ptr<ov::genai::GenerationHandleImpl> handle;
    if (round_config.streamer) {
        const auto& sequence_group = sequence_groups.front();
        handle = std::make_shared<ov::genai::GenerationHandleImpl>(sequence_group->get_generation_stream(),
                                                                   sequence_group->get_sampling_parameters());
    }

    ov::Tensor beam_idx_tensor = make_identity_beam_idx(decoder, prefill_width);
    ov::Tensor fixed_next_input_ids = round_config.retain_physical_rows
                                          ? decoder->create_host_tensor(ov::element::i64, {prefill_width, 1})
                                          : ov::Tensor();

    // Guard sampler state created during prefill or generation.
    SamplerRequestCleanup request_cleanup{sampler};
    request_cleanup.request_ids.reserve(sequence_groups.size());
    for (const auto& sequence_group : sequence_groups) {
        request_cleanup.request_ids.push_back(sequence_group->get_request_id());
    }

    DecoderRoundResetGuard reset_guard{decoder};
    const ov::Tensor input_ids_tensor = build_prefill_input_ids(decoder, sequence_groups);
    const auto infer_start = std::chrono::steady_clock::now();
    reset_guard.mark_started();
    decoder->start_async(encoder_hidden_state, input_ids_tensor, beam_idx_tensor);

    auto logits = decoder->wait();
    reset_guard.mark_waited();
    const auto infer_end = std::chrono::steady_clock::now();
    record_decode_inference_metrics(raw_metrics, whisper_raw_metrics, infer_start, infer_end);
    raw_metrics.m_batch_sizes.emplace_back(prefill_width);

    std::vector<size_t> prefill_rows(prefill_width);
    std::iota(prefill_rows.begin(), prefill_rows.end(), size_t{0});
    process_whisper_logits(logits, config, round_config.return_timestamps, /*initial_step=*/true, prefill_rows, {});

    const int64_t output_sequence_len = logits.get_shape().at(1);
    for (const auto& sequence_group : sequence_groups) {
        sequence_group->schedule_tokens(sequence_group->get_prompt_len());
        sequence_group->set_output_seq_len(output_sequence_len);
    }

    {
        const auto sample_start = std::chrono::steady_clock::now();
        sampler.sample(sequence_groups, logits);
        raw_metrics.m_sampling_durations.emplace_back(
            ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - sample_start));
    }
    stream_generated_tokens(round_config.streamer, handle, round_config.return_timestamps);

    run_decoder_generation_loop(decoder,
                                encoder_hidden_state,
                                sampler,
                                sequence_groups,
                                beam_idx_tensor,
                                fixed_next_input_ids,
                                reset_guard,
                                round_config,
                                config,
                                handle,
                                perf_metrics);

    stream_generated_tokens(round_config.streamer, handle, round_config.return_timestamps);

    // Sequence-group order matches input order.
    DecoderRoundResult result;
    result.tokens.reserve(prefill_width);
    result.cancelled.reserve(prefill_width);
    for (const auto& sequence_group : sequence_groups) {
        const auto& sequences = sequence_group->get_finished_sequences();
        OPENVINO_ASSERT(!sequences.empty(),
                        "Internal error: no finished sequence for request ",
                        sequence_group->get_request_id(),
                        ".");
        result.tokens.push_back(sequences[0]->get_generated_ids());
        result.cancelled.push_back(sequence_group->handle_stopped() || sequence_group->handle_cancelled());
    }

    reset_guard.finish();

    return result;
}

ov::Tensor run_encoder_inference(ov::InferRequest& request,
                                 const ov::Tensor& input_features,
                                 const ov::Shape& release_shape,
                                 ov::genai::RawPerfMetrics& raw_metrics,
                                 ov::genai::WhisperRawPerfMetrics& whisper_raw_metrics) {
    request.set_tensor("input_features", input_features);

    const auto infer_start = std::chrono::steady_clock::now();
    request.infer();
    const auto infer_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);
    raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);
    whisper_raw_metrics.encode_inference_durations.emplace_back(infer_ms);

    // Release caller-owned input memory after synchronous inference.
    request.set_tensor("input_features", ov::Tensor(ov::element::f32, release_shape));

    return request.get_tensor("last_hidden_state");
}

ov::Tensor encode_windows(ov::InferRequest& request,
                          const std::vector<std::vector<float>>& mel_windows,
                          const size_t feature_size,
                          const size_t nb_max_frames,
                          ov::genai::RawPerfMetrics& raw_metrics,
                          ov::genai::WhisperRawPerfMetrics& whisper_raw_metrics) {
    const size_t window_count = mel_windows.size();
    const size_t features_per_window = feature_size * nb_max_frames;

    for (size_t window = 0; window < window_count; window++) {
        OPENVINO_ASSERT(mel_windows[window].size() == features_per_window,
                        "Mel spectrogram required size: ",
                        feature_size,
                        " * ",
                        nb_max_frames,
                        ". Actual size: ",
                        mel_windows[window].size(),
                        " for window ",
                        window,
                        ".");
    }

    // NPU requires release batch 1; dynamic CPU/GPU requests use batch 0.
    auto devices = request.get_compiled_model().get_property(ov::execution_devices);
    OPENVINO_ASSERT(devices.size() > 0, "No execution devices found!");
    const size_t release_batch = (devices[0] == "NPU") ? 1 : 0;

    // Synchronous inference keeps this read-only view valid for the call.
    if (window_count == 1) {
        ov::Tensor input_view(ov::element::f32,
                              {window_count, feature_size, nb_max_frames},
                              const_cast<float*>(mel_windows.front().data()));
        return run_encoder_inference(request,
                                     input_view,
                                     {release_batch, feature_size, nb_max_frames},
                                     raw_metrics,
                                     whisper_raw_metrics);
    }

    // Multiple windows require contiguous owned storage.
    ov::Tensor input_tensor(ov::element::f32, {window_count, feature_size, nb_max_frames});
    auto* input_data = input_tensor.data<float>();
    for (size_t window = 0; window < window_count; window++) {
        std::copy(mel_windows[window].begin(), mel_windows[window].end(), input_data + window * features_per_window);
    }

    return run_encoder_inference(request,
                                 input_tensor,
                                 {release_batch, feature_size, nb_max_frames},
                                 raw_metrics,
                                 whisper_raw_metrics);
}

std::vector<ov::genai::SotTokensResult> prepare_sot_tokens(ov::Tensor& encoder_hidden_state,
                                                           const std::shared_ptr<ov::genai::WhisperDecoder>& decoder,
                                                           const ov::genai::WhisperGenerationConfig& config,
                                                           const size_t batch_size,
                                                           ov::genai::RawPerfMetrics& raw_metrics) {
    if (!config.is_multilingual) {
        // Non-multilingual Whisper models are English-only.
        return std::vector<ov::genai::SotTokensResult>(batch_size,
                                                       {std::vector<int64_t>{config.decoder_start_token_id}, "en"});
    }

    int64_t task_token_id = config.transcribe_token_id;
    if (config.task.has_value() && *config.task == "translate") {
        task_token_id = config.translate_token_id;
    }

    auto to_result = [&](const int64_t language_token_id, const std::string& language) {
        return ov::genai::SotTokensResult{
            std::vector<int64_t>{config.decoder_start_token_id, language_token_id, task_token_id},
            ov::genai::utils::to_unescaped_language(language)};
    };

    if (config.language.has_value()) {
        const std::string language = *config.language;
        const int64_t language_token_id =
            ov::genai::utils::get_or_throw_token_id_by_language(config.lang_to_id, language);
        return std::vector<ov::genai::SotTokensResult>(batch_size, to_result(language_token_id, language));
    }

    auto [language_tokens, infer_ms] = decoder->detect_languages(encoder_hidden_state, config);
    raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);

    OPENVINO_ASSERT(language_tokens.size() == batch_size,
                    "Internal error: detected ",
                    language_tokens.size(),
                    " languages for ",
                    batch_size,
                    " audio inputs.");

    std::vector<ov::genai::SotTokensResult> results;
    results.reserve(batch_size);
    for (size_t batch = 0; batch < batch_size; batch++) {
        results.push_back(
            to_result(language_tokens[batch],
                      ov::genai::utils::find_language_by_token_id(config.lang_to_id, language_tokens[batch])));
    }

    return results;
}

// Per-audio state retained across long-form rounds, in input order.
struct AudioState {
    size_t original_index = 0;
    ov::genai::WhisperFeatures features;
    size_t chunk_offset = 0;
    ov::genai::SotTokensResult sot;
    bool sot_resolved = false;
    std::vector<int64_t> output_tokens;
    std::vector<ov::genai::Segment> segments;
    std::optional<std::vector<ov::genai::WhisperWordTiming>> words;

    bool finished() const {
        return chunk_offset >= features.n_frames;
    }
};

struct SchedulerContext {
    bool retain_physical_rows;
    // Internal timestamp-token generation for this call; distinct from public
    // segment exposure (config.return_timestamps).
    bool generate_timestamps;
    const ov::genai::WhisperGenerationConfig& config;
    const ov::genai::WhisperContextTokens& context_tokens;
    ov::InferRequest& encoder;
    std::shared_ptr<ov::genai::WhisperDecoder> decoder;
    ov::genai::WhisperFeatureExtractor& feature_extractor;
    ov::genai::Sampler& sampler;
    ov::genai::Tokenizer& tokenizer;
    const std::shared_ptr<ov::genai::StreamerBase>& streamer;
    ov::genai::WhisperPerfMetrics& perf_metrics;
    float time_precision = 0.f;
    float frame_length_in_seconds = 0.f;
};

void finalize_audio_after_round(SchedulerContext& ctx,
                                AudioState& st,
                                std::vector<int64_t>& chunk_tokens,
                                bool cancelled,
                                const ov::Tensor& encoder_hidden_state) {
    const auto& config = ctx.config;
    auto& feature_extractor = ctx.feature_extractor;
    auto& perf_metrics = ctx.perf_metrics;
    const size_t window_offset = st.chunk_offset;
    const float chunk_time_offset = window_offset * ctx.frame_length_in_seconds;

    if (ctx.generate_timestamps) {
        auto extracted = ov::genai::extract_segments(chunk_tokens,
                                                     config,
                                                     feature_extractor.nb_max_frames,
                                                     ctx.time_precision,
                                                     chunk_time_offset);
        // Timestamped rounds must advance to avoid an infinite long-form loop.
        OPENVINO_ASSERT(extracted.last_offset > 0,
                        "Whisper long-form seek made no progress (last_offset == 0). Audio index ",
                        st.original_index,
                        ", chunk_offset ",
                        window_offset,
                        ", last_offset ",
                        extracted.last_offset,
                        ", n_frames ",
                        st.features.n_frames,
                        ".");

        if (!ctx.retain_physical_rows) {
            ov::genai::utils::filter_non_segment_metrics(perf_metrics.raw_metrics,
                                                         perf_metrics.whisper_raw_metrics,
                                                         st.output_tokens.size(),
                                                         extracted.segment_ranges);
        }

        st.segments.insert(st.segments.end(), extracted.segments.begin(), extracted.segments.end());
        st.output_tokens.insert(st.output_tokens.end(),
                                extracted.non_timestamp_tokens.begin(),
                                extracted.non_timestamp_tokens.end());

        // Timestamp mode streams extracted non-timestamp tokens after each window.
        if (ctx.streamer &&
            ctx.streamer->write(extracted.non_timestamp_tokens) != ov::genai::StreamingStatus::RUNNING) {
            cancelled = true;
        }

        st.chunk_offset += extracted.last_offset;
    } else {
        // generate_timestamps is false only for an all-short-form call, so every input is a single window.
        st.output_tokens.insert(st.output_tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
        st.chunk_offset = st.features.n_frames;
    }

    if (cancelled) {
        st.chunk_offset = st.features.n_frames;
        return;
    }

    if (config.word_timestamps) {
        const auto n_active_frames =
            std::min(feature_extractor.nb_max_frames, st.features.n_active_frames - window_offset);
        const auto word_timestamps_start = std::chrono::steady_clock::now();
        const auto window_words = ov::genai::add_word_level_timestamps(st.sot,
                                                                       chunk_tokens,
                                                                       ctx.tokenizer,
                                                                       ctx.decoder,
                                                                       encoder_hidden_state,
                                                                       config,
                                                                       n_active_frames,
                                                                       chunk_time_offset);
        perf_metrics.whisper_raw_metrics.word_level_timestamps_processing_durations[0] += MicroSeconds(
            ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - word_timestamps_start));
        if (!st.words.has_value()) {
            st.words = std::vector<ov::genai::WhisperWordTiming>{};
        }
        st.words->insert(st.words->end(), window_words.begin(), window_words.end());
    }
}

// Encoder rows and sequence groups follow cohort order; request IDs preserve input indices.
void run_cohort_round(SchedulerContext& ctx,
                      std::vector<AudioState>& states,
                      const std::vector<size_t>& cohort) {
    const auto& config = ctx.config;
    auto& feature_extractor = ctx.feature_extractor;
    auto& perf_metrics = ctx.perf_metrics;
    ov::genai::RawPerfMetrics& raw_metrics = perf_metrics.raw_metrics;

    std::vector<std::vector<float>> mel_windows;
    mel_windows.reserve(cohort.size());
    for (const size_t a : cohort) {
        mel_windows.emplace_back(
            states[a].features.get_data_with_offset(states[a].chunk_offset, feature_extractor.nb_max_frames));
    }

    ov::Tensor encoder_hidden_state = encode_windows(ctx.encoder,
                                                     mel_windows,
                                                     feature_extractor.feature_size,
                                                     feature_extractor.nb_max_frames,
                                                     raw_metrics,
                                                     perf_metrics.whisper_raw_metrics);

    // All rows in a cohort resolve SOT together.
    const bool sot_resolved = states[cohort.front()].sot_resolved;
    OPENVINO_ASSERT(std::all_of(cohort.begin(),
                                cohort.end(),
                                [&](size_t index) {
                                    return states[index].sot_resolved == sot_resolved;
                                }),
                    "Internal error: a Whisper cohort mixes resolved and unresolved SOT states.");
    if (!sot_resolved) {
        auto sot_results = prepare_sot_tokens(encoder_hidden_state, ctx.decoder, config, cohort.size(), raw_metrics);
        for (size_t j = 0; j < cohort.size(); j++) {
            states[cohort[j]].sot = std::move(sot_results[j]);
            states[cohort[j]].sot_resolved = true;
        }
    }

    // initial_prompt applies only at offset 0; hotwords apply to every window.
    std::vector<ov::genai::SequenceGroup::Ptr> sequence_groups;
    sequence_groups.reserve(cohort.size());
    for (size_t j = 0; j < cohort.size(); j++) {
        AudioState& st = states[cohort[j]];
        std::vector<int64_t> prompt = ov::genai::get_prompt_tokens(ctx.context_tokens, config, st.chunk_offset);
        prompt.insert(prompt.end(), st.sot.tokens.begin(), st.sot.tokens.end());
        if (!ctx.generate_timestamps) {
            prompt.push_back(config.no_timestamps_token_id);
        }
        auto sequence_group = std::make_shared<ov::genai::SequenceGroup>(st.original_index, prompt, config);
        if (config.is_beam_search()) {
            sequence_group->set_logits_type(ov::genai::LogitsType::LOG_PROBS);
        }
        sequence_groups.push_back(std::move(sequence_group));
    }

    const DecoderRoundConfig round_config{ctx.retain_physical_rows, ctx.generate_timestamps, ctx.streamer};
    auto decoded = run_decoder_round(ctx.decoder,
                                     sequence_groups,
                                     encoder_hidden_state,
                                     ctx.sampler,
                                     config,
                                     round_config,
                                     perf_metrics);

    for (size_t j = 0; j < cohort.size(); j++) {
        finalize_audio_after_round(ctx,
                                   states[cohort[j]],
                                   decoded.tokens[j],
                                   decoded.cancelled[j],
                                   encoder_hidden_state);
    }
}

void run_generation_schedule(std::vector<AudioState>& states,
                             const bool retain_physical_rows,
                             const std::shared_ptr<ov::genai::StreamerBase>& streamer,
                             const ov::genai::WhisperGenerationConfig& config,
                             const ov::genai::WhisperConfig& model_config,
                             const ov::genai::WhisperContextTokens& context_tokens,
                             ov::InferRequest& encoder,
                             std::shared_ptr<ov::genai::WhisperDecoder> decoder,
                             ov::genai::WhisperFeatureExtractor& feature_extractor,
                             ov::genai::Sampler& sampler,
                             ov::genai::Tokenizer& tokenizer,
                             ov::genai::WhisperPerfMetrics& perf_metrics) {
    OPENVINO_ASSERT(feature_extractor.sampling_rate != 0, "Sampling Rate for Feature Extractor is 0");
    const float time_precision = static_cast<float>(feature_extractor.chunk_length) / model_config.max_source_positions;
    const float frame_length_in_seconds =
        static_cast<float>(feature_extractor.hop_length) / feature_extractor.sampling_rate;

    // Long-form decoding needs timestamps for seek advancement.
    // Timestamp decoding is batch-wide if any input is long-form.
    const bool any_input_is_long_form =
        std::any_of(states.begin(), states.end(), [&](const AudioState& state) {
            return state.features.n_frames > feature_extractor.nb_max_frames;
        });
    const bool generate_timestamps = config.return_timestamps || any_input_is_long_form;

    SchedulerContext ctx{retain_physical_rows,
                         generate_timestamps,
                         config,
                         context_tokens,
                         encoder,
                         decoder,
                         feature_extractor,
                         sampler,
                         tokenizer,
                         streamer,
                         perf_metrics,
                         time_precision,
                         frame_length_in_seconds};

    std::vector<size_t> active;

    while (true) {
        active.clear();
        for (size_t i = 0; i < states.size(); i++) {
            if (states[i].finished()) {
                continue;
            }
            active.push_back(i);
        }
        if (active.empty()) {
            break;
        }

        run_cohort_round(ctx, states, active);
    }

    if (streamer) {
        streamer->end();
    }
}

}  // namespace

namespace ov {
namespace genai {

std::vector<WhisperGenerateResult> whisper_generate(const ov::genai::WhisperGenerationConfig& config,
                                                    const ov::genai::WhisperConfig& model_config,
                                                    const WhisperContextTokens& context_tokens,
                                                    const RawSpeechInput* raw_speeches,
                                                    size_t batch_size,
                                                    ov::InferRequest& encoder,
                                                    std::shared_ptr<WhisperDecoder> decoder,
                                                    WhisperFeatureExtractor& feature_extractor,
                                                    const std::shared_ptr<StreamerBase> streamer,
                                                    Sampler& sampler,
                                                    Tokenizer& tokenizer) {
    OPENVINO_ASSERT(batch_size > 0, "whisper_generate requires at least one audio input.");
    OPENVINO_ASSERT(raw_speeches != nullptr, "whisper_generate received a null audio input pointer.");

    WhisperPerfMetrics perf_metrics;
    init_whisper_perf_metrics(perf_metrics);
    const size_t max_new_tokens = config.get_max_new_tokens();
    perf_metrics.raw_metrics.m_new_token_times.reserve(max_new_tokens);
    perf_metrics.raw_metrics.m_batch_sizes.reserve(max_new_tokens);
    perf_metrics.raw_metrics.m_token_infer_durations.reserve(max_new_tokens);

    std::vector<AudioState> states;
    states.reserve(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
        const auto extract_start = std::chrono::steady_clock::now();
        auto features = feature_extractor.extract(raw_speeches[i]);
        perf_metrics.whisper_raw_metrics.features_extraction_durations.emplace_back(
            ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - extract_start));

        AudioState state;
        state.original_index = i;
        state.features = std::move(features);
        states.push_back(std::move(state));
    }

    run_generation_schedule(states,
                            /*retain_physical_rows=*/batch_size > 1,
                            streamer,
                            config,
                            model_config,
                            context_tokens,
                            encoder,
                            decoder,
                            feature_extractor,
                            sampler,
                            tokenizer,
                            perf_metrics);

    std::vector<WhisperGenerateResult> results(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
        results[i].output_tokens = std::move(states[i].output_tokens);
        results[i].language = states[i].sot.language;
        results[i].words = std::move(states[i].words);
        // Long-form decoding may use timestamps internally; expose segments only when requested.
        if (config.return_timestamps) {
            results[i].segments = std::move(states[i].segments);
        }
        results[i].perf_metrics = perf_metrics;
    }

    return results;
}
}  // namespace genai
}  // namespace ov
