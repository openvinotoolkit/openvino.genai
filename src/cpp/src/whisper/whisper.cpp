// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "whisper.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <numeric>
#include <openvino/openvino.hpp>
#include <sstream>
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

// TEMPORARY DIAGNOSTIC (macOS 14 CI Concat-exception investigation, GENAI_WHISPER_SHAPE_TRACE=1).
std::atomic<int64_t> g_shape_trace_call_counter{0};

// Convert one logits row (batch_idx) to log-probabilities in place, over the full vocabulary.
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

// row_sequences[i] supplies the timestamp history for physical_rows[i].
// Initial steps do not require sequence history.
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

// Initializes metrics common to scalar and batch generation.
void init_whisper_perf_metrics(ov::genai::WhisperPerfMetrics& perf_metrics) {
    perf_metrics.num_input_tokens = 0;
    perf_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
    perf_metrics.whisper_raw_metrics.word_level_timestamps_processing_durations = {{MicroSeconds(0.0f)}};
}

// Builds a decoder-owned prefill tensor from non-empty prompts of equal length.
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

// Builds the identity state-row mapping used for prefill and fixed-width decoding.
ov::Tensor make_identity_beam_idx(const std::shared_ptr<ov::genai::WhisperDecoder>& decoder, const size_t width) {
    ov::Tensor beam_idx = decoder->create_host_tensor(ov::element::i32, {width});
    std::iota(beam_idx.data<int32_t>(), beam_idx.data<int32_t>() + width, 0);
    return beam_idx;
}

// Call-level feature policy, independent of the physical decoder-row layout.
enum class GenerationMode { SCALAR, BATCH };

// Physical decoder-row policy within a generation round.
enum class DecoderRowLayout {
    PACKED_RUNNING_SEQUENCES,  // Dynamic width, one row per running sequence.
    FIXED_WIDTH_WITH_FILLERS,  // Fixed per-input rows preserve cross-attention identity.
};

// Scalar generation uses packed rows; B > 1 preserves fixed per-input row identity.
DecoderRowLayout select_row_layout(GenerationMode mode) {
    return mode == GenerationMode::SCALAR ? DecoderRowLayout::PACKED_RUNNING_SEQUENCES
                                          : DecoderRowLayout::FIXED_WIDTH_WITH_FILLERS;
}

// Decoder tensors reused across token steps within a round.
struct DecodeLayoutState {
    ov::Tensor beam_idx_tensor;
    ov::Tensor next_input_ids_tensor;
};

// Mutable row plan rebuilt for each token step while retaining vector capacity.
struct DecoderStepPlan {
    // Groups passed to the sampler in consumption order; one group may own multiple packed rows.
    std::vector<ov::genai::SequenceGroup::Ptr> active_groups;

    // live_sequences[i] occupies decoder row live_physical_rows[i].
    std::vector<ov::genai::Sequence::Ptr> live_sequences;
    std::vector<size_t> live_physical_rows;

    // PACKED state-row mapping; empty for FIXED.
    std::vector<int32_t> beam_idx;

    size_t physical_width = 0;
    size_t sampled_row_count = 0;

    // Reserve once, up front, for the worst-case number of packed rows / active groups.
    void reserve(size_t capacity) {
        active_groups.reserve(capacity);
        live_sequences.reserve(capacity);
        live_physical_rows.reserve(capacity);
        beam_idx.reserve(capacity);
    }

    // Reset per-step contents while retaining vector capacity.
    void clear() {
        active_groups.clear();
        live_sequences.clear();
        live_physical_rows.clear();
        beam_idx.clear();
        physical_width = 0;
        sampled_row_count = 0;
    }
};

// Allocates fixed-width decoder tensors once per batch round.
DecodeLayoutState make_fixed_width_layout_state(const std::shared_ptr<ov::genai::WhisperDecoder>& decoder,
                                                const size_t batch_size) {
    DecodeLayoutState state;
    state.beam_idx_tensor = make_identity_beam_idx(decoder, batch_size);
    state.next_input_ids_tensor = decoder->create_host_tensor(ov::element::i64, {batch_size, 1});
    return state;
}

// Builds the row plan for the next token step; no active groups means generation is complete.
void collect_generation_step(DecoderRowLayout layout,
                             const std::vector<ov::genai::SequenceGroup::Ptr>& sequence_groups,
                             ov::genai::Sampler& sampler,
                             DecoderStepPlan& plan) {
    plan.clear();

    if (layout == DecoderRowLayout::PACKED_RUNNING_SEQUENCES) {
        OPENVINO_ASSERT(sequence_groups.size() == 1,
                        "Internal error: packed decoder layout expects exactly one sequence group, got ",
                        sequence_groups.size(),
                        ".");
        const auto& sequence_group = sequence_groups.front();
        if (sequence_group->has_finished() || sequence_group->handle_stopped() || sequence_group->handle_cancelled()) {
            return;
        }

        sequence_group->schedule_tokens(1);

        const std::vector<ov::genai::Sequence::Ptr> running_sequences = sequence_group->get_running_sequences();
        const size_t num_scheduled_tokens = sequence_group->get_num_scheduled_tokens();
        OPENVINO_ASSERT(num_scheduled_tokens == 1,
                        "Internal error: Whisper generation schedules exactly one token per step, got ",
                        num_scheduled_tokens,
                        ".");
        const std::map<size_t, int32_t> beam_idxs = sampler.get_beam_idxs(sequence_group);

        plan.active_groups.push_back(sequence_group);
        for (size_t row = 0; row < running_sequences.size(); row++) {
            const auto& sequence = running_sequences[row];
            plan.live_sequences.push_back(sequence);
            plan.live_physical_rows.push_back(row);
            plan.beam_idx.push_back(beam_idxs.at(sequence->get_id()));
        }
        plan.physical_width = running_sequences.size();
        plan.sampled_row_count = running_sequences.size();
    } else {
        const size_t batch_size = sequence_groups.size();
        plan.physical_width = batch_size;

        for (size_t request = 0; request < batch_size; request++) {
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

            plan.active_groups.push_back(sequence_group);
            plan.live_sequences.push_back(running_sequences.front());
            plan.live_physical_rows.push_back(request);
        }
        plan.sampled_row_count = plan.live_physical_rows.size();
    }

    OPENVINO_ASSERT(plan.live_sequences.size() == plan.live_physical_rows.size() &&
                        plan.live_sequences.size() == plan.sampled_row_count,
                    "Internal error: generation-step row bookkeeping is inconsistent.");
}

// Populates live rows with their next token; FIXED initializes inactive rows with filler_token.
void fill_next_input_ids(ov::Tensor& next_input_ids,
                         const DecoderStepPlan& plan,
                         DecoderRowLayout layout,
                         const int64_t filler_token) {
    auto* data = next_input_ids.data<int64_t>();
    if (layout == DecoderRowLayout::FIXED_WIDTH_WITH_FILLERS) {
        std::fill_n(data, plan.physical_width, filler_token);
    }

    for (size_t i = 0; i < plan.live_sequences.size(); i++) {
        const auto& sequence_group =
            layout == DecoderRowLayout::PACKED_RUNNING_SEQUENCES ? plan.active_groups.front() : plan.active_groups[i];
        const auto& sequence = plan.live_sequences[i];
        const size_t num_processed_tokens = sequence_group->get_num_processed_tokens();
        const size_t request_prompt_len = sequence_group->get_prompt_len();
        data[plan.live_physical_rows[i]] =
            num_processed_tokens < request_prompt_len
                ? sequence_group->get_prompt_ids()[num_processed_tokens]
                : sequence->get_generated_ids()[num_processed_tokens - request_prompt_len];
    }
}

// Updates the PACKED state-row mapping, resizing the tensor as beam width changes.
void sync_beam_idx_tensor(ov::Tensor& beam_idx_tensor, const std::vector<int32_t>& beam_idx) {
    if (beam_idx_tensor.get_shape().at(0) != beam_idx.size()) {
        beam_idx_tensor.set_shape({beam_idx.size()});
    }
    std::copy_n(beam_idx.data(), beam_idx.size(), beam_idx_tensor.data<int32_t>());
}

// Initializes scalar packed layout; its decoder width may change as beams fork or finish.
DecodeLayoutState make_packed_layout_state(const std::shared_ptr<ov::genai::WhisperDecoder>& decoder,
                                           const size_t batch_size) {
    DecodeLayoutState state;
    state.beam_idx_tensor = make_identity_beam_idx(decoder, batch_size);
    return state;
}

// Packs live rows in sampler order, using zero-copy full or prefix views when possible.
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

// Timestamp mode streams during segment extraction; batch generation passes no streamer.
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

// PACKED allocates for the current beam width; FIXED reuses its round-local tensor.
ov::Tensor prepare_next_input_ids(DecodeLayoutState& layout,
                                  const DecoderStepPlan& plan,
                                  const DecoderRowLayout row_layout,
                                  const int64_t filler_token) {
    if (row_layout == DecoderRowLayout::PACKED_RUNNING_SEQUENCES) {
        ov::Tensor next_input_ids(ov::element::i64, {plan.physical_width, 1});
        fill_next_input_ids(next_input_ids, plan, row_layout, filler_token);
        return next_input_ids;
    }
    fill_next_input_ids(layout.next_input_ids_tensor, plan, row_layout, filler_token);
    return layout.next_input_ids_tensor;
}

// PACKED applies the sampler's state-row mapping; FIXED reuses the prefill identity.
ov::Tensor& select_beam_idx_tensor(DecodeLayoutState& layout,
                                   const DecoderStepPlan& plan,
                                   const DecoderRowLayout row_layout) {
    if (row_layout == DecoderRowLayout::PACKED_RUNNING_SEQUENCES) {
        OPENVINO_ASSERT(plan.beam_idx.size() == plan.physical_width,
                        "Internal error: beam-index row count ",
                        plan.beam_idx.size(),
                        " must equal the decoder physical width ",
                        plan.physical_width,
                        ".");
        sync_beam_idx_tensor(layout.beam_idx_tensor, plan.beam_idx);
    }
    return layout.beam_idx_tensor;
}

// Resets decoder state on normal and exceptional exits. During unwinding, it first settles any in-flight request
// and suppresses cleanup failures.
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

    // Mark complete only after reset so the destructor retries if reset_state() throws.
    void finish() {
        decoder->reset_state();
        finished = true;
    }

    ~DecoderRoundResetGuard() {
        if (finished) {
            return;
        }
        // Cleanup failures must not escape the destructor.
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

// Options that differ between scalar PACKED and batched FIXED rounds.
struct DecoderRoundConfig {
    DecoderRowLayout layout;
    bool return_timestamps = false;
    int64_t filler_token = 0;
    std::shared_ptr<ov::genai::StreamerBase> streamer = nullptr;
};

// Shared post-prefill token loop for PACKED and FIXED decoder-row layouts.
void run_decoder_generation_loop(const std::shared_ptr<ov::genai::WhisperDecoder>& decoder,
                                 const ov::Tensor& encoder_hidden_state,
                                 ov::genai::Sampler& sampler,
                                 const std::vector<ov::genai::SequenceGroup::Ptr>& sequence_groups,
                                 DecodeLayoutState& layout,
                                 DecoderStepPlan& plan,
                                 DecoderRoundResetGuard& reset_guard,
                                 const DecoderRoundConfig& round_config,
                                 const ov::genai::WhisperGenerationConfig& config,
                                 const std::shared_ptr<ov::genai::GenerationHandleImpl>& handle,
                                 ov::genai::WhisperPerfMetrics& perf_metrics,
                                 const std::string& trace_ctx) {
    const DecoderRowLayout row_layout = round_config.layout;
    const bool return_timestamps = round_config.return_timestamps;
    ov::genai::RawPerfMetrics& raw_metrics = perf_metrics.raw_metrics;
    ov::genai::WhisperRawPerfMetrics& whisper_raw_metrics = perf_metrics.whisper_raw_metrics;

    size_t trace_step = 0;
    while (true) {
        collect_generation_step(row_layout, sequence_groups, sampler, plan);
        if (plan.active_groups.empty()) {
            break;
        }
        trace_step++;

        const ov::Tensor next_input_ids = prepare_next_input_ids(layout, plan, row_layout, round_config.filler_token);

        ov::genai::utils::whisper_shape_trace(trace_ctx,
                                              " phase=step step=",
                                              trace_step,
                                              " physical_width=",
                                              plan.physical_width,
                                              " live_rows=",
                                              plan.live_physical_rows.size());

        const auto infer_start = std::chrono::steady_clock::now();

        ov::Tensor& beam_idx = select_beam_idx_tensor(layout, plan, row_layout);

        reset_guard.mark_started();
        decoder->start_async(encoder_hidden_state, next_input_ids, beam_idx);

        // Preserve callback/inference overlap; the guard settles the request if streaming throws.
        stream_generated_tokens(round_config.streamer, handle, return_timestamps);

        auto logits = decoder->wait();
        reset_guard.mark_waited();

        const auto infer_end = std::chrono::steady_clock::now();
        record_decode_inference_metrics(raw_metrics, whisper_raw_metrics, infer_start, infer_end);
        raw_metrics.m_batch_sizes.emplace_back(plan.sampled_row_count);

        process_whisper_logits(logits,
                               config,
                               return_timestamps,
                               /*initial_step=*/false,
                               plan.live_physical_rows,
                               plan.live_sequences);

        OPENVINO_ASSERT(logits.get_shape().at(0) == plan.physical_width,
                        "Internal error: decoder logits row count ",
                        logits.get_shape().at(0),
                        " does not match the physical width ",
                        plan.physical_width,
                        ".");

        const ov::Tensor sampled_logits = pack_live_logits(logits, plan.live_physical_rows);
        OPENVINO_ASSERT(sampled_logits.get_shape().at(0) == plan.sampled_row_count,
                        "Internal error: packed logits row count ",
                        sampled_logits.get_shape().at(0),
                        " does not match the sampled row count ",
                        plan.sampled_row_count,
                        ".");

        const auto sample_start = std::chrono::steady_clock::now();
        sampler.sample(plan.active_groups, sampled_logits);
        raw_metrics.m_sampling_durations.emplace_back(
            ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - sample_start));
    }
}

// Token and cancellation results in sequence-group order.
struct DecoderRoundResult {
    std::vector<std::vector<int64_t>> tokens;
    std::vector<bool> cancelled;
};

// Clears per-request sampler state on every exit path; request IDs may be reused by later calls.
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

// Runs prefill and token generation for one PACKED or FIXED decoder round.
DecoderRoundResult run_decoder_round(std::shared_ptr<ov::genai::WhisperDecoder> decoder,
                                     const std::vector<ov::genai::SequenceGroup::Ptr>& sequence_groups,
                                     const ov::Tensor& encoder_hidden_state,
                                     ov::genai::Sampler& sampler,
                                     const ov::genai::WhisperGenerationConfig& config,
                                     const DecoderRoundConfig& round_config,
                                     ov::genai::WhisperPerfMetrics& perf_metrics,
                                     const std::string& trace_ctx) {
    OPENVINO_ASSERT(!sequence_groups.empty(), "Whisper decoder round requires at least one sequence group.");

    OPENVINO_ASSERT(!round_config.streamer || sequence_groups.size() == 1,
                    "Internal error: Whisper streaming requires exactly one sequence group.");

    ov::genai::RawPerfMetrics& raw_metrics = perf_metrics.raw_metrics;
    ov::genai::WhisperRawPerfMetrics& whisper_raw_metrics = perf_metrics.whisper_raw_metrics;

    // Prefill has one decoder row per sequence group.
    const size_t prefill_width = sequence_groups.size();

    // Streaming is scalar-only, so the handle belongs to the sole sequence group.
    std::shared_ptr<ov::genai::GenerationHandleImpl> handle;
    if (round_config.streamer) {
        const auto& sequence_group = sequence_groups.front();
        handle = std::make_shared<ov::genai::GenerationHandleImpl>(sequence_group->get_generation_stream(),
                                                                   sequence_group->get_sampling_parameters());
    }

    DecodeLayoutState layout = round_config.layout == DecoderRowLayout::PACKED_RUNNING_SEQUENCES
                                   ? make_packed_layout_state(decoder, prefill_width)
                                   : make_fixed_width_layout_state(decoder, prefill_width);

    // Cover sampler registrations created during prefill or generation.
    SamplerRequestCleanup request_cleanup{sampler};
    request_cleanup.request_ids.reserve(sequence_groups.size());
    for (const auto& sequence_group : sequence_groups) {
        request_cleanup.request_ids.push_back(sequence_group->get_request_id());
    }

    DecoderRoundResetGuard reset_guard{decoder};
    const ov::Tensor input_ids_tensor = build_prefill_input_ids(decoder, sequence_groups);
    ov::genai::utils::whisper_shape_trace(trace_ctx, " phase=prefill");
    const auto infer_start = std::chrono::steady_clock::now();
    reset_guard.mark_started();
    decoder->start_async(encoder_hidden_state, input_ids_tensor, layout.beam_idx_tensor);

    auto logits = decoder->wait();
    reset_guard.mark_waited();
    const auto infer_end = std::chrono::steady_clock::now();
    record_decode_inference_metrics(raw_metrics, whisper_raw_metrics, infer_start, infer_end);
    // Prefill has one decoder row per sequence group.
    raw_metrics.m_batch_sizes.emplace_back(prefill_width);

    // All prefill rows are live.
    std::vector<size_t> prefill_rows(prefill_width);
    std::iota(prefill_rows.begin(), prefill_rows.end(), size_t{0});
    process_whisper_logits(logits, config, round_config.return_timestamps, /*initial_step=*/true, prefill_rows, {});

    // Sample from the final prompt position.
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

    // Reuse one step plan, reserving for the maximum scalar beam width.
    DecoderStepPlan plan;
    plan.reserve(std::max<size_t>(prefill_width, config.num_beams));
    run_decoder_generation_loop(decoder,
                                encoder_hidden_state,
                                sampler,
                                sequence_groups,
                                layout,
                                plan,
                                reset_guard,
                                round_config,
                                config,
                                handle,
                                perf_metrics,
                                trace_ctx);

    stream_generated_tokens(round_config.streamer, handle, round_config.return_timestamps);

    // Collect results in sequence-group order, which matches input order.
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

    // Perform the observable normal-path reset after collecting results.
    reset_guard.finish();
    ov::genai::utils::whisper_shape_trace(trace_ctx, " action=reset_state_done");

    return result;
}

// Runs encoder inference, releases the bound input, and returns the request-owned hidden states.
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

    // Release caller-owned input memory from the request.
    request.set_tensor("input_features", ov::Tensor(ov::element::f32, release_shape));

    return request.get_tensor("last_hidden_state");
}

// Encodes one window per batch row, preserving window order.
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

    // Avoid a scalar copy by viewing the caller-owned window. Inference is synchronous and the model input is
    // read-only, so the const_cast view remains valid for the call.
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

// Resolves SOT tokens in encoder-row order, detecting one language per input when unspecified.
std::vector<ov::genai::SotTokensResult> prepare_sot_tokens(ov::Tensor& encoder_hidden_state,
                                                           const std::shared_ptr<ov::genai::WhisperDecoder>& decoder,
                                                           const ov::genai::WhisperGenerationConfig& config,
                                                           const size_t batch_size,
                                                           ov::genai::RawPerfMetrics& raw_metrics,
                                                           const std::string& trace_ctx) {
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

    // A configured language applies to every input.
    if (config.language.has_value()) {
        const std::string language = *config.language;
        const int64_t language_token_id =
            ov::genai::utils::get_or_throw_token_id_by_language(config.lang_to_id, language);
        return std::vector<ov::genai::SotTokensResult>(batch_size, to_result(language_token_id, language));
    }

    ov::genai::utils::whisper_shape_trace(trace_ctx, " phase=lang_detect");
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

// Per-input state retained across scheduler rounds. States remain in input order; cohorts store indices into them.
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

// Services and call-level state shared across cohort rounds.
struct SchedulerContext {
    GenerationMode mode;
    DecoderRowLayout row_layout;
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
    // TEMPORARY DIAGNOSTIC fields (GENAI_WHISPER_SHAPE_TRACE=1).
    int64_t trace_call_id = 0;
    int64_t trace_round_id = 0;
};

// Applies one decoder round's output to an input, accumulating results and advancing its seek.
void finalize_audio_after_round(SchedulerContext& ctx,
                                AudioState& st,
                                std::vector<int64_t>& chunk_tokens,
                                bool cancelled,
                                const bool cohort_return_timestamps,
                                const ov::Tensor& encoder_hidden_state) {
    const auto& config = ctx.config;
    auto& feature_extractor = ctx.feature_extractor;
    auto& perf_metrics = ctx.perf_metrics;
    const size_t window_offset = st.chunk_offset;
    const float chunk_time_offset = window_offset * ctx.frame_length_in_seconds;

    if (cohort_return_timestamps) {
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

        if (ctx.mode == GenerationMode::SCALAR) {
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
        // Without timestamps, the cohort contains only single-window inputs.
        st.output_tokens.insert(st.output_tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
        st.chunk_offset = st.features.n_frames;
    }
    ov::genai::utils::whisper_shape_trace("CHUNK_ADVANCE call=",
                                          ctx.trace_call_id,
                                          " round=",
                                          ctx.trace_round_id,
                                          " orig=",
                                          st.original_index,
                                          " new_chunk_offset=",
                                          st.chunk_offset,
                                          " n_frames=",
                                          st.features.n_frames,
                                          " cancelled=",
                                          cancelled);

    if (cancelled) {
        // Stop processing further windows after streamer termination.
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

// Runs a timestamp-compatible cohort in input order. Encoder rows and sequence groups align, while request IDs
// retain original input indices.
void run_cohort_round(SchedulerContext& ctx,
                      std::vector<AudioState>& states,
                      const std::vector<size_t>& cohort,
                      const bool cohort_return_timestamps,
                      std::vector<std::vector<float>>& mel_windows,
                      std::vector<ov::genai::SequenceGroup::Ptr>& sequence_groups) {
    const auto& config = ctx.config;
    auto& feature_extractor = ctx.feature_extractor;
    auto& perf_metrics = ctx.perf_metrics;
    ov::genai::RawPerfMetrics& raw_metrics = perf_metrics.raw_metrics;

    // Diagnostic-only context construction. Swallow-only: the stream objects and .str() calls here
    // allocate, and idx_list.str() is evaluated as a trace argument (before the helper's own guard),
    // so this block must contain its own failures. On failure trace_ctx stays empty and later trace
    // lines simply carry less context; no production operation lives in this block.
    std::string trace_ctx;
    try {
        if (ov::genai::utils::whisper_shape_trace_enabled()) {
            std::ostringstream idx_list;
            for (const size_t a : cohort) {
                idx_list << states[a].original_index << " ";
            }
            ov::genai::utils::whisper_shape_trace("COHORT_ROUND call=",
                                                  ctx.trace_call_id,
                                                  " round=",
                                                  ctx.trace_round_id,
                                                  " return_ts=",
                                                  cohort_return_timestamps,
                                                  " width=",
                                                  cohort.size(),
                                                  " orig_indices=[",
                                                  idx_list.str(),
                                                  "]");

            std::ostringstream ctx_oss;
            ctx_oss << "call=" << ctx.trace_call_id << " round=" << ctx.trace_round_id
                    << " cohort=" << (cohort_return_timestamps ? "ts" : "no_ts") << " width=" << cohort.size();
            trace_ctx = ctx_oss.str();
        }
    } catch (...) {
        trace_ctx.clear();
    }

    mel_windows.clear();
    mel_windows.reserve(cohort.size());
    for (const size_t a : cohort) {
        mel_windows.emplace_back(
            states[a].features.get_data_with_offset(states[a].chunk_offset, feature_extractor.nb_max_frames));
    }

    // Encode cohort windows in cohort order.
    ov::Tensor encoder_hidden_state = encode_windows(ctx.encoder,
                                                     mel_windows,
                                                     feature_extractor.feature_size,
                                                     feature_extractor.nb_max_frames,
                                                     raw_metrics,
                                                     perf_metrics.whisper_raw_metrics);

    ctx.decoder->start_new_round();
    ov::genai::utils::whisper_shape_trace(trace_ctx, " action=start_new_round");

    // Cohorts may shrink but never gain unresolved inputs, so SOT resolution is uniform within a cohort.
    const bool sot_resolved = states[cohort.front()].sot_resolved;
    OPENVINO_ASSERT(std::all_of(cohort.begin(),
                                cohort.end(),
                                [&](size_t index) {
                                    return states[index].sot_resolved == sot_resolved;
                                }),
                    "Internal error: a Whisper cohort mixes resolved and unresolved SOT states.");
    if (!sot_resolved) {
        auto sot_results =
            prepare_sot_tokens(encoder_hidden_state, ctx.decoder, config, cohort.size(), raw_metrics, trace_ctx);
        for (size_t j = 0; j < cohort.size(); j++) {
            states[cohort[j]].sot = std::move(sot_results[j]);
            states[cohort[j]].sot_resolved = true;
        }
    }

    // Build groups in cohort order; initial_prompt applies at offset 0, while hotwords apply every window.
    sequence_groups.clear();
    sequence_groups.reserve(cohort.size());
    for (size_t j = 0; j < cohort.size(); j++) {
        AudioState& st = states[cohort[j]];
        std::vector<int64_t> prompt = ov::genai::get_prompt_tokens(ctx.context_tokens, config, st.chunk_offset);
        prompt.insert(prompt.end(), st.sot.tokens.begin(), st.sot.tokens.end());
        if (!cohort_return_timestamps) {
            prompt.push_back(config.no_timestamps_token_id);
        }
        auto sequence_group = std::make_shared<ov::genai::SequenceGroup>(st.original_index, prompt, config);
        if (config.is_beam_search()) {
            sequence_group->set_logits_type(ov::genai::LogitsType::LOG_PROBS);
        }
        sequence_groups.push_back(std::move(sequence_group));
    }

    const DecoderRoundConfig round_config{ctx.row_layout,
                                          cohort_return_timestamps,
                                          config.decoder_start_token_id,
                                          ctx.streamer};
    auto decoded = run_decoder_round(ctx.decoder,
                                     sequence_groups,
                                     encoder_hidden_state,
                                     ctx.sampler,
                                     config,
                                     round_config,
                                     perf_metrics,
                                     trace_ctx);

    for (size_t j = 0; j < cohort.size(); j++) {
        finalize_audio_after_round(ctx,
                                   states[cohort[j]],
                                   decoded.tokens[j],
                                   decoded.cancelled[j],
                                   cohort_return_timestamps,
                                   encoder_hidden_state);
    }
}

// Repeatedly partitions unfinished inputs by timestamp mode and runs each cohort in original input order.
void run_generation_schedule(std::vector<AudioState>& states,
                             const GenerationMode mode,
                             const std::shared_ptr<ov::genai::StreamerBase>& streamer,
                             const ov::genai::WhisperGenerationConfig& config,
                             const ov::genai::WhisperConfig& model_config,
                             const ov::genai::WhisperContextTokens& context_tokens,
                             ov::InferRequest& encoder,
                             std::shared_ptr<ov::genai::WhisperDecoder> decoder,
                             ov::genai::WhisperFeatureExtractor& feature_extractor,
                             ov::genai::Sampler& sampler,
                             ov::genai::Tokenizer& tokenizer,
                             ov::genai::WhisperPerfMetrics& perf_metrics,
                             const int64_t trace_call_id) {
    OPENVINO_ASSERT(feature_extractor.sampling_rate != 0, "Sampling Rate for Feature Extractor is 0");
    const float time_precision = static_cast<float>(feature_extractor.chunk_length) / model_config.max_source_positions;
    const float frame_length_in_seconds =
        static_cast<float>(feature_extractor.hop_length) / feature_extractor.sampling_rate;

    SchedulerContext ctx{mode,
                         select_row_layout(mode),
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
    ctx.trace_call_id = trace_call_id;

    // Long-form inputs require internal timestamps for seek advancement even when return_timestamps is false.
    const auto wants_timestamps = [&](const AudioState& state) {
        return config.return_timestamps || (state.features.n_frames > feature_extractor.nb_max_frames);
    };

    // Reuse cohort buffers across scheduler rounds.
    std::vector<size_t> ts_cohort;
    std::vector<size_t> no_ts_cohort;
    std::vector<std::vector<float>> mel_windows;
    std::vector<ov::genai::SequenceGroup::Ptr> sequence_groups;

    int64_t round_id = 0;
    while (true) {
        round_id++;
        ctx.trace_round_id = round_id;
        ts_cohort.clear();
        no_ts_cohort.clear();
        for (size_t i = 0; i < states.size(); i++) {
            if (states[i].finished()) {
                continue;
            }
            (wants_timestamps(states[i]) ? ts_cohort : no_ts_cohort).push_back(i);
        }
        if (ov::genai::utils::whisper_shape_trace_enabled()) {
            for (size_t i = 0; i < states.size(); i++) {
                ov::genai::utils::whisper_shape_trace("STATE call=",
                                                      trace_call_id,
                                                      " round=",
                                                      round_id,
                                                      " idx=",
                                                      i,
                                                      " orig=",
                                                      states[i].original_index,
                                                      " chunk_offset=",
                                                      states[i].chunk_offset,
                                                      " n_frames=",
                                                      states[i].features.n_frames,
                                                      " finished=",
                                                      states[i].finished());
            }
            ov::genai::utils::whisper_shape_trace("COHORTS call=",
                                                  trace_call_id,
                                                  " round=",
                                                  round_id,
                                                  " no_ts_size=",
                                                  no_ts_cohort.size(),
                                                  " ts_size=",
                                                  ts_cohort.size());
        }
        if (ts_cohort.empty() && no_ts_cohort.empty()) {
            break;
        }

        if (!no_ts_cohort.empty()) {
            run_cohort_round(ctx,
                             states,
                             no_ts_cohort,
                             /*cohort_return_timestamps=*/false,
                             mel_windows,
                             sequence_groups);
        }
        if (!ts_cohort.empty()) {
            run_cohort_round(ctx, states, ts_cohort, /*cohort_return_timestamps=*/true, mel_windows, sequence_groups);
        }
    }

    if (streamer) {
        streamer->end();
    }
}

}  // namespace

namespace ov {
namespace genai {

WhisperGenerateResult whisper_generate(const ov::genai::WhisperGenerationConfig& config,
                                       const ov::genai::WhisperConfig& model_config,
                                       const WhisperContextTokens& context_tokens,
                                       const RawSpeechInput& raw_speech,
                                       ov::InferRequest& encoder,
                                       std::shared_ptr<WhisperDecoder> decoder,
                                       WhisperFeatureExtractor& feature_extractor,
                                       const std::shared_ptr<StreamerBase> streamer,
                                       Sampler& sampler,
                                       Tokenizer& tokenizer) {
    // Adapt scalar generation to the shared scheduler and shape the result.
    const int64_t trace_call_id = ++g_shape_trace_call_counter;
    utils::whisper_shape_trace("CALL call=", trace_call_id, " mode=SCALAR");
    const size_t max_new_tokens = config.get_max_new_tokens();

    WhisperGenerateResult result;
    init_whisper_perf_metrics(result.perf_metrics);
    RawPerfMetrics& raw_metrics = result.perf_metrics.raw_metrics;
    raw_metrics.m_new_token_times.reserve(max_new_tokens);
    raw_metrics.m_batch_sizes.reserve(max_new_tokens);
    raw_metrics.m_token_infer_durations.reserve(max_new_tokens);

    const auto extract_start = std::chrono::steady_clock::now();
    auto features = feature_extractor.extract(raw_speech);
    result.perf_metrics.whisper_raw_metrics.features_extraction_durations.emplace_back(
        ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - extract_start));

    std::vector<AudioState> states(1);
    states[0].original_index = 0;
    states[0].features = std::move(features);

    run_generation_schedule(states,
                            GenerationMode::SCALAR,
                            streamer,
                            config,
                            model_config,
                            context_tokens,
                            encoder,
                            decoder,
                            feature_extractor,
                            sampler,
                            tokenizer,
                            result.perf_metrics,
                            trace_call_id);

    result.output_tokens = std::move(states[0].output_tokens);
    result.language = states[0].sot.language;
    result.words = std::move(states[0].words);
    // Expose segments only for public return_timestamps; long-form may use timestamps internally.
    if (config.return_timestamps) {
        result.segments = std::move(states[0].segments);
    }

    return result;
}

std::vector<WhisperGenerateResult> whisper_generate_batch(const ov::genai::WhisperGenerationConfig& config,
                                                          const ov::genai::WhisperConfig& model_config,
                                                          const WhisperContextTokens& context_tokens,
                                                          const std::vector<RawSpeechInput>& raw_speeches,
                                                          ov::InferRequest& encoder,
                                                          std::shared_ptr<WhisperDecoder> decoder,
                                                          WhisperFeatureExtractor& feature_extractor,
                                                          Sampler& sampler,
                                                          Tokenizer& tokenizer) {
    // Adapt B > 1 inputs to the shared scheduler and return results in input order.
    const size_t batch_size = raw_speeches.size();
    const int64_t trace_call_id = ++g_shape_trace_call_counter;
    utils::whisper_shape_trace("CALL call=", trace_call_id, " mode=BATCH batch_size=", batch_size);
    OPENVINO_ASSERT(batch_size > 1,
                    "Internal error: whisper_generate_batch expects a batch size greater than one; the pipeline "
                    "delegates single-audio (B == 1) generation to the scalar path. Got ",
                    batch_size,
                    ".");
    // Repeat algorithm-level invariants defensively; device and streamer checks belong to the public pipeline.
    OPENVINO_ASSERT(config.num_beams == 1,
                    "Batched Whisper generation supports greedy decoding only. Got num_beams = ",
                    config.num_beams,
                    ".");
    OPENVINO_ASSERT(!config.do_sample,
                    "Batched Whisper generation supports greedy decoding only ('do_sample' unsupported for B > 1).");
    OPENVINO_ASSERT(!config.word_timestamps, "'word_timestamps' is not supported for batched Whisper generation.");

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
                            GenerationMode::BATCH,
                            /*streamer=*/nullptr,
                            config,
                            model_config,
                            context_tokens,
                            encoder,
                            decoder,
                            feature_extractor,
                            sampler,
                            tokenizer,
                            perf_metrics,
                            trace_call_id);

    std::vector<WhisperGenerateResult> results(batch_size);
    for (size_t i = 0; i < batch_size; i++) {
        results[i].output_tokens = std::move(states[i].output_tokens);
        results[i].language = states[i].sot.language;
        if (config.return_timestamps) {
            results[i].segments = std::move(states[i].segments);
        }
        results[i].perf_metrics = perf_metrics;
    }

    return results;
}
}  // namespace genai
}  // namespace ov
