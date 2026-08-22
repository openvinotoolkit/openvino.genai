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

// Process logits for the given live physical rows only, in physical-row order. Filler rows (finished FIXED
// requests) are excluded by the caller, so a filler never reads another row's timestamp history and its logits
// are left untouched. The history map is keyed by physical decoder row.
void process_whisper_logits(ov::Tensor logits,
                            const ov::genai::WhisperGenerationConfig& config,
                            const bool return_timestamps,
                            const bool initial_step,
                            const std::vector<size_t>& physical_rows,
                            const std::map<size_t, std::vector<int64_t>>& batch_to_generated_ids) {
    // Normalize before masking so beam scores remain comparable across rows, matching HF beam search.
    const bool is_beam_search = config.is_beam_search();

    for (const size_t batch : physical_rows) {
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
                                                            batch_to_generated_ids.at(batch),
                                                            initial_step);
            }
        }
    }
}

// Record decoder inference-timing metrics. m_batch_sizes is recorded by the caller so its value and position
// relative to sampling stay at the call site.
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

// Initialize the accumulator metrics shared by both generate entry points. Callers keep their own reserve()
// calls (only the single-audio path knows max_new_tokens).
void init_whisper_perf_metrics(ov::genai::WhisperPerfMetrics& perf_metrics) {
    perf_metrics.num_input_tokens = 0;
    perf_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
    perf_metrics.whisper_raw_metrics.word_level_timestamps_processing_durations = {{MicroSeconds(0.0f)}};
}

// Build the prefill input_ids tensor from the groups' prompt ids, allocated through the decoder's
// remote-context-safe allocator so it owns its memory for the async inference. Every prompt must be non-empty
// and all prompts must share an identical length.
ov::Tensor build_prefill_input_ids(const std::shared_ptr<ov::genai::WhisperDecoder>& decoder,
                                   const std::vector<ov::genai::SequenceGroup::Ptr>& sequence_groups) {
    OPENVINO_ASSERT(!sequence_groups.empty(), "Whisper prefill requires at least one prompt.");
    const size_t num_prompts = sequence_groups.size();
    const size_t prompt_len = sequence_groups.front()->get_prompt_ids().size();
    OPENVINO_ASSERT(prompt_len != 0, "Whisper prefill requires a non-empty prompt.");

    // All rows must use the same prompt length; only the language token may differ.
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

// Build an identity beam_idx [0, 1, ..., width - 1] in a remote-context-safe host tensor. Greedy decoding
// never reorders state rows, so this identity mapping is the correct prefill beam_idx for both decode paths.
ov::Tensor make_identity_beam_idx(const std::shared_ptr<ov::genai::WhisperDecoder>& decoder, const size_t width) {
    ov::Tensor beam_idx = decoder->create_host_tensor(ov::element::i32, {width});
    std::iota(beam_idx.data<int32_t>(), beam_idx.data<int32_t>() + width, 0);
    return beam_idx;
}

// Semantic contract of a generate call. SCALAR is the single-audio full-feature path (beam search, streaming,
// word timestamps, per-window metric filtering); BATCH is the restricted greedy B>1 path with aggregate metrics.
enum class GenerationMode { SCALAR, BATCH };

// How physical decoder rows are managed within a round. PACKED: every row is a running sequence of one group,
// beam_idx comes from the sampler, and the width tracks the running-sequence count (grows/shrinks with beam
// search). FIXED: the width is pinned to the batch size; finished requests stay as sampling-excluded filler rows
// with the prefill-identity beam_idx. Both feed the one shared generation loop.
enum class DecoderRowLayout {
    PACKED_RUNNING_SEQUENCES,
    FIXED_WIDTH_WITH_FILLERS,
};

// Single place mapping semantic mode to physical row policy. This is only where the mapping lives; future B>1
// beam support would require more than a new selection here (e.g. packed beam bookkeeping spanning groups).
DecoderRowLayout select_row_layout(GenerationMode mode) {
    return mode == GenerationMode::SCALAR ? DecoderRowLayout::PACKED_RUNNING_SEQUENCES
                                          : DecoderRowLayout::FIXED_WIDTH_WITH_FILLERS;
}

// Per-round layout state decided once and reused every token; these tensors are not reallocated per step
// (other per-step allocations, e.g. timestamp history, still occur in the generation loop).
struct DecodeLayoutState {
    // The [physical_width] i32 beam_idx tensor, filled once with the identity and reused every step.
    ov::Tensor beam_idx_tensor;
    // The [physical_width, 1] i64 input_ids tensor, reused (rewritten in place) every generation step.
    ov::Tensor next_input_ids_tensor;
};

// One decoder generation step as reusable mutable data: which sequences to feed, where their logits land, and
// how the packed logits split across groups. Allocated once outside the token loop and repopulated in place
// (clear() retains capacity), so these plan vectors are not reallocated per token. Other per-step allocations
// still occur (e.g. timestamp history and each get_running_sequences() call).
struct DecoderStepPlan {
    // Groups handed to Sampler::sample(), in packed-row order. PACKED holds the single group once; FIXED holds
    // one entry per still-running request.
    std::vector<ov::genai::SequenceGroup::Ptr> active_groups;
    // The running sequence backing each packed logit row, aligned with live_physical_rows. Held by Sequence::Ptr
    // so a sequence stays alive across the step even though Sampler::sample() may fork/remove sequences.
    std::vector<ov::genai::Sequence::Ptr> live_sequences;
    // Physical decoder-row index of each packed logit row, in active-group order. This is also the gather list
    // for pack_live_logits: FIXED lists still-running request indices (non-contiguous once a middle row
    // finishes); PACKED is the identity [0, sampled_row_count).
    std::vector<size_t> live_physical_rows;
    // beam_idx for every physical row; populated only when the producer reorders state (PACKED). FIXED leaves
    // this empty and uses the constant identity tensor in DecodeLayoutState instead.
    std::vector<int32_t> beam_idx;
    // Number of physical decoder rows fed to start_async this step (dim0 of the input_ids and beam_idx tensors).
    size_t physical_width = 0;
    // Logit rows handed to the sampler: sum over active groups of num_running_seqs * num_scheduled_tokens.
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

// Build the immutable fixed-width layout state once per native-batch round: the constant identity beam_idx
// tensor and the reusable next_input_ids tensor. Never called inside the token loop.
DecodeLayoutState make_fixed_width_layout_state(const std::shared_ptr<ov::genai::WhisperDecoder>& decoder,
                                                const size_t batch_size) {
    DecodeLayoutState state;
    state.beam_idx_tensor = make_identity_beam_idx(decoder, batch_size);
    state.next_input_ids_tensor = decoder->create_host_tensor(ov::element::i64, {batch_size, 1});
    return state;
}

// Populate the caller-owned plan for the next generation step: schedule one token per still-running group and
// record the packed-row bookkeeping, sampler spans, and (PACKED only) the sampler-derived beam_idx. No
// inference/sampling/streaming/cleanup; leaves active_groups empty when nothing remains. Cleared first, so
// repeated calls reuse its storage.
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

    // Row bookkeeping is internally consistent.
    OPENVINO_ASSERT(plan.live_sequences.size() == plan.live_physical_rows.size() &&
                        plan.live_sequences.size() == plan.sampled_row_count,
                    "Internal error: generation-step row bookkeeping is inconsistent.");
}

// Fill the [physical_width, 1] i64 input_ids tensor. FIXED fills every row with the filler token then overwrites
// running rows; PACKED writes one token per row. Each row gets its next scheduled token (a prompt token while
// prefilling, else the last generated token). filler_token is unused for PACKED.
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

// Copy the step's beam_idx into the decoder's reusable i32 host tensor, resizing it when the physical width
// changes (packed beam search). FIXED keeps a constant identity width, so this is an identity rewrite of an
// already-identity tensor.
void sync_beam_idx_tensor(ov::Tensor& beam_idx_tensor, const std::vector<int32_t>& beam_idx) {
    if (beam_idx_tensor.get_shape().at(0) != beam_idx.size()) {
        beam_idx_tensor.set_shape({beam_idx.size()});
    }
    std::copy_n(beam_idx.data(), beam_idx.size(), beam_idx_tensor.data<int32_t>());
}

// Build the layout state for the single-audio packed path. The physical width tracks the running-sequence count
// and can grow/shrink with beam search, so only the beam_idx tensor is prebuilt (as the prefill identity) and is
// resynced from the sampler each step; next_input_ids is allocated per step because its width can change.
DecodeLayoutState make_packed_layout_state(const std::shared_ptr<ov::genai::WhisperDecoder>& decoder,
                                           const size_t batch_size) {
    DecodeLayoutState state;
    state.beam_idx_tensor = make_identity_beam_idx(decoder, batch_size);
    return state;
}

// Pack live decoder rows contiguously for sampling. Returns the original tensor when every physical row is live
// (contiguous full set), an ROI slice for a live prefix, and a fresh packed copy for non-contiguous live rows
// (for example [0, 2] after a middle row finishes).
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

// Stream the newest generated token to the caller's streamer, if any. A no-op when timestamps are on (tokens are
// streamed from segment extraction instead), when there is no streamer, or when nothing is ready. The batch path
// passes a null streamer and returns immediately.
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

// Choose and fill the input_ids tensor for this step. Fixed width reuses the call-lifetime tensor (rewritten in
// place); packed width can change with the running-sequence count, so it allocates a right-sized tensor. This is
// a small row-policy leaf, not a second loop body.
ov::Tensor prepare_next_input_ids(DecodeLayoutState& layout,
                                  const DecoderStepPlan& plan,
                                  const DecoderRowLayout row_layout,
                                  const int64_t filler_token) {
    // PACKED width can change with the running-sequence count, so allocate a right-sized tensor; FIXED reuses
    // the call-lifetime tensor rewritten in place.
    if (row_layout == DecoderRowLayout::PACKED_RUNNING_SEQUENCES) {
        ov::Tensor next_input_ids(ov::element::i64, {plan.physical_width, 1});
        fill_next_input_ids(next_input_ids, plan, row_layout, filler_token);
        return next_input_ids;
    }
    fill_next_input_ids(layout.next_input_ids_tensor, plan, row_layout, filler_token);
    return layout.next_input_ids_tensor;
}

// Choose the beam_idx tensor for this step. PACKED reorders/resizes the state rows from the sampler-derived
// mapping; FIXED keeps the prebuilt identity and copies nothing.
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

// Resets decoder KV/encoder state on every exit path. finish() does the normal-path reset (may throw, so a
// failure is observable); the destructor resets only during exceptional unwinding, first settling any in-flight
// request best-effort, and never lets an exception escape. Non-copyable so the reset never runs twice.
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

    // Normal-path reset. reset_state() runs before marking finished so that, if it throws, finished stays false
    // and the destructor retries the reset best-effort during unwinding; the throw still propagates to the caller.
    void finish() {
        decoder->reset_state();
        finished = true;
    }

    ~DecoderRoundResetGuard() {
        if (finished) {
            return;
        }
        // Exceptional exit only: settle an in-flight request (best-effort) then reset, swallowing any
        // failure so no exception escapes during stack unwinding.
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

// Per-round configuration: the row policy plus optional feature resources. PACKED may enable a streamer and
// leaves filler_token unused; FIXED uses filler_token for finished rows and has no streamer. Either layout may
// enable return_timestamps (public timestamps or long-form seek). The streaming handle is not stored here:
// run_decoder_round builds a local one from the sole group only when a streamer is present.
struct DecoderRoundConfig {
    DecoderRowLayout layout;
    bool return_timestamps = false;
    int64_t filler_token = 0;
    std::shared_ptr<ov::genai::StreamerBase> streamer = nullptr;
};

// The one shared decoder token-generation loop, used by both PACKED and FIXED rounds. round_config.layout
// selects how each step's DecoderStepPlan is produced; the loop body consumes only plan/layout data. It performs
// no prefill, result construction, streaming setup, or request cleanup; those stay with the caller.
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
                                 ov::genai::WhisperPerfMetrics& perf_metrics) {
    const DecoderRowLayout row_layout = round_config.layout;
    const bool return_timestamps = round_config.return_timestamps;
    ov::genai::RawPerfMetrics& raw_metrics = perf_metrics.raw_metrics;
    ov::genai::WhisperRawPerfMetrics& whisper_raw_metrics = perf_metrics.whisper_raw_metrics;

    while (true) {
        collect_generation_step(row_layout, sequence_groups, sampler, plan);
        if (plan.active_groups.empty()) {
            break;
        }

        // Each live row's generated ids, captured before this step's inference and keyed by physical decoder
        // row. Only live rows get an entry, so FIXED filler rows never carry a history and are never processed.
        std::map<size_t, std::vector<int64_t>> batch_to_generated_ids{};
        if (return_timestamps) {
            for (size_t row = 0; row < plan.live_sequences.size(); row++) {
                batch_to_generated_ids[plan.live_physical_rows[row]] = plan.live_sequences[row]->get_generated_ids();
            }
        }

        const ov::Tensor next_input_ids = prepare_next_input_ids(layout, plan, row_layout, round_config.filler_token);

        const auto infer_start = std::chrono::steady_clock::now();

        ov::Tensor& beam_idx = select_beam_idx_tensor(layout, plan, row_layout);

        reset_guard.mark_started();
        decoder->start_async(encoder_hidden_state, next_input_ids, beam_idx);

        // Stream while the request is in flight (start_async -> stream -> wait) to preserve callback/inference
        // overlap and the exact step at which STOP/CANCEL is observed. reset_guard is in_flight here, so a throw
        // from streamer.write() still settles the request and resets the decoder.
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
                               batch_to_generated_ids);

        OPENVINO_ASSERT(logits.get_shape().at(0) == plan.physical_width,
                        "Internal error: decoder logits row count ",
                        logits.get_shape().at(0),
                        " does not match the physical width ",
                        plan.physical_width,
                        ".");

        // live_physical_rows is the packed-row gather list; pack_live_logits returns the original tensor for the
        // contiguous full set, an ROI for a live prefix, and a copy only for non-contiguous live rows.
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

// Per-group result of one decoder round, in group (== input) order: the generated token ids and whether the
// group was stopped/cancelled. whisper_generate keeps the public score fixed at 1.f.
struct DecoderRoundResult {
    std::vector<std::vector<int64_t>> tokens;
    std::vector<bool> cancelled;
};

// Clears each round's Sampler request registration once on every exit path. The Sampler is long-lived and reused
// across calls with reused request ids, so a leaked registration would corrupt a later call; clear_request_info()
// is a no-op for an unregistered id, so unconditional clearing is safe. Non-copyable so it never double-cleans.
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

// The one shared decoder round for both PACKED and FIXED callers: unified prefill, the shared generation loop,
// and per-group result collection over caller-created SequenceGroups. The caller owns group creation and request
// ids; this function owns per-round execution, the transient streaming handle, and normal-path sampler cleanup.
// Both layouts share one generation loop and use small layout-specific branches per step.
DecoderRoundResult run_decoder_round(std::shared_ptr<ov::genai::WhisperDecoder> decoder,
                                     const std::vector<ov::genai::SequenceGroup::Ptr>& sequence_groups,
                                     const ov::Tensor& encoder_hidden_state,
                                     ov::genai::Sampler& sampler,
                                     const ov::genai::WhisperGenerationConfig& config,
                                     const DecoderRoundConfig& round_config,
                                     ov::genai::WhisperPerfMetrics& perf_metrics) {
    OPENVINO_ASSERT(!sequence_groups.empty(), "Whisper decoder round requires at least one sequence group.");
    // Streaming is single-audio only; the local handle below is built from the sole group.
    OPENVINO_ASSERT(!round_config.streamer || sequence_groups.size() == 1,
                    "Internal error: Whisper streaming requires exactly one sequence group.");

    ov::genai::RawPerfMetrics& raw_metrics = perf_metrics.raw_metrics;
    ov::genai::WhisperRawPerfMetrics& whisper_raw_metrics = perf_metrics.whisper_raw_metrics;

    // One running sequence per group before beam expansion: the prefill physical width.
    const size_t prefill_width = sequence_groups.size();

    // Streaming handle: built locally from the sole group only when a streamer is present. It borrows the
    // group's generation stream; cancellation is read back from the group's stream status after the round.
    std::shared_ptr<ov::genai::GenerationHandleImpl> handle;
    if (round_config.streamer) {
        const auto& sequence_group = sequence_groups.front();
        handle = std::make_shared<ov::genai::GenerationHandleImpl>(sequence_group->get_generation_stream(),
                                                                   sequence_group->get_sampling_parameters());
    }

    // PACKED tracks the running-sequence width with a sampler beam_idx; FIXED pins the width with an identity
    // beam_idx and filler rows.
    DecodeLayoutState layout = round_config.layout == DecoderRowLayout::PACKED_RUNNING_SEQUENCES
                                   ? make_packed_layout_state(decoder, prefill_width)
                                   : make_fixed_width_layout_state(decoder, prefill_width);

    // Constructed before prefill so it also covers a throw during prefill/generation.
    SamplerRequestCleanup request_cleanup{sampler};
    request_cleanup.request_ids.reserve(sequence_groups.size());
    for (const auto& sequence_group : sequence_groups) {
        request_cleanup.request_ids.push_back(sequence_group->get_request_id());
    }

    // Resets decoder KV/encoder state on every exit path: finish() on the normal path, the destructor on
    // exceptional exits.
    DecoderRoundResetGuard reset_guard{decoder};

    // Prefill: build_prefill_input_ids reads prompts directly from the groups and requires identical lengths.
    const ov::Tensor input_ids_tensor = build_prefill_input_ids(decoder, sequence_groups);

    const auto infer_start = std::chrono::steady_clock::now();
    reset_guard.mark_started();
    decoder->start_async(encoder_hidden_state, input_ids_tensor, layout.beam_idx_tensor);

    auto logits = decoder->wait();
    reset_guard.mark_waited();
    const auto infer_end = std::chrono::steady_clock::now();
    record_decode_inference_metrics(raw_metrics, whisper_raw_metrics, infer_start, infer_end);
    // Prefill batch size == group count (one running sequence per group before beam expansion).
    raw_metrics.m_batch_sizes.emplace_back(prefill_width);

    // Prefill processes every group's single physical row (all rows are live before any request finishes).
    std::vector<size_t> prefill_rows(prefill_width);
    std::iota(prefill_rows.begin(), prefill_rows.end(), size_t{0});
    process_whisper_logits(logits, config, round_config.return_timestamps, /*initial_step=*/true, prefill_rows, {});

    // sample last token only
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

    // Generation phase: the step plan is allocated once and repopulated in place every token; beam search may
    // widen the running-sequence set, so reserve for the configured beam count.
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
                                perf_metrics);

    stream_generated_tokens(round_config.streamer, handle, round_config.return_timestamps);

    // Per-group collection in group (== input) order: generated tokens and cancellation. Sampler cleanup is
    // owned by the SamplerRequestCleanup guard above.
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

    // Normal-path decoder reset, after result collection. May throw so a reset failure stays observable.
    reset_guard.finish();

    return result;
}

// Shared encoder lifecycle: bind the input, run one synchronous inference, record metrics, release the input
// buffer (rebinding an empty tensor of the caller-chosen shape), and return the request-owned hidden state.
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

    // reset input tensor
    request.set_tensor("input_features", ov::Tensor(ov::element::f32, release_shape));

    return request.get_tensor("last_hidden_state");
}

// Pack K current windows into a {K, feature, nb_max_frames} input and run one shared encoder inference.
// K == 1 for scalar/N=1 and K == active_count for native batch. The release shape is device-aware: NPU
// compiles for a static batch dimension of 1; other devices release with a 0 batch dimension (batched
// generation is CPU/GPU only, so K > 1 always releases with 0). Encoder row order matches window order.
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

    // NPU compiles for a static batch dimension of 1; other devices release with a 0 batch dimension.
    auto devices = request.get_compiled_model().get_property(ov::execution_devices);
    OPENVINO_ASSERT(devices.size() > 0, "No execution devices found!");
    const size_t release_batch = (devices[0] == "NPU") ? 1 : 0;

    // Single window (scalar/N == 1): view the caller-owned vector directly, matching the baseline zero-copy path.
    // mel_windows outlives this call and run_encoder_inference runs synchronously, so the view stays valid through
    // request.infer(). const_cast is safe: the encoder only reads input_features.
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

    // Multiple windows (native batch): pack the windows contiguously into an owned tensor.
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

// Resolve SOT tokens for each audio, with language detection performed per row. A single audio is the
// `batch_size == 1` case: row 0 of the returned vector describes it.
std::vector<ov::genai::SotTokensResult> prepare_sot_tokens(ov::Tensor& encoder_hidden_state,
                                                           const std::shared_ptr<ov::genai::WhisperDecoder>& decoder,
                                                           const ov::genai::WhisperGenerationConfig& config,
                                                           const size_t batch_size,
                                                           ov::genai::RawPerfMetrics& raw_metrics) {
    if (!config.is_multilingual) {
        // non-multilingual whisper models are english-only
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

    // An explicitly configured language is shared by the whole batch.
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

// Persistent per-audio state for one generate() call: only data that survives between scheduler rounds. The
// vector is built once in input order and is never reordered or compacted; the cohort vectors are transient
// index views into it. original_index is stable for the whole call; chunk_offset advances monotonically;
// `words` is scalar-only (B>1 word timestamps are unsupported).
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

// Execution context shared across a generate call's cohort rounds: the semantic mode, the derived row policy,
// immutable configuration, the services the rounds use, and the aggregate metrics they update. It shares
// ownership of the decoder (a shared_ptr<WhisperDecoder>) but owns neither the AudioStates nor any round-local
// buffers.
struct SchedulerContext {
    GenerationMode mode;
    DecoderRowLayout row_layout;
    const ov::genai::WhisperGenerationConfig& config;
    const ov::genai::WhisperConfig& model_config;
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

// Per-audio post-round processing for one cohort audio: for a timestamped round extract segments, accumulate the
// transcript, and advance the seek offset; otherwise complete the single short-form window. Then compute scalar
// word timestamps. Mutates `st`. Metric filtering and word timestamps are scalar-only (BATCH rejects both).
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
        // A timestamped round for a still-running audio must advance its seek, or long-form would loop forever.
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

        // Scalar per-window metric filtering, on the transcript size before appending.
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

        // Scalar segment-level streaming (token-by-token streaming is suppressed while timestamps are on).
        if (ctx.streamer &&
            ctx.streamer->write(extracted.non_timestamp_tokens) != ov::genai::StreamingStatus::RUNNING) {
            cancelled = true;
        }

        st.chunk_offset += extracted.last_offset;
    } else {
        // A no-timestamp cohort holds only short-form audios; one window completes each.
        st.output_tokens.insert(st.output_tokens.end(), chunk_tokens.begin(), chunk_tokens.end());
        st.chunk_offset = st.features.n_frames;
    }

    if (cancelled) {
        // Streamer STOP/CANCEL (scalar only): stop this audio with no further window.
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

// Run one compatible cohort (a subset of the active set sharing one timestamp mode, hence one prompt length) as
// a single batched round: pack windows -> one shared encoder inference -> resolve SOT once -> build round-local
// SequenceGroups (encoder row j -> group j -> physical decoder row j) -> one shared decoder round -> per-audio
// finalize. `cohort` holds original indices in original order; request_id == original_index. mel_windows and
// sequence_groups are reused across rounds to avoid per-round reallocation.
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

    mel_windows.clear();
    mel_windows.reserve(cohort.size());
    for (const size_t a : cohort) {
        mel_windows.emplace_back(
            states[a].features.get_data_with_offset(states[a].chunk_offset, feature_extractor.nb_max_frames));
    }

    // One shared encoder inference (K == 1 on the scalar path).
    ov::Tensor encoder_hidden_state = encode_windows(ctx.encoder,
                                                     mel_windows,
                                                     feature_extractor.feature_size,
                                                     feature_extractor.nb_max_frames,
                                                     raw_metrics,
                                                     perf_metrics.whisper_raw_metrics);

    // Resolve SOT/language per audio once. Every audio first appears in round 0; across rounds a cohort may
    // shrink as audios finish but never gains a new unresolved member, so a cohort is all-unresolved on its
    // first round or all-resolved afterwards.
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

    // Build round-local SequenceGroups in cohort order. initial_prompt applies only to a first window
    // (chunk_offset == 0); hotwords apply every window (via get_prompt_tokens).
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

    // One shared decoder round; it resets the decoder and clears sampler state on return.
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
                                     perf_metrics);

    for (size_t j = 0; j < cohort.size(); j++) {
        finalize_audio_after_round(ctx,
                                   states[cohort[j]],
                                   decoded.tokens[j],
                                   decoded.cancelled[j],
                                   cohort_return_timestamps,
                                   encoder_hidden_state);
    }
}

// The one N-ary generation scheduler, shared by the scalar (SCALAR) and batched (BATCH) entry points. Over
// already-extracted persistent AudioStates it repeatedly collects the unfinished states, partitions them (in
// original order) into at most two compatible timestamp-mode cohorts, and runs each cohort as one batched round
// with an independently shrinking active set. One scheduler loop here and one decoder token loop (in
// run_decoder_generation_loop, driven by run_decoder_round); scalar and batch share both.
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
                             ov::genai::WhisperPerfMetrics& perf_metrics) {
    OPENVINO_ASSERT(feature_extractor.sampling_rate != 0, "Sampling Rate for Feature Extractor is 0");
    const float time_precision = static_cast<float>(feature_extractor.chunk_length) / model_config.max_source_positions;
    const float frame_length_in_seconds =
        static_cast<float>(feature_extractor.hop_length) / feature_extractor.sampling_rate;

    SchedulerContext ctx{mode,
                         select_row_layout(mode),
                         config,
                         model_config,
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

    // An audio's internal timestamp mode is constant across its rounds: the public flag, or the audio being
    // long-form (which needs internal timestamps for seek advancement).
    const auto wants_timestamps = [&](const AudioState& state) {
        return config.return_timestamps || (state.features.n_frames > feature_extractor.nb_max_frames);
    };

    // Reused across rounds to avoid per-round reallocation.
    std::vector<size_t> ts_cohort;
    std::vector<size_t> no_ts_cohort;
    std::vector<std::vector<float>> mel_windows;
    std::vector<ov::genai::SequenceGroup::Ptr> sequence_groups;

    while (true) {
        ts_cohort.clear();
        no_ts_cohort.clear();
        for (size_t i = 0; i < states.size(); i++) {
            if (states[i].finished()) {
                continue;
            }
            (wants_timestamps(states[i]) ? ts_cohort : no_ts_cohort).push_back(i);
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
    // Scalar entry: normalize the single input to one AudioState and run the shared scheduler in SCALAR mode.
    // This wrapper only reserves scalar-known capacities and shapes the result.
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
                            result.perf_metrics);

    result.output_tokens = std::move(states[0].output_tokens);
    result.language = states[0].sot.language;
    result.words = std::move(states[0].words);
    // Public chunks are exposed only when the user requested timestamps, even if internal timestamps were
    // used for long-form scheduling.
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
    // Batch entry: normalize each input to one AudioState and run the shared scheduler in BATCH mode. One
    // aggregate metrics object describes the whole call and is copied into every result. This wrapper validates
    // the internal B > 1 invariants (also enforced at the pipeline boundary) and shapes the results in input order.
    const size_t batch_size = raw_speeches.size();
    OPENVINO_ASSERT(batch_size > 1,
                    "Internal error: whisper_generate_batch expects a batch size greater than one; the pipeline "
                    "delegates single-audio (B == 1) generation to the scalar path. Got ",
                    batch_size,
                    ".");
    OPENVINO_ASSERT(config.num_beams == 1,
                    "Batched Whisper generation supports greedy decoding only. Got num_beams = ",
                    config.num_beams,
                    ".");
    OPENVINO_ASSERT(!config.do_sample,
                    "Batched Whisper generation supports greedy decoding only ('do_sample' unsupported for B > 1).");
    OPENVINO_ASSERT(!config.word_timestamps, "'word_timestamps' is not supported for batched Whisper generation.");

    WhisperPerfMetrics perf_metrics;
    init_whisper_perf_metrics(perf_metrics);

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
                            perf_metrics);

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
