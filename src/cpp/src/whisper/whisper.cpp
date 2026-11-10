// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "whisper.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <openvino/openvino.hpp>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/whisper_generation_config.hpp"
#include "openvino/genai/whisper_pipeline.hpp"
#include "sampling/sampler.hpp"
#include "utils.hpp"
#include "whisper/config.hpp"
#include "whisper/context_tokens.hpp"
#include "whisper/decoder_round_guard.hpp"
#include "whisper/decoder_rows.hpp"
#include "whisper/feature_extractor.hpp"
#include "whisper/logit_processor.hpp"
#include "whisper/models.hpp"
#include "whisper/models/decoder.hpp"
#include "whisper/timestamps.hpp"
#include "whisper/whisper_utils.hpp"
#include "whisper/word_level_timestamps.hpp"

using ov::genai::MicroSeconds;

namespace {

constexpr uint64_t single_audio_request_id = 0;

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

void process_whisper_logits(ov::Tensor& logits,
                            const ov::genai::WhisperGenerationConfig& config,
                            const bool return_timestamps,
                            const std::vector<ov::genai::Sequence::Ptr>& row_sequences) {
    OPENVINO_ASSERT(row_sequences.size() == logits.get_shape().at(0),
                    "Whisper logits rows don't match the number of running sequences");

    // Normalize before masking so beam scores remain comparable across rows, matching HF beam search.
    const bool is_beam_search = config.is_beam_search();

    for (size_t batch = 0; batch < row_sequences.size(); batch++) {
        const auto& generated_ids = row_sequences[batch]->get_generated_ids();
        const bool initial_step = generated_ids.empty();

        if (is_beam_search) {
            logits_row_to_logprobs(logits, batch);
        }

        if (initial_step) {
            ov::genai::do_suppress_tokens(logits, batch, config.begin_suppress_tokens);
        }

        ov::genai::do_suppress_tokens(logits, batch, config.suppress_tokens);

        if (return_timestamps) {
            ov::genai::process_whisper_timestamp_logits(logits, batch, config, generated_ids, initial_step);
        }
    }
}

std::pair<ov::genai::EncodedResults, bool> decode(const std::shared_ptr<ov::genai::WhisperDecoder>& decoder,
                                                  const std::vector<int64_t>& input_ids,
                                                  const ov::Tensor& encoder_hidden_state,
                                                  const std::shared_ptr<ov::genai::StreamerBase>& streamer_ptr,
                                                  ov::genai::Sampler& sampler,
                                                  const bool return_timestamps,
                                                  const ov::genai::WhisperGenerationConfig& config,
                                                  ov::genai::RawPerfMetrics& raw_metrics,
                                                  ov::genai::WhisperRawPerfMetrics& whisper_raw_metrics) {
    ov::genai::SequenceGroup::Ptr sequence_group =
        std::make_shared<ov::genai::SequenceGroup>(single_audio_request_id, input_ids, config);

    // Keep the sampler's score representation consistent with process_whisper_logits().
    if (config.is_beam_search()) {
        sequence_group->set_logits_type(ov::genai::LogitsType::LOG_PROBS);
    }

    const auto handle = std::make_shared<ov::genai::GenerationHandleImpl>(sequence_group->get_generation_stream(),
                                                                          sequence_group->get_sampling_parameters());

    auto stream_generated_tokens = [&streamer_ptr, &handle, &return_timestamps]() {
        if (return_timestamps || !streamer_ptr || !handle->can_read()) {
            return;
        }

        std::unordered_map<uint64_t, ov::genai::GenerationOutput> token = handle->read();
        if (token.empty()) {
            // Empty terminator pushed by notify_handle_final()/notify_handle_oom() to unblock readers.
            return;
        }

        auto streaming_status = streamer_ptr->write(token.begin()->second.generated_ids);
        if (streaming_status == ov::genai::StreamingStatus::CANCEL) {
            handle->cancel();
        } else if (streaming_status == ov::genai::StreamingStatus::STOP) {
            handle->stop();
        } else if (streaming_status == ov::genai::StreamingStatus::TOOL_CALL_STOP) {
            handle->stop(ov::genai::GenerationFinishReason::TOOL_CALL);
        }
    };

    const size_t batch_size = 1;

    ov::Tensor beam_idx = decoder->create_host_tensor(ov::element::i32, {batch_size});
    std::fill_n(beam_idx.data<int32_t>(), batch_size, 0);

    // const_cast is safe as ov::Tensor only views the data and doesn't modify it.
    const ov::Tensor input_ids_tensor{ov::element::i64, {1, input_ids.size()}, const_cast<int64_t*>(input_ids.data())};

    const auto infer_start = std::chrono::steady_clock::now();
    decoder->start_async(encoder_hidden_state, input_ids_tensor, beam_idx);

    auto logits = decoder->wait();
    const auto infer_end = std::chrono::steady_clock::now();
    const auto infer_ms = ov::genai::PerfMetrics::get_microsec(infer_end - infer_start);
    raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);
    raw_metrics.m_token_infer_durations.emplace_back(infer_ms);
    raw_metrics.m_new_token_times.emplace_back(infer_end);
    raw_metrics.m_batch_sizes.emplace_back(batch_size);
    whisper_raw_metrics.decode_inference_durations.emplace_back(infer_ms);

    process_whisper_logits(logits, config, return_timestamps, sequence_group->get_running_sequences());

    // sample last token only
    int64_t output_sequence_len = logits.get_shape().at(1);
    sequence_group->schedule_tokens(sequence_group->get_prompt_len());
    sequence_group->set_output_seq_len(output_sequence_len);

    {
        const auto sample_start = std::chrono::steady_clock::now();
        sampler.sample({sequence_group}, logits);
        raw_metrics.m_sampling_durations.emplace_back(
            ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - sample_start));
    }
    sequence_group->notify_handle();
    stream_generated_tokens();

    // "Generation" phase
    while (!sequence_group->has_finished() && !sequence_group->handle_stopped() &&
           !sequence_group->handle_cancelled()) {
        sequence_group->schedule_tokens(1);
        // compute aggregated values
        size_t num_sequences = sequence_group->num_running_seqs();
        size_t total_num_tokens = sequence_group->get_num_scheduled_tokens() * num_sequences;

        ov::Tensor new_input_ids(ov::element::i64, {total_num_tokens, 1});
        int64_t* input_ids_data = new_input_ids.data<int64_t>();

        std::vector<int32_t> next_beams;

        std::vector<ov::genai::Sequence::Ptr> running_sequences = sequence_group->get_running_sequences();
        size_t num_scheduled_tokens = sequence_group->get_num_scheduled_tokens();
        size_t num_processed_tokens = sequence_group->get_num_processed_tokens();

        std::map<size_t, int32_t> beam_idxs = sampler.get_beam_idxs(sequence_group);

        for (auto sequence : running_sequences) {
            for (size_t batch = 0, position_id = num_processed_tokens; batch < num_scheduled_tokens;
                 ++batch, ++position_id) {
                // compute token for current sequence
                if (position_id < sequence_group->get_prompt_len()) {
                    input_ids_data[batch] = sequence_group->get_prompt_ids()[position_id];
                } else {
                    input_ids_data[batch] =
                        sequence->get_generated_ids()[position_id - sequence_group->get_prompt_len()];
                }
            }

            // apply strides to shift to a next sequence
            input_ids_data += num_scheduled_tokens;

            auto beam_idx = beam_idxs[sequence->get_id()];
            next_beams.push_back(beam_idx);
        }

        const auto infer_start = std::chrono::steady_clock::now();

        // align beam_idx shape with next_beams size
        if (beam_idx.get_shape()[0] != next_beams.size()) {
            beam_idx.set_shape({next_beams.size()});
        }
        std::copy_n(next_beams.data(), next_beams.size(), beam_idx.data<int32_t>());

        decoder->start_async(encoder_hidden_state, new_input_ids, beam_idx);

        stream_generated_tokens();

        auto logits = decoder->wait();

        const auto infer_end = std::chrono::steady_clock::now();
        const auto infer_ms = ov::genai::PerfMetrics::get_microsec(infer_end - infer_start);
        raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);
        raw_metrics.m_token_infer_durations.emplace_back(infer_ms);
        raw_metrics.m_new_token_times.emplace_back(infer_end);
        raw_metrics.m_batch_sizes.emplace_back(total_num_tokens);
        whisper_raw_metrics.decode_inference_durations.emplace_back(infer_ms);

        process_whisper_logits(logits, config, return_timestamps, running_sequences);

        {
            const auto sample_start = std::chrono::steady_clock::now();
            sampler.sample({sequence_group}, logits);
            raw_metrics.m_sampling_durations.emplace_back(
                ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - sample_start));
        }
        sequence_group->notify_handle();
    }

    stream_generated_tokens();

    ov::genai::EncodedResults results;

    const auto sampling_params = sequence_group->get_sampling_parameters();

    // there is also check in generation config validate function
    OPENVINO_ASSERT(config.num_return_sequences == 1);
    const auto& sequences = sequence_group->get_finished_sequences();
    const auto& sequence = sequences[0];

    const float score = sampling_params.is_beam_search() ? sequence->get_beam_search_score(sampling_params)
                                                         : sequence->get_cumulative_log_prob();

    results.tokens.push_back(sequence->get_generated_ids());
    results.scores.push_back(score);

    ov::genai::GenerationFinishReason finish_reason = sequence->get_finish_reason();
    if (sequence_group->handle_stopped() && finish_reason == ov::genai::GenerationFinishReason::NONE) {
        finish_reason = sequence_group->get_generation_stream()->get_finish_reason();
    }
    results.finish_reasons.push_back(finish_reason);

    return {results, (sequence_group->handle_stopped() || sequence_group->handle_cancelled())};
}

ov::Tensor encode_windows(ov::InferRequest& request,
                          std::vector<std::vector<float>>& mel_windows,
                          const size_t feature_size,
                          const size_t nb_max_frames,
                          ov::genai::RawPerfMetrics& raw_metrics,
                          ov::genai::WhisperRawPerfMetrics& whisper_raw_metrics) {
    const size_t features_per_window = feature_size * nb_max_frames;
    for (const auto& mel_window : mel_windows) {
        OPENVINO_ASSERT(mel_window.size() == features_per_window,
                        "Mel spectrogram required size: ",
                        feature_size,
                        " * ",
                        nb_max_frames,
                        ". Actual size: ",
                        mel_window.size(),
                        ".");
    }

    ov::Tensor input_tensor;
    if (mel_windows.size() == 1) {
        input_tensor = ov::Tensor(ov::element::f32, {1, feature_size, nb_max_frames}, mel_windows.front().data());
    } else {
        input_tensor = ov::Tensor(ov::element::f32, {mel_windows.size(), feature_size, nb_max_frames});
        for (size_t window = 0; window < mel_windows.size(); ++window) {
            std::copy(mel_windows[window].begin(),
                      mel_windows[window].end(),
                      input_tensor.data<float>() + window * features_per_window);
        }
    }

    request.set_tensor("input_features", input_tensor);

    const auto infer_start = std::chrono::steady_clock::now();
    request.infer();
    const auto infer_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);
    raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);
    whisper_raw_metrics.encode_inference_durations.emplace_back(infer_ms);

    // reset input tensor
    auto devices = request.get_compiled_model().get_property(ov::execution_devices);
    OPENVINO_ASSERT(devices.size() > 0, "No execution devices found!");
    size_t batch_size = (devices[0].rfind("NPU", 0) == 0) ? 1 : 0;
    request.set_tensor("input_features", ov::Tensor(ov::element::f32, {batch_size, feature_size, nb_max_frames}));

    return request.get_tensor("last_hidden_state");
}

std::vector<ov::genai::SotTokensResult> prepare_sot_tokens(const ov::Tensor& encoder_hidden_state,
                                                           const std::shared_ptr<ov::genai::WhisperDecoder>& decoder,
                                                           const ov::genai::WhisperGenerationConfig& config,
                                                           const bool single_audio_request,
                                                           ov::genai::RawPerfMetrics& raw_metrics) {
    const size_t rows = encoder_hidden_state.get_shape().at(0);
    if (!config.is_multilingual) {
        // non-multilingual whisper models are english-only
        return std::vector<ov::genai::SotTokensResult>(rows,
                                                       {std::vector<int64_t>{config.decoder_start_token_id}, "en"});
    }

    const int64_t task_token_id =
        config.task.has_value() && *config.task == "translate" ? config.translate_token_id : config.transcribe_token_id;
    auto to_result = [&](const int64_t language_token_id, const std::string& language) {
        return ov::genai::SotTokensResult{
            std::vector<int64_t>{config.decoder_start_token_id, language_token_id, task_token_id},
            ov::genai::utils::to_unescaped_language(language)};
    };

    if (config.language.has_value()) {
        const int64_t language_token_id =
            ov::genai::utils::get_or_throw_token_id_by_language(config.lang_to_id, *config.language);
        return std::vector<ov::genai::SotTokensResult>(rows, to_result(language_token_id, *config.language));
    }

    // Preserve the original language-detection path for single-audio requests.
    if (single_audio_request) {
        auto [language_token, infer_ms] = decoder->detect_language(encoder_hidden_state, config);
        raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);
        return {
            to_result(language_token, ov::genai::utils::find_language_by_token_id(config.lang_to_id, language_token))};
    }

    auto [language_tokens, infer_ms] = decoder->detect_languages(encoder_hidden_state, config);
    raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);

    std::vector<ov::genai::SotTokensResult> results;
    results.reserve(rows);
    for (const int64_t language_token : language_tokens) {
        results.push_back(
            to_result(language_token, ov::genai::utils::find_language_by_token_id(config.lang_to_id, language_token)));
    }
    return results;
}

std::vector<int64_t> build_window_prompt(const ov::genai::WhisperContextTokens& context_tokens,
                                         const ov::genai::WhisperGenerationConfig& config,
                                         const size_t chunk_offset,
                                         const std::vector<int64_t>& sot_tokens,
                                         const bool return_timestamps) {
    std::vector<int64_t> prompt = ov::genai::get_prompt_tokens(context_tokens, config, chunk_offset);
    prompt.insert(prompt.end(), sot_tokens.begin(), sot_tokens.end());
    if (!return_timestamps) {
        prompt.push_back(config.no_timestamps_token_id);
    }
    return prompt;
}

ov::genai::ExtractedSegments append_window_segments(const std::vector<int64_t>& window_tokens,
                                                    const ov::genai::WhisperGenerationConfig& config,
                                                    const size_t nb_max_frames,
                                                    const float time_precision,
                                                    const float window_time_offset,
                                                    std::vector<ov::genai::Segment>& segments,
                                                    std::vector<int64_t>& output_tokens) {
    auto extracted =
        ov::genai::extract_segments(window_tokens, config, nb_max_frames, time_precision, window_time_offset);
    segments.insert(segments.end(), extracted.segments.begin(), extracted.segments.end());
    output_tokens.insert(output_tokens.end(),
                         extracted.non_timestamp_tokens.begin(),
                         extracted.non_timestamp_tokens.end());
    return extracted;
}

ov::Tensor run_decoder_step(const std::shared_ptr<ov::genai::WhisperDecoder>& decoder,
                            const ov::Tensor& encoder_hidden_state,
                            const ov::Tensor& input_ids,
                            const ov::Tensor& beam_idx,
                            ov::genai::WhisperPerfMetrics& perf_metrics) {
    const auto infer_start = std::chrono::steady_clock::now();
    decoder->start_async(encoder_hidden_state, input_ids, beam_idx);
    ov::Tensor logits = decoder->wait();
    const auto infer_end = std::chrono::steady_clock::now();

    const auto infer_ms = ov::genai::PerfMetrics::get_microsec(infer_end - infer_start);
    auto& raw_metrics = perf_metrics.raw_metrics;
    raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);
    raw_metrics.m_token_infer_durations.emplace_back(infer_ms);
    raw_metrics.m_new_token_times.emplace_back(infer_end);
    raw_metrics.m_batch_sizes.emplace_back(input_ids.get_shape().at(0));
    perf_metrics.whisper_raw_metrics.decode_inference_durations.emplace_back(infer_ms);
    return logits;
}

void sample_logits(ov::genai::Sampler& sampler,
                   const std::vector<ov::genai::SequenceGroup::Ptr>& sequence_groups,
                   const ov::Tensor& logits,
                   ov::genai::RawPerfMetrics& raw_metrics) {
    const auto sample_start = std::chrono::steady_clock::now();
    sampler.sample(sequence_groups, logits);
    raw_metrics.m_sampling_durations.emplace_back(
        ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - sample_start));
}

[[nodiscard]] bool is_running(const ov::genai::SequenceGroup::Ptr& sequence_group) {
    return !sequence_group->has_finished() && !sequence_group->handle_stopped() && !sequence_group->handle_cancelled();
}

/**
 * Greedily decodes one window per group and returns tokens in group order.
 * row_to_group maps active decoder rows to sequence groups. When the batch shrinks,
 * retain_rows() compacts cross-KV, and the next inference passes the same source_rows
 * as beam_idx to reorder self-attention KV.
 */
std::vector<std::vector<int64_t>> run_batched_decoder_round(
    const std::shared_ptr<ov::genai::WhisperDecoder>& decoder,
    const std::vector<ov::genai::SequenceGroup::Ptr>& sequence_groups,
    const ov::Tensor& encoder_hidden_state,
    ov::genai::Sampler& sampler,
    const ov::genai::WhisperGenerationConfig& config,
    const bool return_timestamps,
    ov::genai::WhisperPerfMetrics& perf_metrics) {
    const size_t num_groups = sequence_groups.size();
    const size_t prompt_len = sequence_groups.front()->get_prompt_ids().size();

    for (const auto& sequence_group : sequence_groups) {
        OPENVINO_ASSERT(sequence_group->get_prompt_ids().size() == prompt_len,
                        "Batched Whisper generation requires the same prompt length for every input.");
    }

    ov::Tensor prefill_ids = decoder->create_host_tensor(ov::element::i64, {num_groups, prompt_len});
    for (size_t group = 0; group < num_groups; ++group) {
        const auto& prompt_ids = sequence_groups[group]->get_prompt_ids();
        std::copy(prompt_ids.begin(), prompt_ids.end(), prefill_ids.data<int64_t>() + group * prompt_len);
    }
    ov::Tensor beam_idx = decoder->create_host_tensor(ov::element::i32, {num_groups});
    std::iota(beam_idx.data<int32_t>(), beam_idx.data<int32_t>() + num_groups, 0);

    std::vector<ov::genai::SequenceGroup::Ptr> row_groups;
    row_groups.reserve(num_groups);
    std::vector<ov::genai::Sequence::Ptr> row_sequences;
    row_sequences.reserve(num_groups);
    for (const auto& sequence_group : sequence_groups) {
        row_sequences.push_back(sequence_group->get_running_sequences().front());
    }

    ov::Tensor logits = run_decoder_step(decoder, encoder_hidden_state, prefill_ids, beam_idx, perf_metrics);
    process_whisper_logits(logits, config, return_timestamps, row_sequences);
    for (const auto& sequence_group : sequence_groups) {
        sequence_group->schedule_tokens(sequence_group->get_prompt_len());
        sequence_group->set_output_seq_len(logits.get_shape().at(1));
    }
    sample_logits(sampler, sequence_groups, logits, perf_metrics.raw_metrics);

    std::vector<size_t> row_to_group(num_groups);
    std::iota(row_to_group.begin(), row_to_group.end(), 0);

    while (true) {
        std::vector<bool> survivors(row_to_group.size());
        for (size_t row = 0; row < row_to_group.size(); ++row) {
            survivors[row] = is_running(sequence_groups[row_to_group[row]]);
        }
        const std::vector<int32_t> source_rows = ov::genai::plan_surviving_decoder_rows(survivors);
        if (source_rows.empty()) {
            break;
        }
        if (source_rows.size() != row_to_group.size()) {
            decoder->retain_rows(source_rows);
            std::vector<size_t> retained(source_rows.size());
            for (size_t row = 0; row < source_rows.size(); ++row) {
                retained[row] = row_to_group[source_rows[row]];
            }
            row_to_group = std::move(retained);
        }

        const size_t rows = row_to_group.size();
        ov::Tensor input_ids = decoder->create_host_tensor(ov::element::i64, {rows, 1});
        beam_idx = decoder->create_host_tensor(ov::element::i32, {rows});
        std::copy(source_rows.begin(), source_rows.end(), beam_idx.data<int32_t>());
        row_groups.clear();
        row_sequences.clear();
        for (size_t row = 0; row < rows; ++row) {
            const auto& sequence_group = sequence_groups[row_to_group[row]];
            sequence_group->schedule_tokens(1);
            const auto running_sequences = sequence_group->get_running_sequences();
            OPENVINO_ASSERT(running_sequences.size() == 1,
                            "Batched Whisper generation expects one running sequence per input, got ",
                            running_sequences.size(),
                            ".");
            const auto& sequence = running_sequences.front();
            const size_t processed = sequence_group->get_num_processed_tokens();
            input_ids.data<int64_t>()[row] = processed < prompt_len
                                                 ? sequence_group->get_prompt_ids()[processed]
                                                 : sequence->get_generated_ids()[processed - prompt_len];
            row_groups.push_back(sequence_group);
            row_sequences.push_back(sequence);
        }

        logits = run_decoder_step(decoder, encoder_hidden_state, input_ids, beam_idx, perf_metrics);
        process_whisper_logits(logits, config, return_timestamps, row_sequences);
        sample_logits(sampler, row_groups, logits, perf_metrics.raw_metrics);
    }

    std::vector<std::vector<int64_t>> tokens;
    tokens.reserve(num_groups);
    for (const auto& sequence_group : sequence_groups) {
        const auto& sequences = sequence_group->get_finished_sequences();
        OPENVINO_ASSERT(!sequences.empty(),
                        "Internal error: no finished sequence for request ",
                        sequence_group->get_request_id(),
                        ".");
        tokens.push_back(sequences[0]->get_generated_ids());
    }
    return tokens;
}

struct AudioState {
    size_t input_index = 0;
    ov::genai::WhisperFeatures features;
    size_t chunk_offset = 0;
    bool return_timestamps = false;
    std::optional<ov::genai::SotTokensResult> sot;
    std::vector<int64_t> output_tokens;
    std::vector<ov::genai::Segment> segments;
    std::optional<std::vector<ov::genai::WhisperWordTiming>> words;

    [[nodiscard]] bool finished() const noexcept {
        return chunk_offset >= features.n_frames;
    }
};

struct GenerationContext {
    const ov::genai::WhisperGenerationConfig& config;
    const ov::genai::WhisperContextTokens& context_tokens;
    ov::InferRequest& encoder;
    const std::shared_ptr<ov::genai::WhisperDecoder>& decoder;
    const ov::genai::WhisperFeatureExtractor& feature_extractor;
    ov::genai::Sampler& sampler;
    ov::genai::Tokenizer& tokenizer;
    const std::shared_ptr<ov::genai::StreamerBase>& streamer;
    ov::genai::WhisperPerfMetrics& perf_metrics;
    bool single_audio_request = false;
    float time_precision = 0.f;
    float frame_length_in_seconds = 0.f;
};

void finalize_audio_window(const GenerationContext& ctx,
                           AudioState& state,
                           const std::vector<int64_t>& tokens,
                           bool cancelled,
                           const ov::Tensor& encoder_hidden_state) {
    const auto& config = ctx.config;
    const auto& feature_extractor = ctx.feature_extractor;
    const size_t window_offset = state.chunk_offset;
    const float window_time_offset = window_offset * ctx.frame_length_in_seconds;

    if (!state.return_timestamps) {
        state.output_tokens.insert(state.output_tokens.end(), tokens.begin(), tokens.end());
        state.chunk_offset = state.features.n_frames;
    } else {
        auto extracted = append_window_segments(tokens,
                                                config,
                                                feature_extractor.nb_max_frames,
                                                ctx.time_precision,
                                                window_time_offset,
                                                state.segments,
                                                state.output_tokens);
        if (ctx.streamer &&
            ctx.streamer->write(extracted.non_timestamp_tokens) != ov::genai::StreamingStatus::RUNNING) {
            cancelled = true;
        }
        if (!cancelled) {
            OPENVINO_ASSERT(extracted.last_offset > 0,
                            "Whisper timestamp decoding made no progress for audio ",
                            state.input_index,
                            " at frame ",
                            state.chunk_offset,
                            ": the window ended right after its first timestamp. max_new_tokens may be too small.");
            state.chunk_offset += extracted.last_offset;
        }
    }

    if (cancelled) {
        state.chunk_offset = state.features.n_frames;
        return;
    }

    if (config.word_timestamps) {
        const auto n_active_frames =
            std::min(feature_extractor.nb_max_frames, state.features.n_active_frames - window_offset);
        const auto word_timestamps_processing_start = std::chrono::steady_clock::now();
        const auto word_timestamps = ov::genai::add_word_level_timestamps(*state.sot,
                                                                          tokens,
                                                                          ctx.tokenizer,
                                                                          ctx.decoder,
                                                                          encoder_hidden_state,
                                                                          config,
                                                                          n_active_frames,
                                                                          window_time_offset);
        ctx.perf_metrics.whisper_raw_metrics.word_level_timestamps_processing_durations[0] += MicroSeconds(
            ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - word_timestamps_processing_start));
        if (!state.words.has_value()) {
            state.words = std::vector<ov::genai::WhisperWordTiming>{};
        }
        state.words->insert(state.words->end(), word_timestamps.begin(), word_timestamps.end());
    }
}

void run_cohort_round(const GenerationContext& ctx,
                      std::vector<AudioState>& states,
                      const std::vector<size_t>& cohort) {
    const auto& config = ctx.config;
    const auto& feature_extractor = ctx.feature_extractor;
    auto& perf_metrics = ctx.perf_metrics;
    const bool return_timestamps = states[cohort.front()].return_timestamps;

    std::vector<std::vector<float>> mel_windows;
    mel_windows.reserve(cohort.size());
    for (const size_t index : cohort) {
        mel_windows.push_back(
            states[index].features.get_data_with_offset(states[index].chunk_offset, feature_extractor.nb_max_frames));
    }
    const ov::Tensor encoder_hidden_state = encode_windows(ctx.encoder,
                                                           mel_windows,
                                                           feature_extractor.feature_size,
                                                           feature_extractor.nb_max_frames,
                                                           perf_metrics.raw_metrics,
                                                           perf_metrics.whisper_raw_metrics);

    ov::genai::DecoderRoundGuard round_guard{ctx.decoder, ctx.sampler};

    // Every cohort contains every unfinished audio of its timestamp mode, so its first round resolves all languages.
    if (!states[cohort.front()].sot.has_value()) {
        auto sot_results = prepare_sot_tokens(encoder_hidden_state,
                                              ctx.decoder,
                                              config,
                                              ctx.single_audio_request,
                                              perf_metrics.raw_metrics);
        for (size_t i = 0; i < cohort.size(); ++i) {
            states[cohort[i]].sot = std::move(sot_results[i]);
        }
    }

    if (ctx.single_audio_request) {
        AudioState& state = states[cohort.front()];
        const std::vector<int64_t> prompt =
            build_window_prompt(ctx.context_tokens, config, state.chunk_offset, state.sot->tokens, return_timestamps);
        round_guard.track_request(single_audio_request_id);
        auto [decoded, cancelled] = decode(ctx.decoder,
                                           prompt,
                                           encoder_hidden_state,
                                           ctx.streamer,
                                           ctx.sampler,
                                           return_timestamps,
                                           config,
                                           perf_metrics.raw_metrics,
                                           perf_metrics.whisper_raw_metrics);
        ctx.decoder->reset_state();
        finalize_audio_window(ctx, state, decoded.tokens[0], cancelled, encoder_hidden_state);
    } else {
        std::vector<ov::genai::SequenceGroup::Ptr> sequence_groups;
        sequence_groups.reserve(cohort.size());
        for (const size_t index : cohort) {
            const AudioState& state = states[index];
            const std::vector<int64_t> prompt = build_window_prompt(ctx.context_tokens,
                                                                    config,
                                                                    state.chunk_offset,
                                                                    state.sot->tokens,
                                                                    return_timestamps);
            sequence_groups.push_back(std::make_shared<ov::genai::SequenceGroup>(state.input_index, prompt, config));
            round_guard.track_request(state.input_index);
        }

        const auto tokens = run_batched_decoder_round(ctx.decoder,
                                                      sequence_groups,
                                                      encoder_hidden_state,
                                                      ctx.sampler,
                                                      config,
                                                      return_timestamps,
                                                      perf_metrics);
        ctx.decoder->reset_state();
        for (size_t i = 0; i < cohort.size(); ++i) {
            finalize_audio_window(ctx, states[cohort[i]], tokens[i], false, encoder_hidden_state);
        }
    }

    round_guard.dismiss();
}

}  // namespace

namespace ov {
namespace genai {

std::vector<WhisperGenerateResult> whisper_generate(const ov::genai::WhisperGenerationConfig& config,
                                                    const ov::genai::WhisperConfig& model_config,
                                                    const WhisperContextTokens& context_tokens,
                                                    const AudiosView& audios,
                                                    ov::InferRequest& encoder,
                                                    const std::shared_ptr<WhisperDecoder>& decoder,
                                                    WhisperFeatureExtractor& feature_extractor,
                                                    const std::shared_ptr<StreamerBase>& streamer,
                                                    Sampler& sampler,
                                                    Tokenizer& tokenizer) {
    OPENVINO_ASSERT(!audios.empty(), "Whisper generation requires at least one audio input.");
    const bool single_audio_request = audios.size() == 1;
    if (!single_audio_request) {
        OPENVINO_ASSERT(decoder->supports_dynamic_batch(), "This Whisper decoder does not support batched generation.");
        OPENVINO_ASSERT(!streamer && !config.word_timestamps && config.is_greedy_decoding(),
                        "Internal error: batched Whisper generation supports greedy decoding without a streamer or "
                        "word timestamps only.");
    }
    OPENVINO_ASSERT(feature_extractor.sampling_rate != 0, "Sampling Rate for Feature Extractor is 0");

    const size_t max_new_tokens = config.get_max_new_tokens();
    WhisperPerfMetrics perf_metrics;
    perf_metrics.num_input_tokens = 0;
    perf_metrics.raw_metrics.m_new_token_times.reserve(max_new_tokens);
    perf_metrics.raw_metrics.m_batch_sizes.reserve(max_new_tokens);
    perf_metrics.raw_metrics.m_token_infer_durations.reserve(max_new_tokens);
    perf_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
    perf_metrics.whisper_raw_metrics.word_level_timestamps_processing_durations = {{MicroSeconds(0.0f)}};

    std::vector<AudioState> states(audios.size());
    for (size_t i = 0; i < audios.size(); ++i) {
        const auto extract_start = std::chrono::steady_clock::now();
        states[i].input_index = i;
        states[i].features = feature_extractor.extract(audios[i]);
        states[i].return_timestamps =
            config.return_timestamps || states[i].features.n_frames > feature_extractor.nb_max_frames;
        perf_metrics.whisper_raw_metrics.features_extraction_durations.emplace_back(
            PerfMetrics::get_microsec(std::chrono::steady_clock::now() - extract_start));
    }

    GenerationContext ctx{config,
                          context_tokens,
                          encoder,
                          decoder,
                          feature_extractor,
                          sampler,
                          tokenizer,
                          streamer,
                          perf_metrics,
                          single_audio_request,
                          static_cast<float>(feature_extractor.chunk_length) / model_config.max_source_positions,
                          static_cast<float>(feature_extractor.hop_length) / feature_extractor.sampling_rate};

    // Each audio keeps its single-audio timestamp mode; a round holds one mode because their prompt lengths differ.
    std::vector<size_t> cohort;
    cohort.reserve(states.size());
    for (const bool return_timestamps : {false, true}) {
        while (true) {
            cohort.clear();
            for (size_t i = 0; i < states.size(); ++i) {
                if (!states[i].finished() && states[i].return_timestamps == return_timestamps) {
                    cohort.push_back(i);
                }
            }
            if (cohort.empty()) {
                break;
            }
            run_cohort_round(ctx, states, cohort);
        }
    }

    if (streamer) {
        streamer->end();
    }

    std::vector<WhisperGenerateResult> results(states.size());
    for (size_t i = 0; i < states.size(); ++i) {
        results[i].output_tokens = std::move(states[i].output_tokens);
        results[i].language = states[i].sot.has_value() ? states[i].sot->language : std::string{};
        results[i].words = std::move(states[i].words);
        if (config.return_timestamps) {
            results[i].segments = std::move(states[i].segments);
        }
        results[i].perf_metrics = perf_metrics;
    }
    return results;
}
}  // namespace genai
}  // namespace ov
