// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "whisper.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <openvino/openvino.hpp>
#include <optional>
#include <thread>

#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/whisper_generation_config.hpp"
#include "openvino/genai/whisper_pipeline.hpp"
#include "sampling/sampler.hpp"
#include "utils.hpp"
#include "whisper/config.hpp"
#include "whisper/context_tokens.hpp"
#include "whisper/decoder_round_guard.hpp"
#include "whisper/feature_extractor.hpp"
#include "whisper/logit_processor.hpp"
#include "whisper/models.hpp"
#include "whisper/models/decoder.hpp"
#include "whisper/timestamps.hpp"
#include "whisper/whisper_utils.hpp"
#include "whisper/word_level_timestamps.hpp"

using ov::genai::MicroSeconds;

namespace {

constexpr uint64_t decode_request_id = 0;

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

void process_whisper_logits(ov::Tensor logits,
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
        std::make_shared<ov::genai::SequenceGroup>(decode_request_id, input_ids, config);

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

    const auto& sampling_params = sequence_group->get_sampling_parameters();

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

    return {std::move(results), (sequence_group->handle_stopped() || sequence_group->handle_cancelled())};
}

ov::Tensor encode(ov::InferRequest& request,
                  std::vector<float>& mel_data,
                  const size_t feature_size,
                  const size_t nb_max_frames,
                  ov::genai::RawPerfMetrics& raw_metrics,
                  ov::genai::WhisperRawPerfMetrics& whisper_raw_metrics) {
    OPENVINO_ASSERT(mel_data.size() == feature_size * nb_max_frames,
                    "Mel spectrogram required size: ",
                    feature_size,
                    " * ",
                    nb_max_frames,
                    ". Actual size: ",
                    mel_data.size(),
                    ".");
    ov::Tensor input_tensor(ov::element::f32, {1, feature_size, nb_max_frames}, mel_data.data());

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

ov::genai::SotTokensResult prepare_sot_tokens(const ov::Tensor& encoder_hidden_state,
                                              const std::shared_ptr<ov::genai::WhisperDecoder>& decoder,
                                              const ov::genai::WhisperGenerationConfig& config,
                                              ov::genai::RawPerfMetrics& raw_metrics) {
    if (!config.is_multilingual) {
        // non-multilingual whisper models are english-only
        return {std::vector<int64_t>{config.decoder_start_token_id}, "en"};
    }

    int64_t language_token_id = 0;
    std::string language;
    if (config.language.has_value()) {
        language = *config.language;
        language_token_id = ov::genai::utils::get_or_throw_token_id_by_language(config.lang_to_id, language);
    } else {
        auto [language_token, infer_ms] = decoder->detect_language(encoder_hidden_state, config);
        language_token_id = language_token;
        language = ov::genai::utils::find_language_by_token_id(config.lang_to_id, language_token_id);
        raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);
    }

    int64_t task_token_id = config.transcribe_token_id;
    if (config.task.has_value() && *config.task == "translate") {
        task_token_id = config.translate_token_id;
    }

    return {std::vector<int64_t>{config.decoder_start_token_id, language_token_id, task_token_id},
            ov::genai::utils::to_unescaped_language(language)};
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

struct AudioState {
    ov::genai::WhisperFeatures features;
    size_t chunk_offset = 0;
    bool return_timestamps = false;
    std::optional<ov::genai::SotTokensResult> sot;
    std::vector<int64_t> output_tokens;
    std::vector<ov::genai::Segment> segments;
    std::optional<std::vector<ov::genai::WhisperWordTiming>> words;

    bool finished() const noexcept {
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
    float time_precision;
    float frame_length_in_seconds;
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
        auto extracted = ov::genai::extract_segments(tokens,
                                                     config,
                                                     feature_extractor.nb_max_frames,
                                                     ctx.time_precision,
                                                     window_time_offset);
        state.segments.insert(state.segments.end(), extracted.segments.begin(), extracted.segments.end());
        state.output_tokens.insert(state.output_tokens.end(),
                                   extracted.non_timestamp_tokens.begin(),
                                   extracted.non_timestamp_tokens.end());
        if (ctx.streamer &&
            ctx.streamer->write(extracted.non_timestamp_tokens) != ov::genai::StreamingStatus::RUNNING) {
            cancelled = true;
        }
        if (!cancelled) {
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

void process_audio_window(const GenerationContext& ctx, AudioState& state) {
    const auto& feature_extractor = ctx.feature_extractor;
    auto& perf_metrics = ctx.perf_metrics;

    auto mel_window = state.features.get_data_with_offset(state.chunk_offset, feature_extractor.nb_max_frames);
    ov::Tensor encoder_hidden_state = encode(ctx.encoder,
                                             mel_window,
                                             feature_extractor.feature_size,
                                             feature_extractor.nb_max_frames,
                                             perf_metrics.raw_metrics,
                                             perf_metrics.whisper_raw_metrics);

    ov::genai::DecoderRoundGuard round_guard{ctx.decoder, ctx.sampler};

    // prepare sot_tokens just once for whole input
    if (!state.sot.has_value()) {
        state.sot = prepare_sot_tokens(encoder_hidden_state, ctx.decoder, ctx.config, perf_metrics.raw_metrics);
    }

    const std::vector<int64_t> prompt = build_window_prompt(ctx.context_tokens,
                                                            ctx.config,
                                                            state.chunk_offset,
                                                            state.sot->tokens,
                                                            state.return_timestamps);
    round_guard.track_request(decode_request_id);
    auto [decoded, cancelled] = decode(ctx.decoder,
                                       prompt,
                                       encoder_hidden_state,
                                       ctx.streamer,
                                       ctx.sampler,
                                       state.return_timestamps,
                                       ctx.config,
                                       perf_metrics.raw_metrics,
                                       perf_metrics.whisper_raw_metrics);
    ctx.decoder->reset_state();
    finalize_audio_window(ctx, state, decoded.tokens[0], cancelled, encoder_hidden_state);

    round_guard.dismiss();
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
    size_t max_new_tokens = config.get_max_new_tokens();

    WhisperGenerateResult result;
    RawPerfMetrics& raw_metrics = result.perf_metrics.raw_metrics;
    result.perf_metrics.num_input_tokens = 0;
    raw_metrics.m_new_token_times.reserve(max_new_tokens);
    raw_metrics.m_batch_sizes.reserve(max_new_tokens);
    raw_metrics.m_token_infer_durations.reserve(max_new_tokens);
    raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};

    result.perf_metrics.whisper_raw_metrics.word_level_timestamps_processing_durations = {{MicroSeconds(0.0f)}};

    AudioState state;
    const auto infer_start = std::chrono::steady_clock::now();
    state.features = feature_extractor.extract(raw_speech);
    const auto infer_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);
    result.perf_metrics.whisper_raw_metrics.features_extraction_durations.emplace_back(infer_ms);

    const bool is_shortform = state.features.n_frames <= feature_extractor.nb_max_frames;
    // long-form audio processing requires timestamps to be enabled
    state.return_timestamps = config.return_timestamps || !is_shortform;

    // 0.02 by default
    const float time_precision = static_cast<float>(feature_extractor.chunk_length) / model_config.max_source_positions;

    OPENVINO_ASSERT(feature_extractor.sampling_rate != 0, "Sampling Rate for Feature Extractor is 0");

    GenerationContext ctx{config,
                          context_tokens,
                          encoder,
                          decoder,
                          feature_extractor,
                          sampler,
                          tokenizer,
                          streamer,
                          result.perf_metrics,
                          time_precision,
                          static_cast<float>(feature_extractor.hop_length) / feature_extractor.sampling_rate};

    while (!state.finished()) {
        process_audio_window(ctx, state);
    }

    if (streamer) {
        streamer->end();
    }

    result.output_tokens = std::move(state.output_tokens);
    if (state.sot.has_value()) {
        result.language = state.sot->language;
    }
    result.words = std::move(state.words);

    // if return_timestamps wasn't enabled by user
    if (!config.return_timestamps) {
        return result;
    }

    result.segments = std::move(state.segments);

    return result;
}
}  // namespace genai
}  // namespace ov
