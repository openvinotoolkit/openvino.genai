// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "whisper.hpp"

#include <iostream>
#include <openvino/openvino.hpp>
#include <regex>
#include <thread>

#include "context_tokens.hpp"
#include "logit_processor.hpp"
#include "models/decoder.hpp"
#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/whisper_generation_config.hpp"
#include "openvino/genai/whisper_pipeline.hpp"
#include "timestamps.hpp"
#include "utils.hpp"
#include "whisper_config.hpp"
#include "whisper_feature_extractor.hpp"
#include "whisper_models.hpp"
#include "whisper_utils.hpp"

using ov::genai::MicroSeconds;

namespace {

ov::Tensor encode(ov::InferRequest& request,
                  std::vector<float>& mel_data,
                  const size_t feature_size,
                  const size_t nb_max_frames,
                  ov::genai::RawPerfMetrics& raw_metrics) {
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

    // reset input tensor
    request.set_tensor("input_features", ov::Tensor(ov::element::f32, {0, feature_size, nb_max_frames}));

    return request.get_tensor("last_hidden_state");
}

int64_t decode(ov::Tensor& encoder_hidden_state,
               std::shared_ptr<ov::genai::WhisperDecoder> decoder,
               const std::vector<int64_t>& input_ids,
               const size_t cache_position,
               const ov::genai::WhisperGenerationConfig& config,
               ov::genai::RawPerfMetrics& raw_metrics,
               const bool return_timestamps,
               const bool initial_step,
               const std::vector<int64_t>& generated_tokens) {
    auto [output_tensor, infer_ms] = decoder->decode(encoder_hidden_state, input_ids, cache_position);
    const auto infer_end = std::chrono::steady_clock::now();
    raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);
    raw_metrics.m_token_infer_durations.emplace_back(infer_ms);
    raw_metrics.m_new_token_times.emplace_back(infer_end);
    raw_metrics.m_batch_sizes.emplace_back(1);

    if (initial_step) {
        ov::genai::do_suppress_tokens(output_tensor, 0, config.begin_suppress_tokens);
    }

    ov::genai::do_suppress_tokens(output_tensor, 0, config.suppress_tokens);

    if (return_timestamps) {
        if (initial_step) {
            ov::genai::process_whisper_timestamp_logits(output_tensor, 0, config, {}, true);
        } else {
            ov::genai::process_whisper_timestamp_logits(output_tensor, 0, config, generated_tokens);
        }
    }

    int64_t output_token = ov::genai::utils::argmax(output_tensor, 0);

    return output_token;
}

std::vector<int64_t> prepare_init_ids(ov::Tensor& encoder_hidden_state,
                                      std::shared_ptr<ov::genai::WhisperDecoder> decoder,
                                      const ov::genai::WhisperGenerationConfig& config,
                                      const bool return_timestamps,
                                      ov::genai::RawPerfMetrics& raw_metrics) {
    if (!config.is_multilingual) {
        if (return_timestamps) {
            return std::vector<int64_t>{config.decoder_start_token_id};
        } else {
            return std::vector<int64_t>{config.decoder_start_token_id, config.no_timestamps_token_id};
        }
    }

    int64_t language_token_id;
    if (config.language.has_value()) {
        std::string language = *config.language;
        if (config.lang_to_id.count(language)) {
            language_token_id = config.lang_to_id.at(language);
        }
    } else {
        auto [language, infer_ms] = decoder->detect_language(encoder_hidden_state, config.decoder_start_token_id);
        language_token_id = language;
        raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);
    }

    int64_t task_token_id = config.transcribe_token_id;
    if (config.task.has_value() && *config.task == "translate") {
        task_token_id = config.translate_token_id;
    }

    if (return_timestamps) {
        return std::vector<int64_t>{config.decoder_start_token_id, language_token_id, task_token_id};
    }

    return std::vector<int64_t>{config.decoder_start_token_id,
                                language_token_id,
                                task_token_id,
                                config.no_timestamps_token_id};
}

std::pair<bool, std::vector<int64_t>> full_decode(ov::Tensor& encoder_hidden_state,
                                                  const ov::genai::WhisperGenerationConfig& config,
                                                  std::shared_ptr<ov::genai::WhisperDecoder> decoder,
                                                  std::vector<int64_t> init_ids,
                                                  const size_t max_new_tokens,
                                                  const bool return_timestamps,
                                                  ov::genai::RawPerfMetrics& raw_metrics,
                                                  const std::shared_ptr<ov::genai::StreamerBase> streamer) {
    int64_t output_token =
        decode(encoder_hidden_state, decoder, init_ids, 0, config, raw_metrics, return_timestamps, true, {});

    std::vector<int64_t> output_tokens{output_token};

    if (!return_timestamps && streamer && streamer->put(output_token)) {
        return {true, output_tokens};
    }

    if (max_new_tokens == 1) {
        return {false, output_tokens};
    }

    for (size_t i = 0; i < max_new_tokens - 1; i++) {
        auto output_token = decode(encoder_hidden_state,
                                   decoder,
                                   {output_tokens.back()},
                                   init_ids.size() + i,
                                   config,
                                   raw_metrics,
                                   return_timestamps,
                                   false,
                                   output_tokens);

        if (output_token == config.eos_token_id) {
            break;
        }

        output_tokens.push_back(output_token);

        if (!return_timestamps && streamer && streamer->put(output_token)) {
            return {true, output_tokens};
        }
    }

    return {false, output_tokens};
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
                                       const std::shared_ptr<ChunkStreamerBase> streamer) {
    size_t max_new_tokens = config.get_max_new_tokens();

    WhisperGenerateResult result;
    RawPerfMetrics& raw_metrics = result.perf_metrics.raw_metrics;
    result.perf_metrics.num_input_tokens = 0;
    raw_metrics.m_new_token_times.reserve(max_new_tokens);
    raw_metrics.m_batch_sizes.reserve(max_new_tokens);
    raw_metrics.m_token_infer_durations.reserve(max_new_tokens);
    raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};

    const auto infer_start = std::chrono::steady_clock::now();
    auto input_features = feature_extractor.extract(raw_speech);
    const auto infer_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);
    result.perf_metrics.whisper_raw_metrics.features_extraction_durations.emplace_back(infer_ms);

    const bool is_shortform = input_features.n_frames <= feature_extractor.nb_max_frames;
    // long-form audio processing requires timestamps to be enabled
    const bool return_timestamps = config.return_timestamps || !is_shortform;

    std::vector<int64_t> init_tokens;
    std::vector<int64_t>& output_tokens = result.output_tokens;
    std::vector<Segment> segments;

    // 0.02 by default
    const float time_precision = static_cast<float>(feature_extractor.chunk_length) / model_config.max_source_positions;
    size_t segment_offset = 0;

    for (size_t chunk_offset = 0; chunk_offset < input_features.n_frames; chunk_offset += segment_offset) {
        if (output_tokens.size() >= max_new_tokens) {
            break;
        }

        auto input_features_chunk = input_features.get_data_with_offset(chunk_offset, feature_extractor.nb_max_frames);

        ov::Tensor hidden_state_tensor = encode(encoder,
                                                input_features_chunk,
                                                feature_extractor.feature_size,
                                                feature_extractor.nb_max_frames,
                                                raw_metrics);

        // prepare init_ids just once for whole input
        if (init_tokens.empty()) {
            init_tokens = prepare_init_ids(hidden_state_tensor, decoder, config, return_timestamps, raw_metrics);
        }

        std::vector<int64_t> chunk_init_tokens = ov::genai::get_prompt_tokens(context_tokens, config, chunk_offset);
        chunk_init_tokens.insert(chunk_init_tokens.end(), init_tokens.begin(), init_tokens.end());

        auto [cancelled, chunk_output_tokens] = full_decode(hidden_state_tensor,
                                                            config,
                                                            decoder,
                                                            chunk_init_tokens,
                                                            max_new_tokens - output_tokens.size(),
                                                            return_timestamps,
                                                            raw_metrics,
                                                            streamer);

        decoder->reset_state();

        if (return_timestamps) {
            auto extracted_segments = ov::genai::extract_segments(chunk_output_tokens,
                                                                  config,
                                                                  feature_extractor.nb_max_frames,
                                                                  time_precision);

            utils::filter_non_segment_metrics(raw_metrics, output_tokens.size(), extracted_segments.segment_ranges);

            segments.insert(segments.end(), extracted_segments.segments.begin(), extracted_segments.segments.end());

            output_tokens.insert(output_tokens.end(),
                                 extracted_segments.non_timestamp_tokens.begin(),
                                 extracted_segments.non_timestamp_tokens.end());

            if (streamer && streamer->put_chunk(extracted_segments.non_timestamp_tokens)) {
                cancelled = true;
                break;
            }

            segment_offset = extracted_segments.last_offset;
        } else {
            output_tokens.insert(output_tokens.end(), chunk_output_tokens.begin(), chunk_output_tokens.end());
        }

        if (is_shortform) {
            segment_offset = input_features.n_frames;
        }

        if (cancelled) {
            break;
        }
    }

    if (streamer) {
        streamer->end();
    }

    // if return_timestamps wasn't enabled by user
    if (!config.return_timestamps) {
        return result;
    }

    result.segments = segments;

    return result;
}
}  // namespace genai
}  // namespace ov
