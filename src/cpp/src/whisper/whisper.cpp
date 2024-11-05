// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "whisper.hpp"

#include <iostream>
#include <openvino/openvino.hpp>
#include <regex>
#include <thread>

#include "logit_processor.hpp"
#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/whisper_generation_config.hpp"
#include "openvino/genai/whisper_pipeline.hpp"
#include "timestamps.hpp"
#include "utils.hpp"
#include "whisper_config.hpp"
#include "whisper_feature_extractor.hpp"
#include "whisper_models.hpp"

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

void set_past_key_value(ov::InferRequest& source, ov::InferRequest& dest) {
    // source outputs:
    // present.0.decoder.key
    // present.0.decoder.value
    // present.0.encoder.key
    // present.0.encoder.value

    // dest inputs:
    // past_key_values.0.decoder.key
    // past_key_values.0.decoder.value
    // past_key_values.0.encoder.key
    // past_key_values.0.encoder.value

    for (auto& source_output : source.get_compiled_model().outputs()) {
        std::string source_output_name = source_output.get_any_name();
        if (source_output_name.find("logits") != std::string::npos) {
            continue;
        }

        std::string with_past_input_name =
            std::regex_replace(source_output_name, std::regex("present"), "past_key_values");

        auto kv_tensor = source.get_tensor(source_output_name);
        dest.set_tensor(with_past_input_name, ov::Tensor{kv_tensor});
    }
}

void infer_with_perf_metrics(ov::InferRequest& request, ov::genai::RawPerfMetrics& raw_metrics) {
    const auto infer_start = std::chrono::steady_clock::now();
    request.infer();
    const auto infer_end = std::chrono::steady_clock::now();
    const auto infer_ms = ov::genai::PerfMetrics::get_microsec(infer_end - infer_start);
    raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);
    raw_metrics.m_token_infer_durations.emplace_back(infer_ms);
    raw_metrics.m_new_token_times.emplace_back(infer_end);
    raw_metrics.m_batch_sizes.emplace_back(1);
}

int64_t decode(ov::Tensor& encoder_hidden_state,
               ov::InferRequest& decoder,
               std::vector<int64_t>& input_ids,
               const ov::genai::WhisperGenerationConfig& config,
               ov::genai::RawPerfMetrics& raw_metrics,
               const bool apply_logit_processors = true,
               const bool return_timestamps = false) {
    decoder.set_tensor("encoder_hidden_states", ov::Tensor{encoder_hidden_state});

    ov::Tensor input_ids_tensor(ov::element::i64, {1, input_ids.size()}, input_ids.data());
    decoder.set_tensor("input_ids", input_ids_tensor);

    infer_with_perf_metrics(decoder, raw_metrics);

    auto output_tensor = decoder.get_tensor("logits");

    if (apply_logit_processors) {
        ov::genai::do_suppress_tokens(output_tensor, 0, config.begin_suppress_tokens);
        ov::genai::do_suppress_tokens(output_tensor, 0, config.suppress_tokens);

        if (return_timestamps) {
            ov::genai::process_whisper_timestamp_logits(output_tensor, 0, config, {}, true);
        }
    }

    int64_t output_token = ov::genai::utils::argmax(output_tensor, 0);

    return output_token;
}

int64_t decode_with_past(ov::Tensor& encoder_hidden_state,
                         ov::InferRequest& decoder_with_past,
                         int64_t input_id,
                         const size_t cache_position,
                         const ov::genai::WhisperGenerationConfig& config,
                         ov::genai::RawPerfMetrics& raw_metrics,
                         const bool return_timestamps,
                         const std::vector<int64_t>& generated_tokens) {
    decoder_with_past.set_tensor("encoder_hidden_states", ov::Tensor{encoder_hidden_state});

    std::vector<int64_t> input_ids = {input_id};
    ov::Tensor input_ids_tensor(ov::element::i64, {1, 1}, input_ids.data());
    decoder_with_past.set_tensor("input_ids", input_ids_tensor);

    ov::Tensor cache_position_tensor = decoder_with_past.get_tensor("cache_position");
    cache_position_tensor.set_shape({1});
    cache_position_tensor.data<int64_t>()[0] = cache_position;

    infer_with_perf_metrics(decoder_with_past, raw_metrics);

    auto output_tensor = decoder_with_past.get_tensor("logits");

    ov::genai::do_suppress_tokens(output_tensor, 0, config.suppress_tokens);

    if (return_timestamps) {
        ov::genai::process_whisper_timestamp_logits(output_tensor, 0, config, generated_tokens);
    }

    int64_t output_token = ov::genai::utils::argmax(output_tensor, 0);

    return output_token;
}

int64_t detect_language(ov::Tensor& encoder_hidden_state,
                        ov::InferRequest decoder,
                        const ov::genai::WhisperGenerationConfig& config) {
    std::vector<int64_t> input_ids{config.decoder_start_token_id};

    decoder.set_tensor("encoder_hidden_states", ov::Tensor{encoder_hidden_state});

    ov::Tensor input_ids_tensor(ov::element::i64, {1, input_ids.size()}, input_ids.data());
    decoder.set_tensor("input_ids", input_ids_tensor);

    decoder.infer();

    auto output_tensor = decoder.get_tensor("logits");

    int64_t output_token = ov::genai::utils::argmax(output_tensor, 0);

    return output_token;
}

std::vector<int64_t> prepare_init_ids(ov::Tensor& encoder_hidden_state,
                                      ov::InferRequest decoder,
                                      const ov::genai::WhisperGenerationConfig& config,
                                      const bool return_timestamps) {
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
        language_token_id = detect_language(encoder_hidden_state, decoder, config);
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
                                                  ov::genai::WhisperInitializedModels& models,
                                                  std::vector<int64_t> init_ids,
                                                  const size_t max_new_tokens,
                                                  const bool return_timestamps,
                                                  ov::genai::RawPerfMetrics& raw_metrics,
                                                  const std::shared_ptr<ov::genai::StreamerBase> streamer) {
    int64_t output_token =
        decode(encoder_hidden_state, models.decoder, init_ids, config, raw_metrics, true, return_timestamps);

    std::vector<int64_t> output_tokens{output_token};

    if (!return_timestamps && streamer && streamer->put({output_token})) {
        return {true, output_tokens};
    }

    if (max_new_tokens == 1) {
        return {false, output_tokens};
    }

    set_past_key_value(models.decoder, models.decoder_with_past);

    for (size_t i = 0; i < max_new_tokens - 1; i++) {
        auto output_token = decode_with_past(encoder_hidden_state,
                                             models.decoder_with_past,
                                             output_tokens.back(),
                                             init_ids.size() + i,
                                             config,
                                             raw_metrics,
                                             return_timestamps,
                                             output_tokens);

        if (i == 0) {
            set_past_key_value(models.decoder_with_past, models.decoder_with_past);
        }

        if (output_token == config.eos_token_id) {
            break;
        }

        output_tokens.push_back(output_token);

        if (!return_timestamps && streamer && streamer->put({output_token})) {
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
                                       const RawSpeechInput& raw_speech,
                                       ov::genai::WhisperInitializedModels& models,
                                       WhisperFeatureExtractor& feature_extractor,
                                       const std::shared_ptr<ChunkStreamerBase> streamer) {
    auto input_features = feature_extractor.extract(raw_speech);

    const bool is_shortform = input_features.n_frames <= feature_extractor.nb_max_frames;
    // long-form audio processing requires timestamps to be enabled
    const bool return_timestamps = config.return_timestamps || !is_shortform;

    size_t max_new_tokens = config.get_max_new_tokens();

    WhisperGenerateResult result;
    RawPerfMetrics& raw_metrics = result.perf_metrics.raw_metrics;
    result.perf_metrics.num_input_tokens = 0;
    raw_metrics.m_new_token_times.reserve(max_new_tokens);
    raw_metrics.m_batch_sizes.reserve(max_new_tokens);
    raw_metrics.m_token_infer_durations.reserve(max_new_tokens);
    raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};

    std::vector<int64_t> init_ids;
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

        ov::Tensor hidden_state_tensor = encode(models.encoder,
                                                input_features_chunk,
                                                feature_extractor.feature_size,
                                                feature_extractor.nb_max_frames,
                                                raw_metrics);

        // prepare init_ids just once for whole input
        if (init_ids.empty()) {
            init_ids = prepare_init_ids(hidden_state_tensor, models.decoder, config, return_timestamps);
        }

        auto [cancelled, chunk_output_tokens] = full_decode(hidden_state_tensor,
                                                            config,
                                                            models,
                                                            init_ids,
                                                            max_new_tokens - output_tokens.size(),
                                                            return_timestamps,
                                                            raw_metrics,
                                                            streamer);

        models.decoder_with_past.reset_state();

        if (return_timestamps) {
            auto extracted_segments = ov::genai::extract_segments(chunk_output_tokens,
                                                                  config,
                                                                  feature_extractor.nb_max_frames,
                                                                  time_precision);

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
