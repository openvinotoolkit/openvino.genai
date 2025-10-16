// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "whisper.hpp"

#include <iostream>
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

using ov::genai::MicroSeconds;

namespace {

void process_whisper_logits(ov::Tensor logits,
                            const ov::genai::WhisperGenerationConfig& config,
                            const bool return_timestamps,
                            const std::map<size_t, std::vector<int64_t>>& batch_to_generated_ids) {
    const bool initial_step = batch_to_generated_ids.empty();
    const size_t batch_size = logits.get_shape().at(0);

    for (size_t batch = 0; batch < batch_size; batch++) {
        if (initial_step) {
            ov::genai::do_suppress_tokens(logits, batch, config.begin_suppress_tokens);
        }

        ov::genai::do_suppress_tokens(logits, batch, config.suppress_tokens);

        if (return_timestamps) {
            const auto& generated_ids = initial_step ? std::vector<int64_t>{} : batch_to_generated_ids.at(batch);
            ov::genai::process_whisper_timestamp_logits(logits, batch, config, generated_ids, initial_step);
        }
    }
}

std::pair<ov::genai::EncodedResults, bool> decode(std::shared_ptr<ov::genai::WhisperDecoder> decoder,
                                                  const std::vector<int64_t>& input_ids,
                                                  const ov::Tensor& encoder_hidden_state,
                                                  const std::shared_ptr<ov::genai::StreamerBase> streamer_ptr,
                                                  ov::genai::Sampler& sampler,
                                                  ov::genai::SequenceGroup::Ptr sequence_group,
                                                  const bool return_timestamps,
                                                  const ov::genai::WhisperGenerationConfig& config,
                                                  ov::genai::RawPerfMetrics& raw_metrics) {
    const auto handle = std::make_shared<ov::genai::GenerationHandleImpl>(sequence_group->get_generation_stream(),
                                                                          sequence_group->get_sampling_parameters());

    auto stream_generated_tokens = [&streamer_ptr, &handle, &return_timestamps]() {
        if (return_timestamps || !streamer_ptr || !handle->can_read()) {
            return;
        }

        std::unordered_map<uint64_t, ov::genai::GenerationOutput> token = handle->read();

        auto streaming_status = streamer_ptr->write(token.begin()->second.generated_ids);
        if (streaming_status != ov::genai::StreamingStatus::RUNNING) {
            streaming_status == ov::genai::StreamingStatus::CANCEL ? handle->cancel() : handle->stop();
        }
    };

    const size_t batch_size = 1;

    ov::Tensor beam_idx = decoder->create_host_tensor(ov::element::i32, {batch_size});
    std::fill_n(beam_idx.data<int32_t>(), batch_size, 0);

    const ov::Tensor input_ids_tensor{ov::element::i64, {1, input_ids.size()}, (void*)input_ids.data()};

    const auto infer_start = std::chrono::steady_clock::now();
    decoder->start_async(encoder_hidden_state, input_ids_tensor, beam_idx);

    auto logits = decoder->wait();
    const auto infer_end = std::chrono::steady_clock::now();
    const auto infer_ms = ov::genai::PerfMetrics::get_microsec(infer_end - infer_start);
    raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);
    raw_metrics.m_token_infer_durations.emplace_back(infer_ms);
    raw_metrics.m_new_token_times.emplace_back(infer_end);
    raw_metrics.m_batch_sizes.emplace_back(batch_size);

    process_whisper_logits(logits, config, return_timestamps, {});

    // sample last token only
    int64_t output_sequence_len = logits.get_shape().at(1);
    sequence_group->schedule_tokens(sequence_group->get_prompt_len());
    sequence_group->set_output_seq_len(output_sequence_len);

    sampler.sample({sequence_group}, logits);
    stream_generated_tokens();

    // "Generation" phase
    while (!sequence_group->has_finished() && !sequence_group->handle_stopped() &&
           !sequence_group->handle_cancelled()) {
        std::map<size_t, std::vector<int64_t>> batch_to_generated_ids{};

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
            batch_to_generated_ids[next_beams.size() - 1] = sequence->get_generated_ids();
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

        process_whisper_logits(logits, config, return_timestamps, batch_to_generated_ids);

        sampler.sample({sequence_group}, logits);
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

    sampler.clear_request_info(sequence_group->get_request_id());

    return {results, (sequence_group->handle_stopped() || sequence_group->handle_cancelled())};
}

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

std::vector<int64_t> prepare_init_tokens(ov::Tensor& encoder_hidden_state,
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

    int64_t language_token_id = 0;
    if (config.language.has_value()) {
        std::string language = *config.language;
        if (config.lang_to_id.count(language)) {
            language_token_id = config.lang_to_id.at(language);
        }
    } else {
        auto [language_token, infer_ms] = decoder->detect_language(encoder_hidden_state, config.decoder_start_token_id);
        language_token_id = language_token;
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
                                       Sampler& sampler) {
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

    OPENVINO_ASSERT(feature_extractor.sampling_rate != 0, "Sampling Rate for Feature Extractor is 0");
    const float frame_length_in_seconds =
        static_cast<float>(feature_extractor.hop_length) / feature_extractor.sampling_rate;

    for (size_t chunk_offset = 0; chunk_offset < input_features.n_frames; chunk_offset += segment_offset) {
        const float chunk_time_offset = chunk_offset * frame_length_in_seconds;

        auto input_features_chunk = input_features.get_data_with_offset(chunk_offset, feature_extractor.nb_max_frames);

        ov::Tensor hidden_state_tensor = encode(encoder,
                                                input_features_chunk,
                                                feature_extractor.feature_size,
                                                feature_extractor.nb_max_frames,
                                                raw_metrics);

        // prepare init_tokens just once for whole input
        if (init_tokens.empty()) {
            init_tokens = prepare_init_tokens(hidden_state_tensor, decoder, config, return_timestamps, raw_metrics);
        }

        std::vector<int64_t> chunk_init_tokens = ov::genai::get_prompt_tokens(context_tokens, config, chunk_offset);
        chunk_init_tokens.insert(chunk_init_tokens.end(), init_tokens.begin(), init_tokens.end());

        SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(0, chunk_init_tokens, config, 1);

        auto [result, cancelled] = decode(decoder,
                                          chunk_init_tokens,
                                          hidden_state_tensor,
                                          streamer,
                                          sampler,
                                          sequence_group,
                                          return_timestamps,
                                          config,
                                          raw_metrics);
        decoder->reset_state();
        std::vector<int64_t> chunk_output_tokens = result.tokens[0];

        if (return_timestamps) {
            auto extracted_segments = ov::genai::extract_segments(chunk_output_tokens,
                                                                  config,
                                                                  feature_extractor.nb_max_frames,
                                                                  time_precision,
                                                                  chunk_time_offset);

            utils::filter_non_segment_metrics(raw_metrics, output_tokens.size(), extracted_segments.segment_ranges);

            segments.insert(segments.end(), extracted_segments.segments.begin(), extracted_segments.segments.end());

            output_tokens.insert(output_tokens.end(),
                                 extracted_segments.non_timestamp_tokens.begin(),
                                 extracted_segments.non_timestamp_tokens.end());

            if (streamer &&
                streamer->write(extracted_segments.non_timestamp_tokens) != ov::genai::StreamingStatus::RUNNING) {
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
