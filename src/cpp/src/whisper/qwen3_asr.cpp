// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_asr.hpp"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <openvino/openvino.hpp>

#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/generation_handle.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/whisper_generation_config.hpp"
#include "openvino/genai/whisper_pipeline.hpp"
#include "sampling/sampler.hpp"
#include "utils.hpp"
#include "whisper/config.hpp"
#include "whisper/feature_extractor.hpp"
#include "whisper/models.hpp"
#include "whisper/models/decoder.hpp"
#include "whisper/whisper_utils.hpp"

using ov::genai::MicroSeconds;

namespace {

// Encode audio features through the encoder model
ov::Tensor qwen3_asr_encode(ov::InferRequest& request,
                             std::vector<float>& mel_data,
                             const size_t feature_size,
                             const size_t n_frames,
                             ov::genai::RawPerfMetrics& raw_metrics) {
    ov::Tensor input_tensor(ov::element::f32, {1, feature_size, n_frames}, mel_data.data());
    request.set_tensor("input_features", input_tensor);

    const auto infer_start = std::chrono::steady_clock::now();
    request.infer();
    const auto infer_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);
    raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);

    // Reset input tensor to free memory
    request.set_tensor("input_features", ov::Tensor(ov::element::f32, {0, feature_size, n_frames}));
    return request.get_tensor("last_hidden_state");
}

// Build initial decoder tokens for Qwen3-ASR chat-style prompt
std::vector<int64_t> build_qwen3_asr_prompt(ov::genai::Tokenizer& tokenizer,
                                             const ov::genai::WhisperConfig& model_config,
                                             const size_t num_audio_tokens) {
    // Tokenize the system and user prefix: includes <|im_start|>, system text, <|im_end|>, user prefix
    const std::string prefix_text =
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        "<|im_start|>user\nAudio 1: <|audio_start|>";

    // Tokenize the suffix after audio tokens: <|audio_end|> + closing + assistant prompt
    const std::string suffix_text =
        "<|audio_end|>\n<|im_end|>\n"
        "<|im_start|>assistant\n";

    auto prefix_encoded = tokenizer.encode(prefix_text, ov::genai::add_special_tokens(false));
    auto suffix_encoded = tokenizer.encode(suffix_text, ov::genai::add_special_tokens(false));

    auto prefix_ids_tensor = prefix_encoded.input_ids;
    auto suffix_ids_tensor = suffix_encoded.input_ids;

    std::vector<int64_t> prompt_tokens;
    const size_t total_len = prefix_ids_tensor.get_size() + num_audio_tokens + suffix_ids_tensor.get_size();
    prompt_tokens.reserve(total_len);

    // Append prefix tokens
    auto* prefix_data = prefix_ids_tensor.data<int64_t>();
    prompt_tokens.insert(prompt_tokens.end(), prefix_data, prefix_data + prefix_ids_tensor.get_size());

    // Append audio pad tokens (one per encoder output frame)
    for (size_t i = 0; i < num_audio_tokens; ++i) {
        prompt_tokens.push_back(model_config.audio_token_id);
    }

    // Append suffix tokens
    auto* suffix_data = suffix_ids_tensor.data<int64_t>();
    prompt_tokens.insert(prompt_tokens.end(), suffix_data, suffix_data + suffix_ids_tensor.get_size());

    return prompt_tokens;
}

// Decode a single chunk of audio through the Qwen3-ASR decoder
std::pair<ov::genai::EncodedResults, bool> qwen3_asr_decode(
    std::shared_ptr<ov::genai::WhisperDecoder> decoder,
    const std::vector<int64_t>& input_ids,
    const ov::Tensor& encoder_hidden_state,
    const std::shared_ptr<ov::genai::StreamerBase> streamer_ptr,
    ov::genai::Sampler& sampler,
    ov::genai::SequenceGroup::Ptr sequence_group,
    const ov::genai::WhisperGenerationConfig& config,
    ov::genai::RawPerfMetrics& raw_metrics) {
    const auto handle = std::make_shared<ov::genai::GenerationHandleImpl>(sequence_group->get_generation_stream(),
                                                                          sequence_group->get_sampling_parameters());

    // Stream tokens to the streamer as they are generated
    auto stream_generated_tokens = [&streamer_ptr, &handle]() {
        if (!streamer_ptr || !handle->can_read()) {
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

    // Feed the full prompt as the initial decoder input
    const ov::Tensor input_ids_tensor{ov::element::i64,
                                       {1, input_ids.size()},
                                       const_cast<int64_t*>(input_ids.data())};

    const auto infer_start = std::chrono::steady_clock::now();
    decoder->start_async(encoder_hidden_state, input_ids_tensor, beam_idx);

    auto logits = decoder->wait();
    const auto infer_end = std::chrono::steady_clock::now();
    const auto infer_ms = ov::genai::PerfMetrics::get_microsec(infer_end - infer_start);
    raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);
    raw_metrics.m_token_infer_durations.emplace_back(infer_ms);
    raw_metrics.m_new_token_times.emplace_back(infer_end);
    raw_metrics.m_batch_sizes.emplace_back(batch_size);

    // Schedule prompt tokens and sample the first generated token
    int64_t output_sequence_len = logits.get_shape().at(1);
    sequence_group->schedule_tokens(sequence_group->get_prompt_len());
    sequence_group->set_output_seq_len(output_sequence_len);

    sampler.sample({sequence_group}, logits);
    stream_generated_tokens();

    // Autoregressive generation phase
    while (!sequence_group->has_finished() && !sequence_group->handle_stopped() &&
           !sequence_group->handle_cancelled()) {
        sequence_group->schedule_tokens(1);

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
                if (position_id < sequence_group->get_prompt_len()) {
                    input_ids_data[batch] = sequence_group->get_prompt_ids()[position_id];
                } else {
                    input_ids_data[batch] =
                        sequence->get_generated_ids()[position_id - sequence_group->get_prompt_len()];
                }
            }
            input_ids_data += num_scheduled_tokens;

            auto beam_idx_val = beam_idxs[sequence->get_id()];
            next_beams.push_back(beam_idx_val);
        }

        const auto infer_start = std::chrono::steady_clock::now();

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

        sampler.sample({sequence_group}, logits);
    }

    stream_generated_tokens();

    // Collect results
    ov::genai::EncodedResults results;
    const auto sampling_params = sequence_group->get_sampling_parameters();

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

}  // namespace

namespace ov {
namespace genai {

WhisperGenerateResult qwen3_asr_generate(const ov::genai::WhisperGenerationConfig& config,
                                          const ov::genai::WhisperConfig& model_config,
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

    result.perf_metrics.whisper_raw_metrics.features_extraction_durations = {};
    result.perf_metrics.whisper_raw_metrics.word_level_timestamps_processing_durations = {};

    // Step 1: Extract mel spectrogram features from raw audio
    const auto extract_start = std::chrono::steady_clock::now();
    auto input_features = feature_extractor.extract(raw_speech);
    const auto extract_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - extract_start);
    result.perf_metrics.whisper_raw_metrics.features_extraction_durations.emplace_back(extract_ms);

    // Qwen3-ASR processes the full audio; for long audio, process in chunks matching nb_max_frames
    std::vector<int64_t>& output_tokens = result.output_tokens;

    for (size_t chunk_offset = 0; chunk_offset < input_features.n_frames;
         chunk_offset += feature_extractor.nb_max_frames) {
        // Get the chunk of mel features (padded to nb_max_frames if shorter)
        size_t chunk_frames = std::min(feature_extractor.nb_max_frames,
                                        input_features.n_frames - chunk_offset);
        size_t frames_to_encode = feature_extractor.nb_max_frames;
        auto chunk_data = input_features.get_data_with_offset(chunk_offset, frames_to_encode);

        // Step 2: Run encoder to get hidden states
        ov::Tensor hidden_state_tensor =
            qwen3_asr_encode(encoder, chunk_data, feature_extractor.feature_size, frames_to_encode, raw_metrics);

        // Get the number of audio output tokens from encoder output shape
        const size_t num_audio_tokens = hidden_state_tensor.get_shape().at(1);

        // Step 3: Build decoder input tokens (prompt template + audio pad tokens)
        std::vector<int64_t> prompt_tokens = build_qwen3_asr_prompt(tokenizer, model_config, num_audio_tokens);

        // Step 4: Create a sequence group for sampling and run autoregressive decoding
        SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(0, prompt_tokens, config, 1);

        auto [chunk_result, cancelled] = qwen3_asr_decode(decoder,
                                                           prompt_tokens,
                                                           hidden_state_tensor,
                                                           streamer,
                                                           sampler,
                                                           sequence_group,
                                                           config,
                                                           raw_metrics);

        decoder->reset_state();

        // Append generated tokens
        std::vector<int64_t> chunk_output_tokens = chunk_result.tokens[0];
        output_tokens.insert(output_tokens.end(), chunk_output_tokens.begin(), chunk_output_tokens.end());

        if (cancelled) {
            break;
        }
    }

    if (streamer) {
        streamer->end();
    }

    return result;
}

}  // namespace genai
}  // namespace ov
