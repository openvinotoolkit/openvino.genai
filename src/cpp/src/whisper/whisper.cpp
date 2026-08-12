// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "whisper.hpp"

#include <algorithm>
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

// `initial_step` is passed explicitly so callers that never enable timestamps do not have to materialize that
// map just to signal the first step.
void process_whisper_logits(ov::Tensor logits,
                            const ov::genai::WhisperGenerationConfig& config,
                            const bool return_timestamps,
                            const bool initial_step,
                            const std::map<size_t, std::vector<int64_t>>& batch_to_generated_ids) {
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
                                                  ov::genai::RawPerfMetrics& raw_metrics,
                                                  ov::genai::WhisperRawPerfMetrics& whisper_raw_metrics) {
    const auto handle = std::make_shared<ov::genai::GenerationHandleImpl>(sequence_group->get_generation_stream(),
                                                                          sequence_group->get_sampling_parameters());

    auto stream_generated_tokens = [&streamer_ptr, &handle, &return_timestamps]() {
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

    process_whisper_logits(logits, config, return_timestamps, /*initial_step=*/true, {});

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
        whisper_raw_metrics.decode_inference_durations.emplace_back(infer_ms);

        process_whisper_logits(logits, config, return_timestamps, /*initial_step=*/false, batch_to_generated_ids);

        {
            const auto sample_start = std::chrono::steady_clock::now();
            sampler.sample({sequence_group}, logits);
            raw_metrics.m_sampling_durations.emplace_back(
                ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - sample_start));
        }
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

    sampler.clear_request_info(sequence_group->get_request_id());

    return {results, (sequence_group->handle_stopped() || sequence_group->handle_cancelled())};
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
    size_t batch_size = (devices[0] == "NPU") ? 1 : 0;
    request.set_tensor("input_features", ov::Tensor(ov::element::f32, {batch_size, feature_size, nb_max_frames}));

    return request.get_tensor("last_hidden_state");
}

ov::genai::SotTokensResult prepare_sot_tokens(ov::Tensor& encoder_hidden_state,
                                              std::shared_ptr<ov::genai::WhisperDecoder> decoder,
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

// ---------------------------------------------------------------------------------------------------------------
// Batched short-form generation helpers
// ---------------------------------------------------------------------------------------------------------------

// Run one encoder inference for the batch.
ov::Tensor encode_batch(ov::InferRequest& request,
                        const std::vector<std::vector<float>>& mel_data_per_item,
                        const size_t feature_size,
                        const size_t nb_max_frames,
                        ov::genai::RawPerfMetrics& raw_metrics,
                        ov::genai::WhisperRawPerfMetrics& whisper_raw_metrics) {
    const size_t batch_size = mel_data_per_item.size();
    const size_t features_per_item = feature_size * nb_max_frames;

    ov::Tensor input_tensor(ov::element::f32, {batch_size, feature_size, nb_max_frames});
    auto* input_data = input_tensor.data<float>();

    for (size_t batch = 0; batch < batch_size; batch++) {
        OPENVINO_ASSERT(mel_data_per_item[batch].size() == features_per_item,
                        "Mel spectrogram required size: ",
                        feature_size,
                        " * ",
                        nb_max_frames,
                        ". Actual size: ",
                        mel_data_per_item[batch].size(),
                        " for input at index ",
                        batch,
                        ".");
        std::copy(mel_data_per_item[batch].begin(),
                  mel_data_per_item[batch].end(),
                  input_data + batch * features_per_item);
    }

    request.set_tensor("input_features", input_tensor);

    const auto infer_start = std::chrono::steady_clock::now();
    request.infer();
    const auto infer_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);
    raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);
    whisper_raw_metrics.encode_inference_durations.emplace_back(infer_ms);

    // Drop the infer request's reference to the batched input buffer. The encoder output remains valid until the
    // request is reused for another inference.
    request.set_tensor("input_features", ov::Tensor(ov::element::f32, {0, feature_size, nb_max_frames}));

    return request.get_tensor("last_hidden_state");
}

// Pack live decoder rows contiguously for sampling.
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

// Resolve SOT tokens for each audio, with language detection performed per row.
std::vector<ov::genai::SotTokensResult> prepare_sot_tokens_batch(ov::Tensor& encoder_hidden_state,
                                                                 std::shared_ptr<ov::genai::WhisperDecoder> decoder,
                                                                 const ov::genai::WhisperGenerationConfig& config,
                                                                 const size_t batch_size,
                                                                 ov::genai::RawPerfMetrics& raw_metrics) {
    if (!config.is_multilingual) {
        // non-multilingual whisper models are english-only
        return std::vector<ov::genai::SotTokensResult>(
            batch_size,
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

struct BatchedDecodeResult {
    std::vector<std::vector<int64_t>> tokens;
};

// Run fixed-width greedy decoding, keeping finished requests as filler rows and excluding them from sampling.
BatchedDecodeResult decode_batch(std::shared_ptr<ov::genai::WhisperDecoder> decoder,
                                 const std::vector<std::vector<int64_t>>& prompts,
                                 const ov::Tensor& encoder_hidden_state,
                                 ov::genai::Sampler& sampler,
                                 std::vector<ov::genai::SequenceGroup::Ptr>& sequence_groups,
                                 const ov::genai::WhisperGenerationConfig& config,
                                 ov::genai::RawPerfMetrics& raw_metrics,
                                 ov::genai::WhisperRawPerfMetrics& whisper_raw_metrics) {
    const size_t batch_size = prompts.size();
    const size_t prompt_len = prompts.front().size();

    // All rows must use the same prompt length; only the language token may differ.
    for (size_t request = 0; request < batch_size; request++) {
        OPENVINO_ASSERT(prompts[request].size() == prompt_len,
                        "Batched Whisper generation requires an identical prompt length for every input. Input ",
                        request,
                        " has ",
                        prompts[request].size(),
                        " prompt tokens, expected ",
                        prompt_len,
                        ".");
    }

    // Use a filler token for finished rows, which are no longer sampled.
    const int64_t filler_token = config.decoder_start_token_id;

    // Prefill.
    ov::Tensor input_ids = decoder->create_host_tensor(ov::element::i64, {batch_size, prompt_len});
    auto* input_ids_data = input_ids.data<int64_t>();
    for (size_t request = 0; request < batch_size; request++) {
        std::copy(prompts[request].begin(), prompts[request].end(), input_ids_data + request * prompt_len);
    }

    // Greedy batching never reorders state rows, so beam_idx stays the identity mapping for the whole call.
    ov::Tensor beam_idx = decoder->create_host_tensor(ov::element::i32, {batch_size});
    std::iota(beam_idx.data<int32_t>(), beam_idx.data<int32_t>() + batch_size, 0);

    auto infer_start = std::chrono::steady_clock::now();
    decoder->start_async(encoder_hidden_state, input_ids, beam_idx);
    auto logits = decoder->wait();
    auto infer_end = std::chrono::steady_clock::now();
    auto infer_ms = ov::genai::PerfMetrics::get_microsec(infer_end - infer_start);
    raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);
    raw_metrics.m_token_infer_durations.emplace_back(infer_ms);
    raw_metrics.m_new_token_times.emplace_back(infer_end);
    whisper_raw_metrics.decode_inference_durations.emplace_back(infer_ms);

    process_whisper_logits(logits, config, false, /*initial_step=*/true, {});

    const size_t output_sequence_len = logits.get_shape().at(1);
    for (size_t request = 0; request < batch_size; request++) {
        sequence_groups[request]->schedule_tokens(prompt_len);
        sequence_groups[request]->set_output_seq_len(output_sequence_len);
    }

    {
        const auto sample_start = std::chrono::steady_clock::now();
        auto sampler_output = sampler.sample(sequence_groups, logits);
        raw_metrics.m_sampling_durations.emplace_back(
            ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - sample_start));
        raw_metrics.m_batch_sizes.emplace_back(sampler_output.num_generated_tokens);
    }

    // Reuse the fixed-width input tensor across decoding steps.
    ov::Tensor next_input_ids = decoder->create_host_tensor(ov::element::i64, {batch_size, 1});

    std::vector<ov::genai::SequenceGroup::Ptr> active_groups;
    active_groups.reserve(batch_size);

    std::vector<size_t> live_rows;
    live_rows.reserve(batch_size);

    // Generation.
    while (true) {
        active_groups.clear();
        live_rows.clear();

        auto* next_input_ids_data = next_input_ids.data<int64_t>();
        std::fill_n(next_input_ids_data, batch_size, filler_token);

        for (size_t request = 0; request < batch_size; request++) {
            auto& sequence_group = sequence_groups[request];
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

            const auto& sequence = running_sequences.front();
            const size_t num_processed_tokens = sequence_group->get_num_processed_tokens();
            const size_t request_prompt_len = sequence_group->get_prompt_len();

            next_input_ids_data[request] =
                num_processed_tokens < request_prompt_len
                    ? sequence_group->get_prompt_ids()[num_processed_tokens]
                    : sequence->get_generated_ids()[num_processed_tokens - request_prompt_len];

            live_rows.push_back(request);
            active_groups.push_back(sequence_group);
        }

        if (active_groups.empty()) {
            break;
        }

        infer_start = std::chrono::steady_clock::now();
        decoder->start_async(encoder_hidden_state, next_input_ids, beam_idx);
        logits = decoder->wait();
        infer_end = std::chrono::steady_clock::now();
        infer_ms = ov::genai::PerfMetrics::get_microsec(infer_end - infer_start);
        raw_metrics.m_inference_durations[0] += MicroSeconds(infer_ms);
        raw_metrics.m_token_infer_durations.emplace_back(infer_ms);
        raw_metrics.m_new_token_times.emplace_back(infer_end);
        whisper_raw_metrics.decode_inference_durations.emplace_back(infer_ms);

        // Timestamps are disabled for batched generation, so generated ids are not needed here.
        process_whisper_logits(logits, config, false, /*initial_step=*/false, {});

        const auto sample_start = std::chrono::steady_clock::now();
        auto sampler_output = sampler.sample(active_groups, pack_live_logits(logits, live_rows));
        raw_metrics.m_sampling_durations.emplace_back(
            ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - sample_start));
        raw_metrics.m_batch_sizes.emplace_back(sampler_output.num_generated_tokens);
    }

    BatchedDecodeResult result;
    result.tokens.reserve(batch_size);

    for (size_t request = 0; request < batch_size; request++) {
        auto& sequence_group = sequence_groups[request];
        const auto& sequences = sequence_group->get_finished_sequences();
        OPENVINO_ASSERT(!sequences.empty(), "Internal error: no finished sequence for input at index ", request, ".");

        result.tokens.push_back(sequences[0]->get_generated_ids());

        sampler.clear_request_info(sequence_group->get_request_id());
    }

    return result;
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

    const auto infer_start = std::chrono::steady_clock::now();
    auto input_features = feature_extractor.extract(raw_speech);
    const auto infer_ms = ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - infer_start);
    result.perf_metrics.whisper_raw_metrics.features_extraction_durations.emplace_back(infer_ms);

    const bool is_shortform = input_features.n_frames <= feature_extractor.nb_max_frames;
    // long-form audio processing requires timestamps to be enabled
    const bool return_timestamps = config.return_timestamps || !is_shortform;

    SotTokensResult sot_result;
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
                                                raw_metrics,
                                                result.perf_metrics.whisper_raw_metrics);

        // prepare sot_tokens just once for whole input
        if (sot_result.tokens.empty()) {
            sot_result = prepare_sot_tokens(hidden_state_tensor, decoder, config, raw_metrics);
            result.language = sot_result.language;
        }

        std::vector<int64_t> chunk_sot_tokens = ov::genai::get_prompt_tokens(context_tokens, config, chunk_offset);

        chunk_sot_tokens.insert(chunk_sot_tokens.end(), sot_result.tokens.begin(), sot_result.tokens.end());

        if (!return_timestamps) {
            chunk_sot_tokens.push_back(config.no_timestamps_token_id);
        }

        SequenceGroup::Ptr sequence_group = std::make_shared<SequenceGroup>(0, chunk_sot_tokens, config);

        auto [chunk_result, cancelled] = decode(decoder,
                                                chunk_sot_tokens,
                                                hidden_state_tensor,
                                                streamer,
                                                sampler,
                                                sequence_group,
                                                return_timestamps,
                                                config,
                                                raw_metrics,
                                                result.perf_metrics.whisper_raw_metrics);
        decoder->reset_state();
        std::vector<int64_t> chunk_output_tokens = chunk_result.tokens[0];

        if (return_timestamps) {
            auto extracted_segments = ov::genai::extract_segments(chunk_output_tokens,
                                                                  config,
                                                                  feature_extractor.nb_max_frames,
                                                                  time_precision,
                                                                  chunk_time_offset);

            utils::filter_non_segment_metrics(raw_metrics,
                                              result.perf_metrics.whisper_raw_metrics,
                                              output_tokens.size(),
                                              extracted_segments.segment_ranges);

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
            segment_offset = input_features.n_frames;
        }

        if (cancelled) {
            break;
        }

        if (config.word_timestamps) {
            const auto n_active_frames =
                std::min(feature_extractor.nb_max_frames, input_features.n_active_frames - chunk_offset);

            const auto word_timestamps_processing_start = std::chrono::steady_clock::now();
            const auto word_timestamps = add_word_level_timestamps(sot_result,
                                                                   chunk_output_tokens,
                                                                   tokenizer,
                                                                   decoder,
                                                                   hidden_state_tensor,
                                                                   config,
                                                                   n_active_frames,
                                                                   chunk_time_offset);
            const auto word_timestamps_processing_duration = ov::genai::PerfMetrics::get_microsec(
                std::chrono::steady_clock::now() - word_timestamps_processing_start);

            result.perf_metrics.whisper_raw_metrics.word_level_timestamps_processing_durations[0] +=
                MicroSeconds(word_timestamps_processing_duration);

            if (!result.words.has_value()) {
                result.words = std::vector<WhisperWordTiming>{};
            }
            result.words->insert(result.words->end(), word_timestamps.begin(), word_timestamps.end());
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

std::vector<WhisperGenerateResult> whisper_generate_batch(const ov::genai::WhisperGenerationConfig& config,
                                                          const WhisperContextTokens& context_tokens,
                                                          const std::vector<RawSpeechInput>& raw_speeches,
                                                          ov::InferRequest& encoder,
                                                          std::shared_ptr<WhisperDecoder> decoder,
                                                          WhisperFeatureExtractor& feature_extractor,
                                                          Sampler& sampler) {
    const size_t batch_size = raw_speeches.size();
    OPENVINO_ASSERT(batch_size > 0, "Whisper pipeline expects at least one audio input, got an empty batch.");
    OPENVINO_ASSERT(config.num_beams == 1,
                    "Batched Whisper generation currently supports greedy decoding only. Got num_beams = ",
                    config.num_beams,
                    ".");

    // A single aggregate metrics object describes the whole batched call.
    WhisperPerfMetrics perf_metrics;
    RawPerfMetrics& raw_metrics = perf_metrics.raw_metrics;
    perf_metrics.num_input_tokens = 0;
    raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};
    perf_metrics.whisper_raw_metrics.word_level_timestamps_processing_durations = {{MicroSeconds(0.0f)}};

    // Extract features per audio, then validate the short-form restriction before any inference happens.
    std::vector<std::vector<float>> mel_data_per_item;
    mel_data_per_item.reserve(batch_size);
    for (size_t batch = 0; batch < batch_size; batch++) {
        const auto extract_start = std::chrono::steady_clock::now();
        auto input_features = feature_extractor.extract(raw_speeches[batch]);
        perf_metrics.whisper_raw_metrics.features_extraction_durations.emplace_back(
            ov::genai::PerfMetrics::get_microsec(std::chrono::steady_clock::now() - extract_start));

        OPENVINO_ASSERT(input_features.n_frames <= feature_extractor.nb_max_frames,
                        "Batched Whisper generation supports short-form audio only (up to ",
                        feature_extractor.chunk_length,
                        " seconds). The audio at index ",
                        batch,
                        " is longer and requires the long-form path, which is not supported for batches. Generate "
                        "it with a separate single-audio generate() call.");

        // Short-form features are padded to nb_max_frames so different audio lengths can share one encoder batch.
        mel_data_per_item.emplace_back(input_features.get_data_with_offset(0, feature_extractor.nb_max_frames));
    }

    ov::Tensor hidden_state_tensor = encode_batch(encoder,
                                                  mel_data_per_item,
                                                  feature_extractor.feature_size,
                                                  feature_extractor.nb_max_frames,
                                                  raw_metrics,
                                                  perf_metrics.whisper_raw_metrics);

    auto sot_results = prepare_sot_tokens_batch(hidden_state_tensor, decoder, config, batch_size, raw_metrics);

    const std::vector<int64_t> prefix_tokens = ov::genai::get_prompt_tokens(context_tokens, config, 0);

    std::vector<std::vector<int64_t>> prompts;
    std::vector<SequenceGroup::Ptr> sequence_groups;
    prompts.reserve(batch_size);
    sequence_groups.reserve(batch_size);

    for (size_t batch = 0; batch < batch_size; batch++) {
        std::vector<int64_t> prompt = prefix_tokens;
        prompt.insert(prompt.end(), sot_results[batch].tokens.begin(), sot_results[batch].tokens.end());
        // Timestamps are disabled for batched generation.
        prompt.push_back(config.no_timestamps_token_id);

        prompts.push_back(prompt);
        // One request per audio, with request_id equal to the audio index.
        sequence_groups.push_back(std::make_shared<SequenceGroup>(batch, prompt, config));
    }

    auto decoded = decode_batch(decoder,
                                prompts,
                                hidden_state_tensor,
                                sampler,
                                sequence_groups,
                                config,
                                raw_metrics,
                                perf_metrics.whisper_raw_metrics);

    decoder->reset_state();

    std::vector<WhisperGenerateResult> results(batch_size);
    for (size_t batch = 0; batch < batch_size; batch++) {
        results[batch].output_tokens = std::move(decoded.tokens[batch]);
        results[batch].language = sot_results[batch].language;
        results[batch].perf_metrics = perf_metrics;
    }

    return results;
}
}  // namespace genai
}  // namespace ov
