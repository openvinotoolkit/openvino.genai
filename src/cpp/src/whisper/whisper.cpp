// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "whisper.hpp"

#include <iostream>
#include <openvino/openvino.hpp>
#include <regex>
#include <thread>

#include "../utils.hpp"
#include "logit_processor.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/whisper_generation_config.hpp"
#include "openvino/genai/whisper_pipeline.hpp"
#include "timestamps.hpp"
#include "whisper_config.hpp"
#include "whisper_feature_extractor.hpp"
#include "whisper_models.hpp"

namespace {

ov::Tensor encode(ov::InferRequest& request,
                  std::vector<float>& mel_data,
                  const size_t feature_size,
                  const size_t nb_max_frames) {
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

    request.infer();

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

int64_t decode(ov::Tensor& encoder_hidden_state,
               ov::InferRequest& decoder,
               std::vector<int64_t>& input_ids,
               const ov::genai::WhisperGenerationConfig& config,
               bool apply_logit_processors = true) {
    decoder.set_tensor("encoder_hidden_states", ov::Tensor{encoder_hidden_state});

    ov::Tensor input_ids_tensor(ov::element::i64, {1, input_ids.size()}, input_ids.data());
    decoder.set_tensor("input_ids", input_ids_tensor);

    decoder.infer();

    auto output_tensor = decoder.get_tensor("logits");

    if (apply_logit_processors) {
        ov::genai::do_suppress_tokens(output_tensor, 0, config.begin_suppress_tokens);
        ov::genai::do_suppress_tokens(output_tensor, 0, config.suppress_tokens);

        if (config.return_timestamps) {
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
                         const std::vector<int64_t>& generated_tokens) {
    decoder_with_past.set_tensor("encoder_hidden_states", ov::Tensor{encoder_hidden_state});

    std::vector<int64_t> input_ids = {input_id};
    ov::Tensor input_ids_tensor(ov::element::i64, {1, 1}, input_ids.data());
    decoder_with_past.set_tensor("input_ids", input_ids_tensor);

    ov::Tensor cache_position_tensor = decoder_with_past.get_tensor("cache_position");
    cache_position_tensor.set_shape({1});
    cache_position_tensor.data<int64_t>()[0] = cache_position;

    decoder_with_past.infer();

    auto output_tensor = decoder_with_past.get_tensor("logits");

    ov::genai::do_suppress_tokens(output_tensor, 0, config.suppress_tokens);

    if (config.return_timestamps) {
        ov::genai::process_whisper_timestamp_logits(output_tensor, 0, config, generated_tokens);
    }

    int64_t output_token = ov::genai::utils::argmax(output_tensor, 0);

    return output_token;
}

int64_t detect_language(ov::Tensor& encoder_hidden_state,
                        ov::InferRequest decoder,
                        const ov::genai::WhisperGenerationConfig& config) {
    std::vector<int64_t> input_ids{config.decoder_start_token_id};
    int64_t output_token = decode(encoder_hidden_state, decoder, input_ids, config, false);

    return output_token;
}

std::vector<int64_t> prepare_input_ids(ov::Tensor& encoder_hidden_state,
                                       ov::InferRequest decoder,
                                       const ov::genai::WhisperGenerationConfig& config) {
    if (!config.is_multilingual) {
        return std::vector<int64_t>{config.decoder_start_token_id, config.no_timestamps_token_id};
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

    if (config.return_timestamps) {
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
                                                  const size_t max_new_tokens,
                                                  const std::shared_ptr<ov::genai::StreamerBase> streamer) {
    std::vector<int64_t> input_ids = prepare_input_ids(encoder_hidden_state, models.decoder, config);

    int64_t output_token = decode(encoder_hidden_state, models.decoder, input_ids, config);

    std::vector<int64_t> output_tokens{output_token};

    bool is_timestamp = output_token >= config.begin_timestamps_token_id;
    if (!is_timestamp && streamer && streamer->put(output_token)) {
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
                                             input_ids.size() + output_tokens.size() - 1,
                                             config,
                                             output_tokens);

        if (i == 0) {
            set_past_key_value(models.decoder_with_past, models.decoder_with_past);
        }

        if (output_token == config.eos_token_id) {
            break;
        }

        output_tokens.push_back(output_token);
        bool is_timestamp = output_token >= config.begin_timestamps_token_id;

        if (!is_timestamp && streamer && streamer->put(output_token)) {
            return {true, output_tokens};
        }
    }

    return {false, output_tokens};
}

}  // namespace

namespace ov {
namespace genai {
// hf hash 2 algos for handling long (>30s) audios https://huggingface.co/openai/whisper-large-v3#chunked-long-form
// Sequential: uses a "sliding window" for buffered inference, transcribing 30-second slices one after the other
// Chunked: splits long audio files into shorter ones (with a small overlap between segments), transcribes each segment
// independently, and stitches the resulting transcriptions at the boundaries

// By default, Transformers uses the sequential algorithm. To enable the chunked algorithm, pass the chunk_length_s
// parameter to the pipeline. A chunk length of 30-seconds is optimal. Sequential algo:
// 1. Process whole raw speech into mel spectrogram
// 2. Chunk mel spectrogram into 30s
// 3. Enable timestamps
// 4. Process each chunk sequentially.
// 5. For each chunk stop at first eos token. Start next window from last timestamp found.
//          remove eos tokens if not finished yet
//          remove pad tokens
// 7. Concatenate output tokens
std::pair<std::vector<int64_t>, std::optional<std::vector<Segment>>> whisper_generate(
    const ov::genai::WhisperGenerationConfig& config,
    const ov::genai::WhisperConfig& model_config,
    const RawSpeechInput& raw_speech,
    ov::genai::WhisperInitializedModels& models,
    WhisperFeatureExtractor& feature_extractor,
    const std::shared_ptr<StreamerBase> streamer) {
    std::vector<int64_t> output_tokens;
    size_t max_new_tokens = config.get_max_new_tokens();

    for (size_t chunk_offset = 0; chunk_offset < raw_speech.size(); chunk_offset += feature_extractor.n_samples) {
        if (output_tokens.size() >= max_new_tokens) {
            break;
        }

        // Split audio data into fixed feature_extractor.chunk_size windows.
        size_t copy_size = std::min((raw_speech.size() - chunk_offset), size_t(feature_extractor.n_samples));
        std::vector<float> input_features_sub_chunk(raw_speech.begin() + chunk_offset,
                                                    raw_speech.begin() + chunk_offset + copy_size);

        auto input_features = feature_extractor.extract(input_features_sub_chunk);

        ov::Tensor hidden_state_tensor =
            encode(models.encoder, input_features, feature_extractor.feature_size, feature_extractor.nb_max_frames);

        bool cancelled;
        std::vector<int64_t> chunk_output_tokens;
        std::tie(cancelled, chunk_output_tokens) =
            full_decode(hidden_state_tensor, config, models, max_new_tokens - output_tokens.size(), streamer);

        output_tokens.insert(output_tokens.end(), chunk_output_tokens.begin(), chunk_output_tokens.end());

        if (cancelled) {
            break;
        }
    }

    if (streamer) {
        streamer->end();
    }

    std::optional<std::vector<Segment>> segments = std::nullopt;
    if (config.return_timestamps) {
        // 0.02 by default
        const float time_precision =
            static_cast<float>(feature_extractor.chunk_length) / model_config.max_source_positions;
        std::tie(output_tokens, segments) = ov::genai::extract_segments(output_tokens, config, time_precision);
    }

    return {output_tokens, segments};
}
}  // namespace genai
}  // namespace ov
