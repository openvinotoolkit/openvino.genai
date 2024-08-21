// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <iostream>
#include <openvino/openvino.hpp>
#include <regex>
#include <thread>

#include "../utils.hpp"
#include "audio_processing.hpp"
#include "openvino/genai/audio_utils.hpp"
#include "openvino/genai/streamer_base.hpp"
#include "openvino/genai/whisper_generation_config.hpp"
#include "whisper_models.hpp"

#define AUDIO_CHUNK_SIZE 480000  // 16K sampling X 30 Seconds = 16000 x 30 = 480000

namespace {

// todo: support multi batch
void supress_tokens(ov::Tensor& logits, std::vector<int64_t> supress_tokens) {
    auto data = logits.data<float>();
    for (auto supress_token : supress_tokens) {
        data[supress_token] = -std::numeric_limits<float>::infinity();
    }
}

ov::Tensor encode(ov::InferRequest& request, std::vector<float>& mel_data) {
    ov::Shape input_shape = {1, 80, 3000};
    ov::Tensor input_tensor(ov::element::f32, input_shape, mel_data.data());

    request.set_tensor("input_features", input_tensor);

    request.infer();

    return request.get_tensor("last_hidden_state");
}

void set_past_kev_value(ov::CompiledModel& source_compiled_model, ov::InferRequest& source, ov::InferRequest& dest) {
    // source outputs:
    // present.0.decoder.key
    // present.0.decoder.value
    // present.0.encoder.key
    // present.0.encoder.value

    // dest imputs:
    // past_key_values.0.decoder.key
    // past_key_values.0.decoder.value
    // past_key_values.0.encoder.key
    // past_key_values.0.encoder.value

    for (auto& dec_output : source_compiled_model.outputs()) {
        std::string dec_output_name = dec_output.get_any_name();
        if (dec_output_name.find("logits") != std::string::npos) {
            continue;
        }

        std::string with_past_input_name =
            std::regex_replace(dec_output_name, std::regex("present"), "past_key_values");

        auto kv_tensor = source.get_tensor(dec_output_name);
        dest.set_tensor(with_past_input_name, ov::Tensor{kv_tensor});
    }
}

int64_t decode(ov::Tensor& encoder_hidden_state,
               ov::InferRequest& decoder,
               std::vector<int64_t> input_ids,
               const ov::genai::WhisperGenerationConfig& config) {
    decoder.set_tensor("encoder_hidden_states", ov::Tensor{encoder_hidden_state});

    ov::Tensor input_ids_tensor(ov::element::i64, {1, input_ids.size()}, input_ids.data());
    decoder.set_tensor("input_ids", input_ids_tensor);

    decoder.infer();

    auto output_tensor = decoder.get_tensor("logits");

    supress_tokens(output_tensor, config.begin_suppress_tokens);

    int64_t output_token = ov::genai::utils::argmax(output_tensor, 0);

    return output_token;
}

int64_t decode_with_past(ov::Tensor& encoder_hidden_state,
                         ov::InferRequest& decoder_with_past,
                         ov::CompiledModel& decoder_with_past_compiled,
                         int64_t input_id,
                         size_t cache_position,
                         const ov::genai::WhisperGenerationConfig& config) {
    decoder_with_past.set_tensor("encoder_hidden_states", ov::Tensor{encoder_hidden_state});

    std::vector<int64_t> input_ids = {input_id};
    ov::Tensor input_ids_tensor(ov::element::i64, {1, 1}, input_ids.data());
    decoder_with_past.set_tensor("input_ids", input_ids_tensor);

    ov::Tensor cache_position_tensor = decoder_with_past.get_tensor("cache_position");
    cache_position_tensor.set_shape({1});
    cache_position_tensor.data<int64_t>()[0] = cache_position;

    decoder_with_past.infer();

    auto output_tensor = decoder_with_past.get_tensor("logits");

    supress_tokens(output_tensor, config.suppress_tokens);

    int64_t output_token = ov::genai::utils::argmax(output_tensor, 0);

    set_past_kev_value(decoder_with_past_compiled, decoder_with_past, decoder_with_past);

    return output_token;
}

std::pair<bool, std::vector<int64_t>> full_decode(ov::Tensor& encoder_hidden_state,
                                                  const ov::genai::WhisperGenerationConfig& config,
                                                  ov::genai::WhisperInitializedModels& models,
                                                  size_t max_new_tokens,
                                                  const std::shared_ptr<ov::genai::StreamerBase> streamer) {
    std::vector<int64_t> input_ids = {config.decoder_start_token_id,
                                      config.language_token_id,
                                      config.transcribe_token_id,
                                      config.no_timestamps_token_id};

    int64_t output_token = decode(encoder_hidden_state, models.decoder, input_ids, config);

    std::vector<int64_t> output_tokens{output_token};

    if (streamer && streamer->put(output_token)) {
        return {true, output_tokens};
    }

    if (max_new_tokens == 1) {
        return {false, output_tokens};
    }

    set_past_kev_value(models.decoder_compiled, models.decoder, models.decoder_with_past);

    for (size_t i = 0; i < max_new_tokens - 1; i++) {
        auto output_token = decode_with_past(encoder_hidden_state,
                                             models.decoder_with_past,
                                             models.decoder_with_past_compiled,
                                             output_tokens.back(),
                                             input_ids.size() + output_tokens.size() - 1,
                                             config);

        if (output_token == config.eos_token_id) {
            break;
        }

        output_tokens.push_back(output_token);

        if (streamer && streamer->put(output_token)) {
            return {true, output_tokens};
        }
    }

    return {false, output_tokens};
}

}  // namespace

namespace ov {
namespace genai {
std::vector<int64_t> whisper_generate(const ov::genai::WhisperGenerationConfig& config,
                                      std::vector<float> pcmf32,
                                      ov::genai::WhisperInitializedModels& models,
                                      const std::shared_ptr<StreamerBase> streamer) {
    ov::genai::utils::audio::fill_sin_cos_table();

    std::vector<int64_t> output_tokens;
    size_t max_new_tokens = config.get_max_new_tokens();

    // on a small chunk sizes (eg 1s) the results are starting to diverge from HF
    for (int i = 0; i < pcmf32.size(); i += AUDIO_CHUNK_SIZE) {
        if (output_tokens.size() >= max_new_tokens) {
            break;
        }

        // Split audio data into fixed 30 seconds x 30 seconds window.
        int copy_size = std::min((int)(pcmf32.size() - i), (int)AUDIO_CHUNK_SIZE);
        std::vector<float> pcmf32_sub_chunk(pcmf32.begin() + i, pcmf32.begin() + i + copy_size);

        std::vector<float> mel_data;

        ov::genai::utils::audio::mel_spectrogram_convert_audio(
            pcmf32_sub_chunk.data(),
            pcmf32_sub_chunk.size(),
            WHISPER_SAMPLE_RATE,
            WHISPER_N_FFT,
            WHISPER_HOP_LENGTH,
            std::min(4, (int32_t)std::thread::hardware_concurrency()),
            mel_data);

        ov::Tensor hidden_state_tensor = encode(models.encoder, mel_data);

        bool cancelled;
        std::vector<int64_t> chunk_output_tokens;
        std::tie(cancelled, chunk_output_tokens) =
            full_decode(hidden_state_tensor, config, models, max_new_tokens - output_tokens.size(), streamer);

        output_tokens.insert(output_tokens.end(), chunk_output_tokens.begin(), chunk_output_tokens.end());

        if (cancelled) {
            break;
        }
    }

    return output_tokens;
}
}  // namespace genai
}  // namespace ov
