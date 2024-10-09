// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "whisper_pipeline_static.hpp"
#include "text_callback_streamer.hpp"
#include "utils.hpp"

#include "openvino/runtime/intel_npu/properties.hpp"

#include <chrono>
#include <regex>

namespace {

ov::genai::WhisperGenerationConfig from_config_json_if_exists(const std::filesystem::path& model_path) {
    auto config_file_path = model_path / "generation_config.json";
    if (std::filesystem::exists(config_file_path)) {
        return ov::genai::WhisperGenerationConfig((config_file_path).string());
    } else {
        return ov::genai::WhisperGenerationConfig{};
    }
}

ov::genai::OptionalWhisperGenerationConfig get_config_from_map(const ov::AnyMap& config_map) {
    if (config_map.count("generation_config")) {
        return config_map.at("generation_config").as<ov::genai::WhisperGenerationConfig>();
    } else {
        return std::nullopt;
    }
}

void suppress_tokens(ov::Tensor& logits, const size_t batch_idx, const std::vector<int64_t>& suppress_tokens) {
    OPENVINO_ASSERT(logits.get_shape()[0] >= batch_idx, "logits batch size doesn't match the number of beams");

    size_t vocab_size = logits.get_shape().back();
    size_t batch_offset = batch_idx * logits.get_shape()[1] * vocab_size;
    size_t sequence_offset = (logits.get_shape()[1] - 1) * vocab_size;
    float* logits_data = logits.data<float>() + batch_offset + sequence_offset;

    for (auto supress_token : suppress_tokens) {
        logits_data[supress_token] = -std::numeric_limits<float>::infinity();
    }
}

template <typename T>
void fill_tensor(ov::Tensor tensor, T fill_val) {
    auto* tensor_data = tensor.data<T>();
    std::fill(tensor_data, tensor_data + tensor.get_size(), fill_val);
}

template <typename T>
void copy_to_tensor(const std::vector<T>& src_vec, ov::Tensor dst_tensor) {
    auto* dst_ptr = dst_tensor.data<T>();
    OPENVINO_ASSERT(src_vec.size() == dst_tensor.get_size());
    std::copy(src_vec.begin(), src_vec.end(), dst_ptr);
}

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
    ov::Tensor input_tensor(ov::element::f32, { 1, feature_size, nb_max_frames }, mel_data.data());
    request.set_tensor("input_features", input_tensor);
    request.infer();
    return request.get_tensor("last_hidden_state");
}

// FIXME: Duplicate from llm_pipeline_static.cpp - need to reuse instead of copy-paste
ov::Tensor make_tensor_slice(ov::Tensor tensor, size_t dim, size_t start_pos, size_t end_pos) {
    ov::Shape start_shape(std::vector<size_t>(tensor.get_shape().size(), 0u));
    start_shape[dim] = start_pos;
    ov::Shape end_shape = tensor.get_shape();
    end_shape[dim] = end_pos;
    return ov::Tensor(tensor, start_shape, end_shape);
}

void set_cross_attn_key_value(ov::InferRequest& source, ov::InferRequest& dest) {
    // NB: Source outputs:
    // present_key_values.0.encoder.key
    // present_key_values.0.encoder.value

    // NB: Dest inputs:
    // past_key_values.0.encoder.key
    // past_key_values.0.encoder.value

    for (auto& source_output : source.get_compiled_model().outputs()) {
        std::string source_output_name = source_output.get_any_name();
        if (source_output_name.find("encoder") == std::string::npos) {
            continue;
        }
        std::string with_past_input_name =
            std::regex_replace(source_output_name, std::regex("present"), "past");
        dest.set_tensor(with_past_input_name, source.get_tensor(source_output_name));
    }
}

void update_past_key_value(ov::InferRequest& source,
                           ov::InferRequest& dest,
                           const size_t kv_pos = 0u) {
    // NB: Source outputs:
    // present_key_values.0.decoder.key
    // present_key_values.0.decoder.value

    // NB: Dest inputs:
    // past_key_values.0.decoder.key
    // past_key_values.0.decoder.value

    for (auto& source_output : source.get_compiled_model().outputs()) {
        std::string source_output_name = source_output.get_any_name();
        if (source_output_name.find("decoder") == std::string::npos) {
            continue;
        }

        std::string with_past_input_name =
            std::regex_replace(source_output_name, std::regex("present"), "past");

        auto src_kv_tensor = source.get_tensor(source_output_name);
        auto dst_kv_tensor = dest.get_tensor(with_past_input_name);
        auto kv_size = src_kv_tensor.get_shape()[2];
        // NB: Copy src_kv_tensor into dst_kv_tensor[:, :, kv_pos:kv_pos+kv_size, :]
        auto dst_kv_tensor_slice = make_tensor_slice(dst_kv_tensor, 2u, kv_pos, kv_pos + kv_size);
        src_kv_tensor.copy_to(dst_kv_tensor_slice);
    }
}

int64_t decode(ov::Tensor& encoder_hidden_state,
               ov::InferRequest& decoder,
               std::vector<int32_t>& input_ids,
               const ov::genai::WhisperGenerationConfig& config,
               bool do_suppress_tokens = true) {
    // NB: Fill decoder inputs
    encoder_hidden_state.copy_to(decoder.get_tensor("encoder_hidden_states"));
    copy_to_tensor(input_ids, decoder.get_tensor("input_ids"));
    fill_tensor<ov::float16>(decoder.get_tensor("attention_mask"), 1);

    decoder.infer();

    auto output_tensor = decoder.get_tensor("logits");
    if (do_suppress_tokens) {
        suppress_tokens(output_tensor, 0, config.begin_suppress_tokens);
        suppress_tokens(output_tensor, 0, config.suppress_tokens);
    }
    int64_t output_token = ov::genai::utils::argmax(output_tensor, 0);
    return output_token;
}

int64_t decode_with_past(ov::InferRequest& decoder_with_past,
                         const int64_t input_id,
                         const int64_t position_id,
                         const ov::genai::WhisperGenerationConfig& config) {

    // FIXME: Avoid this cast to i32. Why it's not i64 precision in model?
    decoder_with_past.get_tensor("input_ids").data<int32_t>()[0] = static_cast<int32_t>(input_id);
    // FIXME: Avoid this cast to i32. Why it's not i64 precision in model?
    decoder_with_past.get_tensor("position_ids").data<int32_t>()[0] = static_cast<int32_t>(position_id);
    // FIXME: Is "attention_mask" supposed to be f16?
    decoder_with_past.get_tensor("attention_mask").data<ov::float16>()[position_id-1] = 1u;

    decoder_with_past.infer();

    auto output_tensor = decoder_with_past.get_tensor("logits");
    suppress_tokens(output_tensor, 0, config.suppress_tokens);

    int64_t output_token = ov::genai::utils::argmax(output_tensor, 0);
    return output_token;
}

void zero_past_key_values(ov::InferRequest& request) {
    for (auto& input : request.get_compiled_model().inputs()) {
        std::string past_key_value_decoder_name = input.get_any_name();
        if (past_key_value_decoder_name.find("decoder") == std::string::npos ||
            past_key_value_decoder_name.find("past_key_values") == std::string::npos) {
            continue;
        }
        fill_tensor<float>(request.get_tensor(past_key_value_decoder_name), 0);
    }
}

void prepare_decoder_with_past(ov::InferRequest& decoder_with_past, ov::InferRequest& decoder) {
    // NB: Prepare attetion mask to be in a format [1, 1, 1, 0, 0, 0, 0, ..., 1]
    auto attention_mask = decoder_with_past.get_tensor("attention_mask");
    auto* attention_mask_ptr = attention_mask.data<ov::float16>();
    std::fill(attention_mask_ptr, attention_mask_ptr + 3u, 1);
    std::fill(attention_mask_ptr + 3u, attention_mask_ptr + attention_mask.get_size() - 1, 0);
    attention_mask_ptr[attention_mask.get_size() - 1] = 1;
    // NB: Zero past_key_values.*.decoder.value tensors
    zero_past_key_values(decoder_with_past);
    // NB: Copy KV-caches from decoder
    set_cross_attn_key_value(decoder, decoder_with_past);
    update_past_key_value(decoder, decoder_with_past);
};

std::pair<bool, std::vector<int64_t>> full_decode(ov::Tensor& encoder_hidden_state,
                                                  const ov::genai::WhisperGenerationConfig& config,
                                                  ov::genai::WhisperInitializedModels& models,
                                                  const size_t max_new_tokens,
                                                  const std::shared_ptr<ov::genai::StreamerBase> streamer) {
    // FIXME: These values shouldn't be hardcoded!
    std::vector<int32_t> input_ids = { 50258, 50259, 50359, 50363 };
    int64_t output_token = decode(encoder_hidden_state, models.decoder, input_ids, config);
    std::vector<int64_t> output_tokens{ output_token };

    if (streamer && streamer->put(output_token)) {
        return { true, output_tokens };
    }

    if (max_new_tokens == 1) {
        return { false, output_tokens };
    }

    prepare_decoder_with_past(models.decoder_with_past, models.decoder);

    for (size_t i = 0; i < max_new_tokens - 1; i++) {
        auto output_token =
            decode_with_past(models.decoder_with_past, output_tokens.back(), i + input_ids.size(), config);
        update_past_key_value(models.decoder_with_past, models.decoder_with_past, i + input_ids.size());

        if (output_token == config.eos_token_id) {
            break;
        }
        output_tokens.push_back(output_token);

        if (streamer && streamer->put(output_token)) {
            return { true, output_tokens };
        }
    }

    return { false, output_tokens };
}

}  // namespace

namespace ov {
namespace genai {

StaticWhisperPipeline::StaticWhisperPipeline(const std::filesystem::path& model_path,
                                             const ov::genai::Tokenizer& tokenizer,
                                             const ov::AnyMap& plugin_config)
    : WhisperPipelineImplBase(from_config_json_if_exists(model_path),
                              tokenizer,
                              WhisperFeatureExtractor{(model_path / "preprocessor_config.json").string()}) {
        ov::Core core;

        // FIXME: Only for debug...
        std::string device = "CPU";

        auto encoder_model = core.read_model(model_path / "openvino_encoder_model.xml");
        auto decoder_model = core.read_model(model_path / "openvino_decoder_model.xml");
        auto decoder_with_past_model = core.read_model(model_path / "openvino_decoder_with_past_model.xml");

        // TODO: There must be model reshape to eliminate dynamism!

        m_models.encoder = core.compile_model(encoder_model, device).create_infer_request();
        m_models.decoder = core.compile_model(decoder_model, device).create_infer_request();
        m_models.decoder_with_past = core.compile_model(decoder_with_past_model, device).create_infer_request();

        // If eos_token_id was not provided, take value
        if (m_generation_config.eos_token_id == -1) {
            m_generation_config.set_eos_token_id(m_tokenizer.get_eos_token_id());
        }
}

StaticWhisperPipeline::StaticWhisperPipeline(const std::filesystem::path& model_path,
                                             const ov::AnyMap& plugin_config)
    : StaticWhisperPipeline(model_path, model_path.string(), plugin_config) {
}

DecodedResults StaticWhisperPipeline::generate(const RawSpeechInput& raw_speech_input,
                                               OptionalWhisperGenerationConfig generation_config,
                                               StreamerVariant streamer) {
        WhisperGenerationConfig config = (generation_config.has_value()) ? *generation_config : m_generation_config;
        config.validate();

        std::shared_ptr<StreamerBase> streamer_ptr;
        if (auto streamer_obj = std::get_if<std::monostate>(&streamer)) {
            streamer_ptr = nullptr;
        } else if (auto streamer_obj = std::get_if<std::shared_ptr<StreamerBase>>(&streamer)) {
            streamer_ptr = *streamer_obj;
        } else if (auto callback = std::get_if<std::function<bool(std::string)>>(&streamer)) {
            streamer_ptr = std::make_shared<TextCallbackStreamer>(m_tokenizer, *callback);
        }

        std::vector<int64_t> output_tokens;
        size_t max_new_tokens = config.get_max_new_tokens();

        for (size_t chunk_offset = 0; chunk_offset < raw_speech_input.size(); chunk_offset += m_feature_extractor.n_samples) {
            if (output_tokens.size() >= max_new_tokens) {
                break;
            }

            size_t copy_size = std::min((raw_speech_input.size() - chunk_offset), size_t(m_feature_extractor.n_samples));
            std::vector<float> input_features_sub_chunk(raw_speech_input.begin() + chunk_offset,
                                                        raw_speech_input.begin() + chunk_offset + copy_size);

            auto input_features = m_feature_extractor.extract(input_features_sub_chunk);
            ov::Tensor hidden_state_tensor =
                encode(m_models.encoder, input_features, m_feature_extractor.feature_size, m_feature_extractor.nb_max_frames);

            bool cancelled;
            std::vector<int64_t> chunk_output_tokens;
            std::tie(cancelled, chunk_output_tokens) =
                full_decode(hidden_state_tensor, config, m_models, max_new_tokens - output_tokens.size(), streamer_ptr);

            output_tokens.insert(output_tokens.end(), chunk_output_tokens.begin(), chunk_output_tokens.end());

            if (cancelled) {
                break;
            }
        }

        DecodedResults decoded_results{std::vector{m_tokenizer.decode(output_tokens)}, std::vector{1.f}};
        return decoded_results;
}

}  // namespace genai
}  // namespace ov
