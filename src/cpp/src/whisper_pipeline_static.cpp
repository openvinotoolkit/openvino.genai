// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "whisper_pipeline_static.hpp"

#include <chrono>
#include <regex>

#include "debug_utils.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"
#include "text_callback_streamer.hpp"
#include "utils.hpp"
#include "whisper/logit_processor.hpp"
#include "whisper/timestamps.hpp"
#include "whisper/whisper.hpp"
#include "whisper/whisper_config.hpp"

namespace {

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
    copy_to_tensor(mel_data, request.get_tensor("input_features"));
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
        std::string with_past_input_name = std::regex_replace(source_output_name, std::regex("present"), "past");
        dest.set_tensor(with_past_input_name, source.get_tensor(source_output_name));
    }
}

void update_past_key_value(ov::InferRequest& source, ov::InferRequest& dest, const size_t kv_pos = 0u) {
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

        std::string with_past_input_name = std::regex_replace(source_output_name, std::regex("present"), "past");

        auto src_kv_tensor = source.get_tensor(source_output_name);
        auto dst_kv_tensor = dest.get_tensor(with_past_input_name);
        auto kv_size = src_kv_tensor.get_shape()[2];
        // NB: Copy src_kv_tensor into dst_kv_tensor[:, :, kv_pos:kv_pos+kv_size, :]
        auto dst_kv_tensor_slice = make_tensor_slice(dst_kv_tensor, 2u, kv_pos, kv_pos + kv_size);
        src_kv_tensor.copy_to(dst_kv_tensor_slice);
    }
}

void set_decoder_input_ids_attention_mask(ov::InferRequest& decoder,
                                          const std::vector<int32_t>& init_ids,
                                          const int64_t pad_token) {
    auto input_ids_tensor = decoder.get_tensor("input_ids");
    auto attention_mask_tensor = decoder.get_tensor("attention_mask");

    const size_t seq_length = input_ids_tensor.get_shape()[1];

    OPENVINO_ASSERT(seq_length >= init_ids.size());

    // pad right
    // input_ids [token, token, token, pad_token]
    // attention_mask [1, 1, 1, 0]
    auto input_ids_data = input_ids_tensor.data<int32_t>();
    std::copy(init_ids.begin(), init_ids.end(), input_ids_data);
    std::fill(input_ids_data + init_ids.size(),
              input_ids_data + input_ids_tensor.get_size(),
              static_cast<int32_t>(pad_token));

    auto attention_mask_data = attention_mask_tensor.data<ov::float16>();
    std::fill_n(attention_mask_data, init_ids.size(), 1u);
    std::fill(attention_mask_data + init_ids.size(), attention_mask_data + attention_mask_tensor.get_size(), 0u);
}

int64_t decode(ov::Tensor& encoder_hidden_state,
               ov::InferRequest& decoder,
               const std::vector<int32_t>& init_ids,
               const ov::genai::WhisperGenerationConfig& config,
               const bool apply_logit_processors = true,
               const bool return_timestamps = false) {
    // NB: Fill decoder inputs
    encoder_hidden_state.copy_to(decoder.get_tensor("encoder_hidden_states"));
    set_decoder_input_ids_attention_mask(decoder, init_ids, config.pad_token_id);

    decoder.infer();

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

int64_t decode_with_past(ov::InferRequest& decoder_with_past,
                         const int64_t input_id,
                         const int64_t position_id,
                         const ov::genai::WhisperGenerationConfig& config,
                         const bool return_timestamps,
                         const std::vector<int64_t>& generated_tokens) {
    // FIXME: Avoid this cast to i32. Why it's not i64 precision in model?
    decoder_with_past.get_tensor("input_ids").data<int32_t>()[0] = static_cast<int32_t>(input_id);
    // FIXME: Avoid this cast to i32. Why it's not i64 precision in model?
    decoder_with_past.get_tensor("position_ids").data<int32_t>()[0] = static_cast<int32_t>(position_id);
    // FIXME: Is "attention_mask" supposed to be f16?
    decoder_with_past.get_tensor("attention_mask").data<ov::float16>()[position_id - 1] = 1u;

    decoder_with_past.infer();

    auto output_tensor = decoder_with_past.get_tensor("logits");
    ov::genai::do_suppress_tokens(output_tensor, 0, config.suppress_tokens);

    if (return_timestamps) {
        ov::genai::process_whisper_timestamp_logits(output_tensor, 0, config, generated_tokens);
    }

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

int64_t detect_language(ov::Tensor& encoder_hidden_state,
                        ov::InferRequest decoder,
                        const ov::genai::WhisperGenerationConfig& config) {
    decoder.set_tensor("encoder_hidden_states", ov::Tensor{encoder_hidden_state});

    std::vector<int32_t> init_ids{static_cast<int32_t>(config.decoder_start_token_id)};
    set_decoder_input_ids_attention_mask(decoder, init_ids, config.pad_token_id);

    decoder.infer();

    auto output_tensor = decoder.get_tensor("logits");

    auto logits_data = output_tensor.data<float>();

    int64_t output_token;
    float max_prob = -std::numeric_limits<float>::infinity();

    for (auto [_, lang_token] : config.lang_to_id) {
        auto prob = logits_data[lang_token];
        if (prob > max_prob) {
            max_prob = prob;
            output_token = lang_token;
        }
    }

    return output_token;
}

std::vector<int32_t> prepare_init_ids(ov::Tensor& encoder_hidden_state,
                                      ov::InferRequest& decoder,
                                      const ov::genai::WhisperGenerationConfig& config,
                                      const bool return_timestamps) {
    if (!config.is_multilingual) {
        if (return_timestamps) {
            return std::vector<int32_t>{static_cast<int32_t>(config.decoder_start_token_id)};
        } else {
            return std::vector<int32_t>{static_cast<int32_t>(config.decoder_start_token_id),
                                        static_cast<int32_t>(config.no_timestamps_token_id)};
        }
    }

    int32_t language_token_id;
    if (config.language.has_value()) {
        std::string language = *config.language;
        if (config.lang_to_id.count(language)) {
            language_token_id = static_cast<int32_t>(config.lang_to_id.at(language));
        }
    } else {
        language_token_id = detect_language(encoder_hidden_state, decoder, config);
    }

    int32_t task_token_id = static_cast<int32_t>(config.transcribe_token_id);
    if (config.task.has_value() && *config.task == "translate") {
        task_token_id = static_cast<int32_t>(config.translate_token_id);
    }

    if (return_timestamps) {
        return std::vector<int32_t>{static_cast<int32_t>(config.decoder_start_token_id),
                                    language_token_id,
                                    task_token_id};
    }

    return std::vector<int32_t>{static_cast<int32_t>(config.decoder_start_token_id),
                                language_token_id,
                                task_token_id,
                                static_cast<int32_t>(config.no_timestamps_token_id)};
}

std::pair<bool, std::vector<int64_t>> full_decode(ov::Tensor& encoder_hidden_state,
                                                  const ov::genai::WhisperGenerationConfig& config,
                                                  ov::genai::WhisperInitializedModels& models,
                                                  std::vector<int32_t> init_ids,
                                                  const size_t max_new_tokens,
                                                  const bool return_timestamps,
                                                  const std::shared_ptr<ov::genai::StreamerBase> streamer) {
    int64_t output_token = decode(encoder_hidden_state, models.decoder, init_ids, config, true, return_timestamps);
    std::vector<int64_t> output_tokens{output_token};

    const size_t timestamp_begin = config.no_timestamps_token_id + 1;
    bool is_timestamp = output_token >= timestamp_begin;
    if (!is_timestamp && streamer && streamer->put(output_token)) {
        return {true, output_tokens};
    }

    if (max_new_tokens == 1) {
        return {false, output_tokens};
    }

    prepare_decoder_with_past(models.decoder_with_past, models.decoder);

    for (size_t i = 0; i < max_new_tokens - 1; i++) {
        auto output_token = decode_with_past(models.decoder_with_past,
                                             output_tokens.back(),
                                             i + init_ids.size(),
                                             config,
                                             return_timestamps,
                                             output_tokens);
        update_past_key_value(models.decoder_with_past, models.decoder_with_past, i + init_ids.size());

        if (output_token == config.eos_token_id) {
            break;
        }

        output_tokens.push_back(output_token);
        bool is_timestamp = output_token >= timestamp_begin;

        if (!is_timestamp && streamer && streamer->put(output_token)) {
            return {true, output_tokens};
        }
    }

    return {false, output_tokens};
}

bool check_decoder_model_compatibility(const std::shared_ptr<ov::Model>& decoder) {
    for (auto input : decoder->inputs()) {
        if (input.get_any_name() == "attention_mask") {
            return true;
        }
    }
    return false;
}

}  // namespace

namespace ov {
namespace genai {

WhisperPipeline::StaticWhisperPipeline::StaticWhisperPipeline(const std::filesystem::path& models_path,
                                                              const ov::AnyMap& properties)
    : WhisperPipelineImplBase{models_path} {
    ov::Core core = utils::singleton_core();

    auto encoder_model = core.read_model(models_path / "openvino_encoder_model.xml");
    auto decoder_model = core.read_model(models_path / "openvino_decoder_model.xml");
    auto decoder_with_past_model = core.read_model(models_path / "openvino_decoder_with_past_model.xml");

    // TODO: Support models produced by optimum-cli
    if (!check_decoder_model_compatibility(decoder_model)) {
        OPENVINO_THROW("StaticWhisperPipeline expects decoder model has \"attention_mask\" input!");
    }

    // TODO: There must be model reshape to eliminate dynamism!

    m_models.encoder = core.compile_model(encoder_model, "NPU").create_infer_request();
    m_models.decoder = core.compile_model(decoder_model, "NPU").create_infer_request();
    m_models.decoder_with_past = core.compile_model(decoder_with_past_model, "NPU").create_infer_request();

    // If eos_token_id was not provided, take value
    if (m_generation_config.eos_token_id == -1) {
        m_generation_config.set_eos_token_id(m_tokenizer.get_eos_token_id());
    }
}

WhisperDecodedResults WhisperPipeline::StaticWhisperPipeline::generate(
    const RawSpeechInput& raw_speech_input,
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

    auto input_features = m_feature_extractor.extract(raw_speech_input);

    const bool is_shortform = input_features.n_frames <= m_feature_extractor.nb_max_frames;
    // long-form audio processing requires timestamps to be enabled
    const bool return_timestamps = config.return_timestamps || !is_shortform;

    size_t max_new_tokens = config.get_max_new_tokens();

    std::vector<int32_t> init_ids;
    std::vector<int64_t> output_tokens;
    std::vector<Segment> segments;

    // 0.02 by default
    const float time_precision =
        static_cast<float>(m_feature_extractor.chunk_length) / m_model_config.max_source_positions;
    size_t segment_offset = 0;

    for (size_t chunk_offset = 0; chunk_offset < input_features.n_frames; chunk_offset += segment_offset) {
        if (output_tokens.size() >= max_new_tokens) {
            break;
        }

        auto input_features_chunk =
            input_features.get_data_with_offset(chunk_offset, m_feature_extractor.nb_max_frames);

        ov::Tensor hidden_state_tensor = encode(m_models.encoder,
                                                input_features_chunk,
                                                m_feature_extractor.feature_size,
                                                m_feature_extractor.nb_max_frames);

        // prepare init_ids just once for whole input
        if (init_ids.empty()) {
            init_ids = prepare_init_ids(hidden_state_tensor, m_models.decoder, config, return_timestamps);
        }

        auto [cancelled, chunk_output_tokens] = full_decode(hidden_state_tensor,
                                                            config,
                                                            m_models,
                                                            init_ids,
                                                            max_new_tokens - output_tokens.size(),
                                                            return_timestamps,
                                                            streamer_ptr);

        if (return_timestamps) {
            auto extracted_segments = ov::genai::extract_segments(chunk_output_tokens,
                                                                  config,
                                                                  m_feature_extractor.nb_max_frames,
                                                                  time_precision);

            segments.insert(segments.end(), extracted_segments.segments.begin(), extracted_segments.segments.end());

            output_tokens.insert(output_tokens.end(),
                                 extracted_segments.non_timestamp_tokens.begin(),
                                 extracted_segments.non_timestamp_tokens.end());

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

    m_models.decoder_with_past.reset_state();

    if (streamer_ptr) {
        streamer_ptr->end();
    }

    WhisperDecodedResults result{std::vector{m_tokenizer.decode(output_tokens)}, std::vector{1.f}};

    // if return_timestamps wasn't enabled by user
    if (!config.return_timestamps) {
        return result;
    }

    if (segments.size()) {
        std::vector<WhisperDecodedResultChunk> chunks;
        chunks.reserve(segments.size());

        for (auto& segment : segments) {
            chunks.push_back(
                WhisperDecodedResultChunk{segment.m_start, segment.m_end, m_tokenizer.decode(segment.m_tokens)});
        }

        result.chunks = chunks;
    }

    return result;
}

}  // namespace genai
}  // namespace ov
