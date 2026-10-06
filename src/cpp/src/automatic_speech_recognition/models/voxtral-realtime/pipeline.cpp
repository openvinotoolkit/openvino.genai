// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "automatic_speech_recognition/models/voxtral-realtime/pipeline.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

#include "utils.hpp"
#include "whisper/feature_extractor.hpp"

namespace ov::genai {
namespace {

constexpr size_t TEKKEN_BASE64_BITS = 6;
constexpr size_t AUDIO_FRAMES_PER_TEXT_TOKEN = 8;

struct VoxtralConfig {
    size_t audio_hidden_size;
    size_t audio_num_heads;
    size_t audio_num_layers;
    size_t audio_head_dim;
    size_t text_num_key_value_heads;
    size_t text_num_layers;
    size_t text_head_dim;
    size_t audio_length_per_token;
    size_t left_pad_tokens;
    size_t num_delay_tokens;
    size_t num_special_tokens;
    int64_t bos_token_id;
    int64_t eos_token_id;
    int64_t streaming_pad_token_id;
    size_t first_chunk_samples;
    size_t subsequent_chunk_samples;
    size_t left_pad_samples;
    size_t feature_size;
    size_t sampling_rate;
    size_t n_fft;
    size_t hop_length;
    float global_log_mel_max;
};

std::string base64_decode(const std::string& encoded) {
    static constexpr char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string decoded;
    decoded.reserve((encoded.size() * 3) / 4);
    uint32_t buffer = 0;
    size_t bits = 0;
    for (const char character : encoded) {
        if (character == '=')
            break;
        const char* found = std::find(std::begin(alphabet), std::end(alphabet) - 1, character);
        OPENVINO_ASSERT(found != std::end(alphabet) - 1, "Invalid Base64 token bytes in tekken.json.");
        buffer = (buffer << TEKKEN_BASE64_BITS) | static_cast<uint32_t>(found - std::begin(alphabet));
        bits += TEKKEN_BASE64_BITS;
        if (bits >= 8) {
            bits -= 8;
            decoded.push_back(static_cast<char>((buffer >> bits) & 0xff));
        }
    }
    return decoded;
}

nlohmann::json read_json(const std::filesystem::path& path) {
    std::ifstream input(path);
    OPENVINO_ASSERT(input.is_open(), "Failed to open ", path);
    return nlohmann::json::parse(input);
}

VoxtralConfig load_config(const std::filesystem::path& models_path) {
    const nlohmann::json config = read_json(models_path / "config.json");
    const nlohmann::json tekken = read_json(models_path / "tekken.json");
    const nlohmann::json preprocessor = read_json(models_path / "preprocessor_config.json");
    const auto& audio = config.at("audio_config");
    const auto& text = config.at("text_config");
    const auto& tekken_audio = tekken.at("audio");
    const auto& encoding = tekken_audio.at("audio_encoding_config");

    const size_t audio_length_per_token = config.value("audio_length_per_tok", AUDIO_FRAMES_PER_TEXT_TOKEN);
    const size_t num_delay_tokens = config.value("default_num_delay_tokens", size_t{6});
    const size_t left_pad_tokens = tekken_audio.value("streaming_n_left_pad_tokens", size_t{32});
    const size_t n_fft = preprocessor.at("n_fft").get<size_t>();
    const size_t hop_length = preprocessor.at("hop_length").get<size_t>();
    const size_t first_audio_frames = (num_delay_tokens + 1) * audio_length_per_token;

    int64_t streaming_pad_token_id = -1;
    for (const auto& special_token : tekken.at("special_tokens")) {
        if (special_token.at("token_str") == "[STREAMING_PAD]") {
            streaming_pad_token_id = special_token.at("rank").get<int64_t>();
            break;
        }
    }
    OPENVINO_ASSERT(streaming_pad_token_id >= 0, "[STREAMING_PAD] token is missing from tekken.json.");

    return VoxtralConfig{
        audio.at("hidden_size").get<size_t>(),
        audio.at("num_attention_heads").get<size_t>(),
        audio.at("num_hidden_layers").get<size_t>(),
        audio.at("head_dim").get<size_t>(),
        text.at("num_key_value_heads").get<size_t>(),
        text.at("num_hidden_layers").get<size_t>(),
        text.at("head_dim").get<size_t>(),
        audio_length_per_token,
        left_pad_tokens,
        num_delay_tokens,
        tekken.at("config").value("default_num_special_tokens", size_t{1000}),
        text.at("bos_token_id").get<int64_t>(),
        text.at("eos_token_id").get<int64_t>(),
        streaming_pad_token_id,
        first_audio_frames * hop_length - hop_length + n_fft / 2,
        audio_length_per_token * hop_length + n_fft,
        left_pad_tokens * audio_length_per_token * hop_length,
        preprocessor.at("feature_size").get<size_t>(),
        preprocessor.at("sampling_rate").get<size_t>(),
        n_fft,
        hop_length,
        preprocessor.at("global_log_mel_max").get<float>(),
    };
}

std::vector<std::string> load_tekken_pieces(const std::filesystem::path& tekken_path) {
    const nlohmann::json tekken = read_json(tekken_path);
    const auto& vocab = tekken.at("vocab");
    std::vector<std::string> pieces(vocab.size());
    for (const auto& entry : vocab) {
        const size_t rank = entry.at("rank").get<size_t>();
        if (rank < pieces.size())
            pieces[rank] = base64_decode(entry.at("token_bytes").get<std::string>());
    }
    return pieces;
}

ov::Tensor make_float_tensor(const ov::Shape& shape) {
    ov::Tensor tensor(ov::element::f32, shape);
    std::fill_n(tensor.data<float>(), tensor.get_size(), 0.0f);
    return tensor;
}

ov::Tensor make_i64_tensor(const std::vector<int64_t>& values) {
    ov::Tensor tensor(ov::element::i64, {1, values.size()});
    std::copy(values.begin(), values.end(), tensor.data<int64_t>());
    return tensor;
}

ov::Tensor copy_tensor(const ov::Tensor& source) {
    ov::Tensor destination(source.get_element_type(), source.get_shape());
    source.copy_to(destination);
    return destination;
}

size_t complete_utf8_prefix(const std::string& text) {
    size_t offset = 0;
    size_t complete = 0;
    while (offset < text.size()) {
        const auto lead = static_cast<uint8_t>(text[offset]);
        size_t length = 1;
        if ((lead & 0xe0) == 0xc0)
            length = 2;
        else if ((lead & 0xf0) == 0xe0)
            length = 3;
        else if ((lead & 0xf8) == 0xf0)
            length = 4;
        if (offset + length > text.size())
            break;
        bool valid = true;
        for (size_t continuation = 1; continuation < length; ++continuation) {
            if ((static_cast<uint8_t>(text[offset + continuation]) & 0xc0) != 0x80) {
                valid = false;
                break;
            }
        }
        offset += valid ? length : 1;
        complete = offset;
    }
    return complete;
}

}  // namespace

struct VoxtralRealtimeRuntime {
    VoxtralConfig config;
    WhisperFeatureExtractor feature_extractor;
    std::vector<std::string> tekken_pieces;
    ov::CompiledModel audio_encoder;
    ov::CompiledModel text_decoder;

    VoxtralRealtimeRuntime(const std::filesystem::path& models_path,
                           const std::string& device,
                           const ov::AnyMap& properties)
        : config{load_config(models_path)},
          feature_extractor{config.feature_size,
                            config.sampling_rate,
                            config.n_fft,
                            config.hop_length,
                            config.global_log_mel_max},
          tekken_pieces{load_tekken_pieces(models_path / "tekken.json")} {
        ov::AnyMap device_properties = properties;
        erase_allowed_asr_ctor_properties(device_properties);
        ov::Core core = utils::singleton_core();
        audio_encoder = core.compile_model(models_path / "openvino_audio_encoder.xml", device, device_properties);
        text_decoder = core.compile_model(models_path / "openvino_text_decoder.xml", device, device_properties);
    }

    std::string decode_token(const int64_t token_id) const {
        if (token_id < static_cast<int64_t>(config.num_special_tokens))
            return {};
        const size_t rank = static_cast<size_t>(token_id) - config.num_special_tokens;
        OPENVINO_ASSERT(rank < tekken_pieces.size(), "Voxtral output token is outside the Tekken vocabulary.");
        return tekken_pieces[rank];
    }
};

namespace {

class VoxtralASRStream final : public ASRStreamImplBase {
public:
    VoxtralASRStream(const std::shared_ptr<VoxtralRealtimeRuntime>& runtime,
                     const std::optional<ASRGenerationConfig>& generation_config)
        : m_runtime{runtime},
          m_audio_request{runtime->audio_encoder.create_infer_request()},
          m_text_request{runtime->text_decoder.create_infer_request()} {
        (void)generation_config;
        const auto& config = m_runtime->config;
        m_conv1_cache = make_float_tensor({1, config.feature_size, 2});
        m_conv2_cache = make_float_tensor({1, config.audio_hidden_size, 1});
        m_audio_key_values.reserve(config.audio_num_layers);
        for (size_t layer = 0; layer < config.audio_num_layers; ++layer) {
            m_audio_key_values.emplace_back(
                make_float_tensor({1, config.audio_num_heads, 0, config.audio_head_dim}),
                make_float_tensor({1, config.audio_num_heads, 0, config.audio_head_dim}));
        }
        m_text_key_values.reserve(config.text_num_layers);
        for (size_t layer = 0; layer < config.text_num_layers; ++layer) {
            m_text_key_values.emplace_back(
                make_float_tensor({1, config.text_num_key_value_heads, 0, config.text_head_dim}),
                make_float_tensor({1, config.text_num_key_value_heads, 0, config.text_head_dim}));
        }
    }

    std::string push_audio(const std::vector<float>& audio_chunk) override {
        OPENVINO_ASSERT(!m_finished, "Cannot push audio after ASRStream::finish().");
        m_audio_buffer.insert(m_audio_buffer.end(), audio_chunk.begin(), audio_chunk.end());
        std::string emitted;
        if (!m_prefill_done && m_audio_buffer.size() >= m_runtime->config.first_chunk_samples) {
            std::vector<float> first_chunk(m_audio_buffer.begin(),
                                           m_audio_buffer.begin() + m_runtime->config.first_chunk_samples);
            m_audio_buffer.erase(m_audio_buffer.begin(), m_audio_buffer.begin() + m_runtime->config.first_chunk_samples);
            emitted += process_chunk(first_chunk, true);
        }
        while (m_prefill_done && m_audio_buffer.size() >= m_runtime->config.subsequent_chunk_samples) {
            std::vector<float> chunk(m_audio_buffer.begin(),
                                     m_audio_buffer.begin() + m_runtime->config.subsequent_chunk_samples);
            m_audio_buffer.erase(m_audio_buffer.begin(), m_audio_buffer.begin() + m_runtime->config.subsequent_chunk_samples);
            emitted += process_chunk(chunk, false);
        }
        return emitted;
    }

    ASRDecodedResults finish() override {
        OPENVINO_ASSERT(!m_finished, "ASRStream::finish() may only be called once.");
        if (!m_prefill_done && !m_audio_buffer.empty()) {
            m_audio_buffer.resize(m_runtime->config.first_chunk_samples, 0.0f);
            process_chunk(m_audio_buffer, true);
            m_audio_buffer.clear();
        }
        if (m_prefill_done && !m_audio_buffer.empty()) {
            m_audio_buffer.resize(m_runtime->config.subsequent_chunk_samples, 0.0f);
            process_chunk(m_audio_buffer, false);
            m_audio_buffer.clear();
        }
        m_transcript += m_pending_utf8;
        m_pending_utf8.clear();
        m_finished = true;

        ASRDecodedResults results;
        results.texts.push_back(m_transcript);
        results.scores.push_back(0.0f);
        results.languages.emplace_back();
        return results;
    }

private:
    ov::Tensor run_audio_encoder(const std::vector<float>& samples, const bool is_first_chunk) {
        std::vector<float> waveform;
        if (is_first_chunk)
            waveform.resize(m_runtime->config.left_pad_samples, 0.0f);
        waveform.insert(waveform.end(), samples.begin(), samples.end());

        const WhisperFeatures features = m_runtime->feature_extractor.extract(waveform, false, is_first_chunk);
        ov::Tensor input_features(ov::element::f32, {1, features.feature_size, features.n_frames});
        std::copy(features.data.begin(), features.data.end(), input_features.data<float>());
        m_audio_request.set_tensor("input_features", input_features);
        m_audio_request.set_tensor("conv1_cache", m_conv1_cache);
        m_audio_request.set_tensor("conv2_cache", m_conv2_cache);
        for (size_t layer = 0; layer < m_audio_key_values.size(); ++layer) {
            m_audio_request.set_tensor("past_key_values." + std::to_string(layer) + ".key", m_audio_key_values[layer].first);
            m_audio_request.set_tensor("past_key_values." + std::to_string(layer) + ".value", m_audio_key_values[layer].second);
        }
        m_audio_request.infer();

        m_conv1_cache = copy_tensor(m_audio_request.get_tensor("conv1_cache_1"));
        m_conv2_cache = copy_tensor(m_audio_request.get_tensor("conv2_cache_1"));
        for (size_t layer = 0; layer < m_audio_key_values.size(); ++layer) {
            m_audio_key_values[layer].first = copy_tensor(m_audio_request.get_tensor("present." + std::to_string(layer) + ".key"));
            m_audio_key_values[layer].second = copy_tensor(m_audio_request.get_tensor("present." + std::to_string(layer) + ".value"));
        }
        return m_audio_request.get_tensor("audio_embeddings");
    }

    std::vector<int64_t> build_initial_prompt() const {
        const auto& config = m_runtime->config;
        std::vector<int64_t> input_ids(1 + config.left_pad_tokens + config.num_delay_tokens,
                                       config.streaming_pad_token_id);
        input_ids.front() = config.bos_token_id;
        return input_ids;
    }

    std::string process_chunk(const std::vector<float>& samples, const bool is_first_chunk) {
        ov::Tensor audio_embeddings = run_audio_encoder(samples, is_first_chunk);
        const size_t sequence_length = audio_embeddings.get_shape()[1];
        const size_t past_length = is_first_chunk ? 0 : m_text_key_values.front().first.get_shape()[2];
        const std::vector<int64_t> input_ids = is_first_chunk ? build_initial_prompt() : std::vector<int64_t>{m_last_token};
        OPENVINO_ASSERT(input_ids.size() == sequence_length,
                        "Voxtral audio/text alignment mismatch: ",
                        sequence_length,
                        " audio embeddings for ",
                        input_ids.size(),
                        " text positions.");

        ov::Tensor input_ids_tensor = make_i64_tensor(input_ids);
        ov::Tensor attention_mask(ov::element::i64, {1, past_length + sequence_length});
        std::fill_n(attention_mask.data<int64_t>(), attention_mask.get_size(), int64_t{1});
        std::vector<int64_t> positions(sequence_length);
        for (size_t index = 0; index < sequence_length; ++index)
            positions[index] = static_cast<int64_t>(past_length + index);
        ov::Tensor position_ids_tensor = make_i64_tensor(positions);
        ov::Tensor num_delay_tokens(ov::element::f32, {1});
        num_delay_tokens.data<float>()[0] = static_cast<float>(m_runtime->config.num_delay_tokens);

        m_text_request.set_tensor("input_ids", input_ids_tensor);
        m_text_request.set_tensor("audio_embeddings", audio_embeddings);
        m_text_request.set_tensor("attention_mask", attention_mask);
        m_text_request.set_tensor("position_ids", position_ids_tensor);
        m_text_request.set_tensor("num_delay_tokens", num_delay_tokens);
        for (size_t layer = 0; layer < m_text_key_values.size(); ++layer) {
            m_text_request.set_tensor("past_key_values." + std::to_string(layer) + ".key", m_text_key_values[layer].first);
            m_text_request.set_tensor("past_key_values." + std::to_string(layer) + ".value", m_text_key_values[layer].second);
        }
        m_text_request.infer();

        const ov::Tensor logits = m_text_request.get_tensor("logits");
        const size_t vocab_size = logits.get_shape().back();
        const size_t last_position = logits.get_shape()[1] - 1;
        const float* logits_data = logits.data<const float>() + last_position * vocab_size;
        m_last_token = static_cast<int64_t>(std::max_element(logits_data, logits_data + vocab_size) - logits_data);
        for (size_t layer = 0; layer < m_text_key_values.size(); ++layer) {
            m_text_key_values[layer].first = copy_tensor(m_text_request.get_tensor("present." + std::to_string(layer) + ".key"));
            m_text_key_values[layer].second = copy_tensor(m_text_request.get_tensor("present." + std::to_string(layer) + ".value"));
        }
        if (is_first_chunk)
            m_prefill_done = true;
        if (m_last_token == m_runtime->config.eos_token_id)
            return {};
        return append_token_bytes(m_runtime->decode_token(m_last_token));
    }

    std::string append_token_bytes(const std::string& token_bytes) {
        m_pending_utf8 += token_bytes;
        const size_t complete = complete_utf8_prefix(m_pending_utf8);
        std::string emitted = m_pending_utf8.substr(0, complete);
        m_pending_utf8.erase(0, complete);
        m_transcript += emitted;
        return emitted;
    }

    std::shared_ptr<VoxtralRealtimeRuntime> m_runtime;
    ov::InferRequest m_audio_request;
    ov::InferRequest m_text_request;
    ov::Tensor m_conv1_cache;
    ov::Tensor m_conv2_cache;
    std::vector<std::pair<ov::Tensor, ov::Tensor>> m_audio_key_values;
    std::vector<std::pair<ov::Tensor, ov::Tensor>> m_text_key_values;
    std::vector<float> m_audio_buffer;
    std::string m_pending_utf8;
    std::string m_transcript;
    int64_t m_last_token = -1;
    bool m_prefill_done = false;
    bool m_finished = false;
};

}  // namespace

VoxtralASR::VoxtralASR(const std::filesystem::path& models_path,
                       const std::string& device,
                       const ov::AnyMap& properties)
    : ASRPipelineImplBase(models_path),
      m_runtime{std::make_shared<VoxtralRealtimeRuntime>(models_path, device, properties)} {}

ASRDecodedResults VoxtralASR::generate(const AudioInputs& audio_inputs,
                                       const std::optional<ASRGenerationConfig>& generation_config,
                                       const std::shared_ptr<StreamerBase> streamer) {
    OPENVINO_ASSERT(!streamer,
                    "Voxtral uses input streaming; call create_stream() and collect text from push_audio().");
    auto stream = create_stream(generation_config);
    stream->push_audio(std::get<std::vector<float>>(audio_inputs));
    return stream->finish();
}

std::unique_ptr<ASRStreamImplBase> VoxtralASR::create_stream(
    const std::optional<ASRGenerationConfig>& generation_config) {
    return std::make_unique<VoxtralASRStream>(m_runtime, generation_config);
}

}  // namespace ov::genai