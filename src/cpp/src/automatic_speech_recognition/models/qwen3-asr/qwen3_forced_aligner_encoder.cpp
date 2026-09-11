// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "qwen3_forced_aligner_encoder.hpp"

#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>

#include "json_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/runtime/core.hpp"
#include "utils.hpp"

namespace ov::genai {

namespace {

size_t read_n_window(const std::filesystem::path& models_path) {
    size_t n_window = 100;
    const std::filesystem::path config_path = models_path / "config.json";
    if (std::filesystem::exists(config_path)) {
        std::ifstream stream(config_path);
        const nlohmann::json config = nlohmann::json::parse(stream);
        if (config.contains("thinker_config") && config.at("thinker_config").contains("audio_config") &&
            config.at("thinker_config").at("audio_config").contains("n_window")) {
            const nlohmann::json& n_window_json = config.at("thinker_config").at("audio_config").at("n_window");
            OPENVINO_ASSERT(n_window_json.is_number_integer() && n_window_json.get<int64_t>() > 0,
                            "Forced-aligner thinker_config.audio_config.n_window must be a positive integer. Got: ",
                            n_window_json.dump(),
                            ".");
            n_window = static_cast<size_t>(n_window_json.get<int64_t>());
        }
    }
    return n_window;
}

}  // namespace

Qwen3ForcedAlignerEncoder::Qwen3ForcedAlignerEncoder(const std::filesystem::path& models_path,
                                                     const std::string& device,
                                                     const ov::AnyMap& properties)
    : m_encoder_chunk_frames{read_n_window(models_path) * 2} {
    ov::Core core = utils::singleton_core();
    ov::CompiledModel compiled_model =
        core.compile_model(models_path / "openvino_encoder_model.xml", device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "qwen3 forced-aligner encoder model");
    m_request = compiled_model.create_infer_request();
}

ov::Tensor Qwen3ForcedAlignerEncoder::encode(const WhisperFeatures& features) {
    const size_t remainder_frames = features.n_frames % m_encoder_chunk_frames;

    ov::Tensor input_tensor = chunk_mel_features(features);
    m_request.set_tensor("input_features", input_tensor);

    m_request.infer();

    const ov::Tensor chunked_output = m_request.get_tensor("last_hidden_state");
    ov::Tensor output = merge_chunked_encoder_output(chunked_output, remainder_frames);

    m_request.set_tensor("input_features", ov::Tensor(ov::element::f32, {0, 0, 0}));

    return output;
}

ov::Tensor Qwen3ForcedAlignerEncoder::chunk_mel_features(const WhisperFeatures& features) const {
    const size_t n_features = features.feature_size;
    const size_t n_frames = features.n_frames;
    OPENVINO_ASSERT(n_frames > 0, "Forced-aligner encoder input features must contain at least one frame.");

    const size_t num_full_chunks = n_frames / m_encoder_chunk_frames;
    const size_t remainder_frames = n_frames % m_encoder_chunk_frames;
    const size_t num_chunks = num_full_chunks + (remainder_frames > 0 ? 1 : 0);

    ov::Tensor input_tensor(ov::element::f32, {num_chunks, n_features, m_encoder_chunk_frames});
    float* dst = input_tensor.data<float>();

    for (size_t chunk_index = 0; chunk_index < num_full_chunks; ++chunk_index) {
        const size_t frame_offset = chunk_index * m_encoder_chunk_frames;
        for (size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            const float* src = features.data.data() + feature_index * n_frames + frame_offset;
            float* chunk_dst = dst + (chunk_index * n_features + feature_index) * m_encoder_chunk_frames;
            std::memcpy(chunk_dst, src, m_encoder_chunk_frames * sizeof(float));
        }
    }

    if (remainder_frames > 0) {
        const size_t chunk_index = num_full_chunks;
        const size_t frame_offset = chunk_index * m_encoder_chunk_frames;
        const size_t padding_frames = m_encoder_chunk_frames - remainder_frames;
        for (size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            const float* src = features.data.data() + feature_index * n_frames + frame_offset;
            float* chunk_dst = dst + (chunk_index * n_features + feature_index) * m_encoder_chunk_frames;
            std::memcpy(chunk_dst, src, remainder_frames * sizeof(float));
            std::memset(chunk_dst + remainder_frames, 0, padding_frames * sizeof(float));
        }
    }

    return input_tensor;
}

size_t Qwen3ForcedAlignerEncoder::get_remainder_output_tokens(size_t remainder_frames,
                                                              size_t tokens_per_full_chunk) const {
    // The forced-aligner encoder downsamples the time dimension through three Conv2d layers with
    // kernel=3, stride=2, and padding=1, so each layer produces ceil(input_size / 2) positions.
    size_t output_tokens = remainder_frames;
    for (size_t i = 0; i < 3; ++i) {
        output_tokens = (output_tokens + 1) / 2;
    }

    OPENVINO_ASSERT(output_tokens <= tokens_per_full_chunk,
                    "Forced-aligner encoder tail chunk yields more tokens than a full chunk: ",
                    output_tokens,
                    " > ",
                    tokens_per_full_chunk,
                    " (remainder_frames=",
                    remainder_frames,
                    ", encoder_chunk_frames=",
                    m_encoder_chunk_frames,
                    "). Check that the model config matches the exported encoder geometry.");

    return output_tokens;
}

ov::Tensor Qwen3ForcedAlignerEncoder::merge_chunked_encoder_output(const ov::Tensor& chunked_output,
                                                                   size_t remainder_frames) const {
    const ov::Shape chunked_output_shape = chunked_output.get_shape();
    OPENVINO_ASSERT(chunked_output_shape.size() == 3,
                    "Forced-aligner encoder output must have rank 3 [num_chunks, tokens_per_chunk, hidden_size], "
                    "got rank ",
                    chunked_output_shape.size(),
                    ".");

    const size_t batch_size = chunked_output_shape[0];
    OPENVINO_ASSERT(batch_size > 0, "Forced-aligner encoder output must contain at least one chunk.");
    const size_t tokens_per_full_chunk = chunked_output_shape[1];
    const size_t hidden_dim = chunked_output_shape[2];
    const size_t num_full_chunks = (remainder_frames > 0) ? batch_size - 1 : batch_size;

    const size_t last_chunk_tokens = (remainder_frames > 0)
                                         ? get_remainder_output_tokens(remainder_frames, tokens_per_full_chunk)
                                         : tokens_per_full_chunk;
    const size_t total_tokens =
        num_full_chunks * tokens_per_full_chunk + (remainder_frames > 0 ? last_chunk_tokens : 0);

    ov::Tensor output(ov::element::f32, {1, total_tokens, hidden_dim});
    float* out_dst = output.data<float>();
    const float* chunk_src = chunked_output.data<const float>();
    const size_t chunk_stride = tokens_per_full_chunk * hidden_dim;

    const size_t full_chunks_size = num_full_chunks * chunk_stride;
    if (full_chunks_size > 0) {
        std::memcpy(out_dst, chunk_src, full_chunks_size * sizeof(float));
        out_dst += full_chunks_size;
    }

    if (remainder_frames > 0) {
        const float* last_src = chunk_src + num_full_chunks * chunk_stride;
        std::memcpy(out_dst, last_src, last_chunk_tokens * hidden_dim * sizeof(float));
    }

    return output;
}

}  // namespace ov::genai
