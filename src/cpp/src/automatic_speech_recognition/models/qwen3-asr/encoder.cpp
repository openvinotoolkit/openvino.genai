// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "encoder.hpp"

#include <cstring>

#include "openvino/runtime/core.hpp"
#include "utils.hpp"

namespace ov::genai {

Qwen3ASREncoder::Qwen3ASREncoder(const std::filesystem::path& models_path,
                                 const std::string& device,
                                 const ov::AnyMap& properties)
    : m_model_config{models_path / "config.json"} {
    ov::Core core = utils::singleton_core();
    ov::CompiledModel compiled_model =
        core.compile_model(models_path / "openvino_encoder_model.xml", device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "qwen3-asr encoder model");
    m_request = compiled_model.create_infer_request();
}

ov::Tensor Qwen3ASREncoder::encode(const WhisperFeatures& features) {
    const size_t remainder_frames = features.n_frames % m_encoder_chunk_frames;

    ov::Tensor input_tensor = chunk_mel_features(features);
    m_request.set_tensor("input_features", input_tensor);

    m_request.infer();

    // whisper implementation has remote_tensor optimization when last_hidden_state set to decoder without copy
    // qwen3-asr encoder chunking inference requires merging after inference
    // access to last_hidden_state tensor data -> data copy to host memory -> cannot use remote_tensor optimization
    // consider pre-post processing for chunked inference to avoid data copy and remote_tensor optimization
    const ov::Tensor chunked_output = m_request.get_tensor("last_hidden_state");
    ov::Tensor output = merge_chunked_encoder_output(chunked_output, remainder_frames);

    m_request.set_tensor("input_features", ov::Tensor(ov::element::f32, {0, 0, 0}));

    return output;
}

ov::Tensor Qwen3ASREncoder::chunk_mel_features(const WhisperFeatures& features) {
    const size_t n_features = features.feature_size;
    const size_t n_frames = features.n_frames;
    OPENVINO_ASSERT(n_frames > 0, "Qwen3-ASR encoder input features must contain at least one frame.");

    const size_t num_full_chunks = n_frames / m_encoder_chunk_frames;
    const size_t remainder_frames = n_frames % m_encoder_chunk_frames;
    const size_t num_chunks = num_full_chunks + (remainder_frames > 0 ? 1 : 0);

    ov::Tensor input_tensor(ov::element::f32, {num_chunks, n_features, m_encoder_chunk_frames});
    float* dst = input_tensor.data<float>();

    // Source layout: features.data is [n_features, n_frames] (row = one mel band, contiguous in time).
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

size_t Qwen3ASREncoder::get_remainder_output_tokens(const size_t remainder_frames, const size_t tokens_per_full_chunk) {
    // Integer ceil of: remainder_frames * tokens_per_full_chunk / m_encoder_chunk_frames.
    return (remainder_frames * tokens_per_full_chunk + m_encoder_chunk_frames - 1) / m_encoder_chunk_frames;
}

ov::Tensor Qwen3ASREncoder::merge_chunked_encoder_output(const ov::Tensor& chunked_output, size_t remainder_frames) {
    const ov::Shape chunked_output_shape = chunked_output.get_shape();

    const size_t batch_size = chunked_output_shape[0];
    OPENVINO_ASSERT(batch_size > 0, "Qwen3-ASR encoder output must contain at least one chunk.");
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
