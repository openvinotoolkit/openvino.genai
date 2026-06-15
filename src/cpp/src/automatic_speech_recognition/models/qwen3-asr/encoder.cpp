// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "encoder.hpp"

#include <cstring>

#include "openvino/runtime/core.hpp"
#include "utils.hpp"

namespace {
ov::InferRequest init_model(ov::CompiledModel& compiled) {
    ov::InferRequest request = compiled.create_infer_request();

    try {
        ov::RemoteContext context = compiled.get_context();
        ov::Shape output_shape = request.get_output_tensor().get_shape();
        ov::RemoteTensor remote = context.create_tensor(ov::element::f32, output_shape);
        request.set_tensor("last_hidden_state", remote);
        return request;
    } catch (const ov::Exception&) {
        return request;
    }
}
}  // namespace

namespace ov::genai {

Qwen3ASREncoder::Qwen3ASREncoder(const std::filesystem::path& models_path,
                                 const std::string& device,
                                 const ov::AnyMap& properties) {
    ov::Core core = utils::singleton_core();
    ov::CompiledModel compiled_model =
        core.compile_model(models_path / "openvino_encoder_model.xml", device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "qwen3-asr encoder model");
    m_request = init_model(compiled_model);
}

ov::Tensor Qwen3ASREncoder::encode(const WhisperFeatures& features) {
    const size_t remainder_frames = features.n_frames % ENCODER_CHUNK_FRAMES;

    ov::Tensor input_tensor = chunk_mel_features(features);
    m_request.set_tensor("input_features", input_tensor);

    m_request.infer();

    const ov::Tensor chunked_output = m_request.get_tensor("last_hidden_state");
    ov::Tensor output = merge_chunked_encoder_output(chunked_output, remainder_frames);

    m_request.set_tensor("input_features", ov::Tensor(ov::element::f32, {0, 0, 0}));

    return output;
}

ov::Tensor Qwen3ASREncoder::chunk_mel_features(const WhisperFeatures& features) {
    const size_t n_features = features.feature_size;
    const size_t n_frames = features.n_frames;
    OPENVINO_ASSERT(n_frames > 0, "Qwen3-ASR encoder input features must contain at least one frame.");

    const size_t num_full_chunks = n_frames / ENCODER_CHUNK_FRAMES;
    const size_t remainder_frames = n_frames % ENCODER_CHUNK_FRAMES;
    const size_t num_chunks = num_full_chunks + (remainder_frames > 0 ? 1 : 0);

    ov::Tensor input_tensor(ov::element::f32, {num_chunks, n_features, ENCODER_CHUNK_FRAMES});
    float* dst = input_tensor.data<float>();

    // Source layout: features.data is [n_features, n_frames] (row = one mel band, contiguous in time).
    for (size_t chunk_index = 0; chunk_index < num_full_chunks; ++chunk_index) {
        const size_t frame_offset = chunk_index * ENCODER_CHUNK_FRAMES;
        for (size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            const float* src = features.data.data() + feature_index * n_frames + frame_offset;
            float* chunk_dst = dst + (chunk_index * n_features + feature_index) * ENCODER_CHUNK_FRAMES;
            std::memcpy(chunk_dst, src, ENCODER_CHUNK_FRAMES * sizeof(float));
        }
    }

    if (remainder_frames > 0) {
        const size_t chunk_index = num_full_chunks;
        const size_t frame_offset = chunk_index * ENCODER_CHUNK_FRAMES;
        const size_t padding_frames = ENCODER_CHUNK_FRAMES - remainder_frames;
        for (size_t feature_index = 0; feature_index < n_features; ++feature_index) {
            const float* src = features.data.data() + feature_index * n_frames + frame_offset;
            float* chunk_dst = dst + (chunk_index * n_features + feature_index) * ENCODER_CHUNK_FRAMES;
            std::memcpy(chunk_dst, src, remainder_frames * sizeof(float));
            std::memset(chunk_dst + remainder_frames, 0, padding_frames * sizeof(float));
        }
    }

    return input_tensor;
}

size_t Qwen3ASREncoder::infer_output_frames(const size_t input_frames, const size_t full_chunk_output_frames) {
    return (input_frames * full_chunk_output_frames + ENCODER_CHUNK_FRAMES - 1) / ENCODER_CHUNK_FRAMES;
}

ov::Tensor Qwen3ASREncoder::merge_chunked_encoder_output(const ov::Tensor& chunked_output, size_t remainder_frames) {
    const ov::Shape chunked_output_shape = chunked_output.get_shape();

    const size_t batch_size = chunked_output_shape[0];
    OPENVINO_ASSERT(batch_size > 0, "Qwen3-ASR encoder output must contain at least one chunk.");
    const size_t tokens_per_full_chunk = chunked_output_shape[1];
    const size_t hidden_dim = chunked_output_shape[2];
    const size_t num_full_chunks = (remainder_frames > 0) ? batch_size - 1 : batch_size;

    const size_t last_chunk_tokens =
        (remainder_frames > 0) ? infer_output_frames(remainder_frames, tokens_per_full_chunk) : tokens_per_full_chunk;
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
