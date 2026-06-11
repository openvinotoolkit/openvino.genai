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

ov::Tensor Qwen3ASREncoder::encode(std::vector<WhisperFeatures> features) {
    OPENVINO_ASSERT(features.size() == 1, "Batched encoding of multiple audio samples is not supported yet.");

    const WhisperFeatures& feat = features[0];
    const size_t remainder_frames = feat.n_frames % ENCODER_CHUNK_FRAMES;

    ov::Tensor input_tensor = chunk_mel_features(feat);
    m_request.set_tensor("input_features", input_tensor);

    m_request.infer();

    const ov::Tensor chunked_output = m_request.get_tensor("last_hidden_state");
    ov::Tensor output = merge_chunked_encoder_output(chunked_output, remainder_frames);

    m_request.set_tensor("input_features", ov::Tensor(ov::element::f32, {0, 0, 0}));

    return output;
}

ov::Tensor Qwen3ASREncoder::chunk_mel_features(const WhisperFeatures& feat) {
    const size_t n_features = feat.feature_size;
    const size_t total_frames = feat.n_frames;

    const size_t num_full_chunks = total_frames / ENCODER_CHUNK_FRAMES;
    const size_t remainder_frames = total_frames % ENCODER_CHUNK_FRAMES;
    const size_t num_chunks = num_full_chunks + (remainder_frames > 0 ? 1 : 0);

    ov::Tensor input_tensor(ov::element::f32, {num_chunks, n_features, ENCODER_CHUNK_FRAMES});
    float* dst = input_tensor.data<float>();
    std::memset(dst, 0, num_chunks * n_features * ENCODER_CHUNK_FRAMES * sizeof(float));

    // Source layout: feat.data is [n_features, total_frames] (row = one mel band, contiguous in time).
    for (size_t c = 0; c < num_chunks; ++c) {
        const size_t frame_offset = c * ENCODER_CHUNK_FRAMES;
        const size_t frames_to_copy = (c < num_full_chunks) ? ENCODER_CHUNK_FRAMES : remainder_frames;
        for (size_t f = 0; f < n_features; ++f) {
            const float* src = feat.data.data() + f * total_frames + frame_offset;
            float* chunk_dst = dst + (c * n_features + f) * ENCODER_CHUNK_FRAMES;
            std::memcpy(chunk_dst, src, frames_to_copy * sizeof(float));
        }
    }

    return input_tensor;
}

size_t Qwen3ASREncoder::conv_output_frames(size_t input_frames) {
    size_t t = input_frames;
    for (int i = 0; i < 3; ++i) {
        t = (t + 2 * 1 - 3) / 2 + 1;  // Conv2d: (input + 2*pad - kernel) / stride + 1
    }
    return t;
}

ov::Tensor Qwen3ASREncoder::merge_chunked_encoder_output(const ov::Tensor& chunked_output, size_t remainder_frames) {
    const size_t num_chunks = chunked_output.get_shape()[0];
    const size_t tokens_per_full_chunk = conv_output_frames(ENCODER_CHUNK_FRAMES);
    const size_t hidden_dim = chunked_output.get_shape()[2];
    const size_t num_full_chunks = (remainder_frames > 0) ? num_chunks - 1 : num_chunks;

    const size_t last_chunk_tokens =
        (remainder_frames > 0) ? conv_output_frames(remainder_frames) : tokens_per_full_chunk;
    const size_t total_tokens = num_full_chunks * tokens_per_full_chunk + last_chunk_tokens;

    ov::Tensor output(ov::element::f32, {1, total_tokens, hidden_dim});
    float* out_dst = output.data<float>();
    const float* chunk_src = chunked_output.data<const float>();
    const size_t chunk_stride = tokens_per_full_chunk * hidden_dim;

    for (size_t c = 0; c < num_full_chunks; ++c) {
        std::memcpy(out_dst, chunk_src + c * chunk_stride, tokens_per_full_chunk * hidden_dim * sizeof(float));
        out_dst += tokens_per_full_chunk * hidden_dim;
    }
    if (remainder_frames > 0) {
        const float* last_src = chunk_src + num_full_chunks * chunk_stride;
        std::memcpy(out_dst, last_src, last_chunk_tokens * hidden_dim * sizeof(float));
    }

    return output;
}
}  // namespace ov::genai
