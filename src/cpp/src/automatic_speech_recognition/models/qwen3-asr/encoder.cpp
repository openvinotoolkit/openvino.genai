// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "encoder.hpp"

#include <cstring>
#include <iostream>

#include "openvino/runtime/core.hpp"
#include "utils.hpp"

namespace ov::genai {

namespace {

void reshape_to_static_encoder_for_npu(std::shared_ptr<ov::Model> model,
                                       const size_t feature_size,
                                       const size_t chunk_frames) {
    std::map<std::string, ov::PartialShape> new_shapes;
    for (auto input : model->inputs()) {
        const auto& input_name = input.get_any_name();
        if (input_name.find("input_features") == std::string::npos) {
            continue;
        }
        const auto& partial_shape = input.get_partial_shape();
        OPENVINO_ASSERT(partial_shape.size() >= 3, "Qwen3-ASR encoder input rank must be >= 3");
        ov::PartialShape new_shape = partial_shape;
        new_shape[0] = 1;
        new_shape[1] = feature_size;
        new_shape[2] = chunk_frames;
        new_shapes.emplace(input_name, new_shape);
    }

    OPENVINO_ASSERT(!new_shapes.empty(), "Qwen3-ASR encoder input_features input was not found.");
    model->reshape(new_shapes);
}

}  // namespace

Qwen3ASREncoder::Qwen3ASREncoder(const std::filesystem::path& models_path,
                                 const std::string& device,
                                 const ov::AnyMap& properties)
    : m_model_config{models_path / "config.json"} {
    ov::Core core = utils::singleton_core();
    if (device == "NPU") {
        ov::AnyMap npu_properties = properties;

        const WhisperFeatureExtractor feature_extractor{models_path / "preprocessor_config.json"};
        OPENVINO_ASSERT(feature_extractor.hop_length > 0, "hop_length in preprocessor_config.json must be > 0.");
        OPENVINO_ASSERT(feature_extractor.sampling_rate > 0, "sampling_rate in preprocessor_config.json must be > 0.");

        auto encoder_model =
            core.read_model(models_path / "openvino_encoder_model.xml", {}, std::as_const(npu_properties));
        reshape_to_static_encoder_for_npu(encoder_model, feature_extractor.feature_size, m_encoder_chunk_frames);

        std::cout << "[INFO] Qwen3ASREncoder: NPU static shape configured with batch=1"
                  << ", feature_size=" << feature_extractor.feature_size << ", chunk_frames=" << m_encoder_chunk_frames
                  << std::endl;

        ov::CompiledModel compiled_model = core.compile_model(encoder_model, "NPU", npu_properties);
        ov::genai::utils::print_compiled_model_properties(compiled_model, "qwen3-asr encoder model");
        m_request = compiled_model.create_infer_request();
    } else {
        ov::CompiledModel compiled_model =
            core.compile_model(models_path / "openvino_encoder_model.xml", device, properties);
        ov::genai::utils::print_compiled_model_properties(compiled_model, "qwen3-asr encoder model");
        m_request = compiled_model.create_infer_request();
    }
}

ov::Tensor Qwen3ASREncoder::encode(const WhisperFeatures& features) {
    const size_t remainder_frames = features.n_frames % m_encoder_chunk_frames;

    ov::Tensor input_tensor = chunk_mel_features(features);
    std::cout << "[INFO] Qwen3ASREncoder: input_features shape = " << input_tensor.get_shape() << std::endl;

    ov::Tensor chunked_output;
    const auto execution_devices = m_request.get_compiled_model().get_property(ov::execution_devices);
    const bool is_npu = !execution_devices.empty() && execution_devices[0] == "NPU";
    if (is_npu) {
        const ov::Shape input_shape = input_tensor.get_shape();
        OPENVINO_ASSERT(input_shape.size() == 3, "Unexpected Qwen3-ASR encoder input rank.");

        const size_t num_chunks = input_shape[0];
        const size_t n_features = input_shape[1];
        const size_t chunk_frames = input_shape[2];
        const size_t one_chunk_size = n_features * chunk_frames;

        ov::Tensor one_chunk_input(ov::element::f32, {1, n_features, chunk_frames});
        const float* all_chunks_src = input_tensor.data<const float>();

        size_t tokens_per_chunk = 0;
        size_t hidden_dim = 0;

        for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            std::memcpy(one_chunk_input.data<float>(),
                        all_chunks_src + chunk_idx * one_chunk_size,
                        one_chunk_size * sizeof(float));
            m_request.set_tensor("input_features", one_chunk_input);
            m_request.infer();

            const ov::Tensor one_chunk_output = m_request.get_tensor("last_hidden_state");
            const ov::Shape one_chunk_output_shape = one_chunk_output.get_shape();
            OPENVINO_ASSERT(one_chunk_output_shape.size() == 3 && one_chunk_output_shape[0] == 1,
                            "Unexpected Qwen3-ASR encoder output shape for one chunk.");

            if (chunk_idx == 0) {
                tokens_per_chunk = one_chunk_output_shape[1];
                hidden_dim = one_chunk_output_shape[2];
                chunked_output = ov::Tensor(ov::element::f32, {num_chunks, tokens_per_chunk, hidden_dim});
            } else {
                OPENVINO_ASSERT(
                    one_chunk_output_shape[1] == tokens_per_chunk && one_chunk_output_shape[2] == hidden_dim,
                    "Inconsistent Qwen3-ASR encoder output shape between chunks.");
            }

            const size_t output_stride = tokens_per_chunk * hidden_dim;
            std::memcpy(chunked_output.data<float>() + chunk_idx * output_stride,
                        one_chunk_output.data<const float>(),
                        output_stride * sizeof(float));
        }
    } else {
        m_request.set_tensor("input_features", input_tensor);
        m_request.infer();
        chunked_output = m_request.get_tensor("last_hidden_state");
    }

    std::cout << "[INFO] Qwen3ASREncoder: last_hidden_state shape = " << chunked_output.get_shape() << std::endl;

    // whisper implementation has remote_tensor optimization when last_hidden_state set to decoder without copy
    // qwen3-asr encoder chunking inference requires merging after inference
    // access to last_hidden_state tensor data -> data copy to host memory -> cannot use remote_tensor optimization
    // consider pre-post processing for chunked inference to avoid data copy and remote_tensor optimization
    ov::Tensor output = merge_chunked_encoder_output(chunked_output, remainder_frames);

    const ov::Shape reset_shape = is_npu ? ov::Shape{1, input_tensor.get_shape()[1], input_tensor.get_shape()[2]}
                                         : ov::Shape{0, 0, 0};
    std::cout << "[INFO] Qwen3ASREncoder: resetting input_features with shape = " << reset_shape << std::endl;
    m_request.set_tensor("input_features", ov::Tensor(ov::element::f32, reset_shape));

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
