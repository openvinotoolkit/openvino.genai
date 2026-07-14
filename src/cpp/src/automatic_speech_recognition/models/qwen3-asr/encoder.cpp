// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "encoder.hpp"

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <optional>
#include <string_view>

#include "openvino/core/model.hpp"
#include "openvino/runtime/core.hpp"
#include "utils.hpp"

namespace ov::genai {

namespace {

bool startsWith(std::string_view value, std::string_view prefix) {
    return value.size() >= prefix.size() && value.compare(0, prefix.size(), prefix) == 0;
}

bool isLanguageModelOnlyProperty(std::string_view name) {
    if (startsWith(name, "++")) {
        name.remove_prefix(2);
    }

    if (name == "MAX_PROMPT_LEN" || name == "MIN_RESPONSE_LEN" || name == "BLOB_PATH" || name == "EXPORT_BLOB") {
        return true;
    }

    return name == "NPU_USE_NPUW" || startsWith(name, "NPUW_") || startsWith(name, "PREFILL_") ||
           startsWith(name, "GENERATE_") || startsWith(name, "SHARED_");
}

ov::AnyMap getAudioEncoderProperties(const ov::AnyMap& properties) {
    ov::AnyMap encoder_properties;
    for (const auto& [name, value] : properties) {
        if (!isLanguageModelOnlyProperty(name)) {
            encoder_properties.emplace(name, value);
        }
    }
    return encoder_properties;
}

std::optional<std::filesystem::path> resolveNpuDumpDir(const std::filesystem::path& models_path) {
    const char* flag = std::getenv("OV_GENAI_QWEN3ASR_NPU_DUMP");
    if (flag == nullptr) {
        return std::nullopt;
    }
    const std::string_view value{flag};
    if (value.empty() || value == "0" || value == "false" || value == "FALSE") {
        return std::nullopt;
    }
    auto dump_dir = models_path / "npu_static_dump";
    std::filesystem::create_directories(dump_dir);
    return dump_dir;
}

}  // namespace

Qwen3ASREncoder::Qwen3ASREncoder(const std::filesystem::path& models_path,
                                 const std::string& device,
                                 const ov::AnyMap& properties,
                                 size_t feature_size)
    : m_model_config{models_path / "config.json"},
      m_feature_size{feature_size} {
    ov::Core core = utils::singleton_core();
    ov::CompiledModel compiled_model;
    if (device == "NPU") {
        m_is_npu = true;
        OPENVINO_ASSERT(m_feature_size > 0, "Qwen3-ASR NPU encoder requires a positive feature size");
        OPENVINO_ASSERT(m_encoder_chunk_frames > 0, "Qwen3-ASR NPU encoder requires a positive chunk size");

        auto encoder_model = core.read_model(models_path / "openvino_encoder_model.xml");
        OPENVINO_ASSERT(encoder_model->inputs().size() == 1, "Qwen3-ASR NPU encoder must have exactly one input");
        OPENVINO_ASSERT(encoder_model->outputs().size() == 1, "Qwen3-ASR NPU encoder must have exactly one output");
        OPENVINO_ASSERT(encoder_model->input("input_features").get_element_type() == ov::element::f32,
                        "Qwen3-ASR NPU encoder input_features must have f32 element type");
        OPENVINO_ASSERT(encoder_model->output("last_hidden_state").get_element_type() == ov::element::f32,
                        "Qwen3-ASR NPU encoder last_hidden_state must have f32 element type");

        encoder_model->reshape({{"input_features",
                                 ov::PartialShape{1,
                                                  static_cast<int64_t>(m_feature_size),
                                                  static_cast<int64_t>(m_encoder_chunk_frames)}}});
        const ov::Shape static_input_shape = encoder_model->input("input_features").get_shape();
        OPENVINO_ASSERT(static_input_shape == ov::Shape({1, m_feature_size, m_encoder_chunk_frames}),
                        "Unexpected Qwen3-ASR NPU encoder input shape: ",
                        static_input_shape);

        if (const auto dump_dir = resolveNpuDumpDir(models_path)) {
            ov::save_model(encoder_model, (*dump_dir / "npu_encoder_static.xml").string());
        }

        compiled_model = core.compile_model(encoder_model, "NPU", getAudioEncoderProperties(properties));
    } else {
        compiled_model = core.compile_model(models_path / "openvino_encoder_model.xml", device, properties);
    }
    ov::genai::utils::print_compiled_model_properties(compiled_model, "qwen3-asr encoder model");
    m_request = compiled_model.create_infer_request();
}

ov::Tensor Qwen3ASREncoder::encode(const WhisperFeatures& features) {
    const size_t remainder_frames = features.n_frames % m_encoder_chunk_frames;

    if (m_is_npu) {
        OPENVINO_ASSERT(features.feature_size == m_feature_size,
                        "Qwen3-ASR NPU encoder expected ",
                        m_feature_size,
                        " input features, got ",
                        features.feature_size);
        OPENVINO_ASSERT(features.data.size() == features.feature_size * features.n_frames,
                        "Qwen3-ASR NPU encoder feature data size does not match its shape");
    }

    ov::Tensor input_tensor = chunk_mel_features(features);
    if (!m_is_npu) {
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

    OPENVINO_ASSERT(input_tensor.get_element_type() == ov::element::f32,
                    "Qwen3-ASR NPU encoder chunked input must have f32 element type");
    const ov::Shape input_shape = input_tensor.get_shape();
    OPENVINO_ASSERT(input_shape.size() == 3, "Qwen3-ASR NPU encoder chunked input must have rank 3");
    OPENVINO_ASSERT(input_shape[0] > 0, "Qwen3-ASR NPU encoder input must contain at least one chunk");
    OPENVINO_ASSERT(input_shape[1] == m_feature_size && input_shape[2] == m_encoder_chunk_frames,
                    "Unexpected Qwen3-ASR NPU encoder chunked input shape: ",
                    input_shape);

    const size_t num_chunks = input_shape[0];
    const size_t elements_per_chunk = m_feature_size * m_encoder_chunk_frames;
    OPENVINO_ASSERT(input_tensor.get_size() == num_chunks * elements_per_chunk,
                    "Qwen3-ASR NPU encoder chunked input element count does not "
                    "match its shape");

    ov::Tensor aggregate_output;
    size_t tokens_per_chunk = 0;
    size_t hidden_size = 0;
    for (size_t chunk_index = 0; chunk_index < num_chunks; ++chunk_index) {
        ov::Tensor chunk_input(input_tensor,
                               ov::Coordinate{chunk_index, 0, 0},
                               ov::Coordinate{chunk_index + 1, m_feature_size, m_encoder_chunk_frames});
        OPENVINO_ASSERT(chunk_input.get_shape() == ov::Shape({1, m_feature_size, m_encoder_chunk_frames}),
                        "Unexpected Qwen3-ASR NPU encoder chunk input shape: ",
                        chunk_input.get_shape());

        m_request.set_tensor("input_features", chunk_input);
        m_request.infer();

        const ov::Tensor chunk_output = m_request.get_tensor("last_hidden_state");
        OPENVINO_ASSERT(chunk_output.get_element_type() == ov::element::f32,
                        "Qwen3-ASR NPU encoder output must have f32 element type");
        const ov::Shape chunk_output_shape = chunk_output.get_shape();
        OPENVINO_ASSERT(chunk_output_shape.size() == 3 && chunk_output_shape[0] == 1,
                        "Qwen3-ASR NPU encoder output must have shape [1, T, H], got ",
                        chunk_output_shape);
        OPENVINO_ASSERT(chunk_output_shape[1] > 0 && chunk_output_shape[2] > 0,
                        "Qwen3-ASR NPU encoder output dimensions must be positive");

        if (chunk_index == 0) {
            tokens_per_chunk = chunk_output_shape[1];
            hidden_size = chunk_output_shape[2];
            aggregate_output = ov::Tensor(ov::element::f32, {num_chunks, tokens_per_chunk, hidden_size});
        } else {
            OPENVINO_ASSERT(chunk_output_shape[1] == tokens_per_chunk && chunk_output_shape[2] == hidden_size,
                            "Qwen3-ASR NPU encoder output shape changed between chunks");
        }

        const size_t output_elements = tokens_per_chunk * hidden_size;
        OPENVINO_ASSERT(chunk_output.get_size() == output_elements,
                        "Qwen3-ASR NPU encoder output element count does not match its shape");
        std::memcpy(aggregate_output.data<float>() + chunk_index * output_elements,
                    chunk_output.data<const float>(),
                    chunk_output.get_byte_size());
    }

    OPENVINO_ASSERT(aggregate_output.get_size() == num_chunks * tokens_per_chunk * hidden_size,
                    "Qwen3-ASR NPU encoder aggregate output element count does "
                    "not match its shape");
    return merge_chunked_encoder_output(aggregate_output, remainder_frames);
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
