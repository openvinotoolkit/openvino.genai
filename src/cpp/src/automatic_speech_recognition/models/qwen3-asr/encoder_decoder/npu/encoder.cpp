// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "encoder.hpp"

#include <cstring>
#include <iostream>
#include <map>
#include <set>

#include "utils.hpp"

namespace {

ov::AnyMap encoder_properties(const ov::AnyMap& properties) {
    static const std::set<std::string> decoder_options{
        "MAX_PROMPT_LEN",
        "MIN_RESPONSE_LEN",
        "PREFILL_CONFIG",
        "PREFILL_HINT",
        "GENERATE_CONFIG",
        "GENERATE_HINT",
        "SHARED_HEAD_CONFIG",
        "BLOB_PATH",
        "EXPORT_BLOB",
    };
    auto result = properties;
    for (auto it = result.begin(); it != result.end();) {
        const auto option = it->first.rfind("++", 0) == 0 ? it->first.substr(2) : it->first;
        if (option == "NPUW_LLM" || option.rfind("NPUW_LLM_", 0) == 0 || option.rfind("NPUW_QWEN3_ASR", 0) == 0 ||
            decoder_options.count(option)) {
            it = result.erase(it);
        } else {
            ++it;
        }
    }
    return result;
}

void reshape_encoder(const std::shared_ptr<ov::Model>& model, size_t feature_size, size_t chunk_frames) {
    std::map<std::string, ov::PartialShape> shapes;
    for (const auto& input : model->inputs()) {
        const auto& name = input.get_any_name();
        if (name.find("input_features") == std::string::npos) {
            continue;
        }
        auto shape = input.get_partial_shape();
        OPENVINO_ASSERT(shape.size() >= 3, "Qwen3-ASR encoder input rank must be >= 3");
        shape[0] = 1;
        shape[1] = feature_size;
        shape[2] = chunk_frames;
        shapes.emplace(name, shape);
    }
    OPENVINO_ASSERT(!shapes.empty(), "Qwen3-ASR encoder input_features input was not found.");
    model->reshape(shapes);
}

}  // namespace

namespace ov::genai {

Qwen3ASREncoderNPU::Qwen3ASREncoderNPU(const std::filesystem::path& models_path, const ov::AnyMap& properties)
    : Qwen3ASREncoder(models_path) {
    ov::Core core = utils::singleton_core();
    const auto npu_properties = encoder_properties(properties);
    const WhisperFeatureExtractor feature_extractor{models_path / "preprocessor_config.json"};
    OPENVINO_ASSERT(feature_extractor.hop_length > 0, "hop_length in preprocessor_config.json must be > 0.");
    OPENVINO_ASSERT(feature_extractor.sampling_rate > 0, "sampling_rate in preprocessor_config.json must be > 0.");
    auto model = core.read_model(models_path / "openvino_encoder_model.xml", {}, npu_properties);
    reshape_encoder(model, feature_extractor.feature_size, m_encoder_chunk_frames);
    std::cout << "[INFO] Qwen3ASREncoder: NPU static shape configured with batch=1"
              << ", feature_size=" << feature_extractor.feature_size << ", chunk_frames=" << m_encoder_chunk_frames
              << std::endl;
    auto compiled_model = core.compile_model(model, "NPU", npu_properties);
    m_last_hidden_state_shape = compiled_model.output("last_hidden_state").get_partial_shape();
    utils::print_compiled_model_properties(compiled_model, "qwen3-asr encoder model");
    m_request = compiled_model.create_infer_request();
}

size_t Qwen3ASREncoderNPU::get_max_encoder_len(size_t max_asr_input_seconds,
                                               size_t sampling_rate,
                                               size_t hop_length) const {
    OPENVINO_ASSERT(hop_length > 0, "hop_length must be > 0");
    OPENVINO_ASSERT(m_encoder_chunk_frames > 0, "encoder_chunk_frames must be > 0");
    OPENVINO_ASSERT(m_last_hidden_state_shape.rank().is_static() &&
                        m_last_hidden_state_shape.rank().get_length() >= 2 && m_last_hidden_state_shape[1].is_static(),
                    "Unexpected encoder last_hidden_state shape: ",
                    m_last_hidden_state_shape);
    const size_t tokens_per_chunk = m_last_hidden_state_shape[1].get_length();
    OPENVINO_ASSERT(tokens_per_chunk > 0, "tokens_per_encoder_chunk must be > 0");
    const size_t max_frames = (max_asr_input_seconds * sampling_rate) / hop_length;
    const size_t max_chunks = (max_frames + m_encoder_chunk_frames - 1) / m_encoder_chunk_frames;
    return max_chunks * tokens_per_chunk;
}

ov::Tensor Qwen3ASREncoderNPU::infer(const ov::Tensor& input) {
    const auto& shape = input.get_shape();
    OPENVINO_ASSERT(shape.size() == 3, "Unexpected Qwen3-ASR encoder input rank.");
    const size_t num_chunks = shape[0];
    const size_t chunk_size = shape[1] * shape[2];
    ov::Tensor chunk(ov::element::f32, {1, shape[1], shape[2]});
    ov::Tensor output;
    size_t tokens_per_chunk = 0;
    size_t hidden_dim = 0;
    for (size_t index = 0; index < num_chunks; ++index) {
        std::memcpy(chunk.data<float>(), input.data<const float>() + index * chunk_size, chunk_size * sizeof(float));
        m_request.set_tensor("input_features", chunk);
        m_request.infer();
        const auto result = m_request.get_tensor("last_hidden_state");
        const auto& result_shape = result.get_shape();
        OPENVINO_ASSERT(result_shape.size() == 3 && result_shape[0] == 1,
                        "Unexpected Qwen3-ASR encoder output shape for one chunk.");
        if (index == 0) {
            tokens_per_chunk = result_shape[1];
            hidden_dim = result_shape[2];
            output = ov::Tensor(ov::element::f32, {num_chunks, tokens_per_chunk, hidden_dim});
        } else {
            OPENVINO_ASSERT(result_shape[1] == tokens_per_chunk && result_shape[2] == hidden_dim,
                            "Inconsistent Qwen3-ASR encoder output shape between chunks.");
        }
        const size_t stride = tokens_per_chunk * hidden_dim;
        std::memcpy(output.data<float>() + index * stride, result.data<const float>(), stride * sizeof(float));
    }
    return output;
}

void Qwen3ASREncoderNPU::reset_input() {
    const auto shape = m_request.get_tensor("input_features").get_shape();
    std::cout << "[INFO] Qwen3ASREncoder: resetting input_features with shape = " << shape << std::endl;
    m_request.set_tensor("input_features", ov::Tensor(ov::element::f32, shape));
}

}  // namespace ov::genai
