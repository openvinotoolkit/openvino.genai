// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "audio_encoder.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>

#include "openvino/core/preprocess/pre_post_process.hpp"
#include "properties.hpp"
#include "utils.hpp"

namespace ov::genai {

AudioEncoderQwen3ASRNPU::AudioEncoderQwen3ASRNPU(const VLMConfig& config,
                                                 const std::shared_ptr<ov::Model>& audio_model,
                                                 const std::filesystem::path& config_dir,
                                                 const std::string& device,
                                                 const ov::AnyMap& properties)
    : AudioEncoderQwen3ASR(config, config_dir) {
    OPENVINO_ASSERT(audio_model && audio_model->get_parameters().size() == 4 &&
                        audio_model->get_results().size() == 1 && audio_model->get_variables().empty(),
                    "Qwen3-ASR NPU audio encoder requires four inputs and one stateless output.");
    auto model = audio_model->clone();
    const auto feature = model->input("padded_feature");
    const auto mask = model->input("padded_mask_after_cnn");
    const auto output = model->output().get_partial_shape();
    OPENVINO_ASSERT(
        config.audio_config_num_mel_bins > 0 && config.hidden_size > 0 &&
            config.audio_config_num_mel_bins <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) &&
            config.hidden_size <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()) &&
            feature.get_element_type() == ov::element::f32 && feature.get_partial_shape().rank() == 3 &&
            feature.get_partial_shape()[1].is_static() &&
            feature.get_partial_shape()[1].get_length() == static_cast<int64_t>(config.audio_config_num_mel_bins) &&
            mask.get_element_type() == ov::element::boolean && mask.get_partial_shape().rank() == 2 &&
            model->output().get_element_type() == ov::element::f32 && output.rank() == 3 && output[2].is_static() &&
            output[2].get_length() == static_cast<int64_t>(config.hidden_size),
        "Qwen3-ASR NPU audio encoder has an unsupported input or output shape or type.");
    for (const auto& name : {"aftercnn_lens", "cu_seqlens"}) {
        OPENVINO_ASSERT(model->input(name).get_target_inputs().empty(),
                        "Qwen3-ASR NPU audio encoder requires independent windows; input '",
                        name,
                        "' is used by this export.");
    }

    OPENVINO_ASSERT(config.audio_config_n_window <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max() - 7) / 2,
                    "Qwen3-ASR NPU audio encoder window size is invalid.");
    const size_t frames = 2 * config.audio_config_n_window;
    const size_t tokens = get_feat_extract_output_length(frames);
    OPENVINO_ASSERT(tokens <= static_cast<size_t>(std::numeric_limits<int32_t>::max()),
                    "Qwen3-ASR NPU audio encoder window size is invalid.");
    model->reshape(
        {{"padded_feature",
          ov::PartialShape{1, feature.get_partial_shape()[1], static_cast<ov::Dimension::value_type>(frames)}},
         {"padded_mask_after_cnn", ov::PartialShape{1, static_cast<ov::Dimension::value_type>(tokens)}},
         {"aftercnn_lens", ov::PartialShape{1}},
         {"cu_seqlens", ov::PartialShape{2}}});
    ov::preprocess::PrePostProcessor preprocessor(model);
    preprocessor.input("padded_mask_after_cnn").tensor().set_element_type(ov::element::u8);
    model = preprocessor.build();

    auto model_properties = qwen3_asr_npu_input_model_properties(properties, "audio_encoder", device);
    OPENVINO_ASSERT(utils::is_npuw_enabled(model_properties), "Qwen3-ASR NPU audio encoder requires NPU_USE_NPUW=YES.");
    m_request = utils::singleton_core().compile_model(model, "NPU", model_properties).create_infer_request();
}

EncodedAudio AudioEncoderQwen3ASRNPU::encode_audio(const ov::Tensor& audio) {
    const auto preprocessed = preprocess_audio(audio);
    const size_t num_chunks = preprocessed.chunk_token_counts.size();
    const size_t chunk_frames = 2 * m_config.audio_config_n_window;
    const size_t chunk_tokens = get_feat_extract_output_length(chunk_frames);
    const size_t padded_frames = preprocessed.padded_feature.get_shape()[2];
    ov::Tensor feature(ov::element::f32, {1, m_feature_extractor.feature_size, chunk_frames});
    ov::Tensor mask(ov::element::u8, {1, chunk_tokens});
    ov::Tensor aftercnn_lens(ov::element::i64, {1});
    ov::Tensor cu_seqlens(ov::element::i32, {2});
    ov::Tensor compacted(ov::element::f32, {preprocessed.total_tokens, m_config.hidden_size});
    float* destination = compacted.data<float>();
    const float* source = preprocessed.padded_feature.data<const float>();
    for (size_t chunk_index = 0; chunk_index < num_chunks; ++chunk_index) {
        std::fill_n(feature.data<float>(), feature.get_size(), 0.0f);
        for (size_t mel = 0; mel < m_feature_extractor.feature_size; ++mel) {
            std::memcpy(feature.data<float>() + mel * chunk_frames,
                        source + (chunk_index * m_feature_extractor.feature_size + mel) * padded_frames,
                        padded_frames * sizeof(float));
        }
        const size_t valid_tokens = preprocessed.chunk_token_counts[chunk_index];
        std::fill_n(mask.data<uint8_t>(), chunk_tokens, uint8_t{0});
        std::fill_n(mask.data<uint8_t>(), valid_tokens, uint8_t{1});
        aftercnn_lens.data<int64_t>()[0] = static_cast<int64_t>(valid_tokens);
        cu_seqlens.data<int32_t>()[0] = 0;
        cu_seqlens.data<int32_t>()[1] = static_cast<int32_t>(valid_tokens);
        m_request.set_tensor("padded_feature", feature);
        m_request.set_tensor("padded_mask_after_cnn", mask);
        m_request.set_tensor("aftercnn_lens", aftercnn_lens);
        m_request.set_tensor("cu_seqlens", cu_seqlens);
        m_request.infer();
        const auto encoded = m_request.get_tensor("audio_features");
        OPENVINO_ASSERT(encoded.get_element_type() == ov::element::f32 &&
                            encoded.get_shape() == ov::Shape({1, chunk_tokens, m_config.hidden_size}) &&
                            encoded.is_continuous(),
                        "Qwen3-ASR NPU audio encoder returned an unexpected output shape, type, or stride.");
        const size_t valid_elements = valid_tokens * m_config.hidden_size;
        std::memcpy(destination, encoded.data<const float>(), valid_elements * sizeof(float));
        destination += valid_elements;
    }
    return {std::move(compacted), preprocessed.total_tokens};
}

std::unique_ptr<AudioEncoderQwen3ASR> create_qwen3_asr_audio_encoder(const VLMConfig& config,
                                                                     const std::shared_ptr<ov::Model>& audio_model,
                                                                     const std::filesystem::path& config_dir,
                                                                     const std::string& device,
                                                                     const ov::AnyMap& properties) {
    if (qwen3_asr_input_device(device, properties, "audio_encoder") == "NPU") {
        return std::make_unique<AudioEncoderQwen3ASRNPU>(config, audio_model, config_dir, device, properties);
    }
    return std::make_unique<AudioEncoderQwen3ASR>(config, audio_model, config_dir, device, properties);
}

}  // namespace ov::genai
