// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "automatic_speech_recognition/models/qwen3-asr/split/audio_encoder.hpp"

#include <cstdint>
#include <limits>

#include "properties.hpp"
#include "utils.hpp"

namespace ov::genai {

std::unique_ptr<AudioEncoderQwen3ASR> create_qwen3_asr_audio_encoder(const VLMConfig& config,
                                                                     const std::shared_ptr<ov::Model>& audio_model,
                                                                     const std::filesystem::path& config_dir,
                                                                     const std::string& device,
                                                                     const ov::AnyMap& properties) {
    if (qwen3_asr_input_device(device, properties, "audio_encoder") != "NPU") {
        return std::make_unique<AudioEncoderQwen3ASR>(config, audio_model, config_dir, device, properties);
    }
    auto model_properties = qwen3_asr_input_model_properties(properties, "audio_encoder", device);
    utils::set_config_default(model_properties, "NPU_USE_NPUW", std::string{"YES"});
    OPENVINO_ASSERT(utils::is_npuw_enabled(model_properties), "Qwen3-ASR NPU audio encoder requires NPU_USE_NPUW=YES.");
    utils::set_config_default(model_properties, "NPUW_QWEN3_ASR_AUDIO_ENCODER", std::string{"YES"});
    OPENVINO_ASSERT(utils::is_property_enabled(model_properties, "NPUW_QWEN3_ASR_AUDIO_ENCODER"),
                    "Qwen3-ASR NPU audio encoder requires NPUW_QWEN3_ASR_AUDIO_ENCODER=YES.");
    OPENVINO_ASSERT(config.audio_config_n_window > 0 &&
                        static_cast<uint64_t>(config.audio_config_n_window) <= std::numeric_limits<uint64_t>::max() / 2,
                    "Qwen3-ASR audio window size is invalid.");
    const auto window = static_cast<uint64_t>(config.audio_config_n_window) * 2;
    utils::set_config_default(model_properties, "NPUW_QWEN3_ASR_AUDIO_WINDOW", window);
    OPENVINO_ASSERT(model_properties.at("NPUW_QWEN3_ASR_AUDIO_WINDOW").as<uint64_t>() == window,
                    "Qwen3-ASR NPU audio encoder window must match the model configuration.");
    utils::set_config_default(model_properties, "NPUW_DEVICES", std::string{"NPU"});
    return std::make_unique<AudioEncoderQwen3ASR>(config, audio_model, config_dir, "NPU", model_properties);
}

}  // namespace ov::genai
