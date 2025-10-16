// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "vision_encoder.hpp"
#include "utils.hpp"


#include "visual_language/qwen2vl/classes.hpp"
#include "visual_language/qwen2_5_vl/classes.hpp"
#include "visual_language/phi3_vision/classes.hpp"
#include "visual_language/phi4mm/classes.hpp"
#include "visual_language/minicpm/classes.hpp"
#include "visual_language/llava/classes.hpp"
#include "visual_language/nanollava/classes.hpp"
#include "visual_language/llava_next/classes.hpp"
#include "visual_language/llava_next_video/classes.hpp"
#include "visual_language/internvl_chat/classes.hpp"
#include "visual_language/gemma3/classes.hpp"

namespace ov::genai {

VisionEncoder::VisionEncoder(const std::filesystem::path& model_dir, const std::string& device, const ov::AnyMap properties) {
    auto compiled_model = utils::singleton_core().compile_model(model_dir / "openvino_vision_embeddings_model.xml", device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM vision embeddings model");
    m_ireq_queue_vision_encoder = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
    m_processor_config = utils::from_config_json_if_exists<ProcessorConfig>(model_dir, "preprocessor_config.json");
}

VisionEncoder::VisionEncoder(
    const ModelsMap& models_map,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) {
    const auto& vision_encoder_model = utils::get_model_weights_pair(models_map, "vision_embeddings").first;
    const auto& vision_encoder_weights = utils::get_model_weights_pair(models_map, "vision_embeddings").second;
    auto compiled_model = utils::singleton_core().compile_model(vision_encoder_model, vision_encoder_weights, device, device_config);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM vision embeddings model");
    m_ireq_queue_vision_encoder = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
    m_processor_config = utils::from_config_json_if_exists<ProcessorConfig>(config_dir_path, "preprocessor_config.json");
}

ProcessorConfig VisionEncoder::get_processor_config() const {
    return m_processor_config;
}

VisionEncoder::Ptr VisionEncoder::create(const std::filesystem::path& model_dir, const VLMModelType model_type, const std::string& device, const ov::AnyMap properties) {
    if (model_type == VLMModelType::MINICPM) {
        return std::make_shared<VisionEncoderMiniCPM>(model_dir, device, properties);
    } else if (model_type == VLMModelType::LLAVA) {
        return std::make_shared<VisionEncoderLLaVA>(model_dir, device, properties);
    } else if (model_type == VLMModelType::NANOLLAVA) {
        return std::make_shared<VisionEncoderNanoLLaVA>(model_dir, device, properties);
    } else if (model_type == VLMModelType::LLAVA_NEXT) {
        return std::make_shared<VisionEncoderLLaVANext>(model_dir, device, properties);
    } else if (model_type == VLMModelType::LLAVA_NEXT_VIDEO) {
        return std::make_shared<VisionEncoderLLaVANextVideo>(model_dir, device, properties);
    } else if (model_type == VLMModelType::INTERNVL_CHAT) {
        return std::make_shared<VisionEncoderInternVLChat>(model_dir, device, properties);
    } else if (model_type == VLMModelType::PHI3_V) {
        return std::make_shared<VisionEncoderPhi3V>(model_dir, device, properties);
    } else if (model_type == VLMModelType::PHI4MM) {
        return std::make_shared<VisionEncoderPhi4MM>(model_dir, device, properties);
    } else if (model_type == VLMModelType::QWEN2_VL) {
        return std::make_shared<VisionEncoderQwen2VL>(model_dir, device, properties);
    } else if (model_type == VLMModelType::QWEN2_5_VL) {
        return std::make_shared<VisionEncoderQwen2_5_VL>(model_dir, device, properties);
    } else if (model_type == VLMModelType::GEMMA3) {
        return std::make_shared<VisionEncoderGemma3>(model_dir, device, properties);
    } else {
        OPENVINO_THROW("Unsupported model type in VLM VisionEncoder class. Please, create feature request on new model support");
    }
}

VisionEncoder::Ptr VisionEncoder::create(
    const ModelsMap& models_map,
    const std::filesystem::path& config_dir_path,
    const VLMModelType model_type,
    const std::string& device,
    const ov::AnyMap device_config) {
    if (model_type == VLMModelType::MINICPM) {
        return std::make_shared<VisionEncoderMiniCPM>(models_map, config_dir_path, device, device_config);
    } else if (model_type == VLMModelType::LLAVA) {
        return std::make_shared<VisionEncoderLLaVA>(models_map, config_dir_path, device, device_config);
    } else if (model_type == VLMModelType::NANOLLAVA) {
        return std::make_shared<VisionEncoderNanoLLaVA>(models_map, config_dir_path, device, device_config);
    } else if (model_type == VLMModelType::LLAVA_NEXT) {
        return std::make_shared<VisionEncoderLLaVANext>(models_map, config_dir_path, device, device_config);
    } else if (model_type == VLMModelType::LLAVA_NEXT_VIDEO) {
        return std::make_shared<VisionEncoderLLaVANextVideo>(models_map, config_dir_path, device, device_config);
    } else if (model_type == VLMModelType::INTERNVL_CHAT) {
        return std::make_shared<VisionEncoderInternVLChat>(models_map, config_dir_path, device, device_config);
    } else if (model_type == VLMModelType::PHI3_V) {
        return std::make_shared<VisionEncoderPhi3V>(models_map, config_dir_path, device, device_config);
    } else if (model_type == VLMModelType::PHI4MM) {
        return std::make_shared<VisionEncoderPhi4MM>(models_map, config_dir_path, device, device_config);
    } else if (model_type == VLMModelType::QWEN2_VL) {
        return std::make_shared<VisionEncoderQwen2VL>(models_map, config_dir_path, device, device_config);
    } else if (model_type == VLMModelType::QWEN2_5_VL) {
        return std::make_shared<VisionEncoderQwen2_5_VL>(models_map, config_dir_path, device, device_config);
    } else if (model_type == VLMModelType::GEMMA3) {
        return std::make_shared<VisionEncoderGemma3>(models_map, config_dir_path, device, device_config);
    } else {
        OPENVINO_THROW("Unsupported model type in VLM VisionEncoder class. Please, create feature request on new model support");
    }
}

} // namespace ov::genai
