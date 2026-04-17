// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>

#include "vision_encoder.hpp"
#include "utils.hpp"
#include "logger.hpp"

#include "visual_language/qwen2vl/classes.hpp"
#include "visual_language/qwen2_5_vl/classes.hpp"
#include "visual_language/qwen3_vl/classes.hpp"
#include "visual_language/qwen3_5/classes.hpp"
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
    resolve_processor_configs(model_dir);
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
    resolve_processor_configs(config_dir_path);
}

void VisionEncoder::resolve_processor_configs(const std::filesystem::path& config_dir_path) {
    // TODO Consider using separate class or struct for combined processor_config.json
    const std::string processor_config_filename = "processor_config.json";
    const std::string preprocessor_config_filename = "preprocessor_config.json";
    const std::string video_preprocessor_config_filename = "video_preprocessor_config.json";

    const auto processor_config_path = config_dir_path / processor_config_filename;
    if (std::filesystem::exists(processor_config_path)) {
        std::ifstream stream(processor_config_path);
        OPENVINO_ASSERT(stream.is_open(),
            "Failed to open '", processor_config_path, "' in '", config_dir_path.string(), "'");
        const auto parsed_processor_config = nlohmann::json::parse(stream);

        if (parsed_processor_config.contains("image_processor") && parsed_processor_config.contains("video_processor")) {
            m_processor_config = ProcessorConfig(parsed_processor_config.at("image_processor"));
            m_video_processor_config = VideoProcessorConfig(parsed_processor_config.at("video_processor"));
            return;
        }
    }

    GENAI_INFO("Combined " + processor_config_filename + " not found in '" + config_dir_path.string() + "' or missing image and video processor keys." +
        " Falling back to " + preprocessor_config_filename + " and " + video_preprocessor_config_filename + ".");

    m_processor_config = utils::from_config_json_if_exists<ProcessorConfig>(config_dir_path, preprocessor_config_filename.c_str());

    const auto video_config_path = config_dir_path / video_preprocessor_config_filename;
    if (std::filesystem::exists(video_config_path)) {
        m_video_processor_config = VideoProcessorConfig(video_config_path);
    } else {
        GENAI_INFO(video_preprocessor_config_filename + " not found in '" + config_dir_path.string() +
            "'. Falling back to " + preprocessor_config_filename + " for video processing configuration.");
        m_video_processor_config = utils::from_config_json_if_exists<VideoProcessorConfig>(config_dir_path, preprocessor_config_filename.c_str());
    }
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
    } else if (model_type == VLMModelType::QWEN3_VL) {
        return std::make_shared<VisionEncoderQwen3VL>(model_dir, device, properties);
    } else if (model_type == VLMModelType::QWEN3_5) {
        return std::make_shared<VisionEncoderQwen3_5>(model_dir, device, properties);
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
    } else if (model_type == VLMModelType::QWEN3_VL) {
        return std::make_shared<VisionEncoderQwen3VL>(models_map, config_dir_path, device, device_config);
    } else if (model_type == VLMModelType::QWEN3_5) {
        return std::make_shared<VisionEncoderQwen3_5>(models_map, config_dir_path, device, device_config);
    } else if (model_type == VLMModelType::GEMMA3) {
        return std::make_shared<VisionEncoderGemma3>(models_map, config_dir_path, device, device_config);
    } else {
        OPENVINO_THROW("Unsupported model type in VLM VisionEncoder class. Please, create feature request on new model support");
    }
}

} // namespace ov::genai
