// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "vision_encoder.hpp"

#include <fstream>
#include <type_traits>
#include <utility>

#include "logger.hpp"
#include "utils.hpp"
#include "visual_language/deepseek_ocr2/classes.hpp"
#include "visual_language/gemma3/classes.hpp"
#include "visual_language/gemma3n/classes.hpp"
#include "visual_language/gemma4/classes.hpp"
#include "visual_language/internvl_chat/classes.hpp"
#include "visual_language/lfm2/classes.hpp"
#include "visual_language/llava/classes.hpp"
#include "visual_language/llava_next/classes.hpp"
#include "visual_language/llava_next_video/classes.hpp"
#include "visual_language/minicpm/classes.hpp"
#include "visual_language/minicpmv4_7/classes.hpp"
#include "visual_language/muse_glimmer/classes.hpp"
#include "visual_language/nanollava/classes.hpp"
#include "visual_language/phi3_vision/classes.hpp"
#include "visual_language/phi4mm/classes.hpp"
#include "visual_language/qwen2_5_vl/classes.hpp"
#include "visual_language/qwen2vl/classes.hpp"
#include "visual_language/qwen3_5/classes.hpp"
#include "visual_language/qwen3_omni/classes.hpp"
#include "visual_language/qwen3_vl/classes.hpp"
#include "visual_language/videochat_flash/classes.hpp"

namespace ov::genai {

namespace {
template <typename Model, typename... Args>
VisionEncoder::Ptr construct_model(Args&&... args) {
    if constexpr (std::is_constructible_v<Model, Args...>)
        return std::make_shared<Model>(std::forward<Args>(args)...);
    else
        OPENVINO_THROW("This VLM model type does not support the supplied model source");
}

template <typename... Args>
VisionEncoder::Ptr create_vision_encoder(VLMModelType type, Args&&... args) {
    switch (type) {
    case VLMModelType::MINICPM:
        return construct_model<VisionEncoderMiniCPM>(std::forward<Args>(args)...);
    case VLMModelType::MINICPMV4_7:
        return construct_model<VisionEncoderMiniCPMv4_7>(std::forward<Args>(args)...);
    case VLMModelType::LLAVA:
        return construct_model<VisionEncoderLLaVA>(std::forward<Args>(args)...);
    case VLMModelType::NANOLLAVA:
        return construct_model<VisionEncoderNanoLLaVA>(std::forward<Args>(args)...);
    case VLMModelType::LLAVA_NEXT:
        return construct_model<VisionEncoderLLaVANext>(std::forward<Args>(args)...);
    case VLMModelType::LLAVA_NEXT_VIDEO:
        return construct_model<VisionEncoderLLaVANextVideo>(std::forward<Args>(args)...);
    case VLMModelType::INTERNVL_CHAT:
        return construct_model<VisionEncoderInternVLChat>(std::forward<Args>(args)...);
    case VLMModelType::PHI3_V:
        return construct_model<VisionEncoderPhi3V>(std::forward<Args>(args)...);
    case VLMModelType::PHI4MM:
        return construct_model<VisionEncoderPhi4MM>(std::forward<Args>(args)...);
    case VLMModelType::QWEN2_VL:
        return construct_model<VisionEncoderQwen2VL>(std::forward<Args>(args)...);
    case VLMModelType::QWEN2_5_VL:
        return construct_model<VisionEncoderQwen2_5_VL>(std::forward<Args>(args)...);
    case VLMModelType::QWEN3_VL:
        return construct_model<VisionEncoderQwen3VL>(std::forward<Args>(args)...);
    case VLMModelType::QWEN3_5_MOE:
        return construct_model<VisionEncoderQwen3_5>(std::forward<Args>(args)...);
    case VLMModelType::QWEN3_OMNI:
        return construct_model<VisionEncoderQwen3Omni>(std::forward<Args>(args)...);
    case VLMModelType::GEMMA3:
        return construct_model<VisionEncoderGemma3>(std::forward<Args>(args)...);
    case VLMModelType::GEMMA3N:
        return construct_model<VisionEncoderGemma3n>(std::forward<Args>(args)...);
    case VLMModelType::LFM2_VL:
        return construct_model<VisionEncoderLFM2>(std::forward<Args>(args)...);
    case VLMModelType::GEMMA4:
        return construct_model<VisionEncoderGemma4>(std::forward<Args>(args)...);
    case VLMModelType::GEMMA4_UNIFIED:
        return construct_model<VisionEncoderGemma4>(std::forward<Args>(args)...);
    case VLMModelType::VIDEOCHAT_FLASH_QWEN:
        return construct_model<VisionEncoderVideoChatFlashQwen>(std::forward<Args>(args)...);
    case VLMModelType::DEEPSEEK_OCR2:
        return construct_model<VisionEncoderDeepseekOCR2>(std::forward<Args>(args)...);
    case VLMModelType::MUSE_GLIMMER:
        return construct_model<VisionEncoderMuseGlimmer>(std::forward<Args>(args)...);
    case VLMModelType::QWEN3_5:
        return construct_model<VisionEncoderQwen3_5>(std::forward<Args>(args)...);
    default:
        OPENVINO_THROW("Unsupported VLM model type");
    }
}
}  // namespace

VisionEncoder::VisionEncoder(const VLMModels& models, ConfigOnlyTag)
    : m_processor_config(models.processor_config),
      m_video_processor_config(models.video_processor_config) {}

VisionEncoder::VisionEncoder(const VLMModels& models, const std::string& device, const ov::AnyMap& properties)
    : VisionEncoder(models, ConfigOnlyTag{}) {
    auto compiled = utils::singleton_core().compile_model(
        models.at("vision_embeddings"),
        device,
        utils::get_model_properties(properties, "vision_embeddings", device));
    ov::genai::utils::print_compiled_model_properties(compiled, "VLM vision embeddings model");
    m_ireq_queue_vision_encoder = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled.get_property(ov::optimal_number_of_infer_requests),
        [&compiled] {
            return compiled.create_infer_request();
        });
}

VisionEncoder::VisionEncoder(const std::filesystem::path& model_dir, const std::string& device, const ov::AnyMap properties) {
    auto compiled_model = utils::singleton_core().compile_model(
        model_dir / "openvino_vision_embeddings_model.xml", device,
        utils::get_model_properties(properties, "vision_embeddings", device));
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
    const auto& [vision_encoder_model, vision_encoder_weights] = utils::get_model_weights_pair(models_map, "vision_embeddings");
    auto compiled_model = utils::singleton_core().compile_model(
        vision_encoder_model, vision_encoder_weights, device,
        utils::get_model_properties(device_config, "vision_embeddings", device));
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

        if (parsed_processor_config.contains("image_processor")) {
            m_processor_config = ProcessorConfig(parsed_processor_config.at("image_processor"));
            if (parsed_processor_config.contains("video_processor"))
                m_video_processor_config = VideoProcessorConfig(parsed_processor_config.at("video_processor"));
            else if (std::filesystem::exists(config_dir_path / video_preprocessor_config_filename))
                m_video_processor_config = VideoProcessorConfig(config_dir_path / video_preprocessor_config_filename);
            else
                m_video_processor_config = VideoProcessorConfig(parsed_processor_config.at("image_processor"));
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

VisionEncoder::VisionEncoder(const std::filesystem::path& config_dir, ConfigOnlyTag) {
    resolve_processor_configs(config_dir);
}

VisionEncoder::VisionEncoder(const ModelsMap&, const std::filesystem::path& config_dir, ConfigOnlyTag tag)
    : VisionEncoder(config_dir, tag) {}

ProcessorConfig VisionEncoder::get_processor_config() const {
    return m_processor_config;
}

VideoProcessorConfig VisionEncoder::get_video_processor_config() const {
    return m_video_processor_config;
}

VisionEncoder::Ptr VisionEncoder::create(const std::filesystem::path& model_dir, const VLMModelType model_type, const std::string& device, const ov::AnyMap properties) {
    return create_vision_encoder(model_type, model_dir, device, properties);
}

VisionEncoder::Ptr VisionEncoder::create(
    const ModelsMap& models_map,
    const std::filesystem::path& config_dir_path,
    const VLMModelType model_type,
    const std::string& device,
    const ov::AnyMap device_config) {
    return create_vision_encoder(model_type, models_map, config_dir_path, device, device_config);
}

VisionEncoder::Ptr VisionEncoder::create(const VLMModels& models,
                                         const std::string& device,
                                         const ov::AnyMap& properties) {
    return create_vision_encoder(models.config.model_type, models, device, properties);
}

} // namespace ov::genai
