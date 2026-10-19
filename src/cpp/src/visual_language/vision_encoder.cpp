// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <fstream>
#include <type_traits>

#include "vision_encoder.hpp"

#include "utils.hpp"
#include "logger.hpp"
#include "visual_language/vlm_family.hpp"

namespace ov::genai {

namespace {
template <typename... Args>
VisionEncoder::Ptr create_vision_encoder(VLMModelType type, Args&&... args) {
    return visit_vlm_family(type, [&](auto family) -> VisionEncoder::Ptr {
        using Model = typename decltype(family)::VisionEncoderType;
        if constexpr (std::is_constructible_v<Model, Args...>)
            return std::make_shared<Model>(std::forward<Args>(args)...);
        else
            OPENVINO_THROW("This VLM model type does not support the supplied model source");
    });
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
