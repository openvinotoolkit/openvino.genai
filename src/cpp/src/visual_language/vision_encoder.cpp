// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "vision_encoder.hpp"
#include "utils.hpp"


#include "visual_language/qwen2vl/classes.hpp"
#include "visual_language/phi3_vision/classes.hpp"
#include "visual_language/minicpm/classes.hpp"
#include "visual_language/llava/classes.hpp"
#include "visual_language/llava_next/classes.hpp"
#include "visual_language/internvl_chat/classes.hpp"

namespace ov::genai {

namespace {

ProcessorConfig from_any_map(
    const ov::AnyMap& config_map,
    const ProcessorConfig& initial
) {
    auto iter = config_map.find("processor_config");
    ProcessorConfig extracted_config = config_map.end() != iter ?
        iter->second.as<ProcessorConfig>() : initial;
    using utils::read_anymap_param;
    read_anymap_param(config_map, "patch_size", extracted_config.patch_size);
    read_anymap_param(config_map, "scale_resolution", extracted_config.scale_resolution);
    read_anymap_param(config_map, "max_slice_nums", extracted_config.max_slice_nums);
    read_anymap_param(config_map, "norm_mean", extracted_config.norm_mean);
    read_anymap_param(config_map, "norm_std", extracted_config.norm_std);
    return extracted_config;
}

} // namespace

/// Base VisionEncoder class

VisionEncoder::IVisionEncoder::IVisionEncoder(const std::filesystem::path& model_dir, const std::string& device, const ov::AnyMap properties) {
    auto compiled_model = utils::singleton_core().compile_model(model_dir / "openvino_vision_embeddings_model.xml", device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM vision embeddings model");
    m_vision_encoder = compiled_model.create_infer_request();
    m_processor_config = utils::from_config_json_if_exists<ProcessorConfig>(model_dir, "preprocessor_config.json");
}

VisionEncoder::IVisionEncoder::IVisionEncoder(
    const std::string& model,
    const ov::Tensor& weights,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) {
    auto compiled_model = utils::singleton_core().compile_model(model, weights, device, device_config);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM vision embeddings model");
    m_vision_encoder = compiled_model.create_infer_request();
    m_processor_config = utils::from_config_json_if_exists<ProcessorConfig>(config_dir_path, "preprocessor_config.json");
}

ProcessorConfig VisionEncoder::IVisionEncoder::get_processor_config() const {
    return m_processor_config;
}

/// Public VisionEncoder class

VisionEncoder::VisionEncoder(const std::filesystem::path& model_dir, const VLMModelType model_type, const std::string& device, const ov::AnyMap properties) {
    if (model_type == VLMModelType::MINICPM) {
        m_impl = std::make_shared<VisionEncoderMiniCPM>(model_dir, device, properties);
    } else if (model_type == VLMModelType::LLAVA) {
        m_impl = std::make_shared<VisionEncoderLLaVA>(model_dir, device, properties);
    } else if (model_type == VLMModelType::LLAVA_NEXT) {
        m_impl = std::make_shared<VisionEncoderLLaVANext>(model_dir, device, properties);
    } else if (model_type == VLMModelType::INTERNVL_CHAT) {
        m_impl = std::make_shared<VisionEncoderInternVLChat>(model_dir, device, properties);
    } else if (model_type == VLMModelType::PHI3_V) {
        m_impl = std::make_shared<VisionEncoderPhi3V>(model_dir, device, properties);
    } else if (model_type == VLMModelType::QWEN2_VL) {
        m_impl = std::make_shared<VisionEncoderQwen2VL>(model_dir, device, properties);
    } else {
        OPENVINO_THROW("Unsupported model type in VLM VisionEncoder class. Please, create feature request on new model support");
    }
}

VisionEncoder::VisionEncoder(
    const std::string& model,
    const ov::Tensor& weights,
    const std::filesystem::path& config_dir_path,
    const VLMModelType model_type,
    const std::string& device,
    const ov::AnyMap device_config
) {
    if (model_type == VLMModelType::MINICPM) {
        m_impl = std::make_shared<VisionEncoderMiniCPM>(model, weights, config_dir_path, device, device_config);
    } else if (model_type == VLMModelType::LLAVA) {
        m_impl = std::make_shared<VisionEncoderLLaVA>(model, weights, config_dir_path, device, device_config);
    } else if (model_type == VLMModelType::LLAVA_NEXT) {
        m_impl = std::make_shared<VisionEncoderLLaVANext>(model, weights, config_dir_path, device, device_config);
    } else if (model_type == VLMModelType::INTERNVL_CHAT) {
        m_impl = std::make_shared<VisionEncoderInternVLChat>(model, weights, config_dir_path, device, device_config);
    } else if (model_type == VLMModelType::PHI3_V) {
        m_impl = std::make_shared<VisionEncoderPhi3V>(model, weights, config_dir_path, device, device_config);
    } else if (model_type == VLMModelType::QWEN2_VL) {
        m_impl = std::make_shared<VisionEncoderQwen2VL>(model, weights, config_dir_path, device, device_config);
    } else {
        OPENVINO_THROW("Unsupported model type in VLM VisionEncoder class. Please, create feature request on new model support");
    }
}

EncodedImage VisionEncoder::encode(const ov::Tensor& image, const ov::AnyMap& properties) {
    return m_impl->encode(image, properties);
}

ProcessorConfig VisionEncoder::get_processor_config() const {
    return m_impl->get_processor_config();
}

} // namespace ov::genai
