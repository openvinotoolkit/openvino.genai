// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "module_genai/module_type.hpp"

#include <unordered_map>

namespace ov {
namespace genai {

namespace module {

const std::unordered_map<ModuleType, std::string> ModuleTypeConverter::kTypeToString = {
    {ModuleType::ParameterModule, "ParameterModule"},
    {ModuleType::ImagePreprocessModule, "ImagePreprocessModule"},
    {ModuleType::VisionEncoderModule, "VisionEncoderModule"},
    {ModuleType::TextEncoderModule, "TextEncoderModule"},
    {ModuleType::TextEmbeddingModule, "TextEmbeddingModule"},
    {ModuleType::EmbeddingMergerModule, "EmbeddingMergerModule"},
    {ModuleType::FeaturePrunerModule, "FeaturePrunerModule"},
    {ModuleType::FeatureFusionModule, "FeatureFusionModule"},
    {ModuleType::LLMInferenceModule, "LLMInferenceModule"},
    {ModuleType::ZImageDenoiserLoopModule, "ZImageDenoiserLoopModule"},
    {ModuleType::ResultModule, "ResultModule"},
    {ModuleType::Unknown, "Unknown"}};

const std::unordered_map<std::string, ModuleType> ModuleTypeConverter::kStringToType =
    ModuleTypeConverter::create_string_to_type_map();

}  // namespace module
}  // namespace genai
}  // namespace ov