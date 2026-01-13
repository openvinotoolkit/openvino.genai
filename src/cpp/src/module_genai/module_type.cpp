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
    {ModuleType::VAEDecoderTilingModule, "VAEDecoderTilingModule"},
    {ModuleType::LLMInferenceModule, "LLMInferenceModule"},
    {ModuleType::ZImageDenoiserLoopModule, "ZImageDenoiserLoopModule"},
    {ModuleType::VAEDecoderModule, "VAEDecoderModule"},
    {ModuleType::ClipTextEncoderModule, "ClipTextEncoderModule"},
    {ModuleType::ResultModule, "ResultModule"},
    {ModuleType::SaveImageModule, "SaveImageModule"},
    {ModuleType::Unknown, "Unknown"},

    {ModuleType::FakeModuleA, "FakeModuleA"},
    {ModuleType::FakeModuleB, "FakeModuleB"},
    {ModuleType::FakeModuleC, "FakeModuleC"},
    {ModuleType::FakeModuleD, "FakeModuleD"},
};

const std::unordered_map<std::string, ModuleType> ModuleTypeConverter::kStringToType =
    ModuleTypeConverter::create_string_to_type_map();

const std::unordered_map<ThreadMode, std::string> ThreadModeConverter::kModeToString = {
    {ThreadMode::AUTO, "AUTO"},
    {ThreadMode::AUTO, "auto"},
    {ThreadMode::SYNC, "SYNC"},
    {ThreadMode::SYNC, "sync"},
    {ThreadMode::ASYNC, "ASYNC"},
    {ThreadMode::ASYNC, "async"},
};
const std::unordered_map<std::string, ThreadMode> ThreadModeConverter::kStringToMode =
    ThreadModeConverter::create_string_to_mode_map();

}  // namespace module
}  // namespace genai
}  // namespace ov