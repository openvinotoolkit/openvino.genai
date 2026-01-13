// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "openvino/genai/module_genai/pipeline.hpp"

namespace ov {
namespace genai {

namespace module {

    enum class ModuleType : int {
        // 0. Parameter/Input Modules
        ParameterModule = 0,
        
        // 1. Preprocessing Modules
        ImagePreprocessModule = 10,
        TextEncoderModule = 11,
        RandomLatentImageModule = 12,
        
        // 2. Embedding/Encoder Modules
        VisionEncoderModule = 20,
        TextEmbeddingModule = 21,
        EmbeddingMergerModule = 22,
        ClipTextEncoderModule = 23,
        
        // 3. Fusion/Pruning Modules
        FeaturePrunerModule = 30,
        FeatureFusionModule = 31,
        VAEDecoderTilingModule = 32,
        
        // 4. Inference/Generator Modules
        LLMInferenceModule = 40,
        ZImageDenoiserLoopModule = 41,
        VAEDecoderModule = 42,
        
        // 5. Output/Result Modules
        ResultModule = 50,
        SaveImageModule = 51,
        
        // Default/Unknown
        Unknown = 99,

        FakeModuleA = 10000,
        FakeModuleB = 10001,
        FakeModuleC = 10002,
        FakeModuleD = 10003,
    };

    enum class ThreadMode : int {
        AUTO = 0,   // define AUTO
        SYNC = 1,
        ASYNC = 2
    };

    struct ThreadModeConverter {
    private:
        static const std::unordered_map<ThreadMode, std::string> kModeToString;
        static const std::unordered_map<std::string, ThreadMode> kStringToMode;
        static std::unordered_map<std::string, ThreadMode> create_string_to_mode_map() {
            std::unordered_map<std::string, ThreadMode> map;
            for (const auto& pair : kModeToString) {
                map[pair.second] = pair.first;
            }
            return map;
        }
        public:
        static std::string toString(ThreadMode mode) {
            auto it = kModeToString.find(mode);
            if (it != kModeToString.end()) {
                return it->second;
            }
            throw std::runtime_error("Unknown ThreadMode value: " + std::to_string(static_cast<int>(mode)));
        }
        static ThreadMode fromString(const std::string& str) {
            auto it = kStringToMode.find(str);
            if (it != kStringToMode.end()) {
                return it->second;
            }
            throw std::runtime_error("Unknown ThreadMode string: " + str);
        }
    };

    inline std::ostream& operator<<(std::ostream& os, ThreadMode mode) {
        return os << ThreadModeConverter::toString(mode);
    }

    struct ModuleTypeConverter {
    private:
        static const std::unordered_map<ModuleType, std::string> kTypeToString;

        static const std::unordered_map<std::string, ModuleType> kStringToType;

        static std::unordered_map<std::string, ModuleType> create_string_to_type_map() {
            std::unordered_map<std::string, ModuleType> map;
            for (const auto& pair : kTypeToString) {
                map[pair.second] = pair.first;
            }
            return map;
        }

    public:
        static std::string toString(ModuleType type) {
            auto it = kTypeToString.find(type);
            if (it != kTypeToString.end()) {
                return it->second;
            }
            throw std::runtime_error("Unknown ModuleType value: " + std::to_string(static_cast<int>(type)));
        }

        static ModuleType fromString(const std::string& str) {
            auto it = kStringToType.find(str);
            if (it != kStringToType.end()) {
                return it->second;
            }
            return ModuleType::Unknown;
        }
    };

    inline std::ostream& operator<<(std::ostream& os, ModuleType type) {
        return os << ModuleTypeConverter::toString(type);
    }

}  // namespace module
}  // namespace genai
}  // namespace ov