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

    // Conditional X-macro entry for new-arch modules
    #ifdef ENABLE_OPENVINO_NEW_ARCH
    #define OPENVINO_NEW_ARCH_X(name, val) X(name, val)
    #else
    #define OPENVINO_NEW_ARCH_X(name, val)
    #endif

    // Module type list - maintain only once
    #define GENAI_MODULE_TYPE_LIST \
        X(ParameterModule, 0) \
        X(ImagePreprocessModule, 10) \
        X(VideoPreprocessModule, 13) \
        X(TextEncoderModule, 11) \
        X(RandomLatentImageModule, 12) \
        X(VisionEncoderModule, 20) \
        X(TextEmbeddingModule, 21) \
        X(EmbeddingMergerModule, 22) \
        X(ClipTextEncoderModule, 23) \
        X(FeaturePrunerModule, 30) \
        X(FeatureFusionModule, 31) \
        X(VAEDecoderTilingModule, 32) \
        X(LLMInferenceModule, 40) \
        X(DenoiserLoopModule, 41) \
        X(VAEDecoderModule, 42) \
        OPENVINO_NEW_ARCH_X(LLMInferenceSDPAModule, 43) \
        X(ResultModule, 50) \
        X(SaveImageModule, 51) \
        X(SaveVideoModule, 52) \
        X(Unknown, 99) \
        X(DummyModule, 10000)

    enum class ModuleType : int {
    #define X(name, val) name = val,
        GENAI_MODULE_TYPE_LIST
    #undef X
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