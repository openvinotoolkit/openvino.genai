// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <yaml-cpp/yaml.h>
#include <mutex>
#include <unordered_map>
#include <fstream>
#include <filesystem>

#include "module_genai/module.hpp"
#include "module_genai/module_type.hpp"
#include "module_genai/transformer_config.hpp"
#include "openvino/genai/image_generation/generation_config.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "tokenizer/tokenizer_impl.hpp"


namespace ov {
namespace genai {
namespace module {

// Shared resources for ClipTextEncoderModule with the same model_path
struct ClipTextEncoderSharedResources {
    ov::CompiledModel compiled_model;
    std::shared_ptr<Tokenizer::TokenizerImpl> tokenizer_impl;
    std::shared_ptr<minja::chat_template> minja_template;
    TransformerConfig encoder_config;
    DiffusionModelType model_type;
};

// Compute cache key hash from model XML file content and device
inline std::size_t compute_cache_key(const std::filesystem::path& model_xml_path, const std::string& device) {
    // Read model XML file content
    std::ifstream file(model_xml_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        // Fallback to path-based hash if file cannot be read
        std::string fallback = model_xml_path.string() + device;
        return std::hash<std::string_view>{}(std::string_view(fallback.data(), fallback.size()));
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::string content(size, '\0');
    file.read(content.data(), size);
    content += device;  // Append device to ensure different devices have different keys

    return std::hash<std::string_view>{}(std::string_view(content.data(), content.size()));
}

// Global cache for shared resources
class ClipTextEncoderResourceCache {
public:
    static ClipTextEncoderResourceCache& instance() {
        static ClipTextEncoderResourceCache cache;
        return cache;
    }

    std::shared_ptr<ClipTextEncoderSharedResources> get_or_create(
        std::size_t cache_key,
        std::function<std::shared_ptr<ClipTextEncoderSharedResources>()> creator) {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_cache.find(cache_key);
        if (it != m_cache.end()) {
            return it->second;
        }
        auto resources = creator();
        m_cache[cache_key] = resources;
        return resources;
    }

    bool exists(std::size_t cache_key) {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_cache.find(cache_key) != m_cache.end();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_cache.clear();
    }

private:
    ClipTextEncoderResourceCache() = default;
    std::mutex m_mutex;
    std::unordered_map<std::size_t, std::shared_ptr<ClipTextEncoderSharedResources>> m_cache;
};

class ClipTextEncoderModule : public IBaseModule {
    DeclareModuleConstructor(ClipTextEncoderModule);

private:
    bool initialize();
    bool do_classifier_free_guidance(float guidance_scale);

    // Shared resources (shared across modules with same model_path)
    std::shared_ptr<ClipTextEncoderSharedResources> m_shared_resources;

    // Per-instance infer request (each module has its own)
    ov::InferRequest m_request;

    std::pair<ov::Tensor, ov::Tensor> run(
        const std::vector<std::string>& prompts,
        const std::vector<std::string>& negative_prompts,
        const ImageGenerationConfig &generation_config);
    ov::Tensor encode_prompt(const std::vector<std::string>& prompts, const ImageGenerationConfig &generation_config);
};

REGISTER_MODULE_CONFIG(ClipTextEncoderModule);

}  // namespace module
}  // namespace genai
}  // namespace ov
