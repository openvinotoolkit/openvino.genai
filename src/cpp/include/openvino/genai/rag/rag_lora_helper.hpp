// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include "openvino/genai/lora_adapter.hpp"
#include "openvino/openvino.hpp"

namespace ov {
namespace genai {
namespace rag {

/**
 * @brief Known embedding/reranking model architecture types for LoRA prefix detection
 */
enum class RagModelArchitecture {
    UNKNOWN,
    BGE,        // BAAI General Embedding / Reranker
    BCE,        // Bilingual/Cross-lingual Embedding
    GTE,        // General Text Embedding
    E5,         // Microsoft E5
    JINA,       // Jina embeddings
    NOMIC,      // Nomic embeddings
    INSTRUCTOR, // Instructor embeddings
    STELLA,     // Stella embeddings
    QWEN3,      // Qwen3 reranker
    GENERIC     // Generic BERT-based model
};

/**
 * @brief LoRA configuration for a specific architecture
 */
struct ArchitectureLoraConfig {
    std::string primary_prefix;
    std::vector<std::string> fallback_prefixes;
};

namespace {

inline std::string to_lower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

inline bool contains_ci(const std::string& str, const std::string& substr) {
    return to_lower(str).find(to_lower(substr)) != std::string::npos;
}

/**
 * @brief Get LoRA configuration for a specific architecture
 */
inline ArchitectureLoraConfig get_lora_config(RagModelArchitecture arch) {
    switch (arch) {
        case RagModelArchitecture::BGE:
            return {"base_model.model", {"base_model.model.encoder", "bert", "encoder", "model.encoder", ""}};
        case RagModelArchitecture::BCE:
            return {"base_model.model", {"bert", "encoder", "model", ""}};
        case RagModelArchitecture::GTE:
            return {"base_model.model", {"bert", "encoder", "model.encoder", ""}};
        case RagModelArchitecture::E5:
            return {"base_model.model", {"bert", "encoder", "roberta", "xlm-roberta", ""}};
        case RagModelArchitecture::JINA:
            return {"base_model.model", {"bert", "encoder", "model", ""}};
        case RagModelArchitecture::NOMIC:
            return {"base_model.model", {"model", "bert", "encoder", ""}};
        case RagModelArchitecture::INSTRUCTOR:
            return {"base_model.model", {"encoder", "model.encoder", "t5.encoder", ""}};
        case RagModelArchitecture::STELLA:
            return {"base_model.model", {"model", "bert", "encoder", ""}};
        case RagModelArchitecture::QWEN3:
            return {"base_model.model", {"model", "transformer", ""}};
        case RagModelArchitecture::GENERIC:
        case RagModelArchitecture::UNKNOWN:
        default:
            return {"base_model.model", {"bert", "encoder", "model", "base_model.model.encoder", ""}};
    }
}

/**
 * @brief Detect model architecture from config.json
 */
inline RagModelArchitecture detect_architecture(const std::filesystem::path& model_dir) {
    std::filesystem::path config_path = model_dir / "config.json";

    std::string model_id;
    std::string model_type;

    if (std::filesystem::exists(config_path)) {
        try {
            std::ifstream file(config_path);
            nlohmann::json config = nlohmann::json::parse(file);

            if (config.contains("_name_or_path")) {
                model_id = to_lower(config["_name_or_path"].get<std::string>());
            }

            if (config.contains("model_type")) {
                model_type = to_lower(config["model_type"].get<std::string>());
            }
        } catch (...) {
            // JSON parsing failed, continue with directory name detection
        }
    }

    // Detection based on model_id
    if (!model_id.empty()) {
        if (contains_ci(model_id, "bge-") || contains_ci(model_id, "baai/bge")) {
            return RagModelArchitecture::BGE;
        }
        if (contains_ci(model_id, "bce-") || contains_ci(model_id, "maidalun/bce") ||
            contains_ci(model_id, "netease")) {
            return RagModelArchitecture::BCE;
        }
        if (contains_ci(model_id, "gte-") || contains_ci(model_id, "thenlper/gte") ||
            contains_ci(model_id, "alibaba-nlp/gte")) {
            return RagModelArchitecture::GTE;
        }
        if (contains_ci(model_id, "e5-") || contains_ci(model_id, "intfloat/e5") ||
            contains_ci(model_id, "multilingual-e5")) {
            return RagModelArchitecture::E5;
        }
        if (contains_ci(model_id, "jina")) {
            return RagModelArchitecture::JINA;
        }
        if (contains_ci(model_id, "nomic")) {
            return RagModelArchitecture::NOMIC;
        }
        if (contains_ci(model_id, "instructor")) {
            return RagModelArchitecture::INSTRUCTOR;
        }
        if (contains_ci(model_id, "stella")) {
            return RagModelArchitecture::STELLA;
        }
    }

    // Detection based on model_type
    if (!model_type.empty()) {
        if (model_type == "qwen3") {
            return RagModelArchitecture::QWEN3;
        }
    }

    // Fallback: try directory name
    std::string dir_name = to_lower(model_dir.filename().string());

    if (contains_ci(dir_name, "bge")) return RagModelArchitecture::BGE;
    if (contains_ci(dir_name, "bce")) return RagModelArchitecture::BCE;
    if (contains_ci(dir_name, "gte")) return RagModelArchitecture::GTE;
    if (contains_ci(dir_name, "e5-") || contains_ci(dir_name, "e5_")) return RagModelArchitecture::E5;
    if (contains_ci(dir_name, "jina")) return RagModelArchitecture::JINA;
    if (contains_ci(dir_name, "nomic")) return RagModelArchitecture::NOMIC;
    if (contains_ci(dir_name, "instructor")) return RagModelArchitecture::INSTRUCTOR;
    if (contains_ci(dir_name, "stella")) return RagModelArchitecture::STELLA;
    if (contains_ci(dir_name, "qwen3")) return RagModelArchitecture::QWEN3;

    return RagModelArchitecture::UNKNOWN;
}
   
}  // anonymous namespace

/**
 * @brief Detect appropriate LoRA tensor prefix based on model architecture
 * @param model_dir Path to model directory containing config.json
 * @return Recommended LoRA tensor prefix
 */
inline std::string detect_lora_prefix(const std::filesystem::path& model_dir) {
    auto arch = detect_architecture(model_dir);
    auto config = get_lora_config(arch);
    return config.primary_prefix;
}

/**
 * @brief Get fallback LoRA prefixes for a model
 * @param model_dir Path to model directory
 * @return Vector of fallback prefixes to try if primary doesn't match
 */
inline std::vector<std::string> get_lora_prefix_fallbacks(const std::filesystem::path& model_dir) {
    auto arch = detect_architecture(model_dir);
    auto config = get_lora_config(arch);
    return config.fallback_prefixes;
}


/**
 * @brief Apply LoRA weights to an inference request
 *
 * @param controller The AdapterController managing LoRA state
 * @param request The inference request to apply weights to
 * @param adapters Adapter configuration with weights and alpha values
 */
inline void apply_lora_weights(AdapterController& controller,
                               ov::InferRequest& request,
                               const AdapterConfig& adapters) {
    controller.apply(request, adapters);
}

/**
 * @brief Try multiple LoRA prefixes until one works
 *
 * This is useful when the auto-detected prefix doesn't match the actual
 * tensor naming in the LoRA adapter file.
 *
 * @param model The OpenVINO model
 * @param adapters Adapter configuration
 * @param model_dir Path to model directory
 * @param device Target device
 * @param primary_prefix Primary prefix to try first
 * @param fallbacks List of fallback prefixes
 * @return Configured AdapterController with working prefix
 * @throws std::runtime_error if no prefix works
 */
inline AdapterController setup_lora_with_fallbacks(
    std::shared_ptr<ov::Model>& model,
    AdapterConfig& adapters,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const std::string& primary_prefix,
    const std::vector<std::string>& fallbacks) {

    // Build list of unique prefixes to try
    std::vector<std::string> prefixes_to_try = {primary_prefix};
    for (const auto& fallback : fallbacks) {
        if (std::find(prefixes_to_try.begin(), prefixes_to_try.end(), fallback) == prefixes_to_try.end()) {
            prefixes_to_try.push_back(fallback);
        }
    }

    std::string last_error;

    for (const auto& prefix : prefixes_to_try) {
        try {
            // Set the prefix and try to create controller
            adapters.set_tensor_name_prefix(prefix);

            AdapterController controller(model, adapters, device);

            // If we get here, it worked
            return controller;

        } catch (const std::exception& e) {
            last_error = e.what();
            // Continue to next prefix
        }
    }

    // None of the prefixes worked
    OPENVINO_THROW("Failed to setup LoRA adapters. Tried prefixes: ",
                   primary_prefix, " and ", fallbacks.size(), " fallbacks. ",
                   "Last error: ", last_error);
}


}  // namespace rag
}  // namespace genai
}  // namespace ov
