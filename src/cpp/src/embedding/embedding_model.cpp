// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/embedding/embedding_model.hpp"
#include "openvino/runtime/properties.hpp"  // For ov::log::level

#include <cstdlib>  // For setenv/_putenv_s
#if !defined(NDEBUG) || defined(EMBEDDING_DEBUG)
    #define EMBEDDING_LOGGING_ENABLED 1
#else
    #define EMBEDDING_LOGGING_ENABLED 0
#endif
#if EMBEDDING_LOGGING_ENABLED
    #define EMBEDDING_LOG_INFO(msg) \
        std::cout << "[ INFO ] " << msg << std::endl
    #define EMBEDDING_LOG_DEBUG(msg) \
        std::cout << "[ DEBUG ] " << msg << std::endl
    #define EMBEDDING_LOG_WARNING(msg) \
        std::cout << "[ WARNING ] " << msg << std::endl
    #define EMBEDDING_OV_LOG_LEVEL ov::log::Level::WARNING
#else
    #define EMBEDDING_LOG_INFO(msg) ((void)0)
    #define EMBEDDING_LOG_DEBUG(msg) ((void)0)
    #define EMBEDDING_LOG_WARNING(msg) ((void)0)
    #define EMBEDDING_OV_LOG_LEVEL ov::log::Level::ERR
#endif
namespace {
void set_openvino_log_level() {
    static bool initialized = false;
    if (!initialized) {
#if !EMBEDDING_LOGGING_ENABLED
        #ifdef _WIN32
            _putenv_s("OPENVINO_LOG_LEVEL", "LOG_ERROR");
        #else
            setenv("OPENVINO_LOG_LEVEL", "LOG_ERROR", 0);  // 0 = don't overwrite if already set
        #endif
#endif
        initialized = true;
    }
}
}  // namespace
#include <cmath>
#include <fstream>
#include <limits>
#include <algorithm>
#include <cctype>
#include <iostream>

#include "openvino/openvino.hpp"
#include "json_utils.hpp"
#include "utils.hpp"
#include "lora/helper.hpp"

namespace ov {
namespace genai {

// ============================================
// Helper Functions
// ============================================

namespace {

/**
 * @brief Convert string to lowercase for case-insensitive matching
 */
std::string to_lower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return result;
}

/**
 * @brief Check if string contains substring (case-insensitive)
 */
bool contains_ci(const std::string& str, const std::string& substr) {
    return to_lower(str).find(to_lower(substr)) != std::string::npos;
}

/**
 * @brief Apply pooling to hidden states
 */
ov::Tensor apply_pooling(const ov::Tensor& hidden_states,
                         const ov::Tensor& attention_mask,
                         const std::string& pooling_mode) {
    auto hidden_shape = hidden_states.get_shape();
    size_t batch_size = hidden_shape[0];
    size_t seq_length = hidden_shape[1];
    size_t hidden_size = hidden_shape[2];

    const float* hidden_data = hidden_states.data<float>();
    const int64_t* mask_data = attention_mask.data<int64_t>();

    ov::Tensor output(ov::element::f32, ov::Shape{batch_size, hidden_size});
    float* output_data = output.data<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        if (pooling_mode == "cls") {
            // Use [CLS] token (first token)
            for (size_t h = 0; h < hidden_size; ++h) {
                output_data[b * hidden_size + h] = hidden_data[b * seq_length * hidden_size + h];
            }
        } else if (pooling_mode == "mean") {
            // Mean pooling with attention mask
            std::vector<float> sum(hidden_size, 0.0f);
            float count = 0.0f;

            for (size_t s = 0; s < seq_length; ++s) {
                if (mask_data[b * seq_length + s] > 0) {
                    for (size_t h = 0; h < hidden_size; ++h) {
                        sum[h] += hidden_data[b * seq_length * hidden_size + s * hidden_size + h];
                    }
                    count += 1.0f;
                }
            }

            for (size_t h = 0; h < hidden_size; ++h) {
                output_data[b * hidden_size + h] = (count > 0) ? sum[h] / count : 0.0f;
            }
        } else if (pooling_mode == "max") {
            // Max pooling
            std::vector<float> max_vals(hidden_size, -std::numeric_limits<float>::infinity());

            for (size_t s = 0; s < seq_length; ++s) {
                if (mask_data[b * seq_length + s] > 0) {
                    for (size_t h = 0; h < hidden_size; ++h) {
                        float val = hidden_data[b * seq_length * hidden_size + s * hidden_size + h];
                        max_vals[h] = std::max(max_vals[h], val);
                    }
                }
            }

            for (size_t h = 0; h < hidden_size; ++h) {
                output_data[b * hidden_size + h] = max_vals[h];
            }
        } else if (pooling_mode == "last") {
            // Use last token (for decoder-only models)
            size_t last_idx = 0;
            for (size_t s = 0; s < seq_length; ++s) {
                if (mask_data[b * seq_length + s] > 0) {
                    last_idx = s;
                }
            }
            for (size_t h = 0; h < hidden_size; ++h) {
                output_data[b * hidden_size + h] = hidden_data[b * seq_length * hidden_size + last_idx * hidden_size + h];
            }
        }
    }

    return output;
}

/**
 * @brief L2-normalize embeddings
 */
ov::Tensor normalize_embeddings(const ov::Tensor& embeddings) {
    auto shape = embeddings.get_shape();
    size_t batch_size = shape[0];
    size_t hidden_size = shape[1];

    ov::Tensor output(ov::element::f32, shape);
    const float* input_data = embeddings.data<float>();
    float* output_data = output.data<float>();

    for (size_t b = 0; b < batch_size; ++b) {
        float norm = 0.0f;
        for (size_t h = 0; h < hidden_size; ++h) {
            float val = input_data[b * hidden_size + h];
            norm += val * val;
        }
        norm = std::sqrt(norm);

        for (size_t h = 0; h < hidden_size; ++h) {
            output_data[b * hidden_size + h] = (norm > 0) ? input_data[b * hidden_size + h] / norm : 0.0f;
        }
    }

    return output;
}

}  // namespace

// ============================================
// Architecture String Conversion
// ============================================

std::string architecture_to_string(EmbeddingArchitecture arch) {
    static const std::map<EmbeddingArchitecture, std::string> names = {
        {EmbeddingArchitecture::UNKNOWN, "unknown"},
        {EmbeddingArchitecture::BGE, "bge"},
        {EmbeddingArchitecture::BCE, "bce"},
        {EmbeddingArchitecture::GTE, "gte"},
        {EmbeddingArchitecture::E5, "e5"},
        {EmbeddingArchitecture::INSTRUCTOR, "instructor"},
        {EmbeddingArchitecture::STELLA, "stella"},
        {EmbeddingArchitecture::XIAOBU, "xiaobu"},
        {EmbeddingArchitecture::JINA, "jina"},
        {EmbeddingArchitecture::NOMIC, "nomic"},
        {EmbeddingArchitecture::CUSTOM, "custom"}
    };
    auto it = names.find(arch);
    return it != names.end() ? it->second : "unknown";
}

EmbeddingArchitecture architecture_from_string(const std::string& name) {
    static const std::map<std::string, EmbeddingArchitecture> archs = {
        {"unknown", EmbeddingArchitecture::UNKNOWN},
        {"bge", EmbeddingArchitecture::BGE},
        {"bce", EmbeddingArchitecture::BCE},
        {"gte", EmbeddingArchitecture::GTE},
        {"e5", EmbeddingArchitecture::E5},
        {"instructor", EmbeddingArchitecture::INSTRUCTOR},
        {"stella", EmbeddingArchitecture::STELLA},
        {"xiaobu", EmbeddingArchitecture::XIAOBU},
        {"jina", EmbeddingArchitecture::JINA},
        {"nomic", EmbeddingArchitecture::NOMIC},
        {"custom", EmbeddingArchitecture::CUSTOM}
    };
    auto it = archs.find(to_lower(name));
    return it != archs.end() ? it->second : EmbeddingArchitecture::UNKNOWN;
}

// ============================================
// EmbeddingModelConfig Implementation
// ============================================

void EmbeddingModelConfig::apply_architecture_defaults(EmbeddingArchitecture arch) {
    auto& registry = EmbeddingArchitectureRegistry::instance();
    auto arch_config = registry.get_config(arch);

    architecture = arch;

    if (lora_tensor_prefix.empty()) {
        lora_tensor_prefix = arch_config.lora_tensor_prefix;
    }
    if (lora_prefix_fallbacks.empty()) {
        lora_prefix_fallbacks = arch_config.lora_prefix_fallbacks;
    }
    if (pooling_mode.empty() || pooling_mode == "mean") {
        pooling_mode = arch_config.default_pooling_mode;
    }
    if (!query_instruction && arch_config.query_instruction) {
        query_instruction = arch_config.query_instruction;
    }
    if (!document_instruction && arch_config.document_instruction) {
        document_instruction = arch_config.document_instruction;
    }
    normalize_embeddings = arch_config.normalize_embeddings;
}

// ============================================
// EmbeddingModelConfigBuilder Implementation
// ============================================

EmbeddingModelConfigBuilder& EmbeddingModelConfigBuilder::set_architecture(EmbeddingArchitecture arch) {
    m_config.architecture = arch;
    m_config.apply_architecture_defaults(arch);
    return *this;
}

EmbeddingModelConfigBuilder& EmbeddingModelConfigBuilder::set_hidden_size(size_t size) {
    m_config.hidden_size = size;
    return *this;
}

EmbeddingModelConfigBuilder& EmbeddingModelConfigBuilder::set_max_seq_length(size_t length) {
    m_config.max_seq_length = length;
    return *this;
}

EmbeddingModelConfigBuilder& EmbeddingModelConfigBuilder::set_pooling_mode(const std::string& mode) {
    m_config.pooling_mode = mode;
    return *this;
}

EmbeddingModelConfigBuilder& EmbeddingModelConfigBuilder::set_normalize(bool normalize) {
    m_config.normalize_embeddings = normalize;
    return *this;
}

EmbeddingModelConfigBuilder& EmbeddingModelConfigBuilder::set_lora_prefix(const std::string& prefix) {
    m_config.lora_tensor_prefix = prefix;
    return *this;
}

EmbeddingModelConfigBuilder& EmbeddingModelConfigBuilder::add_lora_prefix_fallback(const std::string& prefix) {
    m_config.lora_prefix_fallbacks.push_back(prefix);
    return *this;
}

EmbeddingModelConfigBuilder& EmbeddingModelConfigBuilder::set_query_instruction(const std::string& instruction) {
    m_config.query_instruction = instruction;
    return *this;
}

EmbeddingModelConfigBuilder& EmbeddingModelConfigBuilder::set_document_instruction(const std::string& instruction) {
    m_config.document_instruction = instruction;
    return *this;
}

EmbeddingModelConfig EmbeddingModelConfigBuilder::build() const {
    return m_config;
}

// ============================================
// EmbeddingArchitectureRegistry Implementation
// ============================================

EmbeddingArchitectureRegistry& EmbeddingArchitectureRegistry::instance() {
    static EmbeddingArchitectureRegistry registry;
    return registry;
}

EmbeddingArchitectureRegistry::EmbeddingArchitectureRegistry() {
    initialize_default_configs();
}

void EmbeddingArchitectureRegistry::initialize_default_configs() {
    // ============================================
    // BGE (BAAI General Embedding) Configuration
    // ============================================
    {
        EmbeddingArchitectureConfig config;
        config.architecture = EmbeddingArchitecture::BGE;
        config.lora_tensor_prefix = "base_model.model";
        config.lora_prefix_fallbacks = {"base_model.model.encoder", "bert", "encoder", "model.encoder", ""};
        config.backbone_type = "bert";
        config.has_pooler = true;
        config.default_pooling_mode = "cls";
        config.normalize_embeddings = true;
        config.query_instruction = "Represent this sentence for searching relevant passages: ";
        m_configs[EmbeddingArchitecture::BGE] = config;
    }

    // ============================================
    // BCE (Bilingual/Cross-lingual Embedding) Configuration
    // ============================================
    {
        EmbeddingArchitectureConfig config;
        config.architecture = EmbeddingArchitecture::BCE;
        config.lora_tensor_prefix = "base_model.model";
        config.lora_prefix_fallbacks = {"bert", "encoder", "model", ""};
        config.backbone_type = "bert";
        config.has_pooler = false;
        config.default_pooling_mode = "cls";
        config.normalize_embeddings = true;
        m_configs[EmbeddingArchitecture::BCE] = config;
    }

    // ============================================
    // GTE (General Text Embedding) Configuration
    // ============================================
    {
        EmbeddingArchitectureConfig config;
        config.architecture = EmbeddingArchitecture::GTE;
        config.lora_tensor_prefix = "base_model.model";
        config.lora_prefix_fallbacks = {"bert", "encoder", "model.encoder", ""};
        config.backbone_type = "bert";
        config.has_pooler = true;
        config.default_pooling_mode = "cls";
        config.normalize_embeddings = true;
        m_configs[EmbeddingArchitecture::GTE] = config;
    }

    // ============================================
    // E5 (Microsoft E5) Configuration
    // ============================================
    {
        EmbeddingArchitectureConfig config;
        config.architecture = EmbeddingArchitecture::E5;
        config.lora_tensor_prefix = "base_model.model";
        config.lora_prefix_fallbacks = {"bert", "encoder", "roberta", "xlm-roberta", ""};
        config.backbone_type = "bert";
        config.has_pooler = false;
        config.default_pooling_mode = "mean";
        config.normalize_embeddings = true;
        config.query_instruction = "query: ";
        config.document_instruction = "passage: ";
        m_configs[EmbeddingArchitecture::E5] = config;
    }

    // ============================================
    // Instructor Embedding Configuration
    // ============================================
    {
        EmbeddingArchitectureConfig config;
        config.architecture = EmbeddingArchitecture::INSTRUCTOR;
        config.lora_tensor_prefix = "base_model.model";
        config.lora_prefix_fallbacks = {"encoder", "model.encoder", "t5.encoder", ""};
        config.backbone_type = "t5";
        config.has_pooler = false;
        config.default_pooling_mode = "mean";
        config.normalize_embeddings = true;
        m_configs[EmbeddingArchitecture::INSTRUCTOR] = config;
    }

    // ============================================
    // Stella Embedding Configuration
    // ============================================
    {
        EmbeddingArchitectureConfig config;
        config.architecture = EmbeddingArchitecture::STELLA;
        config.lora_tensor_prefix = "base_model.model";
        config.lora_prefix_fallbacks = {"model", "bert", "encoder", ""};
        config.backbone_type = "bert";
        config.has_pooler = true;
        config.default_pooling_mode = "cls";
        config.normalize_embeddings = true;
        m_configs[EmbeddingArchitecture::STELLA] = config;
    }

    // ============================================
    // Xiaobu Embedding Configuration
    // ============================================
    {
        EmbeddingArchitectureConfig config;
        config.architecture = EmbeddingArchitecture::XIAOBU;
        config.lora_tensor_prefix = "base_model.model";
        config.lora_prefix_fallbacks = {"bert", "model", "encoder", ""};
        config.backbone_type = "bert";
        config.has_pooler = true;
        config.default_pooling_mode = "cls";
        config.normalize_embeddings = true;
        m_configs[EmbeddingArchitecture::XIAOBU] = config;
    }

    // ============================================
    // Jina Embedding Configuration
    // ============================================
    {
        EmbeddingArchitectureConfig config;
        config.architecture = EmbeddingArchitecture::JINA;
        config.lora_tensor_prefix = "base_model.model";
        config.lora_prefix_fallbacks = {"bert", "encoder", "model", ""};
        config.backbone_type = "bert";
        config.has_pooler = false;
        config.default_pooling_mode = "mean";
        config.normalize_embeddings = true;
        m_configs[EmbeddingArchitecture::JINA] = config;
    }

    // ============================================
    // Nomic Embedding Configuration
    // ============================================
    {
        EmbeddingArchitectureConfig config;
        config.architecture = EmbeddingArchitecture::NOMIC;
        config.lora_tensor_prefix = "base_model.model";
        config.lora_prefix_fallbacks = {"model", "bert", "encoder", ""};
        config.backbone_type = "bert";
        config.has_pooler = false;
        config.default_pooling_mode = "mean";
        config.normalize_embeddings = true;
        config.query_instruction = "search_query: ";
        config.document_instruction = "search_document: ";
        m_configs[EmbeddingArchitecture::NOMIC] = config;
    }

    // ============================================
    // Custom/Unknown - Fallback Configuration
    // ============================================
    {
        EmbeddingArchitectureConfig config;
        config.architecture = EmbeddingArchitecture::CUSTOM;
        config.lora_tensor_prefix = "base_model.model";  // Most common for PEFT-trained adapters
        config.lora_prefix_fallbacks = {"bert", "encoder", "model", "base_model.model.encoder", ""};
        config.backbone_type = "bert";
        config.has_pooler = true;
        config.default_pooling_mode = "cls";
        config.normalize_embeddings = true;
        m_configs[EmbeddingArchitecture::CUSTOM] = config;
        m_configs[EmbeddingArchitecture::UNKNOWN] = config;
    }
}

EmbeddingArchitectureConfig EmbeddingArchitectureRegistry::get_default_config() const {
    EmbeddingArchitectureConfig config;
    config.architecture = EmbeddingArchitecture::UNKNOWN;
    config.lora_tensor_prefix = "base_model.model";  // Most common for PEFT-trained adapters
    config.lora_prefix_fallbacks = {"bert", "encoder", "model", ""};
    config.backbone_type = "bert";
    config.has_pooler = true;
    config.default_pooling_mode = "cls";
    config.normalize_embeddings = true;
    return config;
}

EmbeddingArchitectureConfig EmbeddingArchitectureRegistry::get_config(EmbeddingArchitecture arch) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_configs.find(arch);
    if (it != m_configs.end()) {
        return it->second;
    }
    return get_default_config();
}

void EmbeddingArchitectureRegistry::register_architecture(EmbeddingArchitecture arch,
                                                          const EmbeddingArchitectureConfig& config) {
    // Correct usage: lock_guard is not a template, use parentheses and ensure m_mutex is defined as a member variable
    std::lock_guard<std::mutex> lock(m_mutex);
    m_configs[arch] = config;
}

EmbeddingArchitecture EmbeddingArchitectureRegistry::detect_architecture(
    const std::filesystem::path& model_dir) const {

    // Try to read config.json for model identification
    std::filesystem::path config_path = model_dir / "config.json";

    std::string model_id;
    std::string model_type;

    if (std::filesystem::exists(config_path)) {
        try {
            std::ifstream file(config_path);
            nlohmann::json config = nlohmann::json::parse(file);

            // Check _name_or_path field
            if (config.contains("_name_or_path")) {
                model_id = to_lower(config["_name_or_path"].get<std::string>());
            }

            // Check model_type field
            if (config.contains("model_type")) {
                model_type = to_lower(config["model_type"].get<std::string>());
            }

            // Check for explicit architecture hint in config
            if (config.contains("embedding_architecture")) {
                std::string arch_hint = config["embedding_architecture"].get<std::string>();
                return architecture_from_string(arch_hint);
            }

        } catch (const std::exception& e) {
            // JSON parsing failed, continue with directory name detection
        }
    }

    // Detection logic based on model_id
    if (!model_id.empty()) {
        if (contains_ci(model_id, "bge-") || contains_ci(model_id, "baai/bge")) {
            return EmbeddingArchitecture::BGE;
        }
        if (contains_ci(model_id, "bce-") || contains_ci(model_id, "maidalun/bce")) {
            return EmbeddingArchitecture::BCE;
        }
        if (contains_ci(model_id, "gte-") || contains_ci(model_id, "thenlper/gte") ||
            contains_ci(model_id, "alibaba-nlp/gte")) {
            return EmbeddingArchitecture::GTE;
        }
        if (contains_ci(model_id, "e5-") || contains_ci(model_id, "intfloat/e5") ||
            contains_ci(model_id, "multilingual-e5")) {
            return EmbeddingArchitecture::E5;
        }
        if (contains_ci(model_id, "instructor")) {
            return EmbeddingArchitecture::INSTRUCTOR;
        }
        if (contains_ci(model_id, "stella")) {
            return EmbeddingArchitecture::STELLA;
        }
        if (contains_ci(model_id, "xiaobu")) {
            return EmbeddingArchitecture::XIAOBU;
        }
        if (contains_ci(model_id, "jina")) {
            return EmbeddingArchitecture::JINA;
        }
        if (contains_ci(model_id, "nomic")) {
            return EmbeddingArchitecture::NOMIC;
        }
    }

    // Fallback: try to detect from directory name
    std::string dir_name = to_lower(model_dir.filename().string());

    if (contains_ci(dir_name, "bge")) return EmbeddingArchitecture::BGE;
    if (contains_ci(dir_name, "bce")) return EmbeddingArchitecture::BCE;
    if (contains_ci(dir_name, "gte")) return EmbeddingArchitecture::GTE;
    if (contains_ci(dir_name, "e5-") || contains_ci(dir_name, "e5_")) return EmbeddingArchitecture::E5;
    if (contains_ci(dir_name, "instructor")) return EmbeddingArchitecture::INSTRUCTOR;
    if (contains_ci(dir_name, "stella")) return EmbeddingArchitecture::STELLA;
    if (contains_ci(dir_name, "xiaobu")) return EmbeddingArchitecture::XIAOBU;
    if (contains_ci(dir_name, "jina")) return EmbeddingArchitecture::JINA;
    if (contains_ci(dir_name, "nomic")) return EmbeddingArchitecture::NOMIC;

    return EmbeddingArchitecture::UNKNOWN;
}

std::string EmbeddingArchitectureRegistry::get_lora_prefix(EmbeddingArchitecture arch) const {
    return get_config(arch).lora_tensor_prefix;
}

std::vector<EmbeddingArchitecture> EmbeddingArchitectureRegistry::get_registered_architectures() const {
    std::vector<EmbeddingArchitecture> archs;
    for (const auto& [arch, _] : m_configs) {
        archs.push_back(arch);
    }
    return archs;
}

// ============================================
// Config Loading with Architecture Detection
// ============================================

namespace {

EmbeddingModelConfig load_config_with_architecture(const std::filesystem::path& root_dir) {
    EmbeddingModelConfig config;

    // Auto-detect architecture first
    auto& registry = EmbeddingArchitectureRegistry::instance();
    EmbeddingArchitecture detected_arch = registry.detect_architecture(root_dir);

    // Apply architecture-specific defaults
    config.apply_architecture_defaults(detected_arch);
    config.model_name_or_path = root_dir.string();

    // Load config.json to override defaults
    std::filesystem::path config_path = root_dir / "config.json";
    if (std::filesystem::exists(config_path)) {
        try {
            std::ifstream file(config_path);
            nlohmann::json data = nlohmann::json::parse(file);

            if (data.contains("hidden_size")) {
                config.hidden_size = data["hidden_size"].get<size_t>();
            }
            if (data.contains("max_position_embeddings")) {
                config.max_seq_length = data["max_position_embeddings"].get<size_t>();
            }

            // Check for explicit LoRA prefix in config
            if (data.contains("lora_tensor_prefix")) {
                config.lora_tensor_prefix = data["lora_tensor_prefix"].get<std::string>();
            }

            // Check for explicit query/document instructions
            if (data.contains("query_instruction")) {
                config.query_instruction = data["query_instruction"].get<std::string>();
            }
            if (data.contains("document_instruction")) {
                config.document_instruction = data["document_instruction"].get<std::string>();
            }

        } catch (const std::exception& e) {
            // JSON parsing failed, use defaults
        }
    }

    // Load pooling config if exists (sentence-transformers format)
    std::filesystem::path pooling_config_path = root_dir / "1_Pooling" / "config.json";
    if (std::filesystem::exists(pooling_config_path)) {
        try {
            std::ifstream file(pooling_config_path);
            nlohmann::json data = nlohmann::json::parse(file);

            if (data.contains("pooling_mode_mean_tokens") && data["pooling_mode_mean_tokens"].get<bool>()) {
                config.pooling_mode = "mean";
            } else if (data.contains("pooling_mode_cls_token") && data["pooling_mode_cls_token"].get<bool>()) {
                config.pooling_mode = "cls";
            } else if (data.contains("pooling_mode_max_tokens") && data["pooling_mode_max_tokens"].get<bool>()) {
                config.pooling_mode = "max";
            } else if (data.contains("pooling_mode_lasttoken") && data["pooling_mode_lasttoken"].get<bool>()) {
                config.pooling_mode = "last";
            }
        } catch (const std::exception& e) {
            // JSON parsing failed, use default
        }
    }

    return config;
}

/**
 * @brief Try to create AdapterController with specified prefix
 * @return true if controller was created successfully (doesn't guarantee layers matched)
 */
bool try_create_adapter_controller(
    AdapterController& out_controller,
    std::shared_ptr<ov::Model> model,
    AdapterConfig& adapters,
    const std::string& device,
    const std::string& prefix) {
    
    try {
        adapters.set_tensor_name_prefix(prefix);
        out_controller = AdapterController(model, adapters, device);
        return true;
    } catch (const std::exception& e) {
        EMBEDDING_LOG_DEBUG("LoRA prefix \"" << prefix << "\" failed: " << e.what());
        return false;
    }
}
/**
 * @brief Get the best LoRA prefix for the model
 * 
 * This function returns the primary prefix based on architecture.
 * @return The selected prefix string
 */
std::string select_lora_prefix(
    AdapterController& out_controller,
    std::shared_ptr<ov::Model> model,
    AdapterConfig& adapters,
    const std::string& device,
    const std::string& primary_prefix,
    const std::vector<std::string>& fallback_prefixes) {

    std::vector<std::string> all_prefixes;
    all_prefixes.push_back(primary_prefix);
    for (const auto& p : fallback_prefixes) {
        if (std::find(all_prefixes.begin(), all_prefixes.end(), p) == all_prefixes.end()) {
            all_prefixes.push_back(p);
        }
    }

    for (const auto& prefix : all_prefixes) {
        EMBEDDING_LOG_INFO("Trying LoRA prefix: \"" << prefix << "\"");
            // Set the prefix to try

            // Create controller and check match count
        if (try_create_adapter_controller(out_controller, model, adapters, device, prefix)) {

            EMBEDDING_LOG_INFO("Using LoRA prefix: \"" << prefix << "\"");
            return prefix;
        }
    }

    EMBEDDING_LOG_WARNING("All LoRA prefixes failed. Using primary prefix: \"" 
              << primary_prefix << "\"");
    
    adapters.set_tensor_name_prefix(primary_prefix);
    out_controller = AdapterController(model, adapters, device);

    return primary_prefix;
}

}  // namespace

// ============================================
// EmbeddingModelInferenceBase Interface
// ============================================

class EmbeddingModelInferenceBase {
public:
    virtual ~EmbeddingModelInferenceBase() = default;
    virtual void compile(std::shared_ptr<ov::Model> model, const std::string& device, const ov::AnyMap& properties) = 0;
    virtual void set_adapters(AdapterController& adapter_controller, const AdapterConfig& adapters) = 0;
    virtual ov::Tensor infer(const ov::Tensor& input_ids, const ov::Tensor& attention_mask) = 0;
};

// ============================================
// Dynamic Inference Implementation
// ============================================

class EmbeddingModelInferenceDynamic : public EmbeddingModelInferenceBase {
public:
    void compile(std::shared_ptr<ov::Model> model, const std::string& device, const ov::AnyMap& properties) override {
        ov::Core core;
        ov::AnyMap merged_properties = properties;
        if (merged_properties.find(ov::log::level.name()) == merged_properties.end()) {
            merged_properties[ov::log::level.name()] = EMBEDDING_OV_LOG_LEVEL;
        }
        m_compiled_model = core.compile_model(model, device, merged_properties);
        m_request = m_compiled_model.create_infer_request();
    }

    void set_adapters(AdapterController& adapter_controller, const AdapterConfig& adapters) override {
        OPENVINO_ASSERT(m_request, "Embedding model must be compiled first");
        adapter_controller.apply(m_request, adapters);
    }

    ov::Tensor infer(const ov::Tensor& input_ids, const ov::Tensor& attention_mask) override {
        OPENVINO_ASSERT(m_request, "Embedding model must be compiled first");

        m_request.set_tensor("input_ids", input_ids);
        m_request.set_tensor("attention_mask", attention_mask);

        // Check if token_type_ids is required
        try {
            ov::Shape shape = input_ids.get_shape();
            ov::Tensor token_type_ids(ov::element::i64, shape);
            std::fill_n(token_type_ids.data<int64_t>(), token_type_ids.get_size(), 0);
            m_request.set_tensor("token_type_ids", token_type_ids);
        } catch (const std::exception& e) {
            EMBEDDING_LOG_DEBUG("token_type_ids not set: " << e.what());
        } catch (...) {
            EMBEDDING_LOG_DEBUG("token_type_ids not set due to unknown error");
        }

        m_request.infer();

        return m_request.get_output_tensor(0);
    }

private:
    ov::CompiledModel m_compiled_model;
    ov::InferRequest m_request;
};

// ============================================
// EmbeddingModel Implementation
// ============================================

EmbeddingModel::EmbeddingModel(const std::filesystem::path& root_dir) {
    set_openvino_log_level();
    // Load config with architecture auto-detection
    m_config = load_config_with_architecture(root_dir);
    m_tokenizer = Tokenizer(root_dir);

    ov::Core core;
    std::filesystem::path model_path = root_dir / "openvino_model.xml";
    m_model = core.read_model(model_path);
}

EmbeddingModel::EmbeddingModel(const std::filesystem::path& root_dir,
                               const std::string& device,
                               const ov::AnyMap& properties)
    : EmbeddingModel(root_dir) {
    compile(device, properties);
}

EmbeddingModel::EmbeddingModel(const std::string& model,
                               const Tokenizer& tokenizer,
                               const EmbeddingModelConfig& config)
    : m_tokenizer(tokenizer), m_config(config) {
    set_openvino_log_level();
    ov::Core core;
    m_model = core.read_model(model);
}

EmbeddingModel::EmbeddingModel(const EmbeddingModel&) = default;
EmbeddingModel& EmbeddingModel::operator=(const EmbeddingModel&) = default;
EmbeddingModel::EmbeddingModel(EmbeddingModel&&) noexcept = default;
EmbeddingModel& EmbeddingModel::operator=(EmbeddingModel&&) noexcept = default;
EmbeddingModel::~EmbeddingModel() = default;

EmbeddingModel& EmbeddingModel::reshape(int batch_size, int max_seq_length) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot reshape compiled model");

    std::map<std::string, ov::PartialShape> new_shapes;
    for (const auto& input : m_model->inputs()) {
        std::string name = input.get_any_name();
        new_shapes[name] = ov::PartialShape{batch_size, max_seq_length};
    }
    m_model->reshape(new_shapes);

    return *this;
}

EmbeddingModel& EmbeddingModel::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");

    set_openvino_log_level();
    // Extract adapters from properties for Multi-LoRA support
    std::optional<AdapterConfig> adapters;
    auto filtered_properties = extract_adapters_from_properties(properties, &adapters);

    if (adapters) {
        // Check if user explicitly set a prefix
        auto user_prefix = adapters->get_tensor_name_prefix();

        if (user_prefix.has_value()) {
            // User specified a prefix - use it directly
            EMBEDDING_LOG_INFO("Using user-specified LoRA prefix: \"" << *user_prefix << "\"");
            m_adapter_controller = AdapterController(m_model, *adapters, device);
        } else {
            // Auto-detect prefix based on architecture
            EMBEDDING_LOG_INFO("Auto-detecting LoRA prefix for architecture: "
                      << architecture_to_string(m_config.architecture));

            select_lora_prefix(
                m_adapter_controller,
                m_model,
                *adapters,
                device,
                m_config.lora_tensor_prefix,
                m_config.lora_prefix_fallbacks
            );


                // Still create controller with primary prefix for consistent behavior
        }
    }

    m_impl = std::make_shared<EmbeddingModelInferenceDynamic>();
    m_impl->compile(m_model, device, *filtered_properties);
    if (adapters) {
        EMBEDDING_LOG_INFO("Applying LoRA weights to compiled model...");
        m_impl->set_adapters(m_adapter_controller, *adapters);
    }

    // Release the original model
    m_model.reset();

    return *this;
}

void EmbeddingModel::set_adapters(const std::optional<AdapterConfig>& adapters) {
    OPENVINO_ASSERT(m_impl, "Embedding model must be compiled first");
    if (adapters) {
        m_impl->set_adapters(m_adapter_controller, *adapters);
    }
}

ov::Tensor EmbeddingModel::encode(const std::vector<std::string>& texts) {
    OPENVINO_ASSERT(m_impl, "Embedding model must be compiled first");
    OPENVINO_ASSERT(!texts.empty(), "Input texts cannot be empty");

    auto tokenized = m_tokenizer.encode(texts);
    return infer(tokenized.input_ids, tokenized.attention_mask);
}

ov::Tensor EmbeddingModel::encode(const std::string& text) {
    return encode(std::vector<std::string>{text});
}

ov::Tensor EmbeddingModel::encode_query(const std::vector<std::string>& queries) {
    OPENVINO_ASSERT(m_impl, "Embedding model must be compiled first");

    std::vector<std::string> processed;
    processed.reserve(queries.size());

    for (const auto& q : queries) {
        if (m_config.query_instruction) {
            processed.push_back(*m_config.query_instruction + q);
        } else {
            processed.push_back(q);
        }
    }

    return encode(processed);
}

ov::Tensor EmbeddingModel::encode_query(const std::string& query) {
    return encode_query(std::vector<std::string>{query});
}

ov::Tensor EmbeddingModel::encode_document(const std::vector<std::string>& documents) {
    OPENVINO_ASSERT(m_impl, "Embedding model must be compiled first");

    std::vector<std::string> processed;
    processed.reserve(documents.size());

    for (const auto& d : documents) {
        if (m_config.document_instruction) {
            processed.push_back(*m_config.document_instruction + d);
        } else {
            processed.push_back(d);
        }
    }

    return encode(processed);
}

ov::Tensor EmbeddingModel::encode_document(const std::string& document) {
    return encode_document(std::vector<std::string>{document});
}

ov::Tensor EmbeddingModel::infer(const ov::Tensor& input_ids, const ov::Tensor& attention_mask) {
    OPENVINO_ASSERT(m_impl, "Embedding model must be compiled first");

    ov::Tensor hidden_states = m_impl->infer(input_ids, attention_mask);

    // Apply pooling
    ov::Tensor embeddings = apply_pooling(hidden_states, attention_mask, m_config.pooling_mode);

    // Normalize if configured
    if (m_config.normalize_embeddings) {
        embeddings = normalize_embeddings(embeddings);
    }

    return embeddings;
}

EmbeddingModelConfig EmbeddingModel::get_config() const {
    return m_config;
}

Tokenizer EmbeddingModel::get_tokenizer() const {
    return m_tokenizer;
}

EmbeddingArchitecture EmbeddingModel::get_architecture() const {
    return m_config.architecture;
}

}  // namespace genai
}  // namespace ov
