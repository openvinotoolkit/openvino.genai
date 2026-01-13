// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <memory>
#include <string>
#include <vector>
#include <optional>
#include <map>
#include <mutex>

#include "openvino/core/any.hpp"
#include "openvino/runtime/compiled_model.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/tensor.hpp"
#include "openvino/genai/visibility.hpp"
#include "openvino/genai/lora_adapter.hpp"
#include "openvino/genai/tokenizer.hpp"

namespace ov {
namespace genai {

/**
 * @brief Enumeration of supported embedding model architectures
 * 
 * Used for auto-detection of model type and configuration of
 * architecture-specific settings like LoRA tensor prefixes.
 */
enum class EmbeddingArchitecture {
    UNKNOWN,    ///< Unknown or unrecognized architecture
    BGE,        ///< BAAI General Embedding (bge-small, bge-base, bge-large, bge-m3)
    BCE,        ///< Bilingual/Cross-lingual Embedding (bce-embedding-base)
    GTE,        ///< General Text Embedding (gte-small, gte-base, gte-large)
    E5,         ///< Microsoft E5 (e5-small, e5-base, e5-large, multilingual-e5)
    INSTRUCTOR, ///< Instructor embeddings (based on T5)
    STELLA,     ///< Stella embeddings
    XIAOBU,     ///< Xiaobu embeddings
    JINA,       ///< Jina embeddings
    NOMIC,      ///< Nomic embeddings
    CUSTOM      ///< User-defined architecture
};

/**
 * @brief Convert EmbeddingArchitecture enum to string representation
 * @param arch Architecture enum value
 * @return String name of the architecture
 */
OPENVINO_GENAI_EXPORTS std::string architecture_to_string(EmbeddingArchitecture arch);

/**
 * @brief Convert string to EmbeddingArchitecture enum
 * @param name String name of the architecture
 * @return Architecture enum value, or UNKNOWN if not recognized
 */
OPENVINO_GENAI_EXPORTS EmbeddingArchitecture architecture_from_string(const std::string& name);

/**
 * @brief Architecture-specific configuration for LoRA and model behavior
 */
struct OPENVINO_GENAI_EXPORTS EmbeddingArchitectureConfig {
    EmbeddingArchitecture architecture = EmbeddingArchitecture::UNKNOWN;
    
    /// Primary LoRA tensor name prefix for this architecture
    std::string lora_tensor_prefix;
    
    /// Alternative prefixes to try if primary doesn't match
    std::vector<std::string> lora_prefix_fallbacks;
    
    /// Backbone model type (bert, roberta, t5, etc.)
    std::string backbone_type = "bert";
    
    /// Whether the model has a pooler layer
    bool has_pooler = true;
    
    /// Default pooling mode for this architecture
    std::string default_pooling_mode = "cls";
    
    /// Whether to normalize embeddings by default
    bool normalize_embeddings = true;
    
    /// Query instruction prefix for asymmetric retrieval (optional)
    std::optional<std::string> query_instruction;
    
    /// Document instruction prefix for asymmetric retrieval (optional)
    std::optional<std::string> document_instruction;
};

/**
 * @brief Configuration for embedding models
 * 
 * Extended configuration that includes architecture-specific settings
 * for LoRA adapter support and asymmetric retrieval.
 */
struct OPENVINO_GENAI_EXPORTS EmbeddingModelConfig {
    // ============================================
    // Basic configuration (backward compatible)
    // ============================================
    
    /// Hidden size of the embedding vectors
    size_t hidden_size = 768;
    
    /// Maximum sequence length
    size_t max_seq_length = 512;
    
    /// Pooling mode: "mean", "cls", or "max"
    std::string pooling_mode = "mean";
    
    /// Whether to L2-normalize output embeddings
    bool normalize_embeddings = true;
    
    // ============================================
    // Architecture-specific configuration (new)
    // ============================================
    
    /// Detected or specified model architecture
    EmbeddingArchitecture architecture = EmbeddingArchitecture::UNKNOWN;
    
    /// Original model name or path (for detection)
    std::string model_name_or_path;
    
    /// Primary LoRA tensor name prefix
    std::string lora_tensor_prefix;
    
    /// Fallback prefixes to try for LoRA matching
    std::vector<std::string> lora_prefix_fallbacks;
    
    /// Query instruction for asymmetric retrieval
    std::optional<std::string> query_instruction;
    
    /// Document instruction for asymmetric retrieval
    std::optional<std::string> document_instruction;
    
    /**
     * @brief Apply architecture-specific default settings
     * @param arch Architecture to apply defaults from
     * 
     * This method sets lora_tensor_prefix, lora_prefix_fallbacks,
     * pooling_mode, and instructions based on the architecture.
     */
    void apply_architecture_defaults(EmbeddingArchitecture arch);
};

/**
 * @brief Builder pattern for EmbeddingModelConfig
 * 
 * Provides a fluent interface for constructing configurations:
 * @code
 * auto config = EmbeddingModelConfigBuilder()
 *     .set_architecture(EmbeddingArchitecture::BGE)
 *     .set_hidden_size(384)
 *     .set_lora_prefix("bert")
 *     .build();
 * @endcode
 */
class OPENVINO_GENAI_EXPORTS EmbeddingModelConfigBuilder {
public:
    EmbeddingModelConfigBuilder& set_architecture(EmbeddingArchitecture arch);
    EmbeddingModelConfigBuilder& set_hidden_size(size_t size);
    EmbeddingModelConfigBuilder& set_max_seq_length(size_t length);
    EmbeddingModelConfigBuilder& set_pooling_mode(const std::string& mode);
    EmbeddingModelConfigBuilder& set_normalize(bool normalize);
    EmbeddingModelConfigBuilder& set_lora_prefix(const std::string& prefix);
    EmbeddingModelConfigBuilder& add_lora_prefix_fallback(const std::string& prefix);
    EmbeddingModelConfigBuilder& set_query_instruction(const std::string& instruction);
    EmbeddingModelConfigBuilder& set_document_instruction(const std::string& instruction);
    
    EmbeddingModelConfig build() const;

private:
    EmbeddingModelConfig m_config;
};

/**
 * @brief Registry for embedding model architectures
 * 
 * Singleton registry that stores configurations for known embedding
 * model architectures and provides auto-detection capabilities.
 * 
 * @code
 * auto& registry = EmbeddingArchitectureRegistry::instance();
 * auto config = registry.get_config(EmbeddingArchitecture::BGE);
 * auto detected = registry.detect_architecture("path/to/model");
 * @endcode
 */
class OPENVINO_GENAI_EXPORTS EmbeddingArchitectureRegistry {
public:
    /**
     * @brief Get singleton instance
     */
    static EmbeddingArchitectureRegistry& instance();
    
    /**
     * @brief Get configuration for a known architecture
     * @param arch Architecture enum value
     * @return Configuration for the architecture
     */
    EmbeddingArchitectureConfig get_config(EmbeddingArchitecture arch) const;
    
    /**
     * @brief Register a custom architecture configuration
     * @param arch Architecture enum (typically CUSTOM)
     * @param config Configuration to register
     */
    void register_architecture(EmbeddingArchitecture arch, const EmbeddingArchitectureConfig& config);
    
    /**
     * @brief Auto-detect architecture from model directory
     * @param model_dir Path to model directory
     * @return Detected architecture, or UNKNOWN if not recognized
     * 
     * Detection is based on:
     * 1. config.json "_name_or_path" field
     * 2. config.json "model_type" field
     * 3. Directory name patterns
     */
    EmbeddingArchitecture detect_architecture(const std::filesystem::path& model_dir) const;
    
    /**
     * @brief Get the recommended LoRA prefix for an architecture
     * @param arch Architecture enum value
     * @return Primary LoRA tensor prefix
     */
    std::string get_lora_prefix(EmbeddingArchitecture arch) const;
    
    /**
     * @brief Get all registered architectures
     * @return Vector of registered architecture enums
     */
    std::vector<EmbeddingArchitecture> get_registered_architectures() const;

private:
    EmbeddingArchitectureRegistry();
    void initialize_default_configs();
    EmbeddingArchitectureConfig get_default_config() const;
    
    std::map<EmbeddingArchitecture, EmbeddingArchitectureConfig> m_configs;
    mutable std::mutex m_mutex;
};

// Forward declaration
class EmbeddingModelInferenceBase;

/**
 * @brief Embedding model for generating text embeddings using OpenVINO
 * 
 * This class provides a high-level API for loading and running embedding models
 * with support for:
 * - Multiple embedding model architectures (BGE, BCE, GTE, E5, etc.)
 * - LoRA adapters with automatic tensor prefix detection
 * - Asymmetric query/document encoding for retrieval
 * - Configurable pooling strategies
 * 
 * Basic usage:
 * @code
 * EmbeddingModel model("path/to/bge-small-en");
 * model.compile("CPU");
 * 
 * auto embeddings = model.encode({"Hello world", "Another text"});
 * @endcode
 * 
 * With LoRA adapter:
 * @code
 * Adapter lora("domain_adapter.safetensors");
 * AdapterConfig config;
 * config.add(lora, 1.0f);
 * // Note: prefix is auto-detected based on model architecture
 * 
 * EmbeddingModel model("path/to/model");
 * model.compile("CPU", adapters(config));
 * @endcode
 */
class OPENVINO_GENAI_EXPORTS EmbeddingModel {
public:
    /**
     * @brief Construct embedding model from directory
     * @param root_dir Path to model directory containing openvino_model.xml and tokenizer
     * 
     * The model architecture is auto-detected from config.json.
     */
    EmbeddingModel(const std::filesystem::path& root_dir);
    
    /**
     * @brief Construct and compile embedding model
     * @param root_dir Path to model directory
     * @param device Target device (CPU, GPU, NPU)
     * @param properties Optional compilation properties including LoRA adapters
     */
    EmbeddingModel(const std::filesystem::path& root_dir,
                   const std::string& device,
                   const ov::AnyMap& properties = {});

    /**
     * @brief Construct embedding model from components
     * @param model Path to OpenVINO IR model file
     * @param tokenizer Pre-configured tokenizer
     * @param config Model configuration
     */
    EmbeddingModel(const std::string& model,
                   const Tokenizer& tokenizer,
                   const EmbeddingModelConfig& config);

    EmbeddingModel(const EmbeddingModel&);
    EmbeddingModel& operator=(const EmbeddingModel&);
    EmbeddingModel(EmbeddingModel&&) noexcept;
    EmbeddingModel& operator=(EmbeddingModel&&) noexcept;
    ~EmbeddingModel();

    /**
     * @brief Reshape model for specific batch size and sequence length
     * @param batch_size Fixed batch size
     * @param max_seq_length Maximum sequence length
     * @return Reference to this model for chaining
     * 
     * Must be called before compile(). Cannot reshape after compilation.
     */
    EmbeddingModel& reshape(int batch_size, int max_seq_length);

    /**
     * @brief Compile the model for inference
     * @param device Target device (CPU, GPU, NPU)
     * @param properties Compilation properties, may include LoRA adapters
     * @return Reference to this model for chaining
     * 
     * If LoRA adapters are provided without explicit tensor prefix,
     * the prefix is auto-detected based on model architecture.
     */
    EmbeddingModel& compile(const std::string& device, const ov::AnyMap& properties = {});

    /**
     * @brief Compile with variadic properties
     */
    template <typename... Properties>
    ov::util::EnableIfAllStringAny<EmbeddingModel&, Properties...> compile(
            const std::string& device,
            Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    /**
     * @brief Set or update LoRA adapters
     * @param adapters Optional adapter configuration for Multi-LoRA support
     * 
     * Can be called after compile() to switch adapters at runtime.
     */
    void set_adapters(const std::optional<AdapterConfig>& adapters);

    /**
     * @brief Generate embeddings for input texts
     * @param texts Vector of input texts to embed
     * @return Tensor containing embeddings with shape [batch_size, hidden_size]
     */
    ov::Tensor encode(const std::vector<std::string>& texts);

    /**
     * @brief Generate embedding for a single text
     * @param text Input text to embed
     * @return Tensor containing embedding with shape [1, hidden_size]
     */
    ov::Tensor encode(const std::string& text);

    /**
     * @brief Generate embeddings for queries (asymmetric retrieval)
     * @param queries Vector of query texts
     * @return Tensor containing query embeddings
     * 
     * Automatically prepends query_instruction if configured for the architecture.
     * For BGE models, this adds "Represent this sentence for searching relevant passages: "
     */
    ov::Tensor encode_query(const std::vector<std::string>& queries);

    /**
     * @brief Generate embedding for a single query
     * @param query Query text
     * @return Tensor containing query embedding with shape [1, hidden_size]
     */
    ov::Tensor encode_query(const std::string& query);

    /**
     * @brief Generate embeddings for documents (asymmetric retrieval)
     * @param documents Vector of document texts
     * @return Tensor containing document embeddings
     * 
     * Automatically prepends document_instruction if configured.
     */
    ov::Tensor encode_document(const std::vector<std::string>& documents);

    /**
     * @brief Generate embedding for a single document
     * @param document Document text
     * @return Tensor containing document embedding with shape [1, hidden_size]
     */
    ov::Tensor encode_document(const std::string& document);

    /**
     * @brief Generate embeddings from pre-tokenized inputs
     * @param input_ids Input token IDs tensor
     * @param attention_mask Attention mask tensor
     * @return Tensor containing embeddings
     */
    ov::Tensor infer(const ov::Tensor& input_ids, const ov::Tensor& attention_mask);

    /**
     * @brief Get model configuration
     * @return Current model configuration
     */
    EmbeddingModelConfig get_config() const;

    /**
     * @brief Get the tokenizer
     * @return Tokenizer instance
     */
    Tokenizer get_tokenizer() const;

    /**
     * @brief Get the detected/configured architecture
     * @return Architecture enum value
     */
    EmbeddingArchitecture get_architecture() const;

private:
    std::shared_ptr<EmbeddingModelInferenceBase> m_impl;
    std::shared_ptr<ov::Model> m_model;
    Tokenizer m_tokenizer;
    EmbeddingModelConfig m_config;
    AdapterController m_adapter_controller;
};

}  // namespace genai
}  // namespace ov
