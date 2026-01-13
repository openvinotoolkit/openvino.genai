// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

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
 * @brief Supported reranking model variants
 */
enum class RerankVariant {
    AUTO,   ///< Auto-detect from model config
    BGE,    ///< BGE reranker (BAAI)
    BCE,    ///< BCE reranker (NetEase)
    GENERIC ///< Generic cross-encoder (default fallback)
};

/**
 * @brief Convert string to RerankVariant (case-insensitive)
 * @param variant_str String representation ("bge", "bce", "auto", "generic")
 * @return Corresponding RerankVariant enum value
 * @throws std::invalid_argument if variant string is unknown
 */
OPENVINO_GENAI_EXPORTS RerankVariant parse_rerank_variant(const std::string& variant_str);
/**
 * @brief Get string representation of RerankVariant
 * @param variant The variant enum value
 * @return String representation
 */
OPENVINO_GENAI_EXPORTS std::string to_string(RerankVariant variant);
/**
 * @brief Get list of all supported variant names
 * @return Vector of supported variant name strings
 */
OPENVINO_GENAI_EXPORTS std::vector<std::string> get_supported_rerank_variants();
/**
 * @brief Score normalization method
 */
enum class ScoreNormalization {
    NONE,    ///< Raw scores (no normalization)
    SIGMOID, ///< Sigmoid normalization (0-1 range)
    SOFTMAX, ///< Softmax across batch
    MINMAX   ///< Min-max normalization to 0-1 range
};
/**
 * @brief Pooling strategy for extracting embeddings
 */
enum class PoolingStrategy {
    CLS,     ///< Use [CLS] token embedding
    MEAN,    ///< Mean pooling over all tokens
    LAST,    ///< Use last token embedding
    NONE     ///< No pooling (use raw output)
};
/**
 * @brief Configuration for reranking model
 */
struct RerankingModelConfig {
    RerankVariant variant = RerankVariant::AUTO;
    size_t max_seq_length = 512;
    bool return_scores = true;
    bool normalize_scores = false;
    ScoreNormalization normalization_method = ScoreNormalization::SIGMOID;
    PoolingStrategy pooling = PoolingStrategy::NONE;
    // Variant-specific overrides (empty means use variant defaults)
    std::string query_instruction;
    std::string document_instruction;
    std::string separator = " [SEP] ";
};

/**
 * @brief Result structure for reranking operations
 */
struct RerankingResult {
    std::vector<float> scores;
    std::vector<size_t> sorted_indices;  ///< Indices sorted by score (descending)
    /**
     * @brief Get top-k document indices
     * @param k Number of top results (0 = all)
     * @return Vector of indices for top-k documents
     */
    std::vector<size_t> get_top_k(size_t k = 0) const {
        if (k == 0 || k >= sorted_indices.size()) {
            return sorted_indices;
        }
        return std::vector<size_t>(sorted_indices.begin(), sorted_indices.begin() + k);
    }
};
// Forward declarations
class RerankingModelInferenceBase;
class IRerankingAdapter;

/**
 * @brief Optional logging callback type
 * 
 * Users can set this to receive diagnostic information about
 * reranking operations (variant selection, config, etc.)
 */
using RerankingLogCallback = std::function<void(const std::string& message)>;
/**
 * @brief Cross-encoder reranking model with multi-variant support
 * 
 * Supports multiple reranking model variants (BGE, BCE, etc.) under a unified API.
 * The variant can be auto-detected from model config or explicitly specified.
 * 
 * Example usage:
 * @code
 * // Auto-detect variant
 * RerankingModel model(model_path, "CPU");
 * auto result = model.rerank("query", {"doc1", "doc2", "doc3"});
 * 
 * // Explicit variant selection
 * RerankingModelConfig config;
 * config.variant = RerankVariant::BCE;
 * RerankingModel model(model_path, "CPU", {}, config);
 * @endcode
 */
class OPENVINO_GENAI_EXPORTS RerankingModel {
public:
    /**
     * @brief Construct reranking model from directory (load only, no compilation)
     * @param root_dir Path to model directory containing openvino_model.xml and tokenizer
     */
    RerankingModel(const std::filesystem::path& root_dir);
    
    /**
     * @brief Construct and compile reranking model
     * @param root_dir Path to model directory
     * @param device Target device (CPU, GPU, NPU)
     * @param properties OpenVINO compilation properties
     */
    RerankingModel(const std::filesystem::path& root_dir,
                   const std::string& device,
                   const ov::AnyMap& properties = {});

    /**
     * @brief Construct and compile reranking model with explicit config
     * @param root_dir Path to model directory
     * @param device Target device
     * @param properties OpenVINO compilation properties
     * @param config Reranking model configuration (including variant selection)
     */
    RerankingModel(const std::filesystem::path& root_dir,
                   const std::string& device,
                   const ov::AnyMap& properties,
                   const RerankingModelConfig& config);
    /**
     * @brief Construct from model string and tokenizer
     * @param model Model XML string or path
     * @param tokenizer Tokenizer instance
     * @param config Model configuration
     */
    RerankingModel(const std::string& model,
                   const Tokenizer& tokenizer,
                   const RerankingModelConfig& config);

    RerankingModel(const RerankingModel&);
    
    RerankingModel& operator=(const RerankingModel&);
    
    RerankingModel(RerankingModel&&) noexcept;
    
    RerankingModel& operator=(RerankingModel&&) noexcept;
    
    ~RerankingModel();

    /**
     * @brief Reshape model for specific batch size and sequence length
     * @param batch_size Fixed batch size
     * @param max_seq_length Maximum sequence length
     * @return Reference to this model for chaining
     */
    RerankingModel& reshape(int batch_size, int max_seq_length);

    /**
     * @brief Compile model for target device
     * @param device Target device string
     * @param properties Compilation properties
     * @return Reference to this model for chaining
     */
    RerankingModel& compile(const std::string& device, const ov::AnyMap& properties = {});

    /**
     * @brief Compile model with variadic properties
     */
    template <typename... Properties>
    ov::util::EnableIfAllStringAny<RerankingModel&, Properties...> compile(
            const std::string& device,
            Properties&&... properties) {
        return compile(device, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    /**
     * @brief Set LoRA adapters for the reranking model
     * @param adapters Optional adapter configuration for Multi-LoRA support
     */
    void set_adapters(const std::optional<AdapterConfig>& adapters);

    /**
     * @brief Rerank documents based on relevance to query
     * @param query The query text
     * @param documents Vector of document texts to rerank
     * @param top_k Number of top results to return (0 = all)
     * @return RerankingResult containing scores and sorted indices
     */
    RerankingResult rerank(const std::string& query, 
                           const std::vector<std::string>& documents,
                           size_t top_k = 0);

    /**
     * @brief Compute relevance score for a query-document pair
     * @param query The query text
     * @param document The document text
     * @return Relevance score
     */
    float score(const std::string& query, const std::string& document);

    /**
     * @brief Compute relevance scores from pre-tokenized inputs
     * @param input_ids Input token IDs [batch_size, seq_length]
     * @param attention_mask Attention mask [batch_size, seq_length]
     * @return Tensor containing scores [batch_size] or [batch_size, num_labels]
     */
    ov::Tensor infer(const ov::Tensor& input_ids, const ov::Tensor& attention_mask);

    /**
     * @brief Get current configuration
     * @return Copy of current config
     */
    RerankingModelConfig get_config() const;

    /**
     * @brief Get tokenizer instance
     * @return Copy of tokenizer
     */
    Tokenizer get_tokenizer() const;

    /**
     * @brief Get the active reranking variant
     * @return The resolved variant (never AUTO after initialization)
     */
    RerankVariant get_variant() const;
    /**
     * @brief Set logging callback for diagnostic output
     * @param callback Function to receive log messages (nullptr to disable)
     */
    void set_log_callback(RerankingLogCallback callback);
    /**
     * @brief Get variant-specific adapter (for advanced usage)
     * @return Shared pointer to the adapter interface
     */
    std::shared_ptr<IRerankingAdapter> get_adapter() const;
private:
    void initialize_adapter(const std::filesystem::path& root_dir);
    void log_message(const std::string& message) const;
    void validate_output_shape(const ov::Tensor& output, size_t expected_batch) const;
    std::shared_ptr<RerankingModelInferenceBase> m_impl;
    std::shared_ptr<IRerankingAdapter> m_adapter;
    std::shared_ptr<ov::Model> m_model;
    Tokenizer m_tokenizer;
    RerankingModelConfig m_config;
    AdapterController m_adapter_controller;
    RerankingLogCallback m_log_callback;
};

/**
 * @brief Abstract interface for reranking variant adapters
 * 
 * Implement this interface to add support for new reranking model variants.
 * Each adapter encapsulates variant-specific logic for:
 * - Input formatting (query-document pair construction)
 * - Output parsing (score extraction and normalization)
 * - Default configuration values
 */
class OPENVINO_GENAI_EXPORTS IRerankingAdapter {
public:
    virtual ~IRerankingAdapter() = default;
    /**
     * @brief Get the variant type this adapter handles
     * @return The RerankVariant enum value
     */
    virtual RerankVariant get_variant() const = 0;
    /**
     * @brief Get human-readable variant name
     * @return Variant name string (e.g., "BGE Reranker")
     */
    virtual std::string get_variant_name() const = 0;
    /**
     * @brief Format a query-document pair for tokenization
     * @param query The query text
     * @param document The document text  
     * @param config Current model config (may contain overrides)
     * @return Formatted string ready for tokenization
     */
    virtual std::string format_input(const std::string& query,
                                     const std::string& document,
                                     const RerankingModelConfig& config) const = 0;
    /**
     * @brief Parse model output tensor to extract scores
     * @param output Raw model output tensor
     * @param batch_size Number of query-document pairs
     * @param config Current model config
     * @return Vector of relevance scores
     */
    virtual std::vector<float> parse_output(const ov::Tensor& output,
                                            size_t batch_size,
                                            const RerankingModelConfig& config) const = 0;
    /**
     * @brief Get default max sequence length for this variant
     * @return Default max_seq_length value
     */
    virtual size_t get_default_max_seq_length() const = 0;
    /**
     * @brief Get default score normalization setting
     * @return True if scores should be normalized by default
     */
    virtual bool default_normalize_scores() const = 0;
    /**
     * @brief Get default normalization method
     * @return Default ScoreNormalization enum value
     */
    virtual ScoreNormalization default_normalization_method() const = 0;
    /**
     * @brief Get default pooling strategy
     * @return Default PoolingStrategy enum value
     */
    virtual PoolingStrategy default_pooling() const = 0;
    /**
     * @brief Validate that model output shape is compatible with this variant
     * @param output_shape Shape of the model output tensor
     * @param batch_size Expected batch size
     * @throws std::runtime_error if shape is incompatible
     */
    virtual void validate_output_shape(const ov::Shape& output_shape, 
                                       size_t batch_size) const = 0;
    /**
     * @brief Get description of expected output format for error messages
     * @return Human-readable description of expected output
     */
    virtual std::string get_expected_output_description() const = 0;
};
/**
 * @brief Factory for creating reranking adapters
 */
class OPENVINO_GENAI_EXPORTS RerankingAdapterFactory {
public:
    /**
     * @brief Create adapter for specified variant
     * @param variant The variant to create adapter for
     * @return Shared pointer to adapter instance
     * @throws std::invalid_argument if variant is unknown
     */
    static std::shared_ptr<IRerankingAdapter> create(RerankVariant variant);
    /**
     * @brief Auto-detect variant from model config file
     * @param root_dir Path to model directory
     * @return Detected variant (GENERIC if detection fails)
     */
    static RerankVariant detect_variant(const std::filesystem::path& root_dir);
    /**
     * @brief Register a custom adapter (for extensibility)
     * @param variant Variant enum value to register
     * @param factory_func Function that creates the adapter
     */
    using AdapterCreator = std::function<std::shared_ptr<IRerankingAdapter>()>;
    static void register_adapter(RerankVariant variant, AdapterCreator factory_func);
private:
    static std::map<RerankVariant, AdapterCreator>& get_registry();
};
}  // namespace genai
}  // namespace ov