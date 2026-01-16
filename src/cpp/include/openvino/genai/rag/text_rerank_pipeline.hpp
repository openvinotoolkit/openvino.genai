// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <optional>
#include "openvino/genai/lora_adapter.hpp"
#include "openvino/genai/tokenizer.hpp"

namespace ov {
namespace genai {

class OPENVINO_GENAI_EXPORTS TextRerankPipeline {
public:
    struct OPENVINO_GENAI_EXPORTS Config {
        /**
         * @brief Number of documents to return sorted by score
         */
        size_t top_n = 3;

        /**
         * @brief Maximum length of tokens passed to the rerank model
         */
        std::optional<size_t> max_length;

        /**
         * @brief If 'true', model input tensors are padded to the maximum length
         */
        std::optional<bool> pad_to_max_length;

        /**
         * @brief Side to use for padding "left" or "right"
         */
        std::optional<std::string> padding_side;

        /**
         * @brief LoRA tensor name prefix override.
         * If not specified, the prefix is auto-detected based on model architecture.
         * Common prefixes: "base_model.model", "bert", "encoder"
         */
        std::optional<std::string> lora_tensor_prefix;
        /**
         * @brief Constructs text rerank pipeline configuration
         */
        Config() = default;

        /**
         * @brief Constructs text rerank pipeline configuration
         *
         * @param properties configuration options
         *
         * const ov::AnyMap properties{{"top_n", 3}};
         * ov::genai::TextRerankPipeline::Config config(properties);
         *
         * ov::genai::TextRerankPipeline::Config config({{"top_n", 3}});
         */
        explicit Config(const ov::AnyMap& properties);

        /**
         * @brief checks that are no conflicting parameters
         * @throws Exception if config is invalid.
         */
        void validate() const;
    };

    /**
     * @brief Constructs a pipeline from xml/bin files, tokenizer and configuration in the same dir.
     *
     * @param models_path Path to the directory containing model xml/bin files and tokenizer
     * @param device Device
     * @param config Pipeline configuration
     * @param properties Optional plugin properties to pass to ov::Core::compile_model().
     *                   May include ov::genai::adapters() for LoRA support.
     *
     * @code
     * // Basic usage
     * TextRerankPipeline pipeline(models_path, "CPU", config);
     *
     * // With LoRA adapter
     * Adapter lora("domain_adapter.safetensors");
     * AdapterConfig adapter_config;
     * adapter_config.add(lora, 1.0f);
     * TextRerankPipeline pipeline(models_path, "CPU", config, ov::genai::adapters(adapter_config));
     * @endcode
     */
    TextRerankPipeline(const std::filesystem::path& models_path,
                       const std::string& device,
                       const Config& config,
                       const ov::AnyMap& properties = {});

    /**
     * @brief Constructs a pipeline from xml/bin files, tokenizer and configuration in the same dir.
     *
     * @param models_path Path to the directory containing model xml/bin files and tokenizer
     * @param device Device
     * @param properties Optional plugin and/or config properties.
     *                   May include ov::genai::adapters() for LoRA support.
     */
    TextRerankPipeline(const std::filesystem::path& models_path,
                       const std::string& device,
                       const ov::AnyMap& properties = {});

    /**
     * @brief Constructs a pipeline from xml/bin files, tokenizer and configuration in the same dir.
     *
     * @param models_path Path to the directory containing model xml/bin files and tokenizer
     * @param device Device
     * @param properties Plugin and/or config properties
     */
    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    TextRerankPipeline(const std::filesystem::path& models_path, const std::string& device, Properties&&... properties)
        : TextRerankPipeline(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    /**
     * @brief Reranks a vector of texts based on the query.
     */
    std::vector<std::pair<size_t, float>> rerank(const std::string& query, const std::vector<std::string>& texts);

    /**
     * @brief Asynchronously reranks a vector of texts based on the query. Only one method of async family can be
     * active.
     */
    void start_rerank_async(const std::string& query, const std::vector<std::string>& texts);

    /**
     * @brief Waits for reranked texts.
     */
    std::vector<std::pair<size_t, float>> wait_rerank();

    /**
     * @brief Set or update LoRA adapters at runtime
     * @param adapters Optional adapter configuration. Pass std::nullopt to disable LoRA.
     *
     * @note The pipeline must have been compiled with LoRA support (i.e., adapters
     *       were provided during construction) for this method to work.
     */
    void set_adapters(const std::optional<AdapterConfig>& adapters);
    /**
     * @brief Check if LoRA adapters are currently active
     * @return true if LoRA adapters were configured during construction
     */
    bool has_adapters() const;
    ~TextRerankPipeline();

private:
    class TextRerankPipelineImpl;
    std::unique_ptr<TextRerankPipelineImpl> m_impl;
};

/**
 * @brief Number of documents to return after reranking sorted by score
 */
static constexpr ov::Property<size_t> top_n{"top_n"};

// Note: The following properties are already defined elsewhere and should be used from there:

// - max_length: defined in generation_config.hpp
// - pad_to_max_length: defined in tokenizer.hpp  
// - padding_side: defined in tokenizer.hpp

}  // namespace genai
}  // namespace ov
