// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <optional>
#include <variant>

#include "openvino/genai/tokenizer.hpp"

namespace ov {
namespace genai {

using EmbeddingResult = std::variant<std::vector<float>, std::vector<int8_t>, std::vector<uint8_t>>;
using EmbeddingResults =
    std::variant<std::vector<std::vector<float>>, std::vector<std::vector<int8_t>>, std::vector<std::vector<uint8_t>>>;

class OPENVINO_GENAI_EXPORTS TextEmbeddingPipeline {
public:
    enum class PoolingType {
        /**
         * @brief First token embeddings
         */
        CLS = 0,
        /**
         * @brief The average of all token embeddings
         */
        MEAN = 1,

        /**
         * @brief Last token embeddings
         * 
         * @note Specifying `ov::genai::padding_side = "left"` can optimize performance for this pooling type.
         */
        LAST_TOKEN = 2,
    };

    struct OPENVINO_GENAI_EXPORTS Config {
        /**
         * @brief Maximum length of tokens passed to the embedding model
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
         * @brief Batch size of embedding model.
         * Useful for database population. If set, the pipeline will fix model shape for inference optimization. 
         * Number of documents passed to pipeline should be equal to batch_size.
         * For query embeddings, batch_size should be set to 1 or not set.
         */
        std::optional<size_t> batch_size;

        /**
         * @brief Pooling strategy applied to model output tensor
         */
        PoolingType pooling_type = PoolingType::CLS;

        /**
         * @brief If 'true', L2 normalization is applied to embeddings
         */
        bool normalize = true;

        /**
         * @brief Instruction to use for embedding a query
         */
        std::optional<std::string> query_instruction;

        /**
         * @brief Instruction to use for embedding a document
         */
        std::optional<std::string> embed_instruction;

        /**
         * @brief Constructs text embedding pipeline configuration
         */
        Config() = default;

        /**
         * @brief Constructs text embedding pipeline configuration
         *
         * @param properties configuration options
         *
         * const ov::AnyMap properties{{"normalize", false}};
         * ov::genai::TextEmbeddingPipeline::Config config(properties);
         *
         * ov::genai::TextEmbeddingPipeline::Config config({{"normalize", false}});
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
     */
    TextEmbeddingPipeline(const std::filesystem::path& models_path,
                          const std::string& device,
                          const Config& config,
                          const ov::AnyMap& properties = {});

    /**
     * @brief Constructs a pipeline from xml/bin files, tokenizer and configuration in the same dir.
     *
     * @param models_path Path to the directory containing model xml/bin files and tokenizer
     * @param device Device
     * @param properties Optional plugin and/or config properties
     */
    TextEmbeddingPipeline(const std::filesystem::path& models_path,
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
    TextEmbeddingPipeline(const std::filesystem::path& models_path,
                          const std::string& device,
                          Properties&&... properties)
        : TextEmbeddingPipeline(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    /**
     * @brief Computes embeddings for a vector of texts
     */
    EmbeddingResults embed_documents(const std::vector<std::string>& texts);

    /**
     * @brief Asynchronously computes embeddings for a vector of texts. Only one method of async family can be active.
     */
    void start_embed_documents_async(const std::vector<std::string>& texts);

    /**
     * @brief Waits for computed embeddings of a vector of texts
     */
    EmbeddingResults wait_embed_documents();

    /**
     * @brief Computes embedding for a query
     */
    EmbeddingResult embed_query(const std::string& text);

    /**
     * @brief Asynchronously computes embeddings for a query. Only one method of async family can be active.
     */
    void start_embed_query_async(const std::string& text);

    /**
     * @brief Waits for computed embeddings for a query
     */
    EmbeddingResult wait_embed_query();

    ~TextEmbeddingPipeline();

private:
    class TextEmbeddingPipelineImpl;
    std::unique_ptr<TextEmbeddingPipelineImpl> m_impl;
};

/**
 * @brief If 'true', L2 normalization applied to embeddings
 */
static constexpr ov::Property<bool> normalize{"normalize"};

/**
 * @brief Pooling strategy applied to model output tensor
 */
static constexpr ov::Property<TextEmbeddingPipeline::PoolingType> pooling_type{"pooling_type"};

/**
 * @brief Instruction to use for embedding query
 */
static constexpr ov::Property<std::string> query_instruction{"query_instruction"};

/**
 * @brief Instruction to use for embedding document
 */
static constexpr ov::Property<std::string> embed_instruction{"embed_instruction"};

/**
 * @brief Batch size for embedding model.
 * If batch_size, max_length and pad_to_max_length are set, the pipeline will fix model shape
 * for inference optimization. Number of documents passed to pipeline should be equal to batch_size.
 */
static constexpr ov::Property<size_t> batch_size{"batch_size"};

}  // namespace genai
}  // namespace ov
