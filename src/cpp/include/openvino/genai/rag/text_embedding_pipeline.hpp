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

class OPENVINO_GENAI_EXPORTS TextEmbeddingPipeline {
public:
    enum class OPENVINO_GENAI_EXPORTS PoolingType {
        CLS = 0,
        MEAN = 1,
    };

    struct Config {
        std::optional<size_t> max_length;
        PoolingType pooling_type = PoolingType::CLS;
        bool normalize = false;
        std::optional<std::string> query_instruction;
        std::optional<std::string> embed_instruction;

        Config() = default;
        explicit Config(const ov::AnyMap& properties);
    };

    /**
     * @brief Constructs an pipeline from xml/bin files, tokenizers and configuration in the same dir.
     *
     * @param models_path Path to the dir model xml/bin files, tokenizers
     * @param device optional device
     * @param plugin_config optional plugin_config properties
     */
    TextEmbeddingPipeline(const std::filesystem::path& models_path,
                          const std::string& device,
                          const std::optional<Config>& config = std::nullopt,
                          const ov::AnyMap& properties = {});

    TextEmbeddingPipeline(const std::filesystem::path& models_path,
                          const std::string& device,
                          const ov::AnyMap& properties);

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    TextEmbeddingPipeline(const std::filesystem::path& models_path,
                          const std::string& device,
                          Properties&&... properties)
        : TextEmbeddingPipeline(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    /**
     * @brief Computes embeddings for a vector of texts
     */
    std::vector<EmbeddingResult> embed_documents(std::vector<std::string>& texts);

    void start_embed_documents_async(std::vector<std::string>& texts);
    std::vector<std::vector<float>> wait_embed_documents();

    /**
     * @brief Computes embedding for a query
     */
    EmbeddingResult embed_query(std::string& text);

    void start_embed_query_async(std::string& text);
    EmbeddingResult wait_embed_query();

    ~TextEmbeddingPipeline();

private:
    class TextEmbeddingPipelineImpl;
    std::unique_ptr<TextEmbeddingPipelineImpl> m_impl;
};

static constexpr ov::Property<bool> normalize{"normalize"};
static constexpr ov::Property<TextEmbeddingPipeline::PoolingType> pooling_type{"pooling_type"};
static constexpr ov::Property<std::string> query_instruction{"query_instruction"};
static constexpr ov::Property<std::string> embed_instruction{"embed_instruction"};

}  // namespace genai
}  // namespace ov
