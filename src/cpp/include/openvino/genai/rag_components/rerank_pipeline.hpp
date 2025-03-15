// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <optional>
#include <variant>

#include "openvino/genai/tokenizer.hpp"

namespace ov {
namespace genai {

struct TextDocument {
    std::optional<std::string> id;
    std::string text;
    std::optional<float> score;
};

class OPENVINO_GENAI_EXPORTS TextRerankPipeline {
public:
    struct Config {
        size_t top_n = 3;
        std::optional<size_t> max_length;
    };

    /**
     * @brief Constructs an pipeline from xml/bin files, tokenizers and configuration in the same dir.
     *
     * @param models_path Path to the dir model xml/bin files, tokenizers and generation_configs.json
     * @param device optional device
     * @param properties optional properties
     */
    TextRerankPipeline(const std::filesystem::path& models_path,
                       const std::string& device,
                       const std::optional<Config>& config,
                       const ov::AnyMap& properties = {});

    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    TextRerankPipeline(const std::filesystem::path& models_path, const std::string& device, Properties&&... properties)
        : TextRerankPipeline(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    /**
     * @brief Rerank texts based on query
     *
     * @return vector of top_n texts with the highest score
     */
    std::vector<std::pair<size_t, float>> rerank(const std::string query, const std::vector<std::string>& texts);

    std::vector<TextDocument> rerank(const std::string query, const std::vector<TextDocument>& documents);

    ~TextRerankPipeline();
};

}  // namespace genai
}  // namespace ov
