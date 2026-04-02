// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/vlm_utils.hpp"
#include "openvino/core/except.hpp"
#include "utils.hpp"

#include <algorithm>

namespace ov::genai::vlm_utils {

std::vector<std::variant<ov::Tensor, size_t>> split_tokenize(const std::string& text,
                                                             ov::genai::Tokenizer& tokenizer,
                                                             const std::regex& native_pattern) {
    std::vector<std::variant<ov::Tensor, size_t>> tokenized;
    auto prefix_begin = text.begin();
    bool is_submatch = false;
    for (std::sregex_token_iterator iter{prefix_begin, text.end(), native_pattern, {0, 1}};
         iter != std::sregex_token_iterator{};
         ++iter) {
        if (is_submatch) {
            size_t idx = std::stoul(iter->str());
            OPENVINO_ASSERT(idx != 0);
            tokenized.push_back(idx - 1);
        } else {
            std::string regular_text{prefix_begin, iter->first};
            if (!regular_text.empty()) {
                tokenized.push_back(tokenizer.encode(regular_text, {ov::genai::add_special_tokens(true)}).input_ids);
            }
            prefix_begin = iter->second;
        }
        is_submatch = !is_submatch;
    }
    std::string regular_text{prefix_begin, text.end()};
    if (!regular_text.empty()) {
        tokenized.push_back(tokenizer.encode(regular_text, {ov::genai::add_special_tokens(true)}).input_ids);
    }
    return tokenized;
}

ov::Tensor insert_image_placeholders(const std::vector<std::variant<ov::Tensor, size_t>>& chunks,
                                     const std::vector<size_t>& tokens_per_images) {
    size_t merged_length = 0;
    for (const std::variant<ov::Tensor, size_t>& chunk : chunks) {
        merged_length += std::visit(ov::genai::utils::overloaded{[](const ov::Tensor& token_chunk) {
                                                                      return token_chunk.get_shape().at(1);
                                                                  },
                                                                  [&](size_t image_id) {
                                                                      return tokens_per_images.at(image_id);
                                                                  }},
                                   chunk);
    }

    ov::Tensor merged{ov::element::i64, {1, merged_length}};
    size_t offset = 0;
    for (const std::variant<ov::Tensor, size_t>& chunk : chunks) {
        const size_t written = std::visit(ov::genai::utils::overloaded{[&](const ov::Tensor& token_chunk) {
                                                                            const size_t length = token_chunk.get_shape().at(1);
                                                                            std::copy_n(token_chunk.data<int64_t>(), length, merged.data<int64_t>() + offset);
                                                                            return length;
                                                                        },
                                                                        [&](size_t image_id) {
                                                                            const int64_t fill_value = -(static_cast<int64_t>(image_id)) - 1;
                                                                            std::fill_n(merged.data<int64_t>() + offset,
                                                                                        tokens_per_images.at(image_id),
                                                                                        fill_value);
                                                                            return tokens_per_images.at(image_id);
                                                                        }},
                                         chunk);
        offset += written;
    }
    return merged;
}

std::vector<std::variant<ov::Tensor, size_t>> drop_image_placeholders(const ov::Tensor& tokens) {
    std::vector<std::variant<ov::Tensor, size_t>> chunks;
    const int64_t* tokens_ptr = tokens.data<const int64_t>();
    int64_t last_token = tokens_ptr[0];
    size_t text_start = 0;
    for (size_t offset = 1; offset < tokens.get_shape().at(1); ++offset) {
        const int64_t next_token = tokens_ptr[offset];
        if (last_token < 0 && next_token >= 0) {
            text_start = offset;
            chunks.push_back(size_t(-(last_token + 1)));
        } else if (last_token >= 0 && next_token < 0) {
            chunks.emplace_back(std::in_place_type<ov::Tensor>,
                                ov::element::i64,
                                ov::Shape{1, offset - text_start},
                                const_cast<int64_t*>(tokens_ptr + text_start));
        } else if (last_token < 0 && next_token < 0 && last_token != next_token) {
            chunks.push_back(size_t(-(last_token + 1)));
        }
        last_token = next_token;
    }

    const size_t full_length = tokens.get_shape().at(1);
    if (last_token >= 0) {
        chunks.emplace_back(std::in_place_type<ov::Tensor>,
                            ov::element::i64,
                            ov::Shape{1, full_length - text_start},
                            const_cast<int64_t*>(tokens_ptr + text_start));
    } else {
        chunks.push_back(size_t(-(last_token + 1)));
    }
    return chunks;
}

}  // namespace ov::genai::vlm_utils
