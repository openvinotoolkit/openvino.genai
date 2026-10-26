// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/vlm_utils.hpp"
#include "openvino/core/except.hpp"
#include "utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ov::genai::vlm_utils {

ImageSize find_best_resize(
    const double height,
    const double width,
    const int scale_resolution,
    const int patch_size,
    const int merge_factor,
    const bool allow_upscale) {
    double new_height = height;
    double new_width = width;
    if (height * width > static_cast<double>(scale_resolution) * scale_resolution || allow_upscale) {
        const double aspect_ratio = width / height;
        new_height = scale_resolution / std::sqrt(aspect_ratio);
        new_width = new_height * aspect_ratio;
    }
    const int factor = patch_size * merge_factor;
    const auto align_size = [factor](const double length) {
        return static_cast<size_t>(std::max(static_cast<int>(std::round(length / factor)) * factor, factor));
    };
    return {align_size(new_height), align_size(new_width)};
}

ImageSize get_refine_size(
    const int height,
    const int width,
    const SlicesGrid& grid,
    const int scale_resolution,
    const int patch_size,
    const int merge_factor) {
    OPENVINO_ASSERT(grid.has_slices(), "Refining an image requires a non-empty slices grid");
    const int grid_rows = static_cast<int>(grid.rows);
    const int grid_cols = static_cast<int>(grid.cols);
    const auto align_size = [](const int length, const int divisor) {
        return std::max(static_cast<int>(std::round(static_cast<double>(length) / divisor)) * divisor, divisor);
    };
    const int refine_height = align_size(height, grid_rows);
    const int refine_width = align_size(width, grid_cols);
    const ImageSize best = find_best_resize(
        static_cast<double>(refine_height) / grid_rows,
        static_cast<double>(refine_width) / grid_cols,
        scale_resolution,
        patch_size,
        merge_factor,
        /*allow_upscale=*/true);
    return {best.height * grid.rows, best.width * grid.cols};
}

SlicesGrid get_slices_grid(const int height, const int width, const int max_slice_nums, const int scale_resolution) {
    const double log_ratio = std::log(static_cast<double>(width) / height);
    const double ratio =
        static_cast<double>(width) * height / (static_cast<double>(scale_resolution) * scale_resolution);
    const int multiple = std::min(static_cast<int>(std::ceil(ratio)), max_slice_nums);
    if (multiple <= 1) {
        return {};
    }

    SlicesGrid best_grid{1, 1};
    double min_error = std::numeric_limits<double>::infinity();
    for (int num_slices : {multiple - 1, multiple, multiple + 1}) {
        if (num_slices == 1 || num_slices > max_slice_nums) {
            continue;
        }
        for (int num_rows = 1; num_rows <= num_slices; ++num_rows) {
            if (num_slices % num_rows != 0) {
                continue;
            }
            const int num_cols = num_slices / num_rows;
            const double error = std::abs(log_ratio - std::log(static_cast<double>(num_cols) / num_rows));
            if (error < min_error) {
                best_grid = {static_cast<size_t>(num_rows), static_cast<size_t>(num_cols)};
                min_error = error;
            } else if (error == min_error && static_cast<size_t>(num_rows) > best_grid.rows) {
                best_grid = {static_cast<size_t>(num_rows), static_cast<size_t>(num_cols)};
            }
        }
    }
    return best_grid;
}

std::vector<std::variant<ov::Tensor, size_t>> split_tokenize(const std::string& text,
                                                             ov::genai::Tokenizer& tokenizer,
                                                             const std::regex& native_pattern) {
    std::vector<std::variant<ov::Tensor, size_t>> tokenized;
    auto prefix_begin = text.begin();
    bool is_submatch = false;
    for (std::sregex_token_iterator iter{
        prefix_begin,
        text.end(),
        native_pattern,
        {0, 1}  // Every match emits two values: whole match and submatch
    }; iter != std::sregex_token_iterator{}; ++iter) {
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
                                                                                        fill_value);  // -1 to distinguish 0 token and 0 image id.
                                                                            return tokens_per_images.at(image_id);
                                                                        }},
                                         chunk);
        offset += written;
    }
    return merged;
}

std::vector<std::variant<ov::Tensor, size_t>> drop_image_placeholders(const ov::Tensor& tokens) {
    std::vector<std::variant<ov::Tensor, size_t>> chunks;
    OPENVINO_ASSERT(tokens.get_shape().size() == 2 && tokens.get_shape()[0] == 1,
                    "Expected token tensor with shape [1, N], got ",
                    tokens.get_shape());
    OPENVINO_ASSERT(tokens.get_element_type() == ov::element::i64,
                    "Expected token tensor element type i64, got ",
                    tokens.get_element_type());

    const size_t full_length = tokens.get_shape().at(1);
    if (full_length == 0) {
        return chunks;
    }

    const int64_t* tokens_ptr = tokens.data<const int64_t>();
    int64_t last_token = tokens_ptr[0];
    size_t text_start = 0;
    for (size_t offset = 1; offset < full_length; ++offset) {
            // If last_token and next_token are not negative, it's continuation of the current chunk text - skip
            // If last_token is negative and next_token is not negative, it's a start of text - save the offset, add image placeholder
            // If last token is not negative and next_token is negative, it's an end of text - push_back a chunk
            // If last_token and next_token are negative, it's continuation of an image placeholder - skip
            // if last_token and next_token are negative but different, it's a start of a new image placeholder - save the previous image placeholder
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
