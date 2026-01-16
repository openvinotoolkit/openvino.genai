
// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/videochat_flash/classes.hpp"

#include "openvino/opsets/opset13.hpp"

#include "utils.hpp"

namespace ov::genai {

namespace {

const std::regex NATIVE_PATTERN{R"(<\|image_(\d+)\|>)"};

void write_native(std::ostream& os, size_t idx) {
    os << "<|image_" << idx + 1 << "|>\n";
}
} // namespace

namespace videochat_flash_utils {

std::string normalize_prompt(
    const std::string& prompt, size_t base_id, size_t n_images, const std::regex& native_pattern, void(*write_native)(std::ostream& os, size_t idx)
) {
    std::smatch match;
    std::regex_search(prompt, match, native_pattern);
    auto [image_prompt, image_sequence] = universal_to_native(prompt, write_native);
    if (!image_sequence.empty()) {
        OPENVINO_ASSERT(match.empty(), "Prompt can contain only one type of image tags.");
        verify_ids(image_sequence, base_id, n_images);
        return image_prompt;
    }
    // Restore ids from native tags
    if (!match.empty()) {
        size_t image_id = std::stoul(match.str(1));
        OPENVINO_ASSERT(image_id != 0, "Image tags must be greater than 0");
        image_sequence.push_back(image_id - 1);
        constexpr int submatch_id_to_return = 1;
        for (std::sregex_token_iterator iter{
            match.suffix().first,
            prompt.end(),
            native_pattern,
            submatch_id_to_return
        }; iter != std::sregex_token_iterator{}; ++iter) {
            size_t image_id = std::stoul(*iter);
            OPENVINO_ASSERT(image_id != 0, "Image tags must be greater than 0");
            image_sequence.push_back(image_id - 1);
        }
        if (!image_sequence.empty()) {
            verify_ids(image_sequence, base_id, n_images);
            return image_prompt;
        }
    }
    // Prepend native tags
    std::stringstream stream;
    for (size_t relative_id = 0; relative_id < n_images; relative_id++) {
        image_sequence.push_back(base_id + relative_id);
        write_native(stream, image_sequence.back());
    }
    stream << prompt;
    return stream.str();
}

/// @brief ov::Tensor is tokenized text, size_t is image tag
std::vector<std::variant<ov::Tensor, size_t>> split_tokenize(const std::string& text, ov::genai::Tokenizer& tokenizer, const std::regex& native_pattern) {
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

ov::Tensor insert_image_placeholders(
    const std::vector<std::variant<ov::Tensor, size_t>>& chunks,
    const std::vector<size_t>& tokens_per_images
) {
    size_t merged_length = 0;
    for (const std::variant<ov::Tensor, size_t>& chunk : chunks) {
        merged_length += std::visit(utils::overloaded{
            [](const ov::Tensor& chunk) {
                return chunk.get_shape().at(1);
            },
            [&](size_t image_id) {
                return tokens_per_images.at(image_id);
            }
        }, chunk);
    }
    ov::Tensor merged{ov::element::i64, {1, merged_length}};
    size_t offset = 0;
    for (const std::variant<ov::Tensor, size_t>& chunk : chunks) {
        offset += std::visit(utils::overloaded{
            [&](const ov::Tensor& chunk) {
                size_t length = chunk.get_shape().at(1);
                std::copy_n(
                    chunk.data<int64_t>(),
                    length,
                    merged.data<int64_t>() + offset
                );
                return length;
            },
            [&](size_t image_id) {
                int64_t fill_value = -(static_cast<int64_t>(image_id)) - 1;
                std::fill_n(
                    merged.data<int64_t>() + offset,
                    tokens_per_images.at(image_id),
                    fill_value  // -1 to distinguish 0 token and 0 image id.
                );
                return tokens_per_images.at(image_id);
            }
        }, chunk);
    }
    return merged;
}

std::vector<std::variant<ov::Tensor, size_t>> drop_image_placeholders(const ov::Tensor& tokens) {
    std::vector<std::variant<ov::Tensor, size_t>> chunks;
    int64_t last_token = tokens.data<int64_t>()[0];
    size_t text_start = 0;
    for (size_t offset = 1; offset < tokens.get_shape().at(1); ++offset) {
        // If last_token and next_token are not negative, it's continuation of the current chunk text - skip
        // If last_token is negative and next_token is not negative, it's a start of text - save the offset, add image placeholder
        // If last token is not negative and next_token is negative, it's an end of text - push_back a chunk
        // If last_token and next_token are negative, it's continuation of an image placeholder - skip
        // if last_token and next_token are negative but different, it's a start of a new image placeholder - save the previous image placeholder
        int64_t next_token = tokens.data<int64_t>()[offset];
        if (last_token < 0 && next_token >= 0) {
            text_start = offset;
            chunks.push_back(size_t(-(last_token + 1)));
        } else if (last_token >= 0 && next_token < 0) {
            chunks.emplace_back(
                std::in_place_type<ov::Tensor>,
                ov::element::i64,
                ov::Shape{1, offset - text_start},
                tokens.data<int64_t>() + text_start
            );
        } else if (last_token < 0 && next_token < 0 && last_token != next_token) {
            chunks.push_back(size_t(-(last_token + 1)));
        }
        last_token = next_token;
    }
    // Add the last chunk
    size_t full_length = tokens.get_shape().at(1);
    if (last_token >= 0) {
        chunks.emplace_back(
            std::in_place_type<ov::Tensor>,
            ov::element::i64,
            ov::Shape{1, full_length - text_start},
            tokens.data<int64_t>() + text_start
        );
    } else {
        chunks.push_back(size_t(-(last_token + 1)));
    }
    return chunks;
}

ov::Tensor transpose_video_features(const ov::Tensor& src_tensor, const size_t mm_local_num_frames) {
    // Input feature:  [N, C, H, W]
    // Output feature with reshape & transpose: [N//mm_local_num_frames, C, mm_local_num_frames, H, W]
    const ov::Shape S0 = src_tensor.get_shape();
    if (S0.size() != 4 || S0[0] % 4 != 0) {
        throw std::runtime_error("Input tensor must be 4D (NCHW) and Batch size N must be divisible by 4.");
    }

    const size_t N = S0[0];
    const size_t C = S0[1];
    const size_t H = S0[2];
    const size_t W = S0[3];

    const ov::Shape S2 = {N / mm_local_num_frames, C, mm_local_num_frames, H, W};
    const size_t N_prime = N / mm_local_num_frames;

    ov::Tensor dst_tensor(src_tensor.get_element_type(), S2);

    if (src_tensor.get_element_type() != ov::element::f32) {
        throw std::runtime_error("Only f32 element type is supported in this manual implementation.");
    }

    const float* src_data = src_tensor.data<const float>();
    float* dst_data = dst_tensor.data<float>();

    const size_t MCHW = mm_local_num_frames * C * H * W;
    const size_t CHW = C * H * W;
    const size_t HW = H * W;
    const size_t MHW = mm_local_num_frames * H * W;

    for (size_t n_prime = 0; n_prime < N_prime; ++n_prime) { // N/4
        for (size_t c_prime = 0; c_prime < C; ++c_prime) {   // C
            for (size_t d_prime = 0; d_prime < mm_local_num_frames; ++d_prime) { // 4
                for (size_t h_prime = 0; h_prime < H; ++h_prime) { // H
                    for (size_t w_prime = 0; w_prime < W; ++w_prime) { // W
                        // dst shape [N/4, C, 4, H, W])
                        size_t dst_idx = n_prime * MCHW + 
                                         c_prime * MHW + 
                                         d_prime * HW + 
                                         h_prime * W + 
                                         w_prime;
                        // src shape [N, C, H, W]
                        size_t src_idx = (n_prime * mm_local_num_frames + d_prime) * CHW + 
                                         c_prime * HW + 
                                         h_prime * W + 
                                         w_prime;
                        dst_data[dst_idx] = src_data[src_idx];
                    }
                }
            }
        }
    }

    return dst_tensor;
}

ov::Tensor remove_second_dim_first_element(const ov::Tensor& input) {
    const ov::Shape& input_shape = input.get_shape();
    if (input_shape.size() < 2) {
        throw std::invalid_argument("Input tensor must have at least 2 dimensions");
    }
    if (input_shape[1] < 1) {
        throw std::invalid_argument("Second dimension of input tensor must be at least 1");
    }
    auto input_data = input.data<float>();
    ov::Shape output_shape = input_shape;
    const size_t org_seq_len = output_shape[1];
    output_shape[1] -= 1;
    const size_t seq_len = output_shape[1];
    const size_t head_elements = input_shape[2];
    ov::Tensor output(input.get_element_type(), output_shape);
    auto output_data = output.data<float>();

    for(int i=0; i < input_shape[0]; i++) {
        std::copy(
            input_data + i * org_seq_len * head_elements + head_elements,
            input_data + (i + 1) * org_seq_len * head_elements,
            output_data + i * seq_len * head_elements
        );
    }
    return output;
}

std::shared_ptr<ov::Model> build_bipartite_soft_matching_merge_opt_model(int dim, ov::element::Type dtype = ov::element::f32) {
    // Parameters: x: [B, P, C], size: [B, P, 1]
    auto x_p = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape({-1, -1, -1}));
    x_p->set_friendly_name("hidden_states");
    x_p->output(0).set_names({"hidden_states"});
    auto size_p = std::make_shared<ov::op::v0::Parameter>(dtype, ov::PartialShape({-1, -1, 1}));
    size_p->set_friendly_name("size");
    size_p->output(0).set_names({"size"});

    // Metric Calculation
    // metric4d = x.reshape(0, 0, -1, dim)
    auto reshape_pat = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, std::vector<int32_t>{0, 0, -1, dim});
    auto metric4d = std::make_shared<ov::op::v1::Reshape>(x_p, reshape_pat, true);

    // metric = reduce_mean(metric4d, axis=2) -> [B, P, dim]
    auto axis_2 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int32_t>{2});
    auto metric = std::make_shared<ov::op::v1::ReduceMean>(metric4d, axis_2, false);

    // L2 Normalization
    // metric_n = metric / sqrt(sum(metric^2))
    auto metric_sq = std::make_shared<ov::op::v1::Multiply>(metric, metric);
    auto axis_neg1 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int32_t>{-1});
    auto metric_ss = std::make_shared<ov::op::v1::ReduceSum>(metric_sq, axis_neg1, true);
    auto metric_norm = std::make_shared<ov::op::v0::Sqrt>(metric_ss);
    auto metric_n = std::make_shared<ov::op::v1::Divide>(metric, metric_norm);

    // Bipartite Indices (Even/Odd)
    auto shape_x = std::make_shared<ov::op::v3::ShapeOf>(x_p, ov::element::i64);
    auto axis_0 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int32_t>{0});
    auto p_node = std::make_shared<ov::op::v1::Gather>(shape_x,
                                                       std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int32_t>{1}),
                                                       axis_0);

    // range(0, p, 2) and range(1, p, 2)
    auto const_0 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int32_t>{0});
    auto const_1 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int32_t>{1});
    auto const_2 = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int32_t>{2});

    auto idx_even = std::make_shared<ov::op::v4::Range>(const_0, p_node, const_2, ov::element::i64);
    auto idx_odd = std::make_shared<ov::op::v4::Range>(const_1, p_node, const_2, ov::element::i64);

    // Scoring & Matching
    auto axis_p = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int32_t>{1});
    auto a = std::make_shared<ov::op::v1::Gather>(metric_n, idx_even, axis_p); // [B, P/2, dim]
    auto b = std::make_shared<ov::op::v1::Gather>(metric_n, idx_odd, axis_p);  // [B, P/2, dim]

    // scores = a @ b.T -> [B, P/2, P/2]
    auto b_t = std::make_shared<ov::op::v1::Transpose>(b, std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{3}, std::vector<int32_t>{0, 2, 1}));
    auto scores = std::make_shared<ov::op::v0::MatMul>(a, b_t, false, false);

    // TopK to get ArgMax (k=1)
    auto topk = std::make_shared<ov::op::v1::TopK>(scores,
                                                   std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int32_t>{1}),
                                                   2, // axis
                                                   ov::op::v1::TopK::Mode::MAX,
                                                   ov::op::v1::TopK::SortType::SORT_VALUES,
                                                   ov::element::i64);
    auto node_idx = std::make_shared<ov::op::v0::Squeeze>(topk->output(1),
                                                          std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int32_t>{2}));

    // Merge Subgraph Logic
    auto merge_subgraph = [&](ov::Output<ov::Node> data_3d) {
        auto src = std::make_shared<ov::op::v1::Gather>(data_3d, idx_even, axis_p);
        auto dst = std::make_shared<ov::op::v1::Gather>(data_3d, idx_odd, axis_p);

        // Broadcast node_idx to [B, P/2, C]
        auto idx_u = std::make_shared<ov::op::v0::Unsqueeze>(node_idx,
                                                             std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int32_t>{-1}));
        auto src_shape = std::make_shared<ov::op::v3::ShapeOf>(src, ov::element::i64);
        auto idx_b = std::make_shared<ov::op::v1::Broadcast>(idx_u, src_shape);

        auto merged = std::make_shared<ov::op::v12::ScatterElementsUpdate>(
            dst, idx_b, src, const_1, ov::op::v12::ScatterElementsUpdate::Reduction::SUM
        );
        return merged;
    };

    // Final Weighted Merge
    auto x_weighted = std::make_shared<ov::op::v1::Multiply>(x_p, size_p);
    auto x_m = merge_subgraph(x_weighted);
    auto size_m = merge_subgraph(size_p);
    auto x_out = std::make_shared<ov::op::v1::Divide>(x_m, size_m);

    // Construct Model
    x_out->set_friendly_name("x_out");
    size_m->set_friendly_name("size_out");

    return std::make_shared<ov::Model>(ov::OutputVector{x_out, size_m}, ov::ParameterVector{x_p, size_p}, "bipartite_merge_opt");
}

ov::Tensor merge_tokens(const ov::Tensor& input, ov::InferRequest& merge_embeddings, const size_t target_num_token = 64) {
    const ov::Shape& x_shape = input.get_shape();
    if (x_shape.size() != 3) {
        throw std::invalid_argument("x must be 3D tensor [batch, tokens, channels], got "
            + std::to_string(x_shape.size()) + "D");
    }
    size_t b = x_shape[0];
    size_t p = x_shape[1];
    size_t c = x_shape[2];

    if (p <= target_num_token) {
        throw std::invalid_argument("Current tokens (" + std::to_string(p) +
            ") must be greater than target (" + std::to_string(target_num_token) + ")");
    }

    std::vector<size_t> r_merge_list;
    size_t tmp_p = p;
    while (tmp_p > target_num_token) {
        size_t next_p = std::max(target_num_token, tmp_p / 2);
        r_merge_list.push_back(tmp_p - next_p);
        tmp_p = next_p;
    }

    const ov::Shape size_shape = {b, p, 1};
    ov::Tensor size_tensor(input.get_element_type(), size_shape);
    float* size_data = size_tensor.data<float>();
    size_t num_elements = size_tensor.get_size();
    std::fill(size_data, size_data + num_elements, 1.0f);

    ov::Tensor current_x = input;

    for (int64_t r : r_merge_list) {
        int64_t current_p = current_x.get_shape()[1];
        merge_embeddings.set_tensor("hidden_states", current_x);
        merge_embeddings.set_tensor("size", size_tensor);
        merge_embeddings.infer();
        current_x = merge_embeddings.get_output_tensor(0);
        size_tensor = merge_embeddings.get_output_tensor(1);
    }

    const ov::Shape& final_shape = current_x.get_shape();
    ov::Shape expected_shape = { b, target_num_token, c };
    if (final_shape != expected_shape) {
        throw std::runtime_error("Merge failed: expected shape " + expected_shape.to_string() +
            ", got " + final_shape.to_string());
    }

    return current_x;
}

ov::Tensor efficient_flatten(ov::Tensor& original_tensor) {
    // flatten 3D tensor [N,C,W] to 3D tensor [1, N*C, W]
    const ov::Shape& original_shape = original_tensor.get_shape();
    const ov::element::Type& dtype = original_tensor.get_element_type();
    ov::Shape new_shape = {
        1,
        original_shape[0] * original_shape[1], // N*C
        original_shape[2]                      // W
    };
    ov::Tensor new_tensor(dtype, new_shape);
    if (original_tensor.get_size() != new_tensor.get_size()) {
         OPENVINO_THROW("Flatten error: Element count mismatch during reshape.");
    }
    const void* src_data = original_tensor.data();
    void* dst_data = new_tensor.data();
    std::memcpy(dst_data, src_data, original_tensor.get_byte_size());
    return new_tensor;
}

std::vector<float> get_1d_sincos_pos_embed_from_grid(int embed_dim, const std::vector<float>& pos) {
    assert(embed_dim % 2 == 0);
    int M = pos.size();
    int half_dim = embed_dim / 2;

    std::vector<float> omega(half_dim);
    for (int i = 0; i < half_dim; ++i) {
        omega[i] = 1.0f / std::pow(10000.0f, (float)i / (embed_dim / 2.0f));
    }

    std::vector<float> emb(M * embed_dim);
    for (int m = 0; m < M; ++m) {
        for (int d = 0; d < half_dim; ++d) {
            float out = pos[m] * omega[d];
            emb[m * embed_dim + d] = std::sin(out);            // 前一半是 sin
            emb[m * embed_dim + d + half_dim] = std::cos(out); // 后一半是 cos
        }
    }
    return emb;
}

std::vector<float> get_2d_sincos_pos_embed_from_grid(int embed_dim, const std::vector<std::vector<float>>& grid) {
    assert(embed_dim % 2 == 0);
    int half_dim = embed_dim / 2;
    int num_points = grid[0].size();

    auto emb_h = get_1d_sincos_pos_embed_from_grid(half_dim, grid[0]);
    auto emb_w = get_1d_sincos_pos_embed_from_grid(half_dim, grid[1]);

    std::vector<float> emb(num_points * embed_dim);
    for (int i = 0; i < num_points; ++i) {
        std::copy(emb_h.begin() + i * half_dim, emb_h.begin() + (i + 1) * half_dim, emb.begin() + i * embed_dim);
        std::copy(emb_w.begin() + i * half_dim, emb_w.begin() + (i + 1) * half_dim, emb.begin() + i * embed_dim + half_dim);
    }
    return emb;
}

ov::Tensor get_3d_sincos_pos_embed(int embed_dim, int grid_size, int t_size, bool cls_token = false) {
    assert(embed_dim % 4 == 0);
    int embed_dim_spatial = (embed_dim / 4) * 3;
    int embed_dim_temporal = embed_dim / 4;
    int hw = grid_size * grid_size;
    int total_tokens = t_size * hw;
    int num_rows = (cls_token ? 1 : 0) + total_tokens;

    ov::Shape output_shape = {1, static_cast<size_t>(num_rows), static_cast<size_t>(embed_dim)};
    ov::Tensor pos_tensor(ov::element::f32, output_shape);

    float* out_ptr = pos_tensor.data<float>();
    std::fill(out_ptr, out_ptr + pos_tensor.get_size(), 0.0f);

    std::vector<std::vector<float>> grid(2, std::vector<float>(hw));
    for (int h = 0; h < grid_size; ++h) {
        for (int w = 0; w < grid_size; ++w) {
            grid[0][h * grid_size + w] = (float)w;
            grid[1][h * grid_size + w] = (float)h;
        }
    }
    auto pos_embed_spatial = get_2d_sincos_pos_embed_from_grid(embed_dim_spatial, grid);

    std::vector<float> grid_t(t_size);
    std::iota(grid_t.begin(), grid_t.end(), 0.0f);
    auto pos_embed_temporal = get_1d_sincos_pos_embed_from_grid(embed_dim_temporal, grid_t);

    int output_offset = cls_token ? 1 : 0;
    for (int t = 0; t < t_size; ++t) {
        for (int i = 0; i < hw; ++i) {
            float* token_ptr = out_ptr + (output_offset + t * hw + i) * embed_dim;
            std::memcpy(token_ptr, 
                        &pos_embed_temporal[t * embed_dim_temporal], 
                        embed_dim_temporal * sizeof(float));
            std::memcpy(token_ptr + embed_dim_temporal, 
                        &pos_embed_spatial[i * embed_dim_spatial], 
                        embed_dim_spatial * sizeof(float));
        }
    }

    return pos_tensor;
}

ov::Tensor concatenate_tensors(const std::vector<ov::Tensor>& tensors) {
    if (tensors.empty()) return ov::Tensor();

    ov::Shape single_shape = tensors[0].get_shape();
    auto type = tensors[0].get_element_type();

    ov::Shape final_shape = single_shape;
    final_shape[0] = tensors.size();

    ov::Tensor merged_tensor(type, final_shape);
    uint8_t* dst_ptr = static_cast<uint8_t*>(merged_tensor.data());

    for (const auto& t : tensors) {
        size_t byte_size = t.get_byte_size();
        std::memcpy(dst_ptr, t.data(), byte_size);
        dst_ptr += byte_size;
    }

    return merged_tensor;
}

ov::Tensor cyclic_vit_infer(ov::Tensor& transpose_features, ov::InferRequest& vision_embeddings, ov::Tensor& m_pos_emb) {
    ov::Shape full_shape = transpose_features.get_shape();
    size_t N = full_shape[0];
    size_t single_sample_size = transpose_features.get_size() / N;
    float* src_ptr = transpose_features.data<float>();
    std::vector<ov::Tensor> results_list;
    for (size_t i = 0; i < N; ++i) {
        ov::Shape single_shape = {1, full_shape[1], full_shape[2], full_shape[3], full_shape[4]};
        ov::Tensor single_input(transpose_features.get_element_type(), single_shape, src_ptr + (i * single_sample_size));
        vision_embeddings.set_tensor("hidden_states", single_input);
        vision_embeddings.set_tensor("rotary_pos_emb", m_pos_emb);
        vision_embeddings.infer();
        ov::Tensor out_tensor = vision_embeddings.get_output_tensor();
        ov::Tensor copy_tensor(out_tensor.get_element_type(), out_tensor.get_shape());
        out_tensor.copy_to(copy_tensor);
        results_list.push_back(copy_tensor);
    }
    ov::Tensor final_processed_embeds = concatenate_tensors(results_list);
    return final_processed_embeds;
}

}  // namespace videochat_flash_utils


VisionEncoderVideoChat_Flash::VisionEncoderVideoChat_Flash(
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap properties) : VisionEncoder(model_dir, device, properties) {
    m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(model_dir, "config.json");
    auto compiled_model = utils::singleton_core().compile_model(model_dir / "openvino_vision_projection_model.xml", device, {});
    m_ireq_queue_vision_projection = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });

    auto merge_dim = m_vlm_config.mm_hidden_size / 16;
    auto merge_model = videochat_flash_utils::build_bipartite_soft_matching_merge_opt_model(merge_dim);
    auto compiled_merge_model = utils::singleton_core().compile_model(merge_model, "CPU", {});
    m_ireq_queue_merge_model = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_merge_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_merge_model]() -> ov::InferRequest {
            return compiled_merge_model.create_infer_request();
        });
    
    // init 3d_sincos_pos_embed
    size_t mm_hidden_size = m_vlm_config.mm_hidden_size;
    size_t mm_local_num_frames = m_vlm_config.mm_local_num_frames;
    // Can not obtain this from config for now
    const size_t img_size = 224;
    const size_t patch_size = 14;
    size_t grid_size = img_size / patch_size; // 16
    m_pos_emb = videochat_flash_utils::get_3d_sincos_pos_embed(mm_hidden_size, grid_size, mm_local_num_frames, true);
}

VisionEncoderVideoChat_Flash::VisionEncoderVideoChat_Flash(
    const ModelsMap& models_map,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap properties) : VisionEncoder(models_map, config_dir_path, device, properties) {
    m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(config_dir_path, "config.json");
    const auto& vision_encoder_model = utils::get_model_weights_pair(models_map, "vision_projection").first;
    const auto& vision_encoder_weights = utils::get_model_weights_pair(models_map, "vision_projection").second;
    auto compiled_model = utils::singleton_core().compile_model(vision_encoder_model, vision_encoder_weights, device, properties);
    m_ireq_queue_vision_projection = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });

    auto merge_dim = m_vlm_config.mm_hidden_size / 16;
    auto merge_model = videochat_flash_utils::build_bipartite_soft_matching_merge_opt_model(merge_dim);
    auto compiled_merge_model = utils::singleton_core().compile_model(merge_model, "CPU", {});
    m_ireq_queue_merge_model = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_merge_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_merge_model]() -> ov::InferRequest {
            return compiled_merge_model.create_infer_request();
        });
    
    // init 3d_sincos_pos_embed
    size_t mm_hidden_size = m_vlm_config.mm_hidden_size;
    size_t mm_local_num_frames = m_vlm_config.mm_local_num_frames;
    // Can not obtain this from config for now
    const size_t img_size = 224;
    const size_t patch_size = 14;
    size_t grid_size = img_size / patch_size; // 16
    m_pos_emb = videochat_flash_utils::get_3d_sincos_pos_embed(mm_hidden_size, grid_size, mm_local_num_frames, true);
}

EncodedImage VisionEncoderVideoChat_Flash::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    EncodedImage encoded_feature;
    size_t frame_num = image.get_shape().at(0);
    size_t mm_local_num_frames = m_vlm_config.mm_local_num_frames;
    size_t mm_hidden_size = m_vlm_config.mm_hidden_size;
    auto input_shape = image.get_shape();
    OPENVINO_ASSERT(input_shape.size() == 4, "Input video features must be 4D.");

    // TODO: here suppose passed in frames have been preprocessed, and shape is [N,3,224,224]
    // Consider if we add preprocess in encode_frames in next step

    // transpose video features
    auto transpose_features = videochat_flash_utils::transpose_video_features(image, mm_local_num_frames);
    // video embedding
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& vision_embeddings = infer_request_guard.get();
    bool use_batch_vit = false;
    utils::read_anymap_param(config_map, "use_batch_vit", use_batch_vit);
    ov::Tensor processed_vision_embeds;
    if (!use_batch_vit) {
        processed_vision_embeds = videochat_flash_utils::cyclic_vit_infer(transpose_features, vision_embeddings, m_pos_emb);
    } else {
        vision_embeddings.set_tensor("hidden_states", transpose_features);
        vision_embeddings.set_tensor("rotary_pos_emb", m_pos_emb);
        vision_embeddings.infer();
        processed_vision_embeds = vision_embeddings.get_output_tensor();
    }

    ov::Tensor clipped_vision_embeds = videochat_flash_utils::remove_second_dim_first_element(processed_vision_embeds);

    // merge tokens
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard_merge(this->m_ireq_queue_merge_model.get());
    ov::InferRequest& merge_embeddings = infer_request_guard_merge.get();
    ov::Tensor merged_vision_features = videochat_flash_utils::merge_tokens(clipped_vision_embeds, merge_embeddings);

    // vision projection
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard_proj(this->m_ireq_queue_vision_projection.get());
    ov::InferRequest& vision_projection = infer_request_guard_proj.get();
    vision_projection.set_tensor("hidden_states", merged_vision_features);
    vision_projection.infer();
    // here proj features shape is [N_frames // 4, 4 * 16, 3584]
    ov::Tensor proj_features = vision_projection.get_output_tensor();

    // flatten vision features
    auto final_features = videochat_flash_utils::efficient_flatten(proj_features);
    encoded_feature.images_features_projection = final_features;
    std::cout << "finish encode." << std::endl;
    return encoded_feature;
}


std::vector<ov::genai::EncodedImage> InputsEmbedderVideoChat_Flash::encode_images(const std::vector<ov::Tensor>& images) {
    return encode_images(images, {});
}

std::vector<ov::genai::EncodedImage> InputsEmbedderVideoChat_Flash::encode_images(const std::vector<ov::Tensor>& images, const ov::AnyMap& config_map) {
    std::vector<EncodedImage> embeds;
    for (const ov::Tensor& single_video : images) {
        auto encoded_video = m_vision_encoder->encode(single_video, config_map);
        embeds.emplace_back(encoded_video);
    }
    return embeds;
}

InputsEmbedderVideoChat_Flash::InputsEmbedderVideoChat_Flash(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config
) : IInputsEmbedder(vlm_config, model_dir, device, device_config) {}


InputsEmbedderVideoChat_Flash::InputsEmbedderVideoChat_Flash(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {}


NormalizedPrompt InputsEmbedderVideoChat_Flash::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    return {videochat_flash_utils::normalize_prompt(prompt, base_id, images.size(), NATIVE_PATTERN, write_native), {}};
}

ov::Tensor InputsEmbedderVideoChat_Flash::get_inputs_embeds(const std::string& image_prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& image_sequence) {
    size_t base_id = m_tokens_per_images.size();
    std::vector<ov::Tensor> images_features_proj;
    for (const ov::genai::EncodedImage& encoded_image : images) {
        images_features_proj.push_back(encoded_image.images_features_projection);
        m_tokens_per_images.push_back(images_features_proj.back().get_shape().at(1));
    }
    std::vector<std::variant<ov::Tensor, size_t>> new_chat_tokens;
    if (m_is_chat_conversation) {
        auto start_tokenizer_time = std::chrono::steady_clock::now();
        new_chat_tokens = videochat_flash_utils::split_tokenize(image_prompt, m_tokenizer, NATIVE_PATTERN);
        auto end_tokenizer_time = std::chrono::steady_clock::now();
        metrics.raw_metrics.tokenization_durations.emplace_back(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    } else {
        std::string templated_prompt;
        if (m_apply_chat_template) {
            ChatHistory history({{{"role", "user"}, {"content", std::move(image_prompt)}}});
            constexpr bool add_generation_prompt = true;
            templated_prompt = m_tokenizer.apply_chat_template(history, add_generation_prompt);
        } else {
            templated_prompt = std::move(image_prompt);
        }
        auto start_tokenizer_time = std::chrono::steady_clock::now();
        new_chat_tokens = videochat_flash_utils::split_tokenize(templated_prompt, m_tokenizer, NATIVE_PATTERN);
        auto end_tokenizer_time = std::chrono::steady_clock::now();
        metrics.raw_metrics.tokenization_durations.emplace_back(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    }
    ov::Tensor new_merged_tokens = videochat_flash_utils::insert_image_placeholders(new_chat_tokens, m_tokens_per_images);
    ov::Tensor new_tokens = update_history(new_merged_tokens);
    m_prev_hist_length = m_kv_cache_state.get_state().size();
    m_kv_cache_state.add_inputs(new_tokens);

    std::vector<std::variant<ov::Tensor, size_t>> tokens = videochat_flash_utils::drop_image_placeholders(new_tokens);
    ov::Tensor inputs_embeds{ov::element::f32, {1, new_tokens.get_shape().at(1), m_vlm_config.hidden_size}};
    size_t offset = 0;
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    for (const std::variant<ov::Tensor, size_t>& chunk : tokens) {
        offset += std::visit(utils::overloaded{
            [&](const ov::Tensor& chunk) {
                const ov::Tensor& text_embeds = m_embedding->infer(req, chunk);
                size_t text_length = text_embeds.get_shape().at(1);
                std::copy_n(
                    text_embeds.data<float>(),
                    text_embeds.get_size(),
                    inputs_embeds.data<float>() + offset * m_vlm_config.hidden_size
                );
                return text_length;
            },
            [&](size_t image_id) {
                const ov::Tensor& image_embeds = images_features_proj.at(image_id - base_id);
                size_t im_length = image_embeds.get_shape().at(1);
                std::copy_n(
                    image_embeds.data<float>(),
                    image_embeds.get_size(),
                    inputs_embeds.data<float>() + offset * m_vlm_config.hidden_size
                );
                return im_length;
            }
        }, chunk);
    }

    if (!m_is_chat_conversation) {
        m_tokens_per_images.clear();
    }
    return inputs_embeds;
}

void InputsEmbedderVideoChat_Flash::update_chat_history(const std::string& decoded_results, const ov::genai::GenerationStatus generation_finish_status) {
    IInputsEmbedder::update_chat_history(decoded_results, generation_finish_status);
    if (generation_finish_status == ov::genai::GenerationStatus::CANCEL)
        m_tokens_per_images = m_prev_tokens_per_images;
    else
        m_prev_tokens_per_images = m_tokens_per_images;
}

void InputsEmbedderVideoChat_Flash::start_chat(const std::string& system_message) {
    IInputsEmbedder::start_chat(system_message);
    m_tokens_per_images.clear();
}

void InputsEmbedderVideoChat_Flash::finish_chat() {
    IInputsEmbedder::finish_chat();
    m_tokens_per_images.clear();
}

} // namespace ov::genai
