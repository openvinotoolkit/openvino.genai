// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "vlm_sampling.hpp"
#include "clip.hpp"
#include "text_callback_streamer.hpp"
#include "utils.hpp"
#include "vision_encoder.hpp"
#include "vlm_config.hpp"
#include <openvino/openvino.hpp>
#include <optional>
#include <random>

using namespace ov::genai;

namespace {
template<class... Ts> struct overloaded : Ts... {using Ts::operator()...;};
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

constexpr size_t BATCH_SIZE = 1;

struct Args {
    bool do_sample = false;
    int top_k = 0;
    float top_p = 0.7f;
    float temp = 0.95f;
    float repeat_penalty = 1.0f;
};

int64_t get_out_token_id(const std::vector<int>& input_ids, float* logits, size_t vocab_size, Args args) {
    int64_t out_token;

    // logits pre-process
    if (args.repeat_penalty != 1.f) {
        sampling_repetition_penalty(logits, logits + vocab_size, input_ids, args.repeat_penalty);
    }

    if (args.do_sample)
    {
        if (args.temp > 0) {
            sampling_temperature(logits, logits + vocab_size, args.temp);
        }

        std::vector<TokenIdScore> token_scores(vocab_size);
        for (int i = 0; i < vocab_size; i++) {
            token_scores[i] = TokenIdScore(i, logits[i]);
        }

        // top_k sampling
        if (0 < args.top_k && args.top_k < (int)token_scores.size()) {
            sampling_top_k(token_scores.data(), token_scores.data() + args.top_k,
                token_scores.data() + token_scores.size());
            token_scores.resize(args.top_k);
        }

        // top_p sampling
        if (0.f < args.top_p && args.top_p < 1.f) {
            auto pos = sampling_top_p(token_scores.data(), token_scores.data() + token_scores.size(), args.top_p);
            token_scores.resize(pos - token_scores.data());
        }

        // sample next token
        sampling_softmax_inplace(token_scores.data(), token_scores.data() + token_scores.size());
        for (size_t i = 0; i < token_scores.size(); i++) {
            logits[i] = token_scores[i].score;
        }

        thread_local std::random_device rd;
        thread_local std::mt19937 gen(rd());

        std::discrete_distribution<> dist(logits, logits + token_scores.size());
        out_token = token_scores[dist(gen)].id;
    }
    else {
        out_token = std::max_element(logits, logits + vocab_size) - logits;
    }

    return out_token;
}

ov::Tensor process_prompt(ov::InferRequest& embedding, const ov::Tensor& prompt, float scale_emb) {
    embedding.set_input_tensor(prompt);
    embedding.infer();

    const ov::Tensor& embed_output_tensor = embedding.get_output_tensor();

    ov::Shape out_shape = embed_output_tensor.get_shape();
    float* data = embed_output_tensor.data<float>();

    //embedding * scale_emb
    for (size_t idx = 0; idx < embed_output_tensor.get_size(); idx++) {
        data[idx] = data[idx] * scale_emb;
    }
    return embed_output_tensor;
}

ov::Tensor concatenate_last_dim(const ov::Tensor& first, const ov::Tensor& second) {
    size_t res_d_0 = first.get_shape().at(0);
    size_t res_d_1 = first.get_shape().at(1);
    OPENVINO_ASSERT(second.get_shape().at(0) == res_d_0);
    OPENVINO_ASSERT(second.get_shape().at(1) == res_d_1);
    size_t res_d_2 = first.get_shape().at(2) + second.get_shape().at(2);
    ov::Tensor res{first.get_element_type(), {res_d_0, res_d_1, res_d_2}};
    float* first_data = first.data<float>();
    float* second_data = second.data<float>();
    float* res_data = res.data<float>();
    for (size_t i = 0; i < res_d_0; ++i) {
        for (size_t j = 0; j < res_d_1; ++j) {
            size_t k = 0;
            for (; k < first.get_shape().at(2); ++k) {
                res_data[i * res_d_1 * res_d_2 + j * res_d_2 + k]
                    = first_data[i * res_d_1 * first.get_shape().at(2) + j * first.get_shape().at(2) + k];
            }
            for (size_t l = 0; l < second.get_shape().at(2); ++l, ++k) {
                res_data[i * res_d_1 * res_d_2 + j * res_d_2 + k]
                    = second_data[i * res_d_1 * second.get_shape().at(2) + j * second.get_shape().at(2) + l];
            }
        }
    }
    return res;
}

ov::Tensor concatenate_mid_dim(const ov::Tensor& first, const ov::Tensor& second) {
    size_t res_d_0 = first.get_shape().at(0);
    size_t res_d_2 = first.get_shape().at(2);
    OPENVINO_ASSERT(second.get_shape().at(0) == res_d_0);
    OPENVINO_ASSERT(second.get_shape().at(2) == res_d_2);
    size_t res_d_1 = first.get_shape().at(1) + second.get_shape().at(1);
    ov::Tensor res{first.get_element_type(), {res_d_0, res_d_1, res_d_2}};
    float* first_data = first.data<float>();
    float* second_data = second.data<float>();
    float* res_data = res.data<float>();
    for (size_t i = 0; i < res_d_0; ++i) {
        size_t j = 0;
        for (; j < first.get_shape().at(1); ++j) {
            std::copy_n(
                first_data + i * first.get_shape().at(1) * res_d_2 + j * res_d_2,
                res_d_2,
                res_data + i * res_d_1 * res_d_2 + j * res_d_2
            );
        }
        for (size_t k = 0; k < second.get_shape().at(1); ++k, ++j) {
            std::copy_n(
                second_data + i * second.get_shape().at(1) * res_d_2 + k * res_d_2,
                res_d_2,
                res_data + i * res_d_1 * res_d_2 + j * res_d_2
            );
        }
    }
    return res;
}

/// embed_dim: output dimension for each position
/// pos: a list of positions to be encoded: size (H, W)
/// out: (H, W, D)
ov::Tensor get_1d_sincos_pos_embed_from_grid_new(size_t embed_dim, const ov::Tensor& pos) {
    OPENVINO_ASSERT(embed_dim % 2 == 0);
    ov::Shape pos_shape = pos.get_shape();
    size_t H = pos_shape[0];
    size_t W = pos_shape[1];

    std::vector<float> omega(embed_dim / 2);
    for (size_t i = 0; i < omega.size(); ++i) {
        omega[i] = 1.0f / std::pow(10000.0f, float(i) / (embed_dim / 2));
    }

    std::vector<size_t> out_shape = {H, W, embed_dim};
    ov::Tensor emb(ov::element::f32, out_shape);

    float* pos_data = pos.data<float>();
    float* emb_data = emb.data<float>();

    size_t counter = 0;
    for (size_t h = 0; h < H; ++h) {
        for (size_t w = 0; w < W; ++w) {
            for (size_t d = 0; d < embed_dim / 2; ++d) {
                // Correctly access the 2D position grid
                float value = omega[d] * pos_data[h * W + w];
                // There should be sinf() and cosf(), but they don't exist on default Ubuntu20 gcc.
                emb_data[h * W * embed_dim + w * embed_dim + d] = std::sin(double(value));
                emb_data[h * W * embed_dim + w * embed_dim + d + (embed_dim / 2)] = std::cos(double(value));
            }
        }
    }
    return emb;
}

ov::Tensor get_2d_sincos_pos_embed_from_grid(size_t embed_dim, const ov::Tensor& grid) {
    OPENVINO_ASSERT(embed_dim % 2 == 0);
    ov::Shape grid_shape = grid.get_shape();
    float* grid_data = grid.data<float>();
    ov::Shape plane_shape{grid_shape.at(1), grid_shape.at(2)};
    ov::Tensor emb_h = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, ov::Tensor{
        ov::element::f32,
        plane_shape,
        grid_data
    });  // (H, W, D/2)
    ov::Tensor emb_w = get_1d_sincos_pos_embed_from_grid_new(embed_dim / 2, ov::Tensor{
        ov::element::f32,
        plane_shape,
        grid_data + plane_shape.at(0) * plane_shape.at(1)
    });  // (H, W, D/2)
    return concatenate_last_dim(emb_h, emb_w);
}

/// image_size: image_size or (image_height, image_width)
/// return:
/// pos_embed: [image_height, image_width, embed_dim]
ov::Tensor get_2d_sincos_pos_embed(size_t embed_dim, const ImageSize& image_size) {
    size_t grid_h_size = image_size.height, grid_w_size = image_size.width;
    ov::Tensor grid(ov::element::f32, {2, grid_h_size, grid_w_size});
    float* data = grid.data<float>();
    for (size_t y = 0; y < grid_h_size; ++y) {
        std::iota(data, data + grid_w_size, 0.0f);
        data += grid_w_size;
    }
    for (float y = 0.0f; y < grid_h_size; ++y) {
        std::fill(data, data + grid_w_size, y);
        data += grid_w_size;
    }
    return get_2d_sincos_pos_embed_from_grid(embed_dim, grid);
}

void adjust_pos_cache(
    const std::vector<ImageSize>& target_sizes,
    size_t hidden_size,
    ov::Tensor& pos_embed_cache
) {
    size_t max_h = std::max_element(target_sizes.begin(), target_sizes.end(), [](const ImageSize& left, const ImageSize& right) {
        return left.height < right.height;
    })->height;
    size_t max_w = std::max_element(target_sizes.begin(), target_sizes.end(), [](const ImageSize& left, const ImageSize& right) {
        return left.width < right.width;
    })->width;
    size_t allocated_height, allocated_width;
    if (pos_embed_cache) {
        const ov::Shape& allocated_shape = pos_embed_cache.get_shape();
        allocated_height = allocated_shape.at(0);
        allocated_width = allocated_shape.at(1);
    } else {
        allocated_height = allocated_width = 70;
    }
    if (max_h > allocated_height || max_w > allocated_width) {
        allocated_height = std::max(max_h, allocated_height);
        allocated_width = std::max(max_w, allocated_width);
        pos_embed_cache = get_2d_sincos_pos_embed(
            hidden_size, {allocated_height, allocated_width}
        );
    }
}

ov::Tensor merge_text_and_image_embeddings_llava(
    const ov::Tensor& input_ids,
    const ov::Tensor& text_embeds,
    const ov::Tensor& image_embeds,
    int64_t image_token_id
) {
    auto text_embeds_shape = text_embeds.get_shape();
    auto image_embeds_shape = image_embeds.get_shape();

    OPENVINO_ASSERT(
        text_embeds_shape[2] == image_embeds_shape[2],
        "Incompatible shapes between text_embeds and image_embeds"
    );

    size_t text_embeds_seq_length = text_embeds_shape[1];
    size_t hidden_size = text_embeds_shape[2];
    size_t image_embeds_seq_length = image_embeds_shape[1];

    size_t merged_seq_length = text_embeds_seq_length + (image_embeds_seq_length - 1);

    ov::Tensor merged_embeds(text_embeds.get_element_type(), {BATCH_SIZE, merged_seq_length, hidden_size});

    const int64_t* input_ids_data = input_ids.data<const int64_t>();
    const float* text_embeds_data = text_embeds.data<const float>();
    const float* image_embeds_data = image_embeds.data<const float>();
    float* merged_data = merged_embeds.data<float>();


    size_t merged_idx = 0;
    for (size_t s = 0; s < text_embeds_seq_length; ++s) {
        if (input_ids_data[s] == image_token_id) {
            for (size_t i = 0; i < image_embeds_seq_length; ++i) {
                std::copy_n(image_embeds_data + i * hidden_size,
                            hidden_size,
                            merged_data + merged_idx * hidden_size);
                merged_idx++;
            }
        } else {
            std::copy_n(text_embeds_data + s * hidden_size,
                        hidden_size,
                        merged_data + merged_idx * hidden_size);
            merged_idx++;
        }
    }

    return merged_embeds;
}

ov::Tensor reshape_and_rearrange_image_feature(const ov::Tensor& image_feature, 
                                               int num_patch_height, 
                                               int num_patch_width, 
                                               int height, 
                                               int width) {
    auto shape = image_feature.get_shape();
    OPENVINO_ASSERT(shape.size() == 3, "image_feature tensor must have 3 dimensions");
    
    size_t num_patches = shape[0];
    size_t patch_seq_len = shape[1];
    size_t embed_dim = shape[2];

    OPENVINO_ASSERT(
        num_patches == num_patch_height * num_patch_width,
        "Number of patches does not match the specified grid size"
    );

    OPENVINO_ASSERT(
        patch_seq_len == height * width,
        "Patch sequense length does not match the specified height and width"
    );

    // Reshape tensor data and permute dimensions 
    // [num_patches, patch_seq_len, embed_dim] -> [embed_dim, num_patch_height, height, num_patch_width, width]
    std::vector<float> reshaped_data(num_patches * patch_seq_len * embed_dim);
    const float* image_feature_data = image_feature.data<float>();
    
    for (int p = 0; p < num_patches; ++p) {
        for (int i = 0; i < patch_seq_len; ++i) {
            for (int e = 0; e < embed_dim; ++e) {
                int h = i / width;
                int w = i % width;
                int ph = p / num_patch_width;
                int pw = p % num_patch_width;
                reshaped_data[((((e * num_patch_height + ph) * height + h) * num_patch_width + pw) * width + w)] =
                    image_feature_data[(p * patch_seq_len + i) * embed_dim + e];
            }
        }
    }

    ov::Tensor result(image_feature.get_element_type(), 
                      {static_cast<size_t>(embed_dim), 
                       static_cast<size_t>(num_patch_height * height), 
                       static_cast<size_t>(num_patch_width * width)}
    );
    std::copy(reshaped_data.begin(), reshaped_data.end(), result.data<float>());
    return result;
}

/**
* @brief Unpads an image tensor of a padded and resized image.
* Used for packing image features of llava_next models.
*
* @param tensor An image tensor with a shape (embed_dim, height, width)
* @param original_size A size of original image
* @return An unpadded image tensor with a shape (embed_dim, new_height, new_width)
*/
ov::Tensor unpad_image(const ov::Tensor& tensor, const ImageSize& original_size) {
    size_t original_height = original_size.height;
    size_t original_width = original_size.width;
    auto shape = tensor.get_shape();
    size_t embed_dim = shape[0];
    size_t current_height = shape[1];
    size_t current_width = shape[2];

    float original_aspect_ratio = static_cast<float>(original_width) / original_height;
    float current_aspect_ratio = static_cast<float>(current_width) / current_height;

    ov::Tensor unpadded_tensor;

    if (original_aspect_ratio > current_aspect_ratio) {
        float scale_factor = static_cast<float>(current_width) / original_width;
        size_t new_height = static_cast<size_t>(original_height * scale_factor);
        size_t padding = (current_height - new_height) / 2;
        size_t unpadded_height_dim = new_height + 1;
        unpadded_tensor = ov::Tensor(tensor.get_element_type(), {embed_dim, unpadded_height_dim, current_width});
        
        for (size_t e = 0; e < embed_dim; ++e) {
            for (int h = 0; h < unpadded_height_dim; ++h) {
                std::copy(
                    tensor.data<float>() + (e * current_height * current_width + (padding + h) * current_width),
                    tensor.data<float>() + (e * current_height * current_width + (padding + h) * current_width + current_width),
                    unpadded_tensor.data<float>() + (e * unpadded_height_dim * current_width + h * current_width)
                );
            }
        }
    } else {
        float scale_factor = static_cast<float>(current_height) / original_height;
        size_t new_width = static_cast<size_t>(original_width * scale_factor);
        size_t padding = (current_width - new_width) / 2;
        size_t unpadded_width_dim = new_width + 1;
        unpadded_tensor = ov::Tensor(tensor.get_element_type(), {embed_dim, current_height, unpadded_width_dim});
        
        for (size_t e = 0; e < embed_dim; ++e) {
            for (int h = 0; h < current_height; ++h) {
                std::copy(
                    tensor.data<float>() + (e * current_height * current_width + h * current_width + padding),
                    tensor.data<float>() + (e * current_height * current_width + h * current_width + padding + unpadded_width_dim),
                    unpadded_tensor.data<float>() + (e * current_height * unpadded_width_dim + h * unpadded_width_dim)
                );
            }
        }
    }

    return unpadded_tensor;
}

/**
* @brief Adds image newline tensor to patches image feature tensor.
* Used for packing image features of llava_next models.
*
* @param image_feature A tensor with a shape (embed_dim, height, width)
* @param image_newline A tensor with a shape (embed_dim)
* @return A tensor with a shape (embed_dim, height, width + 1)
*/
ov::Tensor add_image_newline(const ov::Tensor& image_feature, const ov::Tensor& image_newline) {
    auto shape = image_feature.get_shape();

    OPENVINO_ASSERT(shape.size() == 3, "Input image_feature must have 3 dimensions");
    
    size_t embed_dim = shape[0];
    size_t height = shape[1];
    size_t width = shape[2];

    OPENVINO_ASSERT(image_newline.get_shape()[0] == embed_dim, "image_newline dimension must match embed_dim of image_feature");

    const float* image_feature_data = image_feature.data<float>();
    const float* newline_data = image_newline.data<float>();
    
    ov::Tensor feature_with_newline{image_feature.get_element_type(), {embed_dim, height, width + 1}};
    float* feature_with_newline_data = feature_with_newline.data<float>();

    for (size_t e = 0; e < embed_dim; ++e) {
        for (size_t h = 0; h < height; ++h) {
            // Copy original image feature data
            std::copy(
                image_feature_data + (e * height * width + h * width),
                image_feature_data + (e * height * width + (h + 1) * width),
                feature_with_newline_data + (e * height * (width + 1) + h * (width + 1))
            );
            // Add image newline
            feature_with_newline_data[e * height * (width + 1) + h * (width + 1) + width] = newline_data[e];
        }
    }
    
    return feature_with_newline;
}

/**
* @brief Flattens and transposes tensor.
* Used for packing image features of llava_next models.
*
* @param tensor A tensor with a shape (embed_dim, height, width)
* @return A tensor with a shape (height * width, embed_dim)
*/
ov::Tensor flatten_and_transpose(const ov::Tensor& tensor) {
    auto shape = tensor.get_shape();
    OPENVINO_ASSERT(shape.size() == 3, "Flattening tensor must have 3 dimensions");
    const float* data = tensor.data<float>();
    size_t embed_dim = shape[0];
    size_t height = shape[1];
    size_t width = shape[2];
    size_t flatten_dim = height * width;

    ov::Tensor flatten_feature(tensor.get_element_type(), {flatten_dim, embed_dim});
    float* flatten_feature_data = flatten_feature.data<float>();
    
    for (size_t h = 0; h < height; ++h) {
        for (size_t w = 0; w < width; ++w) {
            for (size_t e = 0; e < embed_dim; ++e) {
                flatten_feature_data[(h * width + w) * embed_dim + e] = data[e * flatten_dim + h * width + w];
            }
        }
    }

    return flatten_feature;
}

/**
* @brief Processes base and patches image features extracted from encoded image.
* Used in getting inputs embeds for llava_next models.
*
* @param encoded_image An encoded image retrieved from vision encoder
* @param original_image_size A size of the original image
* @param image_newline An image newline tensor with a shape (embed_dim)
* @return A tensor with a shape (1, new_seq_len, embed_dim)
*/
ov::Tensor pack_image_features_llava_next(
    const EncodedImage& encoded_image,
    const ImageSize& original_image_size, 
    const ov::Tensor& image_newline
) {
    auto image_feature = encoded_image.resized_source;
    auto image_feature_shape = image_feature.get_shape();
    size_t num_patches = image_feature_shape[0];
    size_t patch_seq_len = image_feature_shape[1];
    size_t embed_dim = image_feature_shape[2];

    const float* image_feature_data = image_feature.data<float>();
    const float* newline_data = image_newline.data<float>();

    if (num_patches > 1) {
        // Extract base image feature (first patch)
        ov::Tensor base_image_feature(image_feature.get_element_type(), {1, patch_seq_len, embed_dim});
        const float* src_data = image_feature.data<float>();
        float* dst_data = base_image_feature.data<float>();
        std::copy(src_data, src_data + patch_seq_len * embed_dim, dst_data);

        // Extract other grid patches
        ov::Tensor patches_image_feature(image_feature.get_element_type(), {num_patches - 1, patch_seq_len, embed_dim});
        dst_data = patches_image_feature.data<float>();
        std::copy(src_data + patch_seq_len * embed_dim, 
                  src_data + num_patches * patch_seq_len * embed_dim, 
                  dst_data);
        
        // Process grid patches image feature
        size_t height = encoded_image.resized_source_size.height;
        size_t width = encoded_image.resized_source_size.width;
        size_t num_patch_height = encoded_image.patches_grid.first;
        size_t num_patch_width = encoded_image.patches_grid.second;

        ov::Tensor reshaped_image_feature = reshape_and_rearrange_image_feature(patches_image_feature, num_patch_height, num_patch_width, height, width);
        
        ov::Tensor unpadded_image_feature = unpad_image(reshaped_image_feature, original_image_size);

        ov::Tensor image_feature_with_newline = add_image_newline(unpadded_image_feature, image_newline);

        ov::Tensor processed_image_feature = flatten_and_transpose(image_feature_with_newline);

        // Concatenate base image feature ([1, seq_len_1, emded_dim]) and patches image feature ([seq_len_2, embed_dim])
        auto base_shape = base_image_feature.get_shape();
        auto processed_shape = processed_image_feature.get_shape();

        const float* base_data = base_image_feature.data<float>();
        const float* processed_data = processed_image_feature.data<float>();

        ov::Tensor result(image_feature.get_element_type(), {1, base_shape[1] + processed_shape[0], embed_dim});
        // Copy base image feature data
        std::copy(base_data, base_data + base_shape[1] * embed_dim, result.data<float>());
        // Copy processed image feature data
        std::copy(processed_data,
                  processed_data + processed_shape[0] * embed_dim, 
                  result.data<float>() + base_shape[1] * embed_dim);
        return result;
    } else {
        // If there is only one patch, return the original (base) image feature concatenated with image_newline
        ov::Tensor result(image_feature.get_element_type(), {1, patch_seq_len + 1, embed_dim});
        // Copy base image feature data
        std::copy(image_feature_data + embed_dim, 
                  image_feature_data + patch_seq_len * embed_dim, 
                  result.data<float>());
        // Append image_newline data
        std::copy(newline_data, 
                  newline_data + embed_dim, 
                  result.data<float>() + patch_seq_len * embed_dim);
        return result;
    }
}
}

class ov::genai::VLMPipeline::VLMPipelineImpl {
public:
    // A config to follow for LLM input construction.
    VLMConfig m_vlm_config;
    // A config to follow for text generation.
    GenerationConfig m_generation_config;
    // A tokenizer encoding a prompt.
    Tokenizer m_tokenizer;
    // An encoder to infer embeddings of an image.
    VisionEncoder m_vision_encoder;
    // A resampler model to resample image embeddings.
    // [N, H*W, old_hidden_size] is the input shape.
    // [N, query_num, hidden_size] is the output shape.
    ov::InferRequest m_resampler;
    // A model to compute token embeddings.
    // Input shape: [N, conversation length].
    // Output shape: [1, conversation length, hidden_size].
    ov::InferRequest m_embedding;
    // A language model used to generate a response.
    // Input shapes: inputs_embeds[N, conversation length, hidden_size],
    // position_ids[N, conversation length], beam_idx[N].
    // Output shape: logits[N, conversation length, vocab_size].
    ov::InferRequest m_language;
    // Precomputed positional embeddings for the resampler.
    // [70, 70, hidden_size]. 70 is the initial guess of the image
    // height and width after dividing by patch_size.
    ov::Tensor m_pos_embed_cache;
    // True if chat mode is activated to save conversation
    // history between generate() calls.
    bool m_is_chat_conversation;
    ChatHistory m_history;
    std::string m_templated_chat_history;
    size_t m_image_id;  // Used to insert <image_id>i</image_id> per image (not a slice).

    VLMPipelineImpl(
        const std::filesystem::path& model_dir,
        const std::string& device,
        const ov::AnyMap device_config
    ) :
        m_vlm_config{
            utils::from_config_json_if_exists<ov::genai::VLMConfig>(
                model_dir, "config.json"
            )
        },
        m_tokenizer{Tokenizer(model_dir.string(), device_config)},
        m_vision_encoder(model_dir, m_vlm_config.model_type, device, device_config, ov::Core{}),
        m_is_chat_conversation{false},
        m_image_id{0} {
            if (m_vlm_config.model_type == VLMModelType::MINICPM) {
                m_resampler = ov::Core{}.compile_model(
                    model_dir / "resampler.xml", device, device_config
                ).create_infer_request();

                m_embedding = ov::Core{}.compile_model(
                    model_dir / "embed_tokens.xml", device, device_config
                ).create_infer_request();

                m_language = ov::Core{}.compile_model(
                    model_dir / "language_model.xml", device, device_config
                ).create_infer_request();

                m_pos_embed_cache = get_2d_sincos_pos_embed(m_vlm_config.hidden_size, {70, 70});
            } else if (m_vlm_config.model_type == VLMModelType::LLAVA || m_vlm_config.model_type == VLMModelType::LLAVA_NEXT) {
                m_language = ov::Core{}.compile_model(
                    model_dir / "openvino_language_model.xml", device, device_config
                ).create_infer_request();

                // Reusing the same m_embedding for llava text_embeddings model
                m_embedding = ov::Core{}.compile_model(
                    model_dir / "openvino_text_embeddings_model.xml", device, device_config
                ).create_infer_request();
            }

            m_language.get_tensor("attention_mask").set_shape({1, 0});
    }

    DecodedResults generate(
        const std::string& prompt,
        const std::vector<ov::Tensor>& rgbs,
        const GenerationConfig& generation_config,
        const StreamerVariant& streamer
    ) {
        ov::Tensor inputs_embeds;
        if (m_vlm_config.model_type == VLMModelType::MINICPM) {
            inputs_embeds = get_inputs_embeds_minicpm(prompt, rgbs);
        } else if (m_vlm_config.model_type == VLMModelType::LLAVA) {
            inputs_embeds = get_inputs_embeds_llava(prompt, rgbs);
        } else if (m_vlm_config.model_type == VLMModelType::LLAVA_NEXT) {
            inputs_embeds = get_inputs_embeds_llava_next(prompt, rgbs);
        }

        m_language.set_tensor("inputs_embeds", inputs_embeds);
        size_t history_len = m_language.get_tensor("attention_mask").get_shape().at(1);
        m_language.get_tensor("attention_mask").set_shape({1, history_len + inputs_embeds.get_shape()[1]});
        std::fill_n(m_language.get_tensor("attention_mask").data<int64_t>(), m_language.get_tensor("attention_mask").get_size(), 1);
        
        m_language.get_tensor("position_ids").set_shape({1, inputs_embeds.get_shape().at(1)});
        std::iota(m_language.get_tensor("position_ids").data<int64_t>(), m_language.get_tensor("position_ids").data<int64_t>() + m_language.get_tensor("position_ids").get_size(), history_len);
        
        m_language.get_tensor("beam_idx").set_shape({ BATCH_SIZE });
        m_language.get_tensor("beam_idx").data<int32_t>()[0] = 0;

        m_language.infer();

        ov::Shape logits_shape = m_language.get_tensor("logits").get_shape();
        auto attention_size = m_language.get_tensor("attention_mask").get_size();

        int64_t sequence_len = m_language.get_tensor("logits").get_shape().at(1) - 1;
        size_t vocab_size = m_language.get_tensor("logits").get_shape().back();
        float* logits = m_language.get_tensor("logits").data<float>() + sequence_len * vocab_size;
        int64_t out_token = std::max_element(logits, logits + vocab_size) - logits;

        m_language.get_tensor("inputs_embeds").set_shape({BATCH_SIZE, 1, m_vlm_config.hidden_size});
        m_language.get_tensor("position_ids").set_shape({ BATCH_SIZE, 1 });

        m_embedding.get_input_tensor().set_shape({ 1, 1 });

        int64_t eos_token_id = m_tokenizer.get_eos_token_id();
        std::shared_ptr<StreamerBase> streamer_ptr = std::visit(overloaded{
            [&m_tokenizer = m_tokenizer](
                const std::function<bool(std::string)>& callback
            ) -> std::shared_ptr<StreamerBase> {
                return std::make_shared<TextCallbackStreamer>(m_tokenizer, callback);
            },
            [](const std::shared_ptr<StreamerBase>& ptr) {
                return ptr;
            },
            [](std::monostate) {
                return std::shared_ptr<StreamerBase>{nullptr};
            },
        }, streamer);
        std::vector<int64_t> generated;
        while (true) {  //(out_token != eos_token_id)
            m_embedding.get_input_tensor().data<int64_t>()[0] = out_token;
            m_embedding.infer();
            const ov::Tensor& embed_prompt_tensor = m_embedding.get_output_tensor();
            float* embed_data = embed_prompt_tensor.data<float>();
            for (auto idx = 0; idx < embed_prompt_tensor.get_size(); idx++) {
                embed_data[idx] = embed_data[idx] * m_vlm_config.scale_emb;
            }

            m_language.set_tensor("inputs_embeds", embed_prompt_tensor);
            m_language.get_tensor("attention_mask").set_shape({ BATCH_SIZE, m_language.get_tensor("attention_mask").get_shape()[1] + 1 });
            std::fill_n(m_language.get_tensor("attention_mask").data<int64_t>(), m_language.get_tensor("attention_mask").get_size(), 1);
            m_language.get_tensor("position_ids").data<int64_t>()[0] = int64_t(m_language.get_tensor("attention_mask").get_size() - 1);

            m_language.infer();

            generated.push_back(out_token);
            if (streamer_ptr && streamer_ptr->put(out_token)) {
                break;
            }
            logits = m_language.get_tensor("logits").data<float>();

            out_token = std::max_element(logits, logits + vocab_size) - logits;
            if (out_token == eos_token_id) {
                break;
            }
        }

        if (streamer_ptr) {
            streamer_ptr->end();
        }

        std::string decoded_results = m_tokenizer.decode(generated);
        if (m_is_chat_conversation) {
            // Tail of chat template is missing in KV cache.
            // Find the tail to concatenate it with the next input prompt.
            m_templated_chat_history.append(decoded_results);
            m_history.push_back({{"role", "assistant"}, {"content", decoded_results}});
        } else {
            for (auto& variable : m_language.query_state()) {
                variable.reset();
            }
            m_language.get_tensor("attention_mask").set_shape({1, 0});
        }
        return {{std::move(decoded_results)}};
    }

    DecodedResults generate(
        const std::string& prompt,
        const ov::AnyMap& config_map
    ) {
        auto image = config_map.find(ov::genai::image.name());
        auto images = config_map.find(ov::genai::images.name());
        OPENVINO_ASSERT(
            config_map.end() == image || config_map.end() == images,
            "Only one property can be set: image of images."
        );
        std::vector<ov::Tensor> rgbs;
        if (config_map.end() != image) {
            rgbs = {image->second.as<ov::Tensor>()};
        } if (config_map.end() != images) {
            rgbs = images->second.as<std::vector<ov::Tensor>>();
        }
        ov::genai::OptionalGenerationConfig config_arg = utils::get_config_from_map(config_map);
        GenerationConfig config = (config_arg.has_value()) ? *config_arg : get_generation_config();
        config.update_generation_config(config_map);
        return generate(
            prompt,
            rgbs,
            config,
            utils::get_streamer_from_map(config_map)
        );
    }

    void start_chat(const std::string& system_message) {
        m_is_chat_conversation = true;
        bool have_state = 0 != m_language.get_tensor("attention_mask").get_size();
        if (have_state) {
            // Resetting state may be slow.
            for (ov::VariableState& variable : m_language.query_state()) {
                variable.reset();
            }
            // Since if is already introduced, move all resetting here.
            m_language.get_tensor("attention_mask").set_shape({1, 0});
            m_history.clear();
            m_templated_chat_history.clear();
        }
        if (system_message.empty()) {
            return;
        }
        m_history = {{{"role", "system"}, {"content", system_message}}};
        constexpr bool add_generation_prompt = false;
        m_templated_chat_history = m_tokenizer.apply_chat_template(m_history, add_generation_prompt);
    }

    void finish_chat() {m_is_chat_conversation = false;}

    Tokenizer get_tokenizer() const {
        return m_tokenizer;
    }

    void set_chat_template(const std::string& new_template) {
        m_tokenizer.set_chat_template(new_template);
    }

    GenerationConfig get_generation_config() const {
        return m_generation_config;
    }

    void set_generation_config(const GenerationConfig& new_config) {
        m_generation_config = new_config;
    }

    ov::Tensor get_inputs_embeds_llava(const std::string& prompt, const std::vector<ov::Tensor>& images) {
        std::string image_token = m_vlm_config.im_start;
        // TODO Enable and reuse chat mode
        std::string formatted_prompt = "USER: " + (images.empty() ? prompt : image_token + "\n" + prompt) + " ASSISTANT:";
        ov::Tensor input_ids = m_tokenizer.encode(formatted_prompt).input_ids;
        if (images.empty()) {
            return process_prompt(m_embedding, input_ids, m_vlm_config.scale_emb);
        } else {
            OPENVINO_ASSERT(1 == images.size(), "Only a single image allowed");
            EncodedImage encoded_image = m_vision_encoder.encode(images.at(0));
            ov::Tensor image_embeds = encoded_image.resized_source;
            
            ov::Tensor text_embeds = process_prompt(m_embedding, input_ids, m_vlm_config.scale_emb);

            ov::Tensor encoded_image_token = m_tokenizer.encode(image_token, ov::genai::add_special_tokens(false)).input_ids;
            int64_t image_token_id = encoded_image_token.data<int64_t>()[encoded_image_token.get_size() - 1];

            return merge_text_and_image_embeddings_llava(input_ids, text_embeds, image_embeds, image_token_id);
        }
    }

    ov::Tensor get_inputs_embeds_llava_next(const std::string& prompt, const std::vector<ov::Tensor>& images) {
        std::string image_token = m_vlm_config.im_start;
        // TODO Enable and reuse chat mode
        // TODO Support chat templates for different llava_next models
        std::string formatted_prompt = "USER: " + (images.empty() ? prompt : image_token + "\n" + prompt) + " ASSISTANT:";
        ov::Tensor input_ids = m_tokenizer.encode(formatted_prompt).input_ids;
        if (images.empty()) {
            return process_prompt(m_embedding, input_ids, m_vlm_config.scale_emb);
        } else {
            OPENVINO_ASSERT(1 == images.size(), "Only a single image allowed");
            EncodedImage encoded_image = m_vision_encoder.encode(images.at(0));

            // Create image_newline tensor with data from config
            size_t embed_dim = encoded_image.resized_source.get_shape().at(2);
            ov::Tensor image_newline(encoded_image.resized_source.get_element_type(), {embed_dim});
            float* image_newline_data = image_newline.data<float>();
            std::copy(m_vlm_config.image_newline.begin(), m_vlm_config.image_newline.end(), image_newline_data);

            ImageSize original_image_size{images.at(0).get_shape().at(2), images.at(0).get_shape().at(3)}; // [height, width]

            ov::Tensor image_features = pack_image_features_llava_next(encoded_image, original_image_size, image_newline);

            ov::Tensor text_embeds = process_prompt(m_embedding, input_ids, m_vlm_config.scale_emb);

            ov::Tensor encoded_image_token = m_tokenizer.encode(image_token, ov::genai::add_special_tokens(false)).input_ids;
            int64_t image_token_id = encoded_image_token.data<int64_t>()[encoded_image_token.get_size() - 1];

            return merge_text_and_image_embeddings_llava(input_ids, text_embeds, image_features, image_token_id);
        }
    }

    ov::Tensor get_inputs_embeds_minicpm(const std::string& prompt, const std::vector<ov::Tensor>& images) {
        std::string images_prompt;
        std::vector<EncodedImage> embeds;
        for (const ov::Tensor& rgb : images) {
            ov::Tensor reshaped = rgb;
            ov::Shape rgb_shape = rgb.get_shape();
            switch (rgb_shape.size()) {
                case 3:
                    reshaped.set_shape({1, rgb_shape.at(0), rgb_shape.at(1), rgb_shape.at(2)});
                    break;
                case 4: break;
                default: OPENVINO_THROW("Input image must have [NHWC] or [HWC] layout");
            }
            ov::Shape reshaped_shape = reshaped.get_shape();
            for (size_t batch_idx = 0; batch_idx < reshaped_shape.at(0); ++batch_idx) {
                ov::Tensor single_image{
                    ov::element::u8,
                    {1, reshaped_shape.at(1), reshaped_shape.at(2), reshaped_shape.at(3)},
                    reshaped.data<uint8_t>() + batch_idx * reshaped_shape.at(1) * reshaped_shape.at(1) * reshaped_shape.at(1)
                };
                EncodedImage encoded_image = m_vision_encoder.encode(single_image);
                if (m_vlm_config.use_image_id) {
                    images_prompt += m_vlm_config.im_id_start + std::to_string(m_image_id) + m_vlm_config.im_id_end;
                    ++m_image_id;
                }
                std::string unk64;
                for (size_t idx = 0; idx < m_vlm_config.query_num; ++idx) {
                    unk64 += m_vlm_config.unk;
                }
                images_prompt += m_vlm_config.im_start + unk64 + m_vlm_config.im_end;
                if (encoded_image.slices) {
                    ov::Shape slices_shape = encoded_image.slices.get_shape();
                    for (size_t row_idx = 0; row_idx < slices_shape.at(0); ++row_idx) {
                        for (size_t col_idx = 0; col_idx < slices_shape.at(1); ++col_idx) {
                            images_prompt += m_vlm_config.slice_start + unk64 + m_vlm_config.slice_end;
                        }
                        images_prompt += '\n';
                    }
                }
                if ('\n' != *(images_prompt.end() - 1)) {
                    // Image wasn't sliced, add \n to the end of image anyway.
                    // Strangely, \n isn't placed between </image><slice>.
                    images_prompt += '\n';
                }
                embeds.push_back(std::move(encoded_image));
            }
        }
        images_prompt += prompt;
        ov::Tensor encoded_input;
        if (m_is_chat_conversation) {
            // KV cache in model already contains prompts and answers from previous iterations.
            // So only new prompt wrapped into chat template to be sent into model. Tokenizer always returns
            // token_ids = {<bos token>, ...<valuable tokens>}. So if tokenizer applies only to the new prompt,
            // <bos token> will be inserted on every iteration.
            // So actual pipeline calculates input_ids for whole chat history + for whole chat history without the new prompt
            // and takes only the difference between them.
            // The chat history cannot be saved as already encoded tokens because generate call doesn't return <eos> token, but
            // KV cache contains it. So we have to add it manually or get it by tokenization all chat history.
            m_history.push_back({{"role", "user"}, {"content", images_prompt}});
            constexpr bool add_generation_prompt = true;
            std::string new_templated_chat_history = m_tokenizer.apply_chat_template(m_history, add_generation_prompt);
            ov::Tensor new_chat_tokens = m_tokenizer.encode(new_templated_chat_history).input_ids;
            if (0 == m_language.get_tensor("attention_mask").get_shape().at(1)) {
                encoded_input = new_chat_tokens;
            } else {
                TokenizedInputs prev_chat_tokens = m_tokenizer.encode(
                    m_templated_chat_history
                );
                encoded_input = utils::subtract_chat_tokenized_inputs(
                    {new_chat_tokens}, prev_chat_tokens
                ).input_ids;
            }
            m_templated_chat_history = std::move(new_templated_chat_history);
        } else {
            encoded_input = m_tokenizer.encode(images_prompt).input_ids;
        }
        m_embedding.set_input_tensor(encoded_input);
        m_embedding.infer();
        ov::Tensor inputs_embeds = m_embedding.get_output_tensor();
        OPENVINO_ASSERT(
            m_vlm_config.hidden_size == inputs_embeds.get_shape().at(2),
            "Unexpected embedding size"
        );
        ov::Tensor special_tokens = m_tokenizer.encode(
            m_vlm_config.im_start
            + m_vlm_config.im_end
            + m_vlm_config.slice_start
            + m_vlm_config.slice_end
        ).input_ids;
        OPENVINO_ASSERT(
            4 == special_tokens.get_shape().at(1),
            "Every special token must be represented with a single int."
        );
        int64_t im_start_id = special_tokens.data<int64_t>()[0];
        int64_t im_end_id = special_tokens.data<int64_t>()[1];
        int64_t slice_start_id = special_tokens.data<int64_t>()[2];
        int64_t slice_end_id = special_tokens.data<int64_t>()[3];
        int64_t im_start_pos = 0, slice_start_pos = 0;
        int64_t* begin = encoded_input.data<int64_t>();
        int64_t* ids = begin;
        size_t encoded_input_size = encoded_input.get_size();
        int64_t* end = ids + encoded_input_size;
        float* inputs_embeds_data = inputs_embeds.data<float>();
        for (const EncodedImage& encoded_image : embeds) {
            const ov::Tensor& resampled_source = resample(*this, encoded_image.resized_source, {encoded_image.resized_source_size});
            float* emb = resampled_source.data<float>();
            ids = std::find(ids, end, im_start_id);
            OPENVINO_ASSERT(end != ids);
            ++ids;
            std::copy_n(emb, resampled_source.get_size(), inputs_embeds_data + std::distance(begin, ids) * m_vlm_config.hidden_size);
            ids += m_vlm_config.query_num;
            if (encoded_image.slices) {
                size_t token_idx = 0;
                const ov::Shape& slices_shape = encoded_image.slices.get_shape();
                for (size_t i = 0; i < slices_shape.at(0); ++i) {
                    for (size_t ja = 0; ja < slices_shape.at(1); ++ja) {
                        size_t d2 = slices_shape.at(2);
                        size_t d3 = slices_shape.at(3);
                        ov::Tensor encoded_view{ov::element::f32, {1, d2, d3}, encoded_image.slices.data<float>() + (i * slices_shape.at(1) + ja) * d2 * d3};
                        const ov::Tensor& vision_embed_tensor_i_j = resample(*this, encoded_view, {encoded_image.slices_size});
                        ids = std::find(ids, end, slice_start_id);
                        OPENVINO_ASSERT(end != ids);
                        ++ids;
                        std::copy_n(vision_embed_tensor_i_j.data<float>(), vision_embed_tensor_i_j.get_size(), inputs_embeds_data + std::distance(begin, ids) * m_vlm_config.hidden_size);
                        ids += m_vlm_config.query_num;
                    }
                }
            }
        }

        return inputs_embeds;
    }

    ov::Tensor resample(VLMPipeline::VLMPipelineImpl& pipe, const ov::Tensor& encoded_image, const std::vector<ImageSize>& target_sizes) {
        size_t bs = encoded_image.get_shape().at(0);
        std::vector<size_t> patch_len{target_sizes.size()};
        std::transform(target_sizes.begin(), target_sizes.end(), patch_len.begin(), [](const ImageSize& height_width) {
            return height_width.height * height_width.width;
        });
        adjust_pos_cache(
            target_sizes,
            pipe.m_vlm_config.hidden_size,
            pipe.m_pos_embed_cache
        );
        size_t max_patch_len = *std::max_element(patch_len.begin(), patch_len.end());
        ov::Tensor key_padding_mask(ov::element::boolean, {bs, max_patch_len});
        bool* mask_data = key_padding_mask.data<bool>();
        size_t embed_len = pipe.m_pos_embed_cache.get_shape().at(2);
        ov::Tensor pos_embed(ov::element::f32, {max_patch_len, bs, embed_len});  // BLD => L * B * D
        float* pos_embed_data = pos_embed.data<float>();
        float* cache_data = pipe.m_pos_embed_cache.data<float>();
        size_t _d0 = pipe.m_pos_embed_cache.get_shape().at(0);
        size_t _d1 = pipe.m_pos_embed_cache.get_shape().at(1);
        for (size_t i = 0; i < bs; ++i) {
            size_t target_h = target_sizes.at(i).height;
            size_t target_w = target_sizes.at(i).width;
            for (size_t h_idx = 0; h_idx < target_h; ++h_idx) {
                for (size_t w_idx = 0; w_idx < target_w; ++w_idx) {
                    std::copy_n(
                        cache_data + (h_idx * _d1 + w_idx) * embed_len,
                        embed_len,
                        pos_embed_data + (h_idx * target_w + w_idx) * bs * embed_len + i * embed_len
                    );
                }
            }
            for (size_t flat = target_h * target_w; flat < max_patch_len; ++flat) {
                std::fill_n(pos_embed_data + flat * bs * embed_len + i * embed_len, embed_len, 0.0f);
            }
            std::fill_n(mask_data + i * max_patch_len, patch_len[i], false);
            std::fill_n(mask_data + i * max_patch_len + patch_len[i], max_patch_len - patch_len[i], true);
        }
        pipe.m_resampler.set_tensor("x", encoded_image);  // [N, H*W, old_hidden_size]
        pipe.m_resampler.set_tensor("pos_embed", pos_embed);  // [H*W, N, new_hidden_size]
        pipe.m_resampler.set_tensor("key_padding_mask", key_padding_mask);  // [N, H*W]
        pipe.m_resampler.infer();
        return pipe.m_resampler.get_output_tensor();  // [N, query_num, new_hidden_size]
    }
};

VLMPipeline::VLMPipeline(
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config
) : m_pimpl{std::make_unique<VLMPipelineImpl>(model_dir, device, device_config)} {}

ov::genai::VLMPipeline::~VLMPipeline() = default;

DecodedResults VLMPipeline::generate(
    const std::string& prompt,
    const std::vector<ov::Tensor>& rgbs,
    const GenerationConfig& generation_config,
    const StreamerVariant& streamer
) {
    return m_pimpl->generate(prompt, rgbs, generation_config, streamer);
}

DecodedResults VLMPipeline::generate(
    const std::string& prompt,
    const ov::AnyMap& config_map
) {
    return m_pimpl->generate(prompt, config_map);
}

void VLMPipeline::start_chat(const std::string& system_message) {
    m_pimpl->start_chat(system_message);
}

void VLMPipeline::finish_chat() {
    m_pimpl->finish_chat();
}

void VLMPipeline::set_chat_template(const std::string& new_template) {
    m_pimpl->set_chat_template(new_template);
}

Tokenizer VLMPipeline::get_tokenizer() const {
    return m_pimpl->get_tokenizer();
}

GenerationConfig VLMPipeline::get_generation_config() const {
    return m_pimpl->get_generation_config();
}

void VLMPipeline::set_generation_config(const GenerationConfig& new_config) {
    m_pimpl->set_generation_config(new_config);
}
