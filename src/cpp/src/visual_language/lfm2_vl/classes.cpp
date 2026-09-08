// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/lfm2_vl/classes.hpp"

#include <algorithm>
#include <cmath>
#include <set>

#include "visual_language/clip.hpp"

#include "utils.hpp"

namespace ov::genai {

namespace {

// Fixed special tokens used by Lfm2VlProcessor (see transformers processing_lfm2_vl.py).
constexpr char IMAGE_TOKEN[] = "<image>";
constexpr char IMAGE_START_TOKEN[] = "<|image_start|>";
constexpr char IMAGE_END_TOKEN[] = "<|image_end|>";
constexpr char IMAGE_THUMBNAIL_TOKEN[] = "<|img_thumbnail|>";

int round_by_factor(double number, int factor) {
    return static_cast<int>(std::llround(number / static_cast<double>(factor))) * factor;
}

// Mirrors Lfm2VlImageProcessor.smart_resize. Returns (new_width, new_height).
std::pair<int, int> smart_resize(int height,
                                 int width,
                                 int downsample_factor,
                                 int min_image_tokens,
                                 int max_image_tokens,
                                 int encoder_patch_size) {
    const int total_factor = encoder_patch_size * downsample_factor;
    const double min_pixels = static_cast<double>(min_image_tokens) * encoder_patch_size * encoder_patch_size *
                              downsample_factor * downsample_factor;
    const double max_pixels = static_cast<double>(max_image_tokens) * encoder_patch_size * encoder_patch_size *
                              downsample_factor * downsample_factor;

    int h_bar = std::max(total_factor, round_by_factor(height, total_factor));
    int w_bar = std::max(total_factor, round_by_factor(width, total_factor));

    if (static_cast<double>(h_bar) * w_bar > max_pixels) {
        const double beta = std::sqrt((static_cast<double>(height) * width) / max_pixels);
        h_bar = std::max(total_factor,
                         static_cast<int>(std::floor(height / beta / total_factor)) * total_factor);
        w_bar = std::max(total_factor,
                         static_cast<int>(std::floor(width / beta / total_factor)) * total_factor);
    } else if (static_cast<double>(h_bar) * w_bar < min_pixels) {
        const double beta = std::sqrt(min_pixels / (static_cast<double>(height) * width));
        h_bar = static_cast<int>(std::ceil(height * beta / total_factor)) * total_factor;
        w_bar = static_cast<int>(std::ceil(width * beta / total_factor)) * total_factor;
    }
    return {w_bar, h_bar};
}

// Mirrors Lfm2VlImageProcessor._is_image_too_large.
bool is_image_too_large(int height,
                        int width,
                        int max_image_tokens,
                        int encoder_patch_size,
                        int downsample_factor,
                        float max_pixels_tolerance) {
    const int total_factor = encoder_patch_size * downsample_factor;
    const int h_bar = std::max(encoder_patch_size, round_by_factor(height, total_factor));
    const int w_bar = std::max(encoder_patch_size, round_by_factor(width, total_factor));
    const double limit = static_cast<double>(max_image_tokens) * encoder_patch_size * encoder_patch_size *
                         downsample_factor * downsample_factor * max_pixels_tolerance;
    return static_cast<double>(h_bar) * w_bar > limit;
}

// Mirrors Lfm2VlImageProcessor._target_ratios.
std::vector<std::pair<int, int>> target_ratios(int min_tiles, int max_tiles) {
    std::set<std::pair<int, int>> ratios;
    for (int n = min_tiles; n <= max_tiles; ++n) {
        for (int w = 1; w <= n; ++w) {
            for (int h = 1; h <= n; ++h) {
                if (w * h >= min_tiles && w * h <= max_tiles) {
                    ratios.insert({w, h});
                }
            }
        }
    }
    std::vector<std::pair<int, int>> result(ratios.begin(), ratios.end());
    std::stable_sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return a.first * a.second < b.first * b.second;
    });
    return result;
}

// Mirrors find_closest_aspect_ratio in image_processing_lfm2_vl.py.
std::pair<int, int> find_closest_aspect_ratio(double aspect_ratio,
                                              const std::vector<std::pair<int, int>>& ratios,
                                              int width,
                                              int height,
                                              int image_size) {
    double best_ratio_diff = std::numeric_limits<double>::infinity();
    std::pair<int, int> best_ratio{1, 1};
    const double area = static_cast<double>(width) * height;
    for (const auto& ratio : ratios) {
        const double target_aspect_ratio = static_cast<double>(ratio.first) / ratio.second;
        const double ratio_diff = std::abs(aspect_ratio - target_aspect_ratio);
        if (ratio_diff < best_ratio_diff) {
            best_ratio_diff = ratio_diff;
            best_ratio = ratio;
        } else if (ratio_diff == best_ratio_diff) {
            const double target_area =
                static_cast<double>(image_size) * image_size * ratio.first * ratio.second;
            if (area > 0.5 * target_area) {
                best_ratio = ratio;
            }
        }
    }
    return best_ratio;
}

// Normalize a single tile (RGB u8, HWC) and pack into [num_patches, patch_size*patch_size*3]
// following convert_image_to_patches(): patch order is row-major (ph, pw); within a patch
// the layout is (patch_row, patch_col, channel).
ov::Tensor patchify(const clip_image_u8& tile,
                    int patch_size,
                    const std::array<float, 3>& mean,
                    const std::array<float, 3>& std,
                    int& num_patches_height,
                    int& num_patches_width) {
    const int W = tile.nx;
    const int H = tile.ny;
    num_patches_height = H / patch_size;
    num_patches_width = W / patch_size;
    const int num_patches = num_patches_height * num_patches_width;
    const int patch_dim = patch_size * patch_size * 3;

    ov::Tensor pixel_values(ov::element::f32, ov::Shape{1, static_cast<size_t>(num_patches), static_cast<size_t>(patch_dim)});
    float* dst = pixel_values.data<float>();

    for (int ph = 0; ph < num_patches_height; ++ph) {
        for (int pw = 0; pw < num_patches_width; ++pw) {
            const int patch_idx = ph * num_patches_width + pw;
            float* patch_dst = dst + static_cast<size_t>(patch_idx) * patch_dim;
            for (int py = 0; py < patch_size; ++py) {
                const int y = ph * patch_size + py;
                for (int px = 0; px < patch_size; ++px) {
                    const int x = pw * patch_size + px;
                    const size_t src_off = static_cast<size_t>(3) * (static_cast<size_t>(y) * W + x);
                    for (int c = 0; c < 3; ++c) {
                        const float v = static_cast<float>(tile.buf[src_off + c]) / 255.0f;
                        patch_dst[py * patch_size * 3 + px * 3 + c] = (v - mean[c]) / std[c];
                    }
                }
            }
        }
    }
    return pixel_values;
}

// Run the exported vision embeddings model for a single tile and return its
// projected features [num_downsampled_tokens, hidden_size].
ov::Tensor encode_tile(ov::InferRequest& encoder,
                       const clip_image_u8& tile,
                       const ProcessorConfig& config) {
    int nph = 0;
    int npw = 0;
    ov::Tensor pixel_values = patchify(tile, static_cast<int>(config.encoder_patch_size),
                                       config.image_mean, config.image_std, nph, npw);
    const size_t num_patches = pixel_values.get_shape().at(1);

    ov::Tensor spatial_shapes(ov::element::i64, ov::Shape{1, 2});
    spatial_shapes.data<int64_t>()[0] = nph;
    spatial_shapes.data<int64_t>()[1] = npw;

    const ov::element::Type mask_type = encoder.get_tensor("pixel_attention_mask").get_element_type();
    ov::Tensor pixel_attention_mask(mask_type, ov::Shape{1, num_patches});
    if (mask_type == ov::element::u8 || mask_type == ov::element::boolean) {
        std::fill_n(pixel_attention_mask.data<uint8_t>(), num_patches, static_cast<uint8_t>(1));
    } else if (mask_type == ov::element::i8) {
        std::fill_n(pixel_attention_mask.data<int8_t>(), num_patches, static_cast<int8_t>(1));
    } else if (mask_type == ov::element::i32) {
        std::fill_n(pixel_attention_mask.data<int32_t>(), num_patches, 1);
    } else if (mask_type == ov::element::i64) {
        std::fill_n(pixel_attention_mask.data<int64_t>(), num_patches, static_cast<int64_t>(1));
    } else {
        OPENVINO_THROW("[LFM2-VL] Unsupported pixel_attention_mask element type: ", mask_type);
    }

    encoder.set_tensor("pixel_values", pixel_values);
    encoder.set_tensor("spatial_shapes", spatial_shapes);
    encoder.set_tensor("pixel_attention_mask", pixel_attention_mask);
    encoder.infer();

    const ov::Tensor& out = encoder.get_output_tensor();
    ov::Tensor features(out.get_element_type(), out.get_shape());
    std::memcpy(features.data(), out.data(), out.get_byte_size());
    return features;
}

clip_image_u8 resize_u8(const clip_image_u8& image, int target_width, int target_height) {
    clip_image_u8 resized;
    bicubic_resize(image, resized, target_width, target_height);
    return resized;
}

// Extract a tile_size x tile_size sub-image from a (already resized) image at grid cell (row, col).
clip_image_u8 crop_tile(const clip_image_u8& image, int row, int col, int tile_size) {
    clip_image_u8 tile;
    tile.nx = tile_size;
    tile.ny = tile_size;
    tile.buf.resize(static_cast<size_t>(tile_size) * tile_size * 3);
    const int W = image.nx;
    for (int y = 0; y < tile_size; ++y) {
        const int src_y = row * tile_size + y;
        for (int x = 0; x < tile_size; ++x) {
            const int src_x = col * tile_size + x;
            const size_t src_off = static_cast<size_t>(3) * (static_cast<size_t>(src_y) * W + src_x);
            const size_t dst_off = static_cast<size_t>(3) * (static_cast<size_t>(y) * tile_size + x);
            tile.buf[dst_off + 0] = image.buf[src_off + 0];
            tile.buf[dst_off + 1] = image.buf[src_off + 1];
            tile.buf[dst_off + 2] = image.buf[src_off + 2];
        }
    }
    return tile;
}

} // namespace

EncodedImage VisionEncoderLFM2VL::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();

    ProcessorConfig config = ProcessorConfig::from_any_map(config_map, m_processor_config);

    clip_image_u8 source = tensor_to_clip_image_u8(image);
    const int H = source.ny;
    const int W = source.nx;

    const int ds = static_cast<int>(config.downsample_factor);
    const int enc = static_cast<int>(config.encoder_patch_size);
    const int tile_size = static_cast<int>(config.tile_size);
    const bool do_splitting = config.do_image_splitting && !(config.min_tiles == 1 && config.max_tiles == 1);

    // Thumbnail / single-tile target size.
    auto [new_w, new_h] = smart_resize(H, W, ds, static_cast<int>(config.min_image_tokens),
                                       static_cast<int>(config.max_image_tokens), enc);

    const bool is_large = is_image_too_large(H, W, static_cast<int>(config.max_image_tokens), enc, ds,
                                             config.max_pixels_tolerance);

    std::vector<clip_image_u8> tiles;  // tiles first, thumbnail last (matches HF token order)
    int grid_rows = 1;
    int grid_cols = 1;

    if (is_large && do_splitting) {
        const double aspect_ratio = static_cast<double>(W) / H;
        const auto ratios = target_ratios(static_cast<int>(config.min_tiles), static_cast<int>(config.max_tiles));
        auto [grid_width, grid_height] = find_closest_aspect_ratio(aspect_ratio, ratios, W, H, tile_size);
        grid_cols = grid_width;
        grid_rows = grid_height;

        clip_image_u8 resized = resize_u8(source, tile_size * grid_width, tile_size * grid_height);
        for (int row = 0; row < grid_height; ++row) {
            for (int col = 0; col < grid_width; ++col) {
                tiles.push_back(crop_tile(resized, row, col, tile_size));
            }
        }
        if (config.use_thumbnail && grid_width * grid_height != 1) {
            tiles.push_back(resize_u8(source, new_w, new_h));
        }
    } else {
        tiles.push_back(resize_u8(source, new_w, new_h));
    }

    std::vector<ov::Tensor> tile_features;
    tile_features.reserve(tiles.size());
    size_t total_tokens = 0;
    size_t hidden_size = 0;
    for (const clip_image_u8& tile : tiles) {
        ov::Tensor feat = encode_tile(encoder, tile, config);
        hidden_size = feat.get_shape().back();
        total_tokens += feat.get_shape().at(0);
        tile_features.push_back(std::move(feat));
    }

    // Concatenate all tile features into [1, total_tokens, hidden_size].
    ov::Tensor image_features(ov::element::f32, ov::Shape{1, total_tokens, hidden_size});
    float* dst = image_features.data<float>();
    for (const ov::Tensor& feat : tile_features) {
        std::memcpy(dst, feat.data(), feat.get_byte_size());
        dst += feat.get_size();
    }

    EncodedImage encoded;
    encoded.resized_source = std::move(image_features);
    encoded.num_image_tokens = total_tokens;
    encoded.patches_grid = {grid_rows, grid_cols};
    encoded.original_image_size = ImageSize{static_cast<size_t>(H), static_cast<size_t>(W)};
    return encoded;
}

InputsEmbedderLFM2VL::InputsEmbedderLFM2VL(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, model_dir, tokenizer, device, device_config) { }

InputsEmbedderLFM2VL::InputsEmbedderLFM2VL(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) { }

NormalizedPrompt InputsEmbedderLFM2VL::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    const std::string image_token = IMAGE_TOKEN;
    auto [unified_prompt, images_sequence] = normalize(prompt, image_token, image_token, base_id, images.size());

    const ProcessorConfig& cfg = m_vision_encoder->get_processor_config();
    const size_t patches_per_side = static_cast<size_t>(cfg.tile_size) / cfg.encoder_patch_size;
    const size_t downsampled_per_side =
        (patches_per_side + cfg.downsample_factor - 1) / cfg.downsample_factor;
    const size_t tokens_per_tile = downsampled_per_side * downsampled_per_side;

    size_t search_offset = 0;
    for (size_t new_image_id : images_sequence) {
        const EncodedImage& enc = images.at(new_image_id - base_id);
        const int rows = enc.patches_grid.first;
        const int cols = enc.patches_grid.second;
        const size_t total_tokens = enc.num_image_tokens;
        const bool multi_tile = rows > 1 || cols > 1;

        std::string expanded_tag = IMAGE_START_TOKEN;
        if (multi_tile) {
            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < cols; ++col) {
                    expanded_tag += "<|img_row_" + std::to_string(row + 1) + "_col_" + std::to_string(col + 1) + "|>";
                    for (size_t i = 0; i < tokens_per_tile; ++i) {
                        expanded_tag += image_token;
                    }
                }
            }
            const size_t thumbnail_tokens = total_tokens - static_cast<size_t>(rows) * cols * tokens_per_tile;
            expanded_tag += IMAGE_THUMBNAIL_TOKEN;
            for (size_t i = 0; i < thumbnail_tokens; ++i) {
                expanded_tag += image_token;
            }
        } else {
            for (size_t i = 0; i < total_tokens; ++i) {
                expanded_tag += image_token;
            }
        }
        expanded_tag += IMAGE_END_TOKEN;

        const size_t pos = unified_prompt.find(image_token, search_offset);
        OPENVINO_ASSERT(pos != std::string::npos, "[LFM2-VL] Failed to find image token in prompt during normalization");
        unified_prompt.replace(pos, image_token.length(), expanded_tag);
        search_offset = pos + expanded_tag.size();
    }
    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderLFM2VL::get_inputs_embeds(const std::string& unified_prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& images_sequence) {
    std::vector<const ov::Tensor*> image_embeds;
    image_embeds.reserve(images_sequence.size());
    for (size_t new_image_id : images_sequence) {
        image_embeds.push_back(&images.at(new_image_id).resized_source);
    }

    ov::Tensor input_ids = get_encoded_input_ids(unified_prompt, metrics);
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = get_text_embedding(req, input_ids, metrics);

    ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
    std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());

    if (image_embeds.empty()) {
        return inputs_embeds;
    }

    // masked_scatter: replace embeddings at every image-placeholder position, in order,
    // with the concatenated image feature rows (mirrors transformers modeling_lfm2_vl.py).
    const int64_t image_token_id = m_vlm_config.image_token_id;
    OPENVINO_ASSERT(image_token_id >= 0, "[LFM2-VL] image_token_id is not set in config.json");

    const int64_t* input_ids_data = input_ids.data<const int64_t>();
    const size_t seq_len = input_ids.get_shape().at(1);
    const size_t hidden_size = inputs_embeds.get_shape().at(2);
    float* embeds_data = inputs_embeds.data<float>();

    size_t cur_image = 0;
    size_t cur_row = 0;  // row within current image's features
    for (size_t i = 0; i < seq_len; ++i) {
        if (input_ids_data[i] != image_token_id) {
            continue;
        }
        // Advance to the image that still has feature rows available.
        while (cur_image < image_embeds.size() &&
               cur_row >= image_embeds[cur_image]->get_shape().at(1)) {
            ++cur_image;
            cur_row = 0;
        }
        OPENVINO_ASSERT(cur_image < image_embeds.size(),
                        "[LFM2-VL] Number of image placeholder tokens exceeds available image features");
        const float* src = image_embeds[cur_image]->data<const float>() + cur_row * hidden_size;
        std::memcpy(embeds_data + i * hidden_size, src, hidden_size * sizeof(float));
        ++cur_row;
    }

    return inputs_embeds;
}

} // namespace ov::genai
