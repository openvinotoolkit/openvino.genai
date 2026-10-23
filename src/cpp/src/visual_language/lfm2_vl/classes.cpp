// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/lfm2_vl/classes.hpp"

#include <algorithm>
#include <cmath>

#include "visual_language/clip.hpp"
#include "utils.hpp"

namespace ov::genai {

namespace {

const std::string NATIVE_TAG = "<image>";
const std::string IMAGE_TOKEN = "<image>";
const std::string IMAGE_START_TOKEN = "<|image_start|>";
const std::string IMAGE_END_TOKEN = "<|image_end|>";
const std::string IMAGE_THUMBNAIL_TOKEN = "<|img_thumbnail|>";

// A single preprocessed tile: normalized CHW float image plus its patch-grid size.
struct Lfm2Tile {
    clip_image_f32 chw;
    size_t patches_h = 0;
    size_t patches_w = 0;
};

int round_by_factor(double number, int factor) {
    return static_cast<int>(std::llround(number / factor)) * factor;
}

// Rescale image so both dims are multiples of encoder_patch_size * downsample_factor and the number of
// pixels stays within [min_image_tokens, max_image_tokens] * (patch * downsample)^2, preserving aspect ratio.
// Mirrors Lfm2VlImageProcessor.smart_resize. Returns (new_height, new_width).
std::pair<int, int> smart_resize(int height, int width, size_t downsample_factor, size_t min_image_tokens,
                                 size_t max_image_tokens, size_t encoder_patch_size) {
    const int total_factor = static_cast<int>(encoder_patch_size * downsample_factor);
    const double min_pixels = static_cast<double>(min_image_tokens) * encoder_patch_size * encoder_patch_size *
                              downsample_factor * downsample_factor;
    const double max_pixels = static_cast<double>(max_image_tokens) * encoder_patch_size * encoder_patch_size *
                              downsample_factor * downsample_factor;

    int h_bar = std::max(total_factor, round_by_factor(height, total_factor));
    int w_bar = std::max(total_factor, round_by_factor(width, total_factor));

    if (static_cast<double>(h_bar) * w_bar > max_pixels) {
        double beta = std::sqrt((static_cast<double>(height) * width) / max_pixels);
        h_bar = std::max(total_factor, static_cast<int>(std::floor(height / beta / total_factor)) * total_factor);
        w_bar = std::max(total_factor, static_cast<int>(std::floor(width / beta / total_factor)) * total_factor);
    } else if (static_cast<double>(h_bar) * w_bar < min_pixels) {
        double beta = std::sqrt(min_pixels / (static_cast<double>(height) * width));
        h_bar = static_cast<int>(std::ceil(height * beta / total_factor)) * total_factor;
        w_bar = static_cast<int>(std::ceil(width * beta / total_factor)) * total_factor;
    }
    return {h_bar, w_bar};
}

bool is_image_too_large(int height, int width, size_t max_image_tokens, size_t encoder_patch_size,
                        size_t downsample_factor, float max_pixels_tolerance) {
    const int total_factor = static_cast<int>(encoder_patch_size * downsample_factor);
    const int h_bar = std::max(static_cast<int>(encoder_patch_size), round_by_factor(height, total_factor));
    const int w_bar = std::max(static_cast<int>(encoder_patch_size), round_by_factor(width, total_factor));
    const double limit = static_cast<double>(max_image_tokens) * encoder_patch_size * encoder_patch_size *
                         downsample_factor * downsample_factor * max_pixels_tolerance;
    return static_cast<double>(h_bar) * w_bar > limit;
}

// Choose the tiling grid (cols, rows) that best matches the image aspect ratio within [min_tiles, max_tiles].
// Mirrors Lfm2VlImageProcessor._get_grid_layout / find_closest_aspect_ratio.
std::pair<int, int> get_grid_layout(int height, int width, size_t min_tiles, size_t max_tiles, size_t tile_size) {
    const double aspect_ratio = static_cast<double>(width) / height;
    std::vector<std::pair<int, int>> target_ratios;  // (cols, rows)
    for (size_t n = min_tiles; n <= max_tiles; ++n) {
        for (size_t w = 1; w <= n; ++w) {
            for (size_t h = 1; h <= n; ++h) {
                size_t prod = w * h;
                if (prod >= min_tiles && prod <= max_tiles) {
                    target_ratios.emplace_back(static_cast<int>(w), static_cast<int>(h));
                }
            }
        }
    }
    std::sort(target_ratios.begin(), target_ratios.end(),
              [](const auto& a, const auto& b) { return a.first * a.second < b.first * b.second; });

    double best_diff = std::numeric_limits<double>::max();
    std::pair<int, int> best_ratio{1, 1};
    const double area = static_cast<double>(width) * height;
    for (const auto& ratio : target_ratios) {
        double target_ar = static_cast<double>(ratio.first) / ratio.second;
        double diff = std::abs(aspect_ratio - target_ar);
        if (diff < best_diff) {
            best_diff = diff;
            best_ratio = ratio;
        } else if (diff == best_diff) {
            double target_area = static_cast<double>(tile_size) * tile_size * ratio.first * ratio.second;
            if (area > 0.5 * target_area) {
                best_ratio = ratio;
            }
        }
    }
    return best_ratio;  // (cols, rows)
}

// Bilinear interpolation of a [gh, gw, C] grid to [oh, ow, C], align_corners=False (F.interpolate semantics).
std::vector<float> interpolate_bilinear(const std::vector<float>& src, size_t gh, size_t gw, size_t C,
                                        size_t oh, size_t ow) {
    if (oh == gh && ow == gw) {
        return src;
    }
    std::vector<float> dst(oh * ow * C);
    const double scale_h = static_cast<double>(gh) / oh;
    const double scale_w = static_cast<double>(gw) / ow;
    for (size_t y = 0; y < oh; ++y) {
        double sy = (y + 0.5) * scale_h - 0.5;
        if (sy < 0) sy = 0;
        int y0 = static_cast<int>(std::floor(sy));
        int y1 = std::min(y0 + 1, static_cast<int>(gh) - 1);
        double wy = sy - y0;
        if (y0 > static_cast<int>(gh) - 1) y0 = static_cast<int>(gh) - 1;
        for (size_t x = 0; x < ow; ++x) {
            double sx = (x + 0.5) * scale_w - 0.5;
            if (sx < 0) sx = 0;
            int x0 = static_cast<int>(std::floor(sx));
            int x1 = std::min(x0 + 1, static_cast<int>(gw) - 1);
            double wx = sx - x0;
            if (x0 > static_cast<int>(gw) - 1) x0 = static_cast<int>(gw) - 1;
            for (size_t c = 0; c < C; ++c) {
                float v00 = src[(y0 * gw + x0) * C + c];
                float v01 = src[(y0 * gw + x1) * C + c];
                float v10 = src[(y1 * gw + x0) * C + c];
                float v11 = src[(y1 * gw + x1) * C + c];
                double top = v00 * (1 - wx) + v01 * wx;
                double bot = v10 * (1 - wx) + v11 * wx;
                dst[(y * ow + x) * C + c] = static_cast<float>(top * (1 - wy) + bot * wy);
            }
        }
    }
    return dst;
}

clip_image_u8 crop_region(const clip_image_u8& src, int x0, int y0, int w, int h) {
    clip_image_u8 out;
    out.nx = w;
    out.ny = h;
    out.buf.resize(3 * w * h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < 3; ++c) {
                out.buf[(y * w + x) * 3 + c] = src.buf[((y0 + y) * src.nx + (x0 + x)) * 3 + c];
            }
        }
    }
    return out;
}

// Normalize a resized uint8 image into a tile with CHW float layout and patch-grid metadata.
Lfm2Tile make_tile(const clip_image_u8& img, size_t encoder_patch_size, const clip_ctx_double& mean_std) {
    Lfm2Tile tile;
    tile.chw = normalize_and_convert_to_chw(img, mean_std);
    tile.patches_h = img.ny / encoder_patch_size;
    tile.patches_w = img.nx / encoder_patch_size;
    return tile;
}

// Preprocess one raw image into tiles (grid tiles first, optional thumbnail last), returning grid (rows, cols).
std::vector<Lfm2Tile> preprocess_image(const clip_image_u8& input, const ProcessorConfig& config,
                                       size_t& out_rows, size_t& out_cols) {
    clip_ctx_double mean_std;
    for (size_t c = 0; c < 3; ++c) {
        mean_std.image_mean[c] = static_cast<double>(config.image_mean[c]) * 255.0;
        mean_std.image_std[c] = static_cast<double>(config.image_std[c]) * 255.0;
    }

    const size_t enc_patch = config.encoder_patch_size;
    const int height = input.ny;
    const int width = input.nx;

    auto [new_h, new_w] = smart_resize(height, width, config.downsample_factor, config.min_image_tokens,
                                       config.max_image_tokens, enc_patch);

    const bool do_split = !(config.lfm2_min_tiles == 1 && config.lfm2_max_tiles == 1) && config.do_image_splitting;
    const bool too_large = is_image_too_large(height, width, config.max_image_tokens, enc_patch,
                                              config.downsample_factor, config.max_pixels_tolerance);

    std::vector<Lfm2Tile> tiles;
    out_rows = 1;
    out_cols = 1;

    if (too_large && do_split) {
        auto grid = get_grid_layout(height, width, config.lfm2_min_tiles, config.lfm2_max_tiles, config.tile_size);
        const int grid_cols = grid.first;
        const int grid_rows = grid.second;
        out_rows = grid_rows;
        out_cols = grid_cols;
        const int target_w = static_cast<int>(config.tile_size) * grid_cols;
        const int target_h = static_cast<int>(config.tile_size) * grid_rows;

        clip_image_u8 resized;
        bicubic_resize(input, resized, target_w, target_h);

        for (int r = 0; r < grid_rows; ++r) {
            for (int col = 0; col < grid_cols; ++col) {
                clip_image_u8 crop = crop_region(resized, col * static_cast<int>(config.tile_size),
                                                 r * static_cast<int>(config.tile_size),
                                                 static_cast<int>(config.tile_size),
                                                 static_cast<int>(config.tile_size));
                tiles.push_back(make_tile(crop, enc_patch, mean_std));
            }
        }

        if (config.use_thumbnail) {
            clip_image_u8 thumb;
            bicubic_resize(input, thumb, new_w, new_h);
            tiles.push_back(make_tile(thumb, enc_patch, mean_std));
        }
    } else {
        clip_image_u8 resized;
        bicubic_resize(input, resized, new_w, new_h);
        tiles.push_back(make_tile(resized, enc_patch, mean_std));
    }

    return tiles;
}

// Pack a CHW float tile into Siglip2-naflex patch layout [num_patches, patch*patch*channels].
// Mirrors convert_image_to_patches: per-patch order is (row_in_patch, col_in_patch, channel).
ov::Tensor to_patches(const Lfm2Tile& tile, size_t patch_size) {
    const size_t nph = tile.patches_h;
    const size_t npw = tile.patches_w;
    const size_t H = nph * patch_size;
    const size_t W = npw * patch_size;
    const size_t C = 3;
    const size_t patch_dim = patch_size * patch_size * C;
    const size_t num_patches = nph * npw;

    ov::Tensor out(ov::element::f32, {1, num_patches, patch_dim});
    float* dst = out.data<float>();
    const float* chw = tile.chw.buf.data();

    for (size_t ph = 0; ph < nph; ++ph) {
        for (size_t pw = 0; pw < npw; ++pw) {
            const size_t patch_idx = ph * npw + pw;
            for (size_t r = 0; r < patch_size; ++r) {
                for (size_t col = 0; col < patch_size; ++col) {
                    for (size_t c = 0; c < C; ++c) {
                        const size_t y = ph * patch_size + r;
                        const size_t x = pw * patch_size + col;
                        const float val = chw[c * H * W + y * W + x];
                        dst[patch_idx * patch_dim + (r * patch_size + col) * C + c] = val;
                    }
                }
            }
        }
    }
    return out;
}

ov::Tensor run_single_io(CircularBufferQueue<ov::InferRequest>* queue, const ov::Tensor& input) {
    CircularBufferQueueElementGuard<ov::InferRequest> guard(queue);
    ov::InferRequest& req = guard.get();
    req.set_input_tensor(input);
    req.infer();
    const ov::Tensor& out = req.get_output_tensor();
    ov::Tensor copy(out.get_element_type(), out.get_shape());
    std::memcpy(copy.data(), out.data(), out.get_byte_size());
    return copy;
}

}  // namespace

VisionEncoderLFM2VL::VisionEncoderLFM2VL(const std::filesystem::path& model_dir, const std::string& device,
                                         const ov::AnyMap properties)
    : VisionEncoder(model_dir, device, properties) {
    auto compiled_pos = utils::singleton_core().compile_model(
        model_dir / "openvino_vision_embeddings_pos_model.xml", device,
        utils::get_model_properties(properties, "vision_embeddings_pos", device));
    m_ireq_queue_vision_pos = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_pos.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_pos]() -> ov::InferRequest { return compiled_pos.create_infer_request(); });

    auto compiled_proj = utils::singleton_core().compile_model(
        model_dir / "openvino_multi_modal_projector_model.xml", device,
        utils::get_model_properties(properties, "multi_modal_projector", device));
    m_ireq_queue_projector = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_proj.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_proj]() -> ov::InferRequest { return compiled_proj.create_infer_request(); });

    m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(model_dir, "config.json");
}

VisionEncoderLFM2VL::VisionEncoderLFM2VL(const ModelsMap& models_map, const std::filesystem::path& config_dir_path,
                                         const std::string& device, const ov::AnyMap properties)
    : VisionEncoder(models_map, config_dir_path, device, properties) {
    const auto& [pos_model, pos_weights] = utils::get_model_weights_pair(models_map, "vision_embeddings_pos");
    auto compiled_pos = utils::singleton_core().compile_model(
        pos_model, pos_weights, device, utils::get_model_properties(properties, "vision_embeddings_pos", device));
    m_ireq_queue_vision_pos = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_pos.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_pos]() -> ov::InferRequest { return compiled_pos.create_infer_request(); });

    const auto& [proj_model, proj_weights] = utils::get_model_weights_pair(models_map, "multi_modal_projector");
    auto compiled_proj = utils::singleton_core().compile_model(
        proj_model, proj_weights, device, utils::get_model_properties(properties, "multi_modal_projector", device));
    m_ireq_queue_projector = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_proj.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_proj]() -> ov::InferRequest { return compiled_proj.create_infer_request(); });

    m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(config_dir_path, "config.json");
}

EncodedImage VisionEncoderLFM2VL::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    ProcessorConfig config = ProcessorConfig::from_any_map(config_map, m_processor_config);

    clip_image_u8 input = tensor_to_clip_image_u8(image);
    size_t rows = 1, cols = 1;
    std::vector<Lfm2Tile> tiles = preprocess_image(input, config, rows, cols);

    const size_t num_patches = m_vlm_config.vision_config_num_patches;
    const size_t grid = static_cast<size_t>(std::llround(std::sqrt(static_cast<double>(num_patches))));
    // Query the static position-embedding table once: indices [0, num_patches).
    ov::Tensor idx(ov::element::i64, {1, num_patches});
    int64_t* idx_data = idx.data<int64_t>();
    for (size_t i = 0; i < num_patches; ++i) {
        idx_data[i] = static_cast<int64_t>(i);
    }
    ov::Tensor pos_table = run_single_io(m_ireq_queue_vision_pos.get(), idx);  // [1, num_patches, vis_hidden]
    const size_t vis_hidden = pos_table.get_shape().back();
    const float* pos_src = pos_table.data<float>();
    std::vector<float> pos_grid(pos_src, pos_src + num_patches * vis_hidden);  // [grid, grid, vis_hidden]

    const size_t patch_size = config.encoder_patch_size;
    const size_t downsample = config.downsample_factor;

    std::vector<ov::Tensor> tile_features;
    std::vector<ImageSize> crop_sizes;  // per-tile token grid (patches_h/downsample, patches_w/downsample)
    size_t total_tokens = 0;
    size_t text_hidden = 0;

    for (const auto& tile : tiles) {
        const size_t ph = tile.patches_h;
        const size_t pw = tile.patches_w;

        ov::Tensor pixel_values = to_patches(tile, patch_size);  // [1, ph*pw, patch_dim]

        // Resize the position table to this tile's patch grid and feed it as pos_embeds.
        std::vector<float> pos_resized = interpolate_bilinear(pos_grid, grid, grid, vis_hidden, ph, pw);
        ov::Tensor pos_embeds(ov::element::f32, {1, ph * pw, vis_hidden});
        std::memcpy(pos_embeds.data(), pos_resized.data(), pos_resized.size() * sizeof(float));

        ov::Tensor vision_out;
        {
            CircularBufferQueueElementGuard<ov::InferRequest> guard(m_ireq_queue_vision_encoder.get());
            ov::InferRequest& req = guard.get();
            req.set_tensor("pixel_values", pixel_values);
            req.set_tensor("pos_embeds", pos_embeds);
            req.infer();
            const ov::Tensor& out = req.get_output_tensor();
            vision_out = ov::Tensor(out.get_element_type(), out.get_shape());
            std::memcpy(vision_out.data(), out.data(), out.get_byte_size());
        }  // [1, ph*pw, vis_hidden]

        // Reshape to [1, ph, pw, vis_hidden] (no data movement) and project.
        ov::Tensor projector_in(ov::element::f32, {1, ph, pw, vis_hidden});
        std::memcpy(projector_in.data(), vision_out.data(), vision_out.get_byte_size());
        ov::Tensor projected = run_single_io(m_ireq_queue_projector.get(), projector_in);  // [1, ph/f, pw/f, text_hidden]

        const auto& pshape = projected.get_shape();
        text_hidden = pshape.back();
        size_t tokens = pshape[1] * pshape[2];
        total_tokens += tokens;
        crop_sizes.push_back(ImageSize{ph / downsample, pw / downsample});
        tile_features.push_back(std::move(projected));
    }

    // Concatenate all tile features into [1, total_tokens, text_hidden].
    ov::Tensor resized_source(ov::element::f32, {1, total_tokens, text_hidden});
    float* dst = resized_source.data<float>();
    size_t offset = 0;
    for (const auto& feat : tile_features) {
        std::memcpy(dst + offset, feat.data(), feat.get_byte_size());
        offset += feat.get_size();
    }

    EncodedImage encoded;
    encoded.resized_source = std::move(resized_source);
    encoded.patches_grid = {static_cast<int>(rows), static_cast<int>(cols)};
    encoded.crop_sizes = std::move(crop_sizes);
    encoded.num_image_tokens = total_tokens;
    encoded.original_image_size = ImageSize{static_cast<size_t>(input.ny), static_cast<size_t>(input.nx)};
    return encoded;
}

namespace {

// Build the expanded image-token string for a single image, matching Lfm2VlProcessor._build_image_tokens.
std::string build_image_placeholder(const EncodedImage& image) {
    const size_t rows = static_cast<size_t>(image.patches_grid.first);
    const size_t cols = static_cast<size_t>(image.patches_grid.second);
    const bool multi_tile = rows > 1 || cols > 1;
    const auto& crops = image.crop_sizes;

    std::string result = IMAGE_START_TOKEN;
    if (multi_tile) {
        size_t crop_idx = 0;
        for (size_t r = 0; r < rows; ++r) {
            for (size_t c = 0; c < cols; ++c) {
                result += "<|img_row_" + std::to_string(r + 1) + "_col_" + std::to_string(c + 1) + "|>";
                size_t tokens = crops.at(crop_idx).height * crops.at(crop_idx).width;
                for (size_t t = 0; t < tokens; ++t) {
                    result += IMAGE_TOKEN;
                }
                ++crop_idx;
            }
        }
        if (crop_idx < crops.size()) {  // thumbnail
            result += IMAGE_THUMBNAIL_TOKEN;
            size_t tokens = crops.at(crop_idx).height * crops.at(crop_idx).width;
            for (size_t t = 0; t < tokens; ++t) {
                result += IMAGE_TOKEN;
            }
        }
    } else {
        for (size_t t = 0; t < image.num_image_tokens; ++t) {
            result += IMAGE_TOKEN;
        }
    }
    result += IMAGE_END_TOKEN;
    return result;
}

ov::Tensor merge_text_and_image_embeddings_lfm2(
    const ov::Tensor& input_ids,
    const ov::Tensor& text_embeds,
    const std::vector<ov::Tensor>& image_embeds,
    int64_t image_token_id) {
    auto shape = text_embeds.get_shape();
    size_t batch_size = shape.at(0);
    size_t seq_len = shape.at(1);
    size_t embed_dim = shape.at(2);

    ov::Tensor merged(text_embeds.get_element_type(), shape);
    const float* text_data = text_embeds.data<float>();
    const int64_t* ids = input_ids.data<int64_t>();
    float* merged_data = merged.data<float>();

    size_t flattened = batch_size * seq_len;
    size_t image_idx = 0;
    size_t token_in_image = 0;
    for (size_t i = 0; i < flattened; ++i) {
        size_t offset = i * embed_dim;
        if (ids[i] == image_token_id) {
            OPENVINO_ASSERT(image_idx < image_embeds.size(), "More image placeholder tokens than image embeddings");
            const ov::Tensor& emb = image_embeds[image_idx];
            size_t tokens_in_image = emb.get_shape().at(1);
            const float* emb_data = emb.data<float>();
            std::copy_n(emb_data + token_in_image * embed_dim, embed_dim, merged_data + offset);
            ++token_in_image;
            if (token_in_image == tokens_in_image) {
                ++image_idx;
                token_in_image = 0;
            }
        } else {
            std::copy_n(text_data + offset, embed_dim, merged_data + offset);
        }
    }
    return merged;
}

}  // namespace

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
    auto [unified_prompt, images_sequence] = normalize(prompt, NATIVE_TAG, NATIVE_TAG, base_id, images.size());

    size_t searched_pos = 0;
    for (size_t new_image_id : images_sequence) {
        const EncodedImage& image = images.at(new_image_id - base_id);
        std::string expanded_tag = build_image_placeholder(image);
        searched_pos = unified_prompt.find(NATIVE_TAG, searched_pos);
        OPENVINO_ASSERT(searched_pos != std::string::npos, "Failed to find image placeholder in the prompt");
        unified_prompt.replace(searched_pos, NATIVE_TAG.length(), expanded_tag);
        searched_pos += expanded_tag.length();
    }

    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderLFM2VL::get_inputs_embeds(const std::string& unified_prompt, const std::vector<ov::genai::EncodedImage>& images, ov::genai::VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& images_sequence) {
    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    for (size_t new_image_id : images_sequence) {
        image_embeds.push_back(images.at(new_image_id).resized_source);
    }

    ov::Tensor input_ids = get_encoded_input_ids(unified_prompt, metrics);
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = get_text_embedding(req, input_ids, metrics);

    if (images.empty()) {
        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
        return inputs_embeds;
    }

    int64_t image_token_id = m_vlm_config.image_token_id;
    if (image_token_id < 0) {
        auto start_tokenizer_time = std::chrono::steady_clock::now();
        ov::Tensor encoded_image_token = m_tokenizer.encode(IMAGE_TOKEN, ov::genai::add_special_tokens(false)).input_ids;
        auto end_tokenizer_time = std::chrono::steady_clock::now();
        OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
        metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] += ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
        image_token_id = encoded_image_token.data<int64_t>()[encoded_image_token.get_size() - 1];
    }

    return merge_text_and_image_embeddings_lfm2(input_ids, text_embeds, image_embeds, image_token_id);
}

} // namespace ov::genai
