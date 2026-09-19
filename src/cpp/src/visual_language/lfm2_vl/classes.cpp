// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/lfm2_vl/classes.hpp"

#include <cmath>
#include <cstring>
#include <fstream>
#include <algorithm>
#include <array>
#include <limits>
#include <set>
#include <vector>

#include "visual_language/clip.hpp"

#include "json_utils.hpp"
#include "utils.hpp"

namespace ov::genai {

namespace {

// Image placeholder tokens used by the Lfm2VlProcessor. These are fixed strings
// that the processor inserts around the vision embeddings and are single tokens
// in the tokenizer.
const std::string IMAGE_TOKEN = "<image>";
const std::string IMAGE_START_TOKEN = "<|image_start|>";
const std::string IMAGE_END_TOKEN = "<|image_end|>";
const std::string IMAGE_THUMBNAIL_TOKEN = "<|img_thumbnail|>";

int round_by_factor(double number, int factor) {
    return static_cast<int>(std::llround(number / factor)) * factor;
}

// Ported from Lfm2VlImageProcessor.smart_resize. Returns (new_height, new_width).
std::pair<int, int> smart_resize(int height, int width, const Lfm2VlPreprocessConfig& cfg) {
    const int total_factor = static_cast<int>(cfg.encoder_patch_size * cfg.downsample_factor);
    const double min_pixels = static_cast<double>(cfg.min_image_tokens) * cfg.encoder_patch_size * cfg.encoder_patch_size *
                              cfg.downsample_factor * cfg.downsample_factor;
    const double max_pixels = static_cast<double>(cfg.max_image_tokens) * cfg.encoder_patch_size * cfg.encoder_patch_size *
                              cfg.downsample_factor * cfg.downsample_factor;

    int h_bar = std::max(total_factor, round_by_factor(height, total_factor));
    int w_bar = std::max(total_factor, round_by_factor(width, total_factor));

    if (static_cast<double>(h_bar) * w_bar > max_pixels) {
        const double beta = std::sqrt((static_cast<double>(height) * width) / max_pixels);
        h_bar = std::max(total_factor, static_cast<int>(std::floor(height / beta / total_factor)) * total_factor);
        w_bar = std::max(total_factor, static_cast<int>(std::floor(width / beta / total_factor)) * total_factor);
    } else if (static_cast<double>(h_bar) * w_bar < min_pixels) {
        const double beta = std::sqrt(min_pixels / (static_cast<double>(height) * width));
        h_bar = static_cast<int>(std::ceil(height * beta / total_factor)) * total_factor;
        w_bar = static_cast<int>(std::ceil(width * beta / total_factor)) * total_factor;
    }
    return {h_bar, w_bar};
}

bool is_image_too_large(int height, int width, const Lfm2VlPreprocessConfig& cfg) {
    const int total_factor = static_cast<int>(cfg.encoder_patch_size * cfg.downsample_factor);
    const int h_bar = std::max(static_cast<int>(cfg.encoder_patch_size), round_by_factor(height, total_factor));
    const int w_bar = std::max(static_cast<int>(cfg.encoder_patch_size), round_by_factor(width, total_factor));
    const double threshold = static_cast<double>(cfg.max_image_tokens) * cfg.encoder_patch_size * cfg.encoder_patch_size *
                             cfg.downsample_factor * cfg.downsample_factor * cfg.max_pixels_tolerance;
    return static_cast<double>(h_bar) * w_bar > threshold;
}

// Ported from find_closest_aspect_ratio (target ratios are (width, height)).
std::pair<int, int> find_closest_aspect_ratio(double aspect_ratio,
                                              const std::vector<std::pair<int, int>>& target_ratios,
                                              int width, int height, int image_size) {
    double best_ratio_diff = std::numeric_limits<double>::infinity();
    std::pair<int, int> best_ratio{1, 1};
    const long long area = static_cast<long long>(width) * height;
    for (const auto& ratio : target_ratios) {
        const double target_aspect_ratio = static_cast<double>(ratio.first) / ratio.second;
        const double ratio_diff = std::abs(aspect_ratio - target_aspect_ratio);
        if (ratio_diff < best_ratio_diff) {
            best_ratio_diff = ratio_diff;
            best_ratio = ratio;
        } else if (ratio_diff == best_ratio_diff) {
            const double target_area = static_cast<double>(image_size) * image_size * ratio.first * ratio.second;
            if (area > 0.5 * target_area) {
                best_ratio = ratio;
            }
        }
    }
    return best_ratio;
}

std::vector<std::pair<int, int>> target_ratios(int min_tiles, int max_tiles) {
    std::set<std::pair<int, int>> ratios;
    for (int n = min_tiles; n <= max_tiles; ++n) {
        for (int w = 1; w <= n; ++w) {
            for (int h = 1; h <= n; ++h) {
                if (min_tiles <= w * h && w * h <= max_tiles) {
                    ratios.insert({w, h});
                }
            }
        }
    }
    std::vector<std::pair<int, int>> result(ratios.begin(), ratios.end());
    std::sort(result.begin(), result.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
        return a.first * a.second < b.first * b.second;
    });
    return result;
}

// A resized RGB sub-image (single tile or thumbnail) kept as u8 HWC.
struct SubImage {
    clip_image_u8 image;  // resized RGB, sizes divisible by encoder_patch_size
};

// Crop an RGB u8 region [y0,y0+h) x [x0,x0+w) from a source image.
clip_image_u8 crop_region(const clip_image_u8& src, int x0, int y0, int w, int h) {
    clip_image_u8 out;
    out.nx = w;
    out.ny = h;
    out.buf.resize(static_cast<size_t>(w) * h * 3);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            for (int c = 0; c < 3; ++c) {
                out.buf[3 * (static_cast<size_t>(y) * w + x) + c] =
                    src.buf[3 * (static_cast<size_t>(y0 + y) * src.nx + (x0 + x)) + c];
            }
        }
    }
    return out;
}

// Produce the ordered list of sub-images (tiles then optional thumbnail) plus the
// grid layout (rows, cols) following Lfm2VlImageProcessor.resize_and_split.
std::vector<SubImage> resize_and_split(const clip_image_u8& src, const Lfm2VlPreprocessConfig& cfg,
                                       int& out_rows, int& out_cols,
                                       int& thumb_h, int& thumb_w) {
    const bool do_image_splitting = cfg.do_image_splitting && !(cfg.min_tiles == 1 && cfg.max_tiles == 1);
    const bool large = is_image_too_large(src.ny, src.nx, cfg);

    auto [new_h, new_w] = smart_resize(src.ny, src.nx, cfg);
    thumb_h = new_h;
    thumb_w = new_w;

    std::vector<SubImage> sub_images;

    if (large && do_image_splitting) {
        const double aspect_ratio = static_cast<double>(src.nx) / src.ny;
        auto ratios = target_ratios(static_cast<int>(cfg.min_tiles), static_cast<int>(cfg.max_tiles));
        auto [grid_w, grid_h] = find_closest_aspect_ratio(aspect_ratio, ratios, src.nx, src.ny, static_cast<int>(cfg.tile_size));
        const int target_w = static_cast<int>(cfg.tile_size) * grid_w;
        const int target_h = static_cast<int>(cfg.tile_size) * grid_h;
        out_rows = grid_h;
        out_cols = grid_w;

        clip_image_u8 resized;
        bicubic_resize(src, resized, target_w, target_h);
        // Split into row-major tiles of tile_size x tile_size.
        for (int r = 0; r < grid_h; ++r) {
            for (int c = 0; c < grid_w; ++c) {
                SubImage si;
                si.image = crop_region(resized, c * static_cast<int>(cfg.tile_size), r * static_cast<int>(cfg.tile_size),
                                       static_cast<int>(cfg.tile_size), static_cast<int>(cfg.tile_size));
                sub_images.push_back(std::move(si));
            }
        }
        if (cfg.use_thumbnail && grid_w * grid_h != 1) {
            SubImage thumb;
            bicubic_resize(src, thumb.image, new_w, new_h);
            sub_images.push_back(std::move(thumb));
        }
    } else {
        out_rows = 1;
        out_cols = 1;
        SubImage si;
        bicubic_resize(src, si.image, new_w, new_h);
        sub_images.push_back(std::move(si));
    }
    return sub_images;
}

// Build the bilinear (align_corners=False) interpolation operator that maps the
// vision tower's learned source_size x source_size positional grid onto a
// target grid of (patch_h x patch_w), matching Siglip2 resize_positional_embeddings
// as reproduced by optimum-intel's _build_pos_emb_interp. Because interpolation is
// linear, applying it to the identity basis yields exactly the sampling weights.
// The result is written into pos_emb (row-major [max_len, num_source]); rows beyond
// patch_h*patch_w are filled with the first valid row (padded, later masked out).
void build_pos_emb_interp(float* pos_emb, size_t max_len, int source_size, int patch_h, int patch_w) {
    const int num_source = source_size * source_size;
    const double scale_y = static_cast<double>(source_size) / patch_h;
    const double scale_x = static_cast<double>(source_size) / patch_w;
    const size_t valid = static_cast<size_t>(patch_h) * patch_w;

    std::fill(pos_emb, pos_emb + max_len * num_source, 0.0f);

    for (int oy = 0; oy < patch_h; ++oy) {
        double iy = (oy + 0.5) * scale_y - 0.5;
        int y0f = static_cast<int>(std::floor(iy));
        double wy = iy - y0f;
        int y0 = std::min(std::max(y0f, 0), source_size - 1);
        int y1 = std::min(std::max(y0f + 1, 0), source_size - 1);
        for (int ox = 0; ox < patch_w; ++ox) {
            double ix = (ox + 0.5) * scale_x - 0.5;
            int x0f = static_cast<int>(std::floor(ix));
            double wx = ix - x0f;
            int x0 = std::min(std::max(x0f, 0), source_size - 1);
            int x1 = std::min(std::max(x0f + 1, 0), source_size - 1);

            const size_t o = static_cast<size_t>(oy) * patch_w + ox;
            float* row = pos_emb + o * num_source;
            row[y0 * source_size + x0] += static_cast<float>((1.0 - wy) * (1.0 - wx));
            row[y0 * source_size + x1] += static_cast<float>((1.0 - wy) * wx);
            row[y1 * source_size + x0] += static_cast<float>(wy * (1.0 - wx));
            row[y1 * source_size + x1] += static_cast<float>(wy * wx);
        }
    }
    // Fill padded rows with the first valid row.
    for (size_t o = valid; o < max_len; ++o) {
        std::copy(pos_emb, pos_emb + num_source, pos_emb + o * num_source);
    }
}

} // namespace

void VisionEncoderLFM2VL::load_preprocess_config(const std::filesystem::path& config_dir_path) {
    using ov::genai::utils::read_json_param;
    Lfm2VlPreprocessConfig& c = m_preprocess_config;

    auto apply = [&](const nlohmann::json& parsed) {
        read_json_param(parsed, "downsample_factor", c.downsample_factor);
        read_json_param(parsed, "encoder_patch_size", c.encoder_patch_size);
        read_json_param(parsed, "tile_size", c.tile_size);
        read_json_param(parsed, "min_image_tokens", c.min_image_tokens);
        read_json_param(parsed, "max_image_tokens", c.max_image_tokens);
        read_json_param(parsed, "min_tiles", c.min_tiles);
        read_json_param(parsed, "max_tiles", c.max_tiles);
        read_json_param(parsed, "do_image_splitting", c.do_image_splitting);
        read_json_param(parsed, "use_thumbnail", c.use_thumbnail);
        read_json_param(parsed, "max_pixels_tolerance", c.max_pixels_tolerance);
        utils::read_mean_std_params(parsed, "image_mean", c.image_mean);
        utils::read_mean_std_params(parsed, "image_std", c.image_std);
    };

    const auto processor_config_path = config_dir_path / "processor_config.json";
    if (std::filesystem::exists(processor_config_path)) {
        std::ifstream stream(processor_config_path);
        auto parsed = nlohmann::json::parse(stream);
        if (parsed.contains("image_processor")) {
            apply(parsed.at("image_processor"));
        } else {
            apply(parsed);
        }
    } else {
        const auto preprocessor_config_path = config_dir_path / "preprocessor_config.json";
        if (std::filesystem::exists(preprocessor_config_path)) {
            std::ifstream stream(preprocessor_config_path);
            apply(nlohmann::json::parse(stream));
        }
    }

    const auto config_path = config_dir_path / "config.json";
    if (std::filesystem::exists(config_path)) {
        std::ifstream stream(config_path);
        auto parsed = nlohmann::json::parse(stream);
        read_json_param(parsed, "vision_config.num_patches", c.vision_num_patches);
    }
}

VisionEncoderLFM2VL::VisionEncoderLFM2VL(const std::filesystem::path& model_dir,
                                         const std::string& device,
                                         const ov::AnyMap properties)
    : VisionEncoder(model_dir, device, properties) {
    auto compiled_model = utils::singleton_core().compile_model(
        model_dir / "openvino_multi_modal_projector_model.xml", device,
        utils::get_model_properties(properties, "multi_modal_projector", device));
    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM multi modal projector model");
    m_ireq_queue_projector = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
    load_preprocess_config(model_dir);
}

VisionEncoderLFM2VL::VisionEncoderLFM2VL(const ModelsMap& models_map,
                                         const std::filesystem::path& config_dir_path,
                                         const std::string& device,
                                         const ov::AnyMap properties)
    : VisionEncoder(models_map, config_dir_path, device, properties) {
    const auto& [projector_model, projector_weights] = utils::get_model_weights_pair(models_map, "multi_modal_projector");
    auto compiled_model = utils::singleton_core().compile_model(
        projector_model, projector_weights, device,
        utils::get_model_properties(properties, "multi_modal_projector", device));
    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM multi modal projector model");
    m_ireq_queue_projector = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
    load_preprocess_config(config_dir_path);
}

EncodedImage VisionEncoderLFM2VL::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    const Lfm2VlPreprocessConfig& cfg = m_preprocess_config;
    const int patch = static_cast<int>(cfg.encoder_patch_size);
    const int ds = static_cast<int>(cfg.downsample_factor);
    const size_t max_num_patches = cfg.max_num_patches();
    const int source_size = static_cast<int>(std::llround(std::sqrt(static_cast<double>(cfg.vision_num_patches))));
    const int num_source = source_size * source_size;

    clip_image_u8 input_image = tensor_to_clip_image_u8(image);

    int rows = 1, cols = 1, thumb_h = 0, thumb_w = 0;
    std::vector<SubImage> sub_images = resize_and_split(input_image, cfg, rows, cols, thumb_h, thumb_w);

    CircularBufferQueueElementGuard<ov::InferRequest> encoder_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = encoder_guard.get();
    CircularBufferQueueElementGuard<ov::InferRequest> projector_guard(this->m_ireq_queue_projector.get());
    ov::InferRequest& projector = projector_guard.get();

    // Accumulated projected tokens across all sub-images: [total_tokens, hidden].
    std::vector<float> all_tokens;
    size_t hidden_size = 0;
    size_t total_tokens = 0;

    for (const SubImage& si : sub_images) {
        const int H = si.image.ny;
        const int W = si.image.nx;
        const int ph = H / patch;
        const int pw = W / patch;
        const size_t valid = static_cast<size_t>(ph) * pw;
        OPENVINO_ASSERT(valid <= max_num_patches, "LFM2-VL: patch count exceeds max_num_patches");

        // pixel_values [1, max_num_patches, patch*patch*3], normalized, patchified.
        ov::Tensor pixel_values(ov::element::f32, {1, max_num_patches, static_cast<size_t>(patch) * patch * 3});
        float* pv = pixel_values.data<float>();
        std::fill(pv, pv + pixel_values.get_size(), 0.0f);

        for (int pr = 0; pr < ph; ++pr) {
            for (int pc = 0; pc < pw; ++pc) {
                const size_t patch_idx = static_cast<size_t>(pr) * pw + pc;
                float* dst = pv + patch_idx * (static_cast<size_t>(patch) * patch * 3);
                for (int yy = 0; yy < patch; ++yy) {
                    for (int xx = 0; xx < patch; ++xx) {
                        const int src_y = pr * patch + yy;
                        const int src_x = pc * patch + xx;
                        const size_t base = 3 * (static_cast<size_t>(src_y) * W + src_x);
                        for (int cc = 0; cc < 3; ++cc) {
                            const float px = static_cast<float>(si.image.buf[base + cc]) / 255.0f;
                            const float norm = (px - cfg.image_mean[cc]) / cfg.image_std[cc];
                            dst[(static_cast<size_t>(yy) * patch + xx) * 3 + cc] = norm;
                        }
                    }
                }
            }
        }

        // pixel_attention_mask [1, max_num_patches]
        ov::Tensor pixel_attention_mask(ov::element::i64, {1, max_num_patches});
        int64_t* mask = pixel_attention_mask.data<int64_t>();
        for (size_t i = 0; i < max_num_patches; ++i) {
            mask[i] = (i < valid) ? 1 : 0;
        }

        // pos_emb_interp [1, max_num_patches, num_source]
        ov::Tensor pos_emb_interp(ov::element::f32, {1, max_num_patches, static_cast<size_t>(num_source)});
        build_pos_emb_interp(pos_emb_interp.data<float>(), max_num_patches, source_size, ph, pw);

        encoder.set_tensor("pixel_values", pixel_values);
        encoder.set_tensor("pixel_attention_mask", pixel_attention_mask);
        encoder.set_tensor("pos_emb_interp", pos_emb_interp);
        encoder.infer();
        const ov::Tensor last_hidden = encoder.get_tensor("last_hidden_state");  // [1, max_num_patches, vis_hidden]
        const size_t vis_hidden = last_hidden.get_shape().at(2);
        const float* lh = last_hidden.data<const float>();

        // Unpad to [valid, vis_hidden] and reshape to [1, ph, pw, vis_hidden] for the projector.
        ov::Tensor feature(ov::element::f32, {1, static_cast<size_t>(ph), static_cast<size_t>(pw), vis_hidden});
        std::memcpy(feature.data<float>(), lh, valid * vis_hidden * sizeof(float));

        projector.set_input_tensor(feature);
        projector.infer();
        const ov::Tensor proj = projector.get_output_tensor();  // [1, ph/ds, pw/ds, out_hidden]
        const ov::Shape proj_shape = proj.get_shape();
        const size_t out_hidden = proj_shape.back();
        const size_t tokens = proj.get_size() / out_hidden;
        hidden_size = out_hidden;

        const float* pd = proj.data<const float>();
        all_tokens.insert(all_tokens.end(), pd, pd + tokens * out_hidden);
        total_tokens += tokens;
    }

    ov::Tensor resized_source(ov::element::f32, {1, total_tokens, hidden_size});
    std::memcpy(resized_source.data<float>(), all_tokens.data(), all_tokens.size() * sizeof(float));

    EncodedImage encoded_image;
    encoded_image.resized_source = std::move(resized_source);
    encoded_image.num_image_tokens = total_tokens;
    encoded_image.patches_grid = {rows, cols};
    // Store the downsampled thumbnail token grid so the embedder can reconstruct the
    // exact placeholder layout. Only multi-tile images with use_thumbnail carry a
    // thumbnail sub-image; single-tile images leave this as {0, 0}.
    const bool has_thumbnail = (rows * cols > 1) && cfg.use_thumbnail;
    if (has_thumbnail) {
        const int thumb_ph = static_cast<int>(std::ceil((thumb_h / patch) / static_cast<double>(ds)));
        const int thumb_pw = static_cast<int>(std::ceil((thumb_w / patch) / static_cast<double>(ds)));
        encoded_image.resized_source_size = ImageSize{static_cast<size_t>(thumb_ph), static_cast<size_t>(thumb_pw)};
    } else {
        encoded_image.resized_source_size = ImageSize{0, 0};
    }
    return encoded_image;
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
    auto [unified_prompt, images_sequence] = normalize(prompt, IMAGE_TOKEN, IMAGE_TOKEN, base_id, images.size());

    size_t search_offset = 0;
    for (size_t new_image_id : images_sequence) {
        const EncodedImage& enc = images.at(new_image_id - base_id);
        const size_t total_tokens = enc.num_image_tokens;
        const int rows = enc.patches_grid.first;
        const int cols = enc.patches_grid.second;
        const size_t thumb_tokens = enc.resized_source_size.height * enc.resized_source_size.width;

        std::string expanded;
        expanded += IMAGE_START_TOKEN;
        if (rows * cols > 1) {
            // Multi-tile: per-tile row/col markers followed by an optional thumbnail.
            const bool has_thumbnail = thumb_tokens > 0;
            const size_t num_tiles = static_cast<size_t>(rows) * cols;
            const size_t tile_tokens = has_thumbnail
                ? (total_tokens - thumb_tokens) / num_tiles
                : total_tokens / num_tiles;
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    expanded += "<|img_row_" + std::to_string(r + 1) + "_col_" + std::to_string(c + 1) + "|>";
                    for (size_t t = 0; t < tile_tokens; ++t) {
                        expanded += IMAGE_TOKEN;
                    }
                }
            }
            if (has_thumbnail) {
                expanded += IMAGE_THUMBNAIL_TOKEN;
                for (size_t t = 0; t < thumb_tokens; ++t) {
                    expanded += IMAGE_TOKEN;
                }
            }
        } else {
            for (size_t t = 0; t < total_tokens; ++t) {
                expanded += IMAGE_TOKEN;
            }
        }
        expanded += IMAGE_END_TOKEN;

        search_offset = unified_prompt.find(IMAGE_TOKEN, search_offset);
        OPENVINO_ASSERT(search_offset != std::string::npos, "Failed to find image token in prompt during normalization");
        unified_prompt.replace(search_offset, IMAGE_TOKEN.length(), expanded);
        search_offset += expanded.length();
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

    auto start_tokenizer_time = std::chrono::steady_clock::now();
    ov::Tensor encoded_image_token = m_tokenizer.encode(IMAGE_TOKEN, ov::genai::add_special_tokens(false)).input_ids;
    auto end_tokenizer_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
    metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] += ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    int64_t image_token_id = encoded_image_token.data<int64_t>()[encoded_image_token.get_size() - 1];
    return utils::merge_text_and_image_embeddings_llava(input_ids, text_embeds, image_embeds, image_token_id);
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderLFM2VL::get_position_ids(const size_t inputs_embeds_size, const size_t history_size) {
    // LFM2 hybrid LM has no position_ids input; return an empty tensor to signal
    // the LM encoding loop to skip setting position_ids.
    return {ov::Tensor{ov::element::i64, {1, 0}}, std::nullopt};
}

} // namespace ov::genai
