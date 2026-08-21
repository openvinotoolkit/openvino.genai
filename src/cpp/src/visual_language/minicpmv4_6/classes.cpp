// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/minicpmv4_6/classes.hpp"

#include <cmath>
#include <fstream>
#include <nlohmann/json.hpp>

#include "visual_language/clip.hpp"
#include "utils.hpp"

namespace ov::genai {

namespace {

// The chat template renders an image as a single "<|image_pad|>" token. GenAI's
// universal/native tag machinery inserts this native tag; normalize_prompt then
// expands it into the full MiniCPM-V-4.6 placeholder structure.
const std::string NATIVE_TAG = "<|image_pad|>";

int ensure_divide(int length, int divisor) {
    return std::max(static_cast<int>(std::round(static_cast<float>(length) / divisor) * divisor), divisor);
}

// Mirrors transformers MiniCPMV4_6ImageProcessor.find_best_resize.
// image_size = {height, width}. Returns {best_height, best_width}.
std::pair<int, int> find_best_resize(std::pair<int, int> image_size, int scale_resolution, int patch_size, bool allow_upscale) {
    int height = image_size.first;
    int width = image_size.second;
    if ((static_cast<int64_t>(height) * width > static_cast<int64_t>(scale_resolution) * scale_resolution) || allow_upscale) {
        double aspect_ratio = static_cast<double>(width) / height;
        height = static_cast<int>(scale_resolution / std::sqrt(aspect_ratio));
        width = static_cast<int>(height * aspect_ratio);
    }
    // factor 4 = two successive 2x2 spatial merges (ViT insert merger + downsample MLP)
    int best_width = ensure_divide(width, patch_size * 4);
    int best_height = ensure_divide(height, patch_size * 4);
    return {best_height, best_width};
}

// Mirrors MiniCPMV4_6ImageProcessor.get_refine_size. grid = {grid_y, grid_x}.
std::pair<int, int> get_refine_size(std::pair<int, int> image_size, std::pair<int, int> grid, int scale_resolution, int patch_size, bool allow_upscale) {
    int height = image_size.first;
    int width = image_size.second;
    int grid_y = grid.first;
    int grid_x = grid.second;

    int refine_width = ensure_divide(width, grid_x);
    int refine_height = ensure_divide(height, grid_y);

    auto best = find_best_resize({refine_height / grid_y, refine_width / grid_x}, scale_resolution, patch_size, allow_upscale);
    return {best.first * grid_y, best.second * grid_x};
}

// Mirrors MiniCPMV4_6ImageProcessor.get_sliced_grid. image_size = {height, width}.
// Returns best_grid = {num_cols, num_rows}; {0, 0} means no slicing.
std::pair<int, int> get_sliced_grid(std::pair<int, int> image_size, int max_slice_nums, int scale_resolution) {
    int original_height = image_size.first;
    int original_width = image_size.second;
    double log_ratio = std::log(static_cast<double>(original_width) / original_height);
    double ratio = static_cast<double>(original_width) * original_height / (static_cast<double>(scale_resolution) * scale_resolution);
    int multiple = std::min(static_cast<int>(std::ceil(ratio)), max_slice_nums);
    if (multiple <= 1) {
        return {0, 0};
    }

    std::pair<int, int> best_grid{1, 1};
    double min_error = std::numeric_limits<double>::infinity();
    for (int num_slices : {multiple - 1, multiple, multiple + 1}) {
        if (num_slices == 1 || num_slices > max_slice_nums) {
            continue;
        }
        for (int num_rows = 1; num_rows <= num_slices; ++num_rows) {
            if (num_slices % num_rows == 0) {
                int num_cols = num_slices / num_rows;
                double error = std::abs(log_ratio - std::log(static_cast<double>(num_rows) / num_cols));
                if (error < min_error) {
                    best_grid = {num_cols, num_rows};
                    min_error = error;
                }
            }
        }
    }
    return best_grid;
}

// Crop [y0:y0+ph, x0:x0+pw] from a u8 RGB (HWC) image.
clip_image_u8 crop_patch(const clip_image_u8& img, int y0, int x0, int ph, int pw) {
    clip_image_u8 patch;
    patch.nx = pw;
    patch.ny = ph;
    patch.buf.resize(3 * ph * pw);
    for (int y = 0; y < ph; ++y) {
        for (int x = 0; x < pw; ++x) {
            int src = 3 * ((y0 + y) * img.nx + (x0 + x));
            int dst = 3 * (y * pw + x);
            patch.buf[dst] = img.buf[src];
            patch.buf[dst + 1] = img.buf[src + 1];
            patch.buf[dst + 2] = img.buf[src + 2];
        }
    }
    return patch;
}

// Mirrors MiniCPMV4_6ImageProcessor.reshape_by_patch.
// Input: normalized CHW f32 image (H, W). Output: pixel_values tensor
// [1, C, patch_size, (H/patch_size)*(W/patch_size)*patch_size].
ov::Tensor reshape_by_patch(const clip_image_f32& img, int patch_size) {
    const int C = 3;
    const int H = img.ny;
    const int W = img.nx;
    const int grid_h = H / patch_size;
    const int grid_w = W / patch_size;
    const int L = grid_h * grid_w;  // number of patches
    ov::Tensor pixel_values(ov::element::f32, ov::Shape{1, static_cast<size_t>(C), static_cast<size_t>(patch_size), static_cast<size_t>(L) * patch_size});
    float* dst = pixel_values.data<float>();
    const float* src = img.buf.data();  // CHW: src[c*H*W + y*W + x]
    const size_t HW = static_cast<size_t>(H) * W;
    // reshape_by_patch semantics: patches[c, kh, l*patch_size + kw] where
    // l = row*grid_w + col enumerates patches row-major and (kh, kw) is the
    // intra-patch coordinate.
    const size_t out_last = static_cast<size_t>(L) * patch_size;
    for (int c = 0; c < C; ++c) {
        for (int row = 0; row < grid_h; ++row) {
            for (int col = 0; col < grid_w; ++col) {
                int l = row * grid_w + col;
                for (int kh = 0; kh < patch_size; ++kh) {
                    for (int kw = 0; kw < patch_size; ++kw) {
                        int y = row * patch_size + kh;
                        int x = col * patch_size + kw;
                        float v = src[c * HW + static_cast<size_t>(y) * W + x];
                        size_t out_idx = (static_cast<size_t>(c) * patch_size + kh) * out_last + static_cast<size_t>(l) * patch_size + kw;
                        dst[out_idx] = v;
                    }
                }
            }
        }
    }
    return pixel_values;
}

// Mirrors _OVMiniCPMV4_6ForCausalLM._position_ids (bucketized interpolated positions).
std::vector<int64_t> compute_position_ids(int grid_h, int grid_w, int num_patches_per_side) {
    std::vector<double> boundaries;
    for (int i = 1; i < num_patches_per_side; ++i) {
        boundaries.push_back(static_cast<double>(i) / num_patches_per_side);
    }
    auto bucketize = [&](int n) {
        // torch.arange(0, 1 - 1e-6, 1/n)
        std::vector<int64_t> buckets;
        const double step = 1.0 / n;
        for (double frac = 0.0; frac < 1.0 - 1e-6; frac += step) {
            // bucketize with right=True -> number of boundaries <= frac
            int b = 0;
            while (b < static_cast<int>(boundaries.size()) && boundaries[b] <= frac) {
                ++b;
            }
            buckets.push_back(b);
        }
        return buckets;
    };
    std::vector<int64_t> bh = bucketize(grid_h);
    std::vector<int64_t> bw = bucketize(grid_w);
    std::vector<int64_t> pos_ids;
    pos_ids.reserve(static_cast<size_t>(grid_h) * grid_w);
    for (int64_t h : bh) {
        for (int64_t w : bw) {
            pos_ids.push_back(h * num_patches_per_side + w);
        }
    }
    return pos_ids;
}

// Mirrors _OVMiniCPMV4_6ForCausalLM._grouping_index (permutation grouping a h x w grid
// into row-major kernel_h x kernel_w blocks).
std::vector<int64_t> grouping_index(int h, int w, int kernel_h, int kernel_w) {
    std::vector<int64_t> out;
    out.reserve(static_cast<size_t>(h) * w);
    int H = h / kernel_h;
    int W = w / kernel_w;
    // index.reshape(H, kh, W, kw).permute(0, 2, 1, 3).reshape(-1)
    for (int a = 0; a < H; ++a) {
        for (int b = 0; b < W; ++b) {
            for (int i = 0; i < kernel_h; ++i) {
                for (int j = 0; j < kernel_w; ++j) {
                    int row = a * kernel_h + i;
                    int col = b * kernel_w + j;
                    out.push_back(static_cast<int64_t>(row) * w + col);
                }
            }
        }
    }
    return out;
}

} // namespace

VisionEncoderMiniCPMV4_6::VisionEncoderMiniCPMV4_6(
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap properties) :
    VisionEncoder(model_dir, device, properties) {
    load_vision_params(model_dir);
}

VisionEncoderMiniCPMV4_6::VisionEncoderMiniCPMV4_6(
    const ModelsMap& models_map,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    VisionEncoder(models_map, config_dir_path, device, device_config) {
    load_vision_params(config_dir_path);
}

void VisionEncoderMiniCPMV4_6::load_vision_params(const std::filesystem::path& config_dir_path) {
    m_patch_size = m_processor_config.patch_size;
    m_scale_resolution = m_processor_config.scale_resolution;
    if (m_processor_config.max_slice_nums != 0) {
        m_max_slice_nums = m_processor_config.max_slice_nums;
    }
    m_image_mean = m_processor_config.image_mean;
    m_image_std = m_processor_config.image_std;

    const auto config_path = config_dir_path / "config.json";
    if (std::filesystem::exists(config_path)) {
        std::ifstream stream(config_path);
        if (stream.is_open()) {
            auto cfg = nlohmann::json::parse(stream);
            if (cfg.contains("vision_config")) {
                const auto& vc = cfg.at("vision_config");
                size_t vision_image_size = vc.value("image_size", static_cast<size_t>(980));
                size_t patch = vc.value("patch_size", m_patch_size);
                m_num_patches_per_side = vision_image_size / patch;
                if (vc.contains("window_kernel_size") && vc.at("window_kernel_size").is_array() && vc.at("window_kernel_size").size() == 2) {
                    m_window_kh = vc.at("window_kernel_size")[0].get<size_t>();
                    m_window_kw = vc.at("window_kernel_size")[1].get<size_t>();
                }
            }
            if (cfg.contains("merge_kernel_size") && cfg.at("merge_kernel_size").is_array() && cfg.at("merge_kernel_size").size() == 2) {
                m_merge_kh = cfg.at("merge_kernel_size")[0].get<size_t>();
                m_merge_kw = cfg.at("merge_kernel_size")[1].get<size_t>();
            }
            std::string downsample_mode = cfg.value("downsample_mode", std::string("16x"));
            m_token_divisor = (downsample_mode == "4x") ? 4 : 16;
        }
    }
}

EncodedImage VisionEncoderMiniCPMV4_6::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    ProcessorConfig config = ProcessorConfig::from_any_map(config_map, m_processor_config);
    const int patch_size = static_cast<int>(m_patch_size);
    const int scale_resolution = static_cast<int>(m_scale_resolution);
    const int max_slice_nums = static_cast<int>(m_max_slice_nums);

    clip_image_u8 source = tensor_to_clip_image_u8(image);
    const std::pair<int, int> image_size{source.ny, source.nx};  // {height, width}

    std::pair<int, int> best_grid{0, 0};
    if (m_slice_mode) {
        best_grid = get_sliced_grid(image_size, max_slice_nums, scale_resolution);
    }
    const bool has_slices = best_grid.first > 0 && best_grid.second > 0;

    // Build the ordered list of resized u8 patches: [source, *slices] and their
    // patch-grid sizes {grid_h, grid_w}.
    std::vector<clip_image_u8> patches_u8;
    std::vector<std::pair<int, int>> patch_grids;  // {grid_h, grid_w} in patches

    // Source image (always present).
    auto src_size = find_best_resize(image_size, scale_resolution, patch_size, /*allow_upscale=*/!has_slices);
    clip_image_u8 source_resized;
    bicubic_resize(source, source_resized, src_size.second, src_size.first);  // (target_width, target_height)
    patches_u8.push_back(std::move(source_resized));
    patch_grids.push_back({src_size.first / patch_size, src_size.second / patch_size});

    int slice_grid_h = 0;
    int slice_grid_w = 0;
    if (has_slices) {
        auto refine = get_refine_size(image_size, best_grid, scale_resolution, patch_size, /*allow_upscale=*/true);
        clip_image_u8 refine_img;
        bicubic_resize(source, refine_img, refine.second, refine.first);  // (target_width, target_height)
        int grid_y = best_grid.first;   // num rows of the slice grid
        int grid_x = best_grid.second;  // num cols of the slice grid
        int patch_h = refine.first / grid_y;
        int patch_w = refine.second / grid_x;
        slice_grid_h = patch_h / patch_size;
        slice_grid_w = patch_w / patch_size;
        for (int y = 0; y + patch_h <= refine.first; y += patch_h) {
            for (int x = 0; x + patch_w <= refine.second; x += patch_w) {
                patches_u8.push_back(crop_patch(refine_img, y, x, patch_h, patch_w));
                patch_grids.push_back({slice_grid_h, slice_grid_w});
            }
        }
    }

    // Run the vision encoder for each patch and concatenate the visual tokens.
    clip_ctx_double mean_std;
    for (int c = 0; c < 3; ++c) {
        mean_std.image_mean[c] = static_cast<double>(m_image_mean[c]) * 255.0;
        mean_std.image_std[c] = static_cast<double>(m_image_std[c]) * 255.0;
    }

    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();

    std::vector<ov::Tensor> patch_features;
    size_t total_tokens = 0;
    size_t hidden_size = 0;
    for (size_t p = 0; p < patches_u8.size(); ++p) {
        int grid_h = patch_grids[p].first;
        int grid_w = patch_grids[p].second;
        clip_image_f32 normalized = normalize_and_convert_to_chw(patches_u8[p], mean_std);
        ov::Tensor pixel_values = reshape_by_patch(normalized, patch_size);

        std::vector<int64_t> position_ids = compute_position_ids(grid_h, grid_w, static_cast<int>(m_num_patches_per_side));
        std::vector<int64_t> window_index = grouping_index(grid_h, grid_w, static_cast<int>(m_window_kh), static_cast<int>(m_window_kw));
        std::vector<int64_t> merge_index = grouping_index(grid_h / static_cast<int>(m_window_kh), grid_w / static_cast<int>(m_window_kw), static_cast<int>(m_merge_kh), static_cast<int>(m_merge_kw));

        ov::Tensor position_ids_t(ov::element::i64, ov::Shape{1, position_ids.size()});
        std::copy(position_ids.begin(), position_ids.end(), position_ids_t.data<int64_t>());
        ov::Tensor window_index_t(ov::element::i64, ov::Shape{window_index.size()});
        std::copy(window_index.begin(), window_index.end(), window_index_t.data<int64_t>());
        ov::Tensor merge_index_t(ov::element::i64, ov::Shape{merge_index.size()});
        std::copy(merge_index.begin(), merge_index.end(), merge_index_t.data<int64_t>());

        encoder.set_tensor("pixel_values", pixel_values);
        encoder.set_tensor("position_ids", position_ids_t);
        encoder.set_tensor("window_index", window_index_t);
        encoder.set_tensor("merge_index", merge_index_t);
        encoder.infer();

        const ov::Tensor& out = encoder.get_output_tensor();  // [num_tokens, hidden]
        ov::Tensor feat(out.get_element_type(), out.get_shape());
        std::memcpy(feat.data(), out.data(), out.get_byte_size());
        total_tokens += feat.get_shape().at(0);
        hidden_size = feat.get_shape().at(1);
        patch_features.push_back(std::move(feat));
    }

    // Concatenate along the token dimension -> [total_tokens, hidden_size].
    ov::Tensor resized_source(ov::element::f32, ov::Shape{total_tokens, hidden_size});
    float* dst = resized_source.data<float>();
    size_t offset = 0;
    for (const auto& feat : patch_features) {
        std::memcpy(dst + offset, feat.data(), feat.get_byte_size());
        offset += feat.get_size();
    }

    EncodedImage encoded;
    encoded.resized_source = std::move(resized_source);
    encoded.resized_source_size = ImageSize{static_cast<size_t>(patch_grids[0].first), static_cast<size_t>(patch_grids[0].second)};
    encoded.original_image_size = ImageSize{static_cast<size_t>(source.ny), static_cast<size_t>(source.nx)};
    encoded.num_image_tokens = total_tokens;
    size_t num_source_tokens = static_cast<size_t>(patch_grids[0].first) * patch_grids[0].second / m_token_divisor;
    size_t per_slice_tokens = has_slices ? (static_cast<size_t>(slice_grid_h) * slice_grid_w / m_token_divisor) : 0;
    // num_rows = best_grid[0] (grid_y), num_cols = best_grid[1] (grid_x); {0,0} if no slices.
    encoded.patches_grid = has_slices ? std::make_pair(best_grid.first, best_grid.second) : std::make_pair(0, 0);
    encoded.slices_shape = ov::Shape{num_source_tokens, per_slice_tokens};
    return encoded;
}

InputsEmbedderMiniCPMV4_6::InputsEmbedderMiniCPMV4_6(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, model_dir, tokenizer, device, device_config) { }

InputsEmbedderMiniCPMV4_6::InputsEmbedderMiniCPMV4_6(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) { }

std::string InputsEmbedderMiniCPMV4_6::build_image_placeholder(const EncodedImage& image, size_t image_id) const {
    const std::string& image_pad = m_vlm_config.image_pad_token;
    size_t num_source_tokens = image.slices_shape.size() > 0 ? image.slices_shape[0] : image.num_image_tokens;
    size_t per_slice_tokens = image.slices_shape.size() > 1 ? image.slices_shape[1] : 0;
    int num_rows = image.patches_grid.first;
    int num_cols = image.patches_grid.second;

    std::string placeholder;
    if (m_vlm_config.use_image_id) {
        placeholder += m_vlm_config.im_id_start + std::to_string(image_id) + m_vlm_config.im_id_end;
    }
    placeholder += m_vlm_config.im_start;
    for (size_t i = 0; i < num_source_tokens; ++i) {
        placeholder += image_pad;
    }
    placeholder += m_vlm_config.im_end;

    if (num_rows > 0 && num_cols > 0) {
        std::string slice_placeholder = m_vlm_config.slice_start;
        for (size_t i = 0; i < per_slice_tokens; ++i) {
            slice_placeholder += image_pad;
        }
        slice_placeholder += m_vlm_config.slice_end;
        std::string slices;
        for (int r = 0; r < num_rows; ++r) {
            if (r > 0) {
                slices += "\n";
            }
            for (int c = 0; c < num_cols; ++c) {
                slices += slice_placeholder;
            }
        }
        placeholder += slices;
    }
    return placeholder;
}

NormalizedPrompt InputsEmbedderMiniCPMV4_6::normalize_prompt(
    const std::string& prompt,
    size_t base_id,
    const std::vector<EncodedImage>& images
) const {
    auto [unified_prompt, images_sequence] = normalize(prompt, NATIVE_TAG, NATIVE_TAG + '\n', base_id, images.size());

    size_t searched_pos = 0;
    size_t local_image_index = 0;
    for (size_t new_image_id : images_sequence) {
        const EncodedImage& enc = images.at(new_image_id - base_id);
        std::string expanded = build_image_placeholder(enc, local_image_index);
        searched_pos = unified_prompt.find(NATIVE_TAG, searched_pos);
        OPENVINO_ASSERT(searched_pos != std::string::npos, "Image placeholder not found in prompt.");
        unified_prompt.replace(searched_pos, NATIVE_TAG.length(), expanded);
        searched_pos += expanded.length();
        ++local_image_index;
    }

    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

namespace {

ov::Tensor merge_text_and_image_embeddings(
    const ov::Tensor& input_ids,
    const ov::Tensor& text_embeds,
    const std::vector<const EncodedImage*>& image_embeds,
    int64_t image_pad_token_id) {
    ov::Shape shape = text_embeds.get_shape();
    size_t seq_len = shape.at(1);
    size_t embed_dim = shape.at(2);

    ov::Tensor merged(text_embeds.get_element_type(), shape);
    const float* text_data = text_embeds.data<float>();
    const int64_t* ids = input_ids.data<int64_t>();
    float* out = merged.data<float>();

    size_t image_idx = 0;
    size_t token_in_image = 0;
    const float* cur_image_data = image_embeds.empty() ? nullptr : image_embeds[0]->resized_source.data<float>();
    size_t cur_image_tokens = image_embeds.empty() ? 0 : image_embeds[0]->resized_source.get_shape().at(0);

    for (size_t j = 0; j < seq_len; ++j) {
        size_t offset = j * embed_dim;
        if (ids[j] == image_pad_token_id && image_idx < image_embeds.size()) {
            std::copy_n(cur_image_data + token_in_image * embed_dim, embed_dim, out + offset);
            ++token_in_image;
            if (token_in_image == cur_image_tokens) {
                ++image_idx;
                token_in_image = 0;
                if (image_idx < image_embeds.size()) {
                    cur_image_data = image_embeds[image_idx]->resized_source.data<float>();
                    cur_image_tokens = image_embeds[image_idx]->resized_source.get_shape().at(0);
                }
            }
        } else {
            std::copy_n(text_data + offset, embed_dim, out + offset);
        }
    }
    return merged;
}

} // namespace

ov::Tensor InputsEmbedderMiniCPMV4_6::get_inputs_embeds(
    const std::string& unified_prompt,
    const std::vector<ov::genai::EncodedImage>& images,
    ov::genai::VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& images_sequence) {
    ov::Tensor input_ids = get_encoded_input_ids(unified_prompt, metrics);
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = get_text_embedding(req, input_ids, metrics);

    if (images.empty()) {
        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
        return inputs_embeds;
    }

    std::vector<const EncodedImage*> image_embeds;
    image_embeds.reserve(images_sequence.size());
    for (size_t new_image_id : images_sequence) {
        image_embeds.push_back(&images.at(new_image_id));
    }

    auto start_tokenizer_time = std::chrono::steady_clock::now();
    ov::Tensor encoded_pad = m_tokenizer.encode(m_vlm_config.image_pad_token, ov::genai::add_special_tokens(false)).input_ids;
    auto end_tokenizer_time = std::chrono::steady_clock::now();
    OPENVINO_ASSERT(metrics.raw_metrics.tokenization_durations.size() > 0);
    metrics.raw_metrics.tokenization_durations[metrics.raw_metrics.tokenization_durations.size() - 1] += ov::genai::MicroSeconds(PerfMetrics::get_microsec(end_tokenizer_time - start_tokenizer_time));
    int64_t image_pad_token_id = encoded_pad.data<int64_t>()[encoded_pad.get_size() - 1];

    return merge_text_and_image_embeddings(input_ids, text_embeds, image_embeds, image_pad_token_id);
}

} // namespace ov::genai
