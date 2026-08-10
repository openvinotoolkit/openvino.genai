// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/minicpmv4_6/classes.hpp"

#include <cmath>
#include <fstream>
#include <numeric>

#include <nlohmann/json.hpp>

#include "visual_language/clip.hpp"
#include "json_utils.hpp"

namespace ov::genai {

namespace {

const std::string NATIVE_TAG = "<|image_pad|>";  // image_token placeholder in templated prompt

// ---------------------------------------------------------------------------
// MiniCPMV4_6ImageProcessor port (transformers image_processing_minicpmv4_6.py)
// ---------------------------------------------------------------------------

int ensure_divide(int length, int divisor) {
    return std::max(static_cast<int>(std::round(static_cast<double>(length) / divisor) * divisor), divisor);
}

// Returns {height, width}.
std::pair<int, int> find_best_resize(int height, int width, int scale_resolution, int patch_size, bool allow_upscale) {
    if ((static_cast<int64_t>(height) * width > static_cast<int64_t>(scale_resolution) * scale_resolution) || allow_upscale) {
        double aspect_ratio = static_cast<double>(width) / height;
        height = static_cast<int>(scale_resolution / std::sqrt(aspect_ratio));
        width = static_cast<int>(height * aspect_ratio);
    }
    int best_width = ensure_divide(width, patch_size * 4);
    int best_height = ensure_divide(height, patch_size * 4);
    return {best_height, best_width};
}

// Returns {height, width}.
std::pair<int, int> get_refine_size(int height, int width, int grid_y, int grid_x, int scale_resolution, int patch_size, bool allow_upscale) {
    int refine_width = ensure_divide(width, grid_x);
    int refine_height = ensure_divide(height, grid_y);
    auto [best_height, best_width] = find_best_resize(refine_height / grid_y, refine_width / grid_x, scale_resolution, patch_size, allow_upscale);
    return {best_height * grid_y, best_width * grid_x};
}

// Returns {grid_x, grid_y} matching HF [num_cols, num_rows]. {0,0} means no slicing.
std::pair<int, int> get_sliced_grid(int height, int width, int max_slice_nums, int scale_resolution) {
    double log_ratio = std::log(static_cast<double>(width) / height);
    double ratio = static_cast<double>(width) * height / (static_cast<double>(scale_resolution) * scale_resolution);
    int multiple = std::min(static_cast<int>(std::ceil(ratio)), max_slice_nums);
    if (multiple <= 1) {
        return {0, 0};
    }
    std::pair<int, int> best_grid{1, 1};  // {num_cols, num_rows}
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

// Crop a region [y0, y0+h) x [x0, x0+w) from an RGB u8 image.
clip_image_u8 crop_region(const clip_image_u8& img, int x0, int y0, int w, int h) {
    clip_image_u8 out;
    out.nx = w;
    out.ny = h;
    out.buf.resize(static_cast<size_t>(w) * h * 3);
    for (int dy = 0; dy < h; ++dy) {
        for (int dx = 0; dx < w; ++dx) {
            for (int c = 0; c < 3; ++c) {
                out.buf[(static_cast<size_t>(dy) * w + dx) * 3 + c] =
                    img.buf[(static_cast<size_t>(y0 + dy) * img.nx + (x0 + dx)) * 3 + c];
            }
        }
    }
    return out;
}

// Rescale to [0,1], normalize with mean/std, output CHW float. Mirrors HF
// rescale_and_normalize on a single [C,H,W] tensor.
std::vector<float> rescale_normalize_chw(const clip_image_u8& img, const std::array<float, 3>& mean, const std::array<float, 3>& std) {
    int h = img.ny, w = img.nx;
    std::vector<float> out(static_cast<size_t>(3) * h * w);
    for (int c = 0; c < 3; ++c) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                float v = img.buf[(static_cast<size_t>(y) * w + x) * 3 + c] / 255.0f;
                out[(static_cast<size_t>(c) * h + y) * w + x] = (v - mean[c]) / std[c];
            }
        }
    }
    return out;
}

// reshape_by_patch: CHW float [3,H,W] -> NaViT patchified [3, patch, H*W/patch].
// Mirrors torch.nn.functional.unfold followed by reshape/permute.
// For a patch grid (H/patch) x (W/patch), unfold produces, for each output
// column (a patch), the patch_size*patch_size values per channel in row-major
// order within the patch. Then reshape(C, patch, patch, num_patches).permute(
// 0,1,3,2).reshape(C, patch, -1) => [C, patch, num_patches*patch].
std::vector<float> reshape_by_patch(const std::vector<float>& chw, int h, int w, int patch) {
    int gh = h / patch, gw = w / patch;
    int num_patches = gh * gw;
    // unfold column index p corresponds to patch (py, px) with py=p/gw, px=p%gw.
    // patches[c, i, j, p] where i,j in [0,patch): value at (py*patch+i, px*patch+j)
    // then permute(0,1,3,2): [c, i, p, j] -> reshape [c, patch, num_patches*patch]
    // final layout: out[c][i][p*patch + j]
    std::vector<float> out(static_cast<size_t>(3) * patch * (num_patches * patch));
    size_t last_dim = static_cast<size_t>(num_patches) * patch;
    for (int c = 0; c < 3; ++c) {
        for (int p = 0; p < num_patches; ++p) {
            int py = p / gw, px = p % gw;
            for (int i = 0; i < patch; ++i) {
                for (int j = 0; j < patch; ++j) {
                    int src_y = py * patch + i;
                    int src_x = px * patch + j;
                    float v = chw[(static_cast<size_t>(c) * h + src_y) * w + src_x];
                    out[(static_cast<size_t>(c) * patch + i) * last_dim + (static_cast<size_t>(p) * patch + j)] = v;
                }
            }
        }
    }
    return out;
}

struct ImagePatches {
    // Each patch stored as reshape_by_patch output [3, patch, tokens_w*patch]
    // together with its target size {h_patches, w_patches}.
    std::vector<std::vector<float>> pv;              // per-patch reshaped pixel values
    std::vector<std::array<int, 2>> target_sizes;    // per-patch {h/patch, w/patch}
    std::pair<int, int> grid{0, 0};                  // {num_cols, num_rows}
};

ImagePatches preprocess_image(const clip_image_u8& image, const ProcessorConfig& cfg) {
    int patch_size = static_cast<int>(cfg.patch_size);
    int scale_resolution = static_cast<int>(cfg.scale_resolution);
    int max_slice_nums = static_cast<int>(cfg.max_slice_nums);
    std::array<float, 3> mean = cfg.image_mean;
    std::array<float, 3> std = cfg.image_std;

    int orig_h = image.ny, orig_w = image.nx;
    auto grid = get_sliced_grid(orig_h, orig_w, max_slice_nums, scale_resolution);  // {cols, rows}
    bool has_slices = (grid.first > 0 && grid.second > 0);

    ImagePatches result;
    result.grid = grid;

    // Source (thumbnail) image.
    auto [source_h, source_w] = find_best_resize(orig_h, orig_w, scale_resolution, patch_size, /*allow_upscale=*/!has_slices);
    clip_image_u8 source_img;
    bicubic_resize(image, source_img, source_w, source_h);
    auto source_chw = rescale_normalize_chw(source_img, mean, std);
    result.pv.push_back(reshape_by_patch(source_chw, source_h, source_w, patch_size));
    result.target_sizes.push_back({source_h / patch_size, source_w / patch_size});

    if (has_slices) {
        int grid_x = grid.first, grid_y = grid.second;
        auto [refine_h, refine_w] = get_refine_size(orig_h, orig_w, grid_y, grid_x, scale_resolution, patch_size, /*allow_upscale=*/true);
        clip_image_u8 refine_img;
        bicubic_resize(image, refine_img, refine_w, refine_h);
        int patch_h = refine_h / grid_y;
        int patch_w = refine_w / grid_x;
        // divide_to_patches: row-major over grid_y rows then grid_x cols.
        for (int gy = 0; gy < grid_y; ++gy) {
            for (int gx = 0; gx < grid_x; ++gx) {
                clip_image_u8 slice = crop_region(refine_img, gx * patch_w, gy * patch_h, patch_w, patch_h);
                auto slice_chw = rescale_normalize_chw(slice, mean, std);
                result.pv.push_back(reshape_by_patch(slice_chw, patch_h, patch_w, patch_size));
                result.target_sizes.push_back({patch_h / patch_size, patch_w / patch_size});
            }
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// NaViT index / mask precompute (optimum-intel _OVMiniCPMV4_6ForCausalLM)
// ---------------------------------------------------------------------------

// Block-diagonal additive attention mask [1, total, total]: 0 within a block,
// -inf across blocks.
ov::Tensor block_diagonal_mask(const std::vector<int>& seqlens) {
    int total = std::accumulate(seqlens.begin(), seqlens.end(), 0);
    ov::Tensor mask(ov::element::f32, ov::Shape{1, static_cast<size_t>(total), static_cast<size_t>(total)});
    float* d = mask.data<float>();
    std::fill(d, d + static_cast<size_t>(total) * total, -std::numeric_limits<float>::infinity());
    int offset = 0;
    for (int len : seqlens) {
        for (int i = 0; i < len; ++i) {
            for (int j = 0; j < len; ++j) {
                d[static_cast<size_t>(offset + i) * total + (offset + j)] = 0.0f;
            }
        }
        offset += len;
    }
    return mask;
}

std::vector<int64_t> patch_position_ids(const std::vector<std::array<int, 2>>& target_sizes, int num_side) {
    // boundaries = arange(1/num_side, 1.0, 1/num_side); bucketize(right=True).
    std::vector<float> boundaries;
    for (int i = 1; i < num_side; ++i) {
        boundaries.push_back(static_cast<float>(i) / num_side);
    }
    std::vector<int64_t> pos_ids;
    for (const auto& ts : target_sizes) {
        int nb_h = ts[0], nb_w = ts[1];
        std::vector<int64_t> bucket_h(nb_h), bucket_w(nb_w);
        for (int i = 0; i < nb_h; ++i) {
            float frac = static_cast<float>(i) / nb_h;
            // bucketize right=True: number of boundaries <= frac
            int b = 0;
            while (b < static_cast<int>(boundaries.size()) && boundaries[b] <= frac) ++b;
            bucket_h[i] = b;
        }
        for (int j = 0; j < nb_w; ++j) {
            float frac = static_cast<float>(j) / nb_w;
            int b = 0;
            while (b < static_cast<int>(boundaries.size()) && boundaries[b] <= frac) ++b;
            bucket_w[j] = b;
        }
        for (int i = 0; i < nb_h; ++i) {
            for (int j = 0; j < nb_w; ++j) {
                pos_ids.push_back(bucket_h[i] * num_side + bucket_w[j]);
            }
        }
    }
    return pos_ids;
}

// Returns {window_index, window_seqlens}.
std::pair<std::vector<int64_t>, std::vector<int>> window_index_and_seqlens(
        const std::vector<std::array<int, 2>>& target_sizes, int window_h, int window_w) {
    std::vector<int64_t> window_index;
    std::vector<int> window_seqlens;
    int token_offset = 0;
    for (const auto& ts : target_sizes) {
        int height = ts[0], width = ts[1];
        int num_windows_h = height / window_h;
        int num_windows_w = width / window_w;
        // index reshaped [nwh, wh, nww, ww] permute(0,2,1,3) -> [nwh, nww, wh, ww]
        for (int wy = 0; wy < num_windows_h; ++wy) {
            for (int wx = 0; wx < num_windows_w; ++wx) {
                for (int iy = 0; iy < window_h; ++iy) {
                    for (int ix = 0; ix < window_w; ++ix) {
                        int r = wy * window_h + iy;
                        int c = wx * window_w + ix;
                        window_index.push_back(static_cast<int64_t>(r) * width + c + token_offset);
                    }
                }
            }
        }
        int num_windows = num_windows_h * num_windows_w;
        for (int i = 0; i < num_windows; ++i) {
            window_seqlens.push_back(window_h * window_w);
        }
        token_offset += height * width;
    }
    return {window_index, window_seqlens};
}

// Gather index that reorders a per-window contiguous layout (used for both the
// spatial-merge gather over full-resolution patches and the final merge gather
// over downsampled patches). kernel = {kh, kw}.
std::vector<int64_t> merge_gather_index(const std::vector<std::array<int, 2>>& sizes, int kh, int kw) {
    std::vector<int64_t> out;
    int offset = 0;
    for (const auto& ts : sizes) {
        int height = ts[0], width = ts[1];
        int mh = height / kh, mw = width / kw;
        // arange(h*w).reshape(mh,kh,mw,kw).permute(0,2,1,3).reshape(-1)
        for (int a = 0; a < mh; ++a) {
            for (int b = 0; b < mw; ++b) {
                for (int i = 0; i < kh; ++i) {
                    for (int j = 0; j < kw; ++j) {
                        int r = a * kh + i;
                        int c = b * kw + j;
                        out.push_back(static_cast<int64_t>(r) * width + c + offset);
                    }
                }
            }
        }
        offset += height * width;
    }
    return out;
}

template <typename T>
ov::Tensor to_1d_tensor(const std::vector<T>& v, ov::element::Type type) {
    ov::Tensor t(type, ov::Shape{v.size()});
    std::copy(v.begin(), v.end(), t.data<T>());
    return t;
}

} // namespace

// ---------------------------------------------------------------------------
// VisionEncoderMiniCPMV4_6
// ---------------------------------------------------------------------------

void VisionEncoderMiniCPMV4_6::read_vision_params(const std::filesystem::path& config_dir_path) {
    std::ifstream stream(config_dir_path / "config.json");
    if (!stream.is_open()) {
        return;
    }
    nlohmann::json cfg = nlohmann::json::parse(stream);
    using ov::genai::utils::read_json_param;
    if (cfg.contains("vision_config")) {
        const auto& vc = cfg.at("vision_config");
        size_t image_size = 980, patch_size = 14;
        read_json_param(vc, "image_size", image_size);
        read_json_param(vc, "patch_size", patch_size);
        m_patch_size = patch_size;
        m_num_patches_per_side = image_size / patch_size;
        if (vc.contains("window_kernel_size") && vc.at("window_kernel_size").is_array()) {
            auto wk = vc.at("window_kernel_size").get<std::vector<size_t>>();
            if (wk.size() == 2) m_window_kernel_size = {wk[0], wk[1]};
        }
    }
    if (cfg.contains("merge_kernel_size") && cfg.at("merge_kernel_size").is_array()) {
        auto mk = cfg.at("merge_kernel_size").get<std::vector<size_t>>();
        if (mk.size() == 2) m_merge_kernel_size = {mk[0], mk[1]};
    }
    std::string downsample_mode = "16x";
    read_json_param(cfg, "downsample_mode", downsample_mode);
    m_token_divisor = (downsample_mode == "4x") ? 4 : 16;
}

VisionEncoderMiniCPMV4_6::VisionEncoderMiniCPMV4_6(
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap properties) :
    VisionEncoder(model_dir, device, properties) {
    read_vision_params(model_dir);
}

VisionEncoderMiniCPMV4_6::VisionEncoderMiniCPMV4_6(
    const ModelsMap& models_map,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    VisionEncoder(models_map, config_dir_path, device, device_config) {
    read_vision_params(config_dir_path);
}

EncodedImage VisionEncoderMiniCPMV4_6::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    ProcessorConfig config = ProcessorConfig::from_any_map(config_map, m_processor_config);

    clip_image_u8 input_image = tensor_to_clip_image_u8(image);
    ImagePatches patches = preprocess_image(input_image, config);

    const int patch = static_cast<int>(config.patch_size);
    const int window_h = static_cast<int>(m_window_kernel_size[0]);
    const int window_w = static_cast<int>(m_window_kernel_size[1]);
    const int merge_h = static_cast<int>(m_merge_kernel_size[0]);
    const int merge_w = static_cast<int>(m_merge_kernel_size[1]);

    // Concatenate per-patch reshape_by_patch outputs along the last dim to build
    // pixel_values [1, 3, patch, sum(patch * tokens_per_patch)].
    size_t total_last = 0;
    for (const auto& ts : patches.target_sizes) {
        total_last += static_cast<size_t>(ts[0]) * ts[1] * patch;  // tokens * patch
    }
    ov::Tensor pixel_values(ov::element::f32, ov::Shape{1, 3, static_cast<size_t>(patch), total_last});
    float* pv_data = pixel_values.data<float>();
    // For each channel c and row i (patch), fill columns sequentially per patch.
    for (int c = 0; c < 3; ++c) {
        for (int i = 0; i < patch; ++i) {
            size_t col_offset = 0;
            for (size_t p = 0; p < patches.pv.size(); ++p) {
                const auto& src = patches.pv[p];
                size_t tokens = static_cast<size_t>(patches.target_sizes[p][0]) * patches.target_sizes[p][1];
                size_t src_last = tokens * patch;
                const float* src_row = src.data() + (static_cast<size_t>(c) * patch + i) * src_last;
                float* dst = pv_data + (static_cast<size_t>(c) * patch + i) * total_last + col_offset;
                std::copy(src_row, src_row + src_last, dst);
                col_offset += src_last;
            }
        }
    }

    // Index / mask tensors.
    std::vector<int64_t> pos_ids = patch_position_ids(patches.target_sizes, static_cast<int>(m_num_patches_per_side));
    std::vector<int> encoder_seqlens;
    for (const auto& ts : patches.target_sizes) {
        encoder_seqlens.push_back(ts[0] * ts[1]);
    }
    ov::Tensor encoder_mask = block_diagonal_mask(encoder_seqlens);

    auto [window_index, window_seqlens] = window_index_and_seqlens(patches.target_sizes, window_h, window_w);
    std::vector<int64_t> reverse_window_index(window_index.size());
    {
        std::vector<size_t> order(window_index.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return window_index[a] < window_index[b]; });
        for (size_t k = 0; k < order.size(); ++k) {
            reverse_window_index[k] = static_cast<int64_t>(order[k]);
        }
    }
    ov::Tensor window_mask = block_diagonal_mask(window_seqlens);

    std::vector<int64_t> merge_gather = merge_gather_index(patches.target_sizes, window_h, window_w);

    std::vector<std::array<int, 2>> downsampled_sizes;
    for (const auto& ts : patches.target_sizes) {
        downsampled_sizes.push_back({ts[0] / 2, ts[1] / 2});
    }
    std::vector<int> downsampled_seqlens;
    for (const auto& ds : downsampled_sizes) {
        downsampled_seqlens.push_back(ds[0] * ds[1]);
    }
    ov::Tensor downsampled_mask = block_diagonal_mask(downsampled_seqlens);
    std::vector<int64_t> final_gather = merge_gather_index(downsampled_sizes, merge_h, merge_w);

    encoder.set_tensor("pixel_values", pixel_values);
    encoder.set_tensor("pos_ids", to_1d_tensor(pos_ids, ov::element::i64));
    encoder.set_tensor("encoder_attention_mask", encoder_mask);
    encoder.set_tensor("downsampled_attention_mask", downsampled_mask);
    encoder.set_tensor("window_index", to_1d_tensor(window_index, ov::element::i64));
    encoder.set_tensor("reverse_window_index", to_1d_tensor(reverse_window_index, ov::element::i64));
    encoder.set_tensor("window_attention_mask", window_mask);
    encoder.set_tensor("merge_gather_index", to_1d_tensor(merge_gather, ov::element::i64));
    encoder.set_tensor("final_gather_index", to_1d_tensor(final_gather, ov::element::i64));
    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_output_tensor();  // [total_tokens, hidden]
    ov::Tensor image_features(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    // Per-patch image-token counts: (h*w) / token_divisor.
    // Source patch is target_sizes[0]; slices (if any) share target_sizes[1].
    EncodedImage encoded;
    encoded.resized_source = std::move(image_features);
    size_t source_tokens = static_cast<size_t>(patches.target_sizes[0][0]) * patches.target_sizes[0][1] / m_token_divisor;
    size_t slice_tokens = 0;
    if (patches.target_sizes.size() > 1) {
        slice_tokens = static_cast<size_t>(patches.target_sizes[1][0]) * patches.target_sizes[1][1] / m_token_divisor;
    }
    encoded.patches_grid = patches.grid;  // {num_cols, num_rows}
    // Repurpose resized_source_size to carry per-patch token counts for
    // normalize_prompt (height=source tokens, width=slice tokens).
    encoded.resized_source_size = ImageSize{source_tokens, slice_tokens};
    size_t num_rows = patches.grid.second > 0 ? static_cast<size_t>(patches.grid.second) : 0;
    size_t num_cols = patches.grid.first > 0 ? static_cast<size_t>(patches.grid.first) : 0;
    encoded.num_image_tokens = source_tokens + num_rows * num_cols * slice_tokens;
    return encoded;
}

// ---------------------------------------------------------------------------
// InputsEmbedderMiniCPMV4_6
// ---------------------------------------------------------------------------

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

NormalizedPrompt InputsEmbedderMiniCPMV4_6::normalize_prompt(
        const std::string& prompt,
        size_t base_id,
        const std::vector<EncodedImage>& images) const {
    auto [unified_prompt, images_sequence] = normalize(prompt, NATIVE_TAG, NATIVE_TAG, base_id, images.size());

    const std::string image_pad = m_vlm_config.image_pad_token;  // <|image_pad|>
    const std::string image_start = m_vlm_config.im_start;       // <image>
    const std::string image_end = m_vlm_config.im_end;           // </image>
    const std::string slice_start = m_vlm_config.slice_start;    // <slice>
    const std::string slice_end = m_vlm_config.slice_end;        // </slice>
    const std::string id_start = m_vlm_config.im_id_start;       // <image_id>
    const std::string id_end = m_vlm_config.im_id_end;           // </image_id>
    const bool use_image_id = m_vlm_config.use_image_id;

    size_t searched_pos = 0;
    size_t local_image_index = 0;
    for (size_t new_image_id : images_sequence) {
        const EncodedImage& img = images.at(new_image_id - base_id);
        size_t source_tokens = img.resized_source_size.height;
        size_t slice_tokens = img.resized_source_size.width;
        int num_cols = img.patches_grid.first;
        int num_rows = img.patches_grid.second;

        std::string placeholder;
        if (use_image_id) {
            placeholder += id_start + std::to_string(local_image_index) + id_end;
        }
        placeholder += image_start;
        for (size_t t = 0; t < source_tokens; ++t) {
            placeholder += image_pad;
        }
        placeholder += image_end;
        if (num_rows > 0 && num_cols > 0) {
            std::string slice_ph = slice_start;
            for (size_t t = 0; t < slice_tokens; ++t) {
                slice_ph += image_pad;
            }
            slice_ph += slice_end;
            for (int r = 0; r < num_rows; ++r) {
                if (r > 0) {
                    placeholder += "\n";
                }
                for (int c = 0; c < num_cols; ++c) {
                    placeholder += slice_ph;
                }
            }
        }

        searched_pos = unified_prompt.find(NATIVE_TAG, searched_pos);
        OPENVINO_ASSERT(searched_pos != std::string::npos, "Image placeholder not found in prompt for MiniCPM-V-4.6");
        unified_prompt.replace(searched_pos, NATIVE_TAG.length(), placeholder);
        searched_pos += placeholder.length();
        ++local_image_index;
    }

    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderMiniCPMV4_6::get_inputs_embeds(
        const std::string& unified_prompt,
        const std::vector<ov::genai::EncodedImage>& images,
        ov::genai::VLMPerfMetrics& metrics,
        bool recalculate_merged_embeddings,
        const std::vector<size_t>& images_sequence) {
    ov::Tensor input_ids = get_encoded_input_ids(unified_prompt, metrics);
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = m_embedding->infer(req, input_ids);

    if (images.empty()) {
        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
        return inputs_embeds;
    }

    auto text_shape = text_embeds.get_shape();
    size_t seq_len = text_shape.at(1);
    size_t embed_dim = text_shape.at(2);

    ov::Tensor merged(text_embeds.get_element_type(), text_shape);
    const float* text_data = text_embeds.data<float>();
    float* merged_data = merged.data<float>();
    std::memcpy(merged_data, text_data, text_embeds.get_byte_size());

    const int64_t image_token_id = m_vlm_config.image_token_id;
    const int64_t* ids = input_ids.data<int64_t>();

    // Concatenate image features in the order they appear in images_sequence and
    // masked-scatter them onto image_token positions (matches HF masked_scatter).
    size_t img_ptr = 0;  // index into current image's feature rows
    size_t seq_idx = 0;  // which image in images_sequence
    const float* cur_features = nullptr;
    size_t cur_rows = 0;
    auto load_image = [&](size_t seq_pos) {
        size_t img_id = images_sequence.at(seq_pos);
        const ov::Tensor& feats = images.at(img_id).resized_source;
        cur_features = feats.data<float>();
        cur_rows = feats.get_shape().at(0);
        img_ptr = 0;
    };
    if (!images_sequence.empty()) {
        load_image(0);
    }

    for (size_t j = 0; j < seq_len; ++j) {
        if (ids[j] == image_token_id) {
            OPENVINO_ASSERT(cur_features != nullptr, "MiniCPM-V-4.6: image tokens present but no image features");
            if (img_ptr == cur_rows) {
                ++seq_idx;
                OPENVINO_ASSERT(seq_idx < images_sequence.size(),
                    "MiniCPM-V-4.6: number of image tokens exceeds available image features");
                load_image(seq_idx);
            }
            std::copy_n(cur_features + img_ptr * embed_dim, embed_dim, merged_data + j * embed_dim);
            ++img_ptr;
        }
    }
    return merged;
}

} // namespace ov::genai
