// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/molmo2/classes.hpp"

#include <algorithm>
#include <array>
#include <limits>
#include <numeric>

#include "visual_language/clip.hpp"

#include "utils.hpp"

namespace ov::genai {
namespace {

// ---------------------------------------------------------------------------------------------
// Molmo2 multi-crop tiling + pooling preprocessing.
//
// Ported from the model's own remote code (`image_processing_molmo2.py` /
// `processing_molmo2.py`, published at huggingface.co/allenai/MolmoWeb-4B): an image is split
// into up to `max_crops` overlapping high-resolution crops plus one low-resolution global
// thumbnail; a `pixel_values` tensor holds all (patchified) crops, and an `image_token_pooling`
// tensor tells the vision model which patches to average-pool (2x2 groups, by default) into
// each of the visual tokens that get spliced into the prompt.
// ---------------------------------------------------------------------------------------------

// A single normalized RGB float crop, HWC layout (row-major: for each row, for each column, R/G/B).
struct Molmo2Crop {
    std::vector<float> hwc;
    int height = 0;
    int width = 0;
};

float normalize_channel(uint8_t value, float mean, float std) {
    return (static_cast<float>(value) / 255.0f - mean) / std;
}

// Resizes (nearest structural equivalent of Python's torchvision-based resize, already used
// throughout this codebase for other VLMs) and normalizes an image to a HWC float buffer.
Molmo2Crop resize_and_normalize(const clip_image_u8& src, int target_width, int target_height, const std::array<float, 3>& mean, const std::array<float, 3>& std_dev) {
    clip_image_u8 resized;
    bilinear_resize(src, resized, target_width, target_height);

    Molmo2Crop out;
    out.height = target_height;
    out.width = target_width;
    out.hwc.resize(static_cast<size_t>(target_height) * target_width * 3);
    for (size_t i = 0; i < out.hwc.size(); i += 3) {
        for (size_t c = 0; c < 3; c++) {
            out.hwc[i + c] = normalize_channel(resized.buf[i + c], mean[c], std_dev[c]);
        }
    }
    return out;
}

// Port of Python's `select_tiling`: picks an (num_rows, num_cols) tiling of `patch_size`-aligned
// crop windows (with `num_rows*num_cols <= max_crops`) that best matches the image's aspect
// ratio, preferring the tiling that requires the least up/downscaling.
std::pair<int, int> select_tiling(int h, int w, int patch_size, int max_crops) {
    std::vector<std::pair<int, int>> tilings;
    for (int i = 1; i <= max_crops; i++) {
        for (int j = 1; j <= max_crops; j++) {
            if (i * j <= max_crops) {
                tilings.emplace_back(i, j);
            }
        }
    }
    // Sort so argmin/argmax favour smaller tilings in the event of a tie (matches Python's
    // `tilings.sort(key=lambda x: (x[0]*x[1], x[0]))`).
    std::sort(tilings.begin(), tilings.end(), [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
        int area_a = a.first * a.second, area_b = b.first * b.second;
        return area_a != area_b ? area_a < area_b : a.first < b.first;
    });

    std::vector<float> required_scale(tilings.size());
    bool all_below_one = true;
    for (size_t k = 0; k < tilings.size(); k++) {
        float scale_h = static_cast<float>(tilings[k].first * patch_size) / static_cast<float>(h);
        float scale_w = static_cast<float>(tilings[k].second * patch_size) / static_cast<float>(w);
        required_scale[k] = std::min(scale_h, scale_w);
        if (required_scale[k] >= 1.0f) {
            all_below_one = false;
        }
    }

    size_t best_ix = 0;
    if (all_below_one) {
        // Forced to downscale regardless of tiling: minimize the amount of downscaling (argmax).
        for (size_t k = 1; k < required_scale.size(); k++) {
            if (required_scale[k] > required_scale[best_ix]) {
                best_ix = k;
            }
        }
    } else {
        // Pick the resolution that required the least upscaling (argmin among scale >= 1).
        for (size_t k = 0; k < required_scale.size(); k++) {
            if (required_scale[k] < 1.0f) {
                required_scale[k] = std::numeric_limits<float>::max();
            }
        }
        for (size_t k = 1; k < required_scale.size(); k++) {
            if (required_scale[k] < required_scale[best_ix]) {
                best_ix = k;
            }
        }
    }
    return tilings[best_ix];
}

// Result of splitting an image into overlapping high-resolution crops: the crops themselves,
// and a de-overlapped patch-id mosaic (in image scanline order) pointing into those crops'
// flattened patch space.
struct OverlappingCrops {
    std::vector<Molmo2Crop> crops;
    std::vector<int32_t> patch_idx;  // flattened [mosaic_h * mosaic_w], row-major
    int mosaic_h = 0;
    int mosaic_w = 0;
};

// Port of Python's `build_overlapping_crops`.
OverlappingCrops build_overlapping_crops(
    const clip_image_u8& image,
    int max_crops,
    int left_margin,
    int right_margin,
    int crop_size,
    int patch_size,
    const std::array<float, 3>& mean,
    const std::array<float, 3>& std_dev
) {
    const int total_margin_pixels = patch_size * (left_margin + right_margin);
    const int crop_patches = crop_size / patch_size;
    const int crop_window_patches = crop_patches - (left_margin + right_margin);
    const int crop_window_size = crop_window_patches * patch_size;

    const std::pair<int, int> tiling = select_tiling(
        std::max(image.ny - total_margin_pixels, 1),
        std::max(image.nx - total_margin_pixels, 1),
        crop_window_size,
        max_crops);
    const int tiling_h = tiling.first;
    const int tiling_w = tiling.second;

    const int resized_h = tiling_h * crop_window_size + total_margin_pixels;
    const int resized_w = tiling_w * crop_window_size + total_margin_pixels;
    const Molmo2Crop src = resize_and_normalize(image, resized_w, resized_h, mean, std_dev);

    const int n_crops = tiling_h * tiling_w;
    OverlappingCrops result;
    result.crops.resize(n_crops);
    // crop_patch_idx[on_crop] is a [crop_patches x crop_patches] grid of unique global patch ids,
    // masked to -1 in this crop's overlap regions (matches Python's `patch_idx_arr` per-crop).
    std::vector<std::vector<int32_t>> crop_patch_idx(n_crops, std::vector<int32_t>(static_cast<size_t>(crop_patches) * crop_patches));

    int on_crop = 0;
    for (int i = 0; i < tiling_h; i++) {
        const int y0 = i * crop_window_size;
        for (int j = 0; j < tiling_w; j++) {
            const int x0 = j * crop_window_size;

            Molmo2Crop& crop = result.crops[on_crop];
            crop.height = crop_size;
            crop.width = crop_size;
            crop.hwc.resize(static_cast<size_t>(crop_size) * crop_size * 3);
            for (int y = 0; y < crop_size; y++) {
                for (int x = 0; x < crop_size; x++) {
                    for (int c = 0; c < 3; c++) {
                        crop.hwc[(static_cast<size_t>(y) * crop_size + x) * 3 + c] =
                            src.hwc[(static_cast<size_t>(y0 + y) * resized_w + (x0 + x)) * 3 + c];
                    }
                }
            }

            std::vector<int32_t>& idx = crop_patch_idx[on_crop];
            for (int py = 0; py < crop_patches; py++) {
                for (int px = 0; px < crop_patches; px++) {
                    idx[py * crop_patches + px] = on_crop * crop_patches * crop_patches + py * crop_patches + px;
                }
            }
            // Mask out patch ids that fall in the overlap region shared with a neighboring crop.
            if (i != 0) {
                for (int py = 0; py < left_margin; py++) {
                    for (int px = 0; px < crop_patches; px++) {
                        idx[py * crop_patches + px] = -1;
                    }
                }
            }
            if (j != 0) {
                for (int py = 0; py < crop_patches; py++) {
                    for (int px = 0; px < left_margin; px++) {
                        idx[py * crop_patches + px] = -1;
                    }
                }
            }
            if (i != tiling_h - 1) {
                for (int py = crop_patches - right_margin; py < crop_patches; py++) {
                    for (int px = 0; px < crop_patches; px++) {
                        idx[py * crop_patches + px] = -1;
                    }
                }
            }
            if (j != tiling_w - 1) {
                for (int py = 0; py < crop_patches; py++) {
                    for (int px = crop_patches - right_margin; px < crop_patches; px++) {
                        idx[py * crop_patches + px] = -1;
                    }
                }
            }
            on_crop++;
        }
    }

    // Reorder from crop-major order into image-scanline order (matches Python's
    // reshape->transpose([0,2,1,3])->reshape), then drop masked (-1) entries.
    std::vector<int32_t> scanline;
    scanline.reserve(static_cast<size_t>(tiling_h) * crop_patches * tiling_w * crop_patches);
    for (int ti = 0; ti < tiling_h; ti++) {
        for (int py = 0; py < crop_patches; py++) {
            for (int tj = 0; tj < tiling_w; tj++) {
                const int crop_id = ti * tiling_w + tj;
                for (int px = 0; px < crop_patches; px++) {
                    scanline.push_back(crop_patch_idx[crop_id][py * crop_patches + px]);
                }
            }
        }
    }

    const int mosaic_h = resized_h / patch_size;
    const int mosaic_w = resized_w / patch_size;
    result.patch_idx.reserve(static_cast<size_t>(mosaic_h) * mosaic_w);
    for (int32_t v : scanline) {
        if (v >= 0) {
            result.patch_idx.push_back(v);
        }
    }
    OPENVINO_ASSERT(result.patch_idx.size() == static_cast<size_t>(mosaic_h) * mosaic_w,
        "Molmo2 image preprocessing: unexpected patch count after de-overlap filtering");
    result.mosaic_h = mosaic_h;
    result.mosaic_w = mosaic_w;
    return result;
}

// Port of Python's `build_resized_image`: the low-resolution global thumbnail.
struct GlobalThumbnail {
    Molmo2Crop image;
    std::vector<int32_t> patch_idx;
    int patch_h = 0;
    int patch_w = 0;
};

GlobalThumbnail build_resized_image(const clip_image_u8& image, int target_size, int patch_size, const std::array<float, 3>& mean, const std::array<float, 3>& std_dev) {
    GlobalThumbnail result;
    result.image = resize_and_normalize(image, target_size, target_size, mean, std_dev);
    result.patch_h = target_size / patch_size;
    result.patch_w = target_size / patch_size;
    result.patch_idx.resize(static_cast<size_t>(result.patch_h) * result.patch_w);
    std::iota(result.patch_idx.begin(), result.patch_idx.end(), 0);
    return result;
}

// Port of Python's `arange_for_pooling`: pads a [h, w] patch-id grid with -1 to align both
// dimensions to multiples of (pool_h, pool_w), then regroups into [out_h, out_w, pool_h*pool_w]
// pooling windows (row-major within each window).
struct PooledIdx {
    std::vector<int32_t> data;
    int out_h = 0;
    int out_w = 0;
};

PooledIdx arange_for_pooling(const std::vector<int32_t>& idx, int h, int w, int pool_h, int pool_w) {
    const int h_pad = pool_h * ((h + pool_h - 1) / pool_h) - h;
    const int w_pad = pool_w * ((w + pool_w - 1) / pool_w) - w;
    const int pad_top = h_pad / 2;
    const int pad_left = w_pad / 2;
    const int padded_h = h + h_pad;
    const int padded_w = w + w_pad;

    std::vector<int32_t> padded(static_cast<size_t>(padded_h) * padded_w, -1);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            padded[static_cast<size_t>(y + pad_top) * padded_w + (x + pad_left)] = idx[static_cast<size_t>(y) * w + x];
        }
    }

    PooledIdx result;
    result.out_h = padded_h / pool_h;
    result.out_w = padded_w / pool_w;
    result.data.resize(static_cast<size_t>(result.out_h) * result.out_w * pool_h * pool_w);
    for (int gy = 0; gy < result.out_h; gy++) {
        for (int gx = 0; gx < result.out_w; gx++) {
            const size_t out_base = (static_cast<size_t>(gy) * result.out_w + gx) * (pool_h * pool_w);
            for (int dh = 0; dh < pool_h; dh++) {
                for (int dw = 0; dw < pool_w; dw++) {
                    const int src_y = gy * pool_h + dh;
                    const int src_x = gx * pool_w + dw;
                    result.data[out_base + dh * pool_w + dw] = padded[static_cast<size_t>(src_y) * padded_w + src_x];
                }
            }
        }
    }
    return result;
}

// Port of Python's `batch_pixels_to_patches` for a single crop: reshapes a HWC image into
// [n_patches, patch_size*patch_size*3] (patches enumerated row-major, pixels within each patch
// enumerated row-major with channel fastest-varying).
std::vector<float> patchify(const Molmo2Crop& crop, int patch_size) {
    const int h_patches = crop.height / patch_size;
    const int w_patches = crop.width / patch_size;
    std::vector<float> out(static_cast<size_t>(h_patches) * w_patches * patch_size * patch_size * 3);
    size_t out_idx = 0;
    for (int ph = 0; ph < h_patches; ph++) {
        for (int pw = 0; pw < w_patches; pw++) {
            for (int dy = 0; dy < patch_size; dy++) {
                const int y = ph * patch_size + dy;
                for (int dx = 0; dx < patch_size; dx++) {
                    const int x = pw * patch_size + dx;
                    for (int c = 0; c < 3; c++) {
                        out[out_idx++] = crop.hwc[(static_cast<size_t>(y) * crop.width + x) * 3 + c];
                    }
                }
            }
        }
    }
    return out;
}

// Final assembled inputs for the vision embeddings OpenVINO model, plus the pooled grid
// dimensions needed to build the prompt's image token expansion.
struct Molmo2ImageFeatures {
    ov::Tensor pixel_values;
    ov::Tensor image_token_pooling;
    int low_res_h = 0, low_res_w = 0;
    int high_res_h = 0, high_res_w = 0;
};

// Port of Python's `image_to_patches_and_grids` (single image; GenAI's VisionEncoder API
// processes one image per `encode()` call).
Molmo2ImageFeatures image_to_patches_and_grids(
    const clip_image_u8& image,
    int max_crops,
    int left_margin,
    int right_margin,
    int crop_size,
    int patch_size,
    int pool_h,
    int pool_w,
    const std::array<float, 3>& mean,
    const std::array<float, 3>& std_dev
) {
    OverlappingCrops overlapping = build_overlapping_crops(image, max_crops, left_margin, right_margin, crop_size, patch_size, mean, std_dev);
    PooledIdx high_res_pooling = arange_for_pooling(overlapping.patch_idx, overlapping.mosaic_h, overlapping.mosaic_w, pool_h, pool_w);

    GlobalThumbnail thumbnail = build_resized_image(image, crop_size, patch_size, mean, std_dev);
    PooledIdx low_res_pooling = arange_for_pooling(thumbnail.patch_idx, thumbnail.patch_h, thumbnail.patch_w, pool_h, pool_w);

    const int crop_patches = crop_size / patch_size;
    const int global_patch_count = crop_patches * crop_patches;  // thumbnail occupies patch ids [0, global_patch_count)

    // The global thumbnail is prepended as "crop 0", so high-resolution patch ids (which were
    // computed assuming they start at 0) must be shifted past the thumbnail's own patches.
    for (int32_t& v : high_res_pooling.data) {
        if (v >= 0) {
            v += global_patch_count;
        }
    }

    const int n_crops_total = 1 + static_cast<int>(overlapping.crops.size());
    const int patch_dim = patch_size * patch_size * 3;
    ov::Tensor pixel_values(ov::element::f32, {static_cast<size_t>(n_crops_total), static_cast<size_t>(global_patch_count), static_cast<size_t>(patch_dim)});
    float* pixel_data = pixel_values.data<float>();

    const std::vector<float> thumb_patches = patchify(thumbnail.image, patch_size);
    OPENVINO_ASSERT(thumb_patches.size() == static_cast<size_t>(global_patch_count) * patch_dim);
    std::copy(thumb_patches.begin(), thumb_patches.end(), pixel_data);

    for (size_t k = 0; k < overlapping.crops.size(); k++) {
        const std::vector<float> crop_patches_data = patchify(overlapping.crops[k], patch_size);
        std::copy(crop_patches_data.begin(), crop_patches_data.end(),
            pixel_data + (k + 1) * static_cast<size_t>(global_patch_count) * patch_dim);
    }

    const int pool_dim = pool_h * pool_w;
    const size_t total_pooled_tokens = static_cast<size_t>(low_res_pooling.out_h) * low_res_pooling.out_w
        + static_cast<size_t>(high_res_pooling.out_h) * high_res_pooling.out_w;
    ov::Tensor image_token_pooling(ov::element::i64, {1, total_pooled_tokens, static_cast<size_t>(pool_dim)});
    int64_t* pooling_data = image_token_pooling.data<int64_t>();
    size_t pos = 0;
    for (int32_t v : low_res_pooling.data) {
        pooling_data[pos++] = v;
    }
    for (int32_t v : high_res_pooling.data) {
        pooling_data[pos++] = v;
    }

    Molmo2ImageFeatures result;
    result.pixel_values = std::move(pixel_values);
    result.image_token_pooling = std::move(image_token_pooling);
    result.low_res_h = low_res_pooling.out_h;
    result.low_res_w = low_res_pooling.out_w;
    result.high_res_h = high_res_pooling.out_h;
    result.high_res_w = high_res_pooling.out_w;
    return result;
}

// Molmo2's structural prompt tokens (matches IMAGE_TOKENS subset relevant to static images in
// `processing_molmo2.py`; video-only frame tokens are intentionally excluded since video input
// is not supported here, matching sibling VLM implementations in this codebase).
constexpr const char* IM_PATCH_TOKEN = "<im_patch>";
constexpr const char* IM_COL_TOKEN = "<im_col>";
constexpr const char* IM_START_TOKEN = "<im_start>";
constexpr const char* IM_END_TOKEN = "<im_end>";
constexpr const char* LOW_RES_IM_START_TOKEN = "<low_res_im_start>";

}  // namespace

EncodedImage VisionEncoderMolmo2::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();

    ProcessorConfig config = ProcessorConfig::from_any_map(config_map, m_processor_config);
    OPENVINO_ASSERT(config.size_height == config.size_width, "Molmo2 requires a square base image input size");

    clip_image_u8 input_image = tensor_to_clip_image_u8(image);

    Molmo2ImageFeatures features = image_to_patches_and_grids(
        input_image,
        static_cast<int>(config.max_crops),
        static_cast<int>(config.overlap_margins[0]),
        static_cast<int>(config.overlap_margins[1]),
        static_cast<int>(config.size_height),
        static_cast<int>(config.patch_size),
        static_cast<int>(config.pooling_size[0]),
        static_cast<int>(config.pooling_size[1]),
        config.image_mean,
        config.image_std);

    encoder.set_tensor("pixel_values", features.pixel_values);
    encoder.set_tensor("image_token_pooling", features.image_token_pooling);
    encoder.infer();

    const ov::Tensor& infer_output = encoder.get_output_tensor();
    ov::Tensor image_features(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(image_features.data(), infer_output.data(), infer_output.get_byte_size());

    EncodedImage encoded;
    encoded.resized_source = std::move(image_features);
    encoded.patches_grid = {features.high_res_h, features.high_res_w};
    encoded.low_res_patches_grid = {features.low_res_h, features.low_res_w};
    encoded.num_image_tokens = static_cast<size_t>(features.high_res_h) * features.high_res_w
        + static_cast<size_t>(features.low_res_h) * features.low_res_w;
    return encoded;
}

InputsEmbedderMolmo2::InputsEmbedderMolmo2(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, model_dir, tokenizer, device, device_config) {}

InputsEmbedderMolmo2::InputsEmbedderMolmo2(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {}

bool InputsEmbedderMolmo2::has_token_type_ids() const {
    // The exported language model (optimum-intel Molmo2 export, upstream PR #1812) does not
    // currently trace a `token_type_ids` input, so bidirectional image-token attention is not
    // available yet -- report false to match what is actually exported and avoid the LM infer
    // request failing to find that port. See compute_merged_embeds_and_token_type_ids() for
    // details; this is a known, documented limitation to revisit once the exporter adds the
    // corresponding input.
    return false;
}

std::vector<ov::genai::EncodedImage> InputsEmbedderMolmo2::encode_images(const std::vector<ov::Tensor>& images) {
    std::vector<EncodedImage> embeds;
    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    embeds.reserve(single_images.size());
    for (const ov::Tensor& image : single_images) {
        embeds.emplace_back(m_vision_encoder->encode(image));
    }
    return embeds;
}

NormalizedPrompt InputsEmbedderMolmo2::normalize_prompt(const std::string& prompt, size_t base_id, const std::vector<EncodedImage>& images) const {
    const std::string& image_tag = m_vlm_config.image_token;  // "<|image|>"

    auto [unified_prompt, images_sequence] = normalize(prompt, image_tag, image_tag, base_id, images.size());

    size_t search_offset = 0;
    for (size_t new_image_id : images_sequence) {
        const EncodedImage& encoded = images.at(new_image_id - base_id);
        const int low_res_h = encoded.low_res_patches_grid.first;
        const int low_res_w = encoded.low_res_patches_grid.second;
        const int high_res_h = encoded.patches_grid.first;
        const int high_res_w = encoded.patches_grid.second;

        // Low-resolution (global thumbnail) section always precedes the high-resolution
        // crop-mosaic section (matches Python's `get_image_tokens`).
        std::string expanded_tag = LOW_RES_IM_START_TOKEN;
        for (int row = 0; row < low_res_h; row++) {
            for (int col = 0; col < low_res_w; col++) {
                expanded_tag += IM_PATCH_TOKEN;
            }
            expanded_tag += IM_COL_TOKEN;
        }
        expanded_tag += IM_END_TOKEN;

        expanded_tag += IM_START_TOKEN;
        for (int row = 0; row < high_res_h; row++) {
            for (int col = 0; col < high_res_w; col++) {
                expanded_tag += IM_PATCH_TOKEN;
            }
            expanded_tag += IM_COL_TOKEN;
        }
        expanded_tag += IM_END_TOKEN;

        size_t pos = unified_prompt.find(image_tag, search_offset);
        OPENVINO_ASSERT(pos != std::string::npos, "Failed to find image token in prompt during normalization");
        unified_prompt.replace(pos, image_tag.length(), expanded_tag);
        search_offset = pos + expanded_tag.size();
    }
    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderMolmo2::get_inputs_embeds(const std::string& prompt, const std::vector<EncodedImage>& images, VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& images_sequence) {
    auto [inputs_embeds, token_type_ids] = compute_merged_embeds_and_token_type_ids(prompt, images, metrics, images_sequence);
    return inputs_embeds;
}

std::pair<ov::Tensor, ov::Tensor> InputsEmbedderMolmo2::get_inputs_embeds_with_token_type_ids(const std::string& unified_prompt, const std::vector<EncodedImage>& images, VLMPerfMetrics& metrics, bool recalculate_merged_embeddings, const std::vector<size_t>& images_sequence) {
    return compute_merged_embeds_and_token_type_ids(unified_prompt, images, metrics, images_sequence);
}

std::pair<ov::Tensor, ov::Tensor> InputsEmbedderMolmo2::compute_merged_embeds_and_token_type_ids(const std::string& unified_prompt, const std::vector<EncodedImage>& images, VLMPerfMetrics& metrics, const std::vector<size_t>& images_sequence) {
    ov::Tensor input_ids = get_encoded_input_ids(unified_prompt, metrics);

    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = get_text_embedding(req, input_ids, metrics);

    ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
    std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());

    const int64_t* input_ids_data = input_ids.data<const int64_t>();
    const size_t seq_len = input_ids.get_size();

    const int64_t image_patch_id = m_vlm_config.image_patch_id;
    const int64_t image_col_id = m_vlm_config.image_col_id;
    const int64_t image_start_token_id = m_vlm_config.image_start_token_id;
    const int64_t image_end_token_id = m_vlm_config.image_end_token_id;
    const int64_t low_res_image_start_token_id = m_vlm_config.low_res_image_start_token_id;

    ov::Tensor token_type_ids(ov::element::i64, input_ids.get_shape());
    int64_t* token_type_data = token_type_ids.data<int64_t>();
    for (size_t i = 0; i < seq_len; i++) {
        const int64_t id = input_ids_data[i];
        const bool is_image_token = id == image_patch_id || id == image_col_id || id == image_start_token_id ||
            id == image_end_token_id || id == low_res_image_start_token_id;
        token_type_data[i] = is_image_token ? 1 : 0;
    }

    if (images.empty()) {
        return {inputs_embeds, token_type_ids};
    }

    // Additive merge: unlike llava-style VLMs, Molmo2 adds the pooled vision feature onto the
    // `<im_patch>` placeholder token's own (learned) text embedding rather than replacing it.
    const size_t hidden_size = inputs_embeds.get_shape().at(2);
    float* embeds_data = inputs_embeds.data<float>();

    size_t global_pos = 0;
    for (size_t new_image_id : images_sequence) {
        const EncodedImage& encoded = images.at(new_image_id);
        const ov::Shape& feature_shape = encoded.resized_source.get_shape();
        // The vision embeddings model returns a rank-2 [num_visual_tokens, hidden_size] tensor
        // (already flattened over any batch dim by the exported model); index from the back to
        // stay robust if a future export variant adds a leading batch dimension.
        OPENVINO_ASSERT(feature_shape.size() >= 2, "Unexpected Molmo2 vision feature tensor rank");
        OPENVINO_ASSERT(feature_shape.at(feature_shape.size() - 1) == hidden_size,
            "Molmo2 vision feature hidden size does not match text embedding hidden size");
        const float* feature_data = encoded.resized_source.data<const float>();
        const size_t num_features = feature_shape.at(feature_shape.size() - 2);

        size_t feature_row = 0;
        for (; global_pos < seq_len && feature_row < num_features; global_pos++) {
            if (input_ids_data[global_pos] == image_patch_id) {
                float* dst = embeds_data + global_pos * hidden_size;
                const float* src = feature_data + feature_row * hidden_size;
                for (size_t d = 0; d < hidden_size; d++) {
                    dst[d] += src[d];
                }
                feature_row++;
            }
        }
        OPENVINO_ASSERT(feature_row == num_features,
            "Molmo2: number of '<im_patch>' tokens in the prompt does not match the number of vision features "
            "produced for the corresponding image");
    }

    return {inputs_embeds, token_type_ids};
}

} // namespace ov::genai
