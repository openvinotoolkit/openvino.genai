// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/minicpmv4_7/classes.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <optional>

#include "utils.hpp"
#include "visual_language/clip.hpp"

namespace ov::genai {

namespace {

const std::string NATIVE_TAG = "<image>./</image>";
const std::string NATIVE_VIDEO_TAG = "<video>./</video>";

// Slices layout (how original image is adaptively sliced). Holds the number of slices in rows and columns.
struct SlicesGrid {
    size_t rows = 0;
    size_t cols = 0;
};

size_t calc_vision_downsample_factor(size_t window_kernel_size, size_t merge_kernel_size) {
    return window_kernel_size * merge_kernel_size;
}

int ensure_divide(double length, int divisor) {
    return std::max(static_cast<int>(std::round(length / divisor)) * divisor, divisor);
}

// Resizes so the area fits scale_resolution^2 while preserving aspect ratio, snapping each side to a
// multiple of patch_size * 4 (two 2x2 spatial merges).
ImageSize find_best_resize(double height, double width, int scale_resolution, int patch_size, bool allow_upscale) {
    if (height * width > static_cast<double>(scale_resolution) * scale_resolution || allow_upscale) {
        const double aspect_ratio = width / height;
        height = scale_resolution / std::sqrt(aspect_ratio);
        width = height * aspect_ratio;
    }
    const int factor = patch_size * 4;
    return {static_cast<size_t>(ensure_divide(height, factor)), static_cast<size_t>(ensure_divide(width, factor))};
}

// Chooses the slice grid, or nullopt when the image is small enough to skip slicing.
std::optional<SlicesGrid> get_sliced_grid(int height, int width, int max_slice_nums, int scale_resolution) {
    const double log_ratio = std::log(static_cast<double>(width) / height);
    const double ratio =
        static_cast<double>(width) * height / (static_cast<double>(scale_resolution) * scale_resolution);
    const int multiple = std::min(static_cast<int>(std::ceil(ratio)), max_slice_nums);
    if (multiple <= 1) {
        return std::nullopt;
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

// Returns the size the source image is resized to before being tiled into the slice grid.
ImageSize get_refine_size(int height, int width, SlicesGrid grid, int scale_resolution, int patch_size) {
    const int grid_rows = static_cast<int>(grid.rows);
    const int grid_cols = static_cast<int>(grid.cols);
    const int refine_height = ensure_divide(height, grid_rows);
    const int refine_width = ensure_divide(width, grid_cols);
    const bool allow_upscale = true;
    const ImageSize best = find_best_resize(
        static_cast<double>(refine_height) / grid_rows,
        static_cast<double>(refine_width) / grid_cols,
        scale_resolution,
        patch_size,
        allow_upscale
    );
    return {best.height * grid.rows, best.width * grid.cols};
}

// Splits a resized image into grid tiles of size (tile_height, tile_width), row-major.
std::vector<clip_image_u8> divide_to_patches(const clip_image_u8& image, int tile_height, int tile_width) {
    const int grid_y = image.ny / tile_height;
    const int grid_x = image.nx / tile_width;
    std::vector<clip_image_u8> tiles;
    tiles.reserve(static_cast<size_t>(grid_y) * grid_x);
    for (int row = 0; row < grid_y; ++row) {
        for (int col = 0; col < grid_x; ++col) {
            clip_image_u8 tile;
            tile.nx = tile_width;
            tile.ny = tile_height;
            tile.buf.resize(static_cast<size_t>(3) * tile_width * tile_height);
            for (int y = 0; y < tile_height; ++y) {
                for (int x = 0; x < tile_width; ++x) {
                    const int src_idx = 3 * ((row * tile_height + y) * image.nx + (col * tile_width + x));
                    const int dst_idx = 3 * (y * tile_width + x);
                    tile.buf[dst_idx] = image.buf[src_idx];
                    tile.buf[dst_idx + 1] = image.buf[src_idx + 1];
                    tile.buf[dst_idx + 2] = image.buf[src_idx + 2];
                }
            }
            tiles.push_back(std::move(tile));
        }
    }
    return tiles;
}

// Result of adaptive slicing: thumbnail crop first, then detail slices row-major.
struct SlicedImage {
    std::vector<clip_image_u8> crops;
    std::vector<ImageSize> crop_grids;  // patch grid (height, width) per crop
    ov::Shape slices_shape;             // {rows, cols}; empty when the image is not sliced
};

// Splits a source image into crops (low-res thumbnail + optional higher-res detail slices)
SlicedImage slice_image(const clip_image_u8& source, const ProcessorConfig& config) {
    const int patch_size = static_cast<int>(config.patch_size);
    const size_t patch_side = config.patch_size;
    const int scale_resolution = static_cast<int>(config.scale_resolution);
    const int max_slice_nums = static_cast<int>(config.max_slice_nums);
    const int height = source.ny;
    const int width = source.nx;

    const std::optional<SlicesGrid> best_grid = get_sliced_grid(height, width, max_slice_nums, scale_resolution);

    
    const bool allow_upscale = !best_grid.has_value();
    const ImageSize thumbnail_size = find_best_resize(height, width, scale_resolution, patch_size, allow_upscale);
    clip_image_u8 thumbnail;
    bicubic_resize(source, thumbnail, static_cast<int>(thumbnail_size.width), static_cast<int>(thumbnail_size.height));
    
    SlicedImage result;
    result.crops.push_back(std::move(thumbnail));
    result.crop_grids.push_back({thumbnail_size.height / patch_side, thumbnail_size.width / patch_side});

    if (best_grid.has_value()) {
        const ImageSize refine_size = get_refine_size(height, width, *best_grid, scale_resolution, patch_size);
        clip_image_u8 refine_img;
        bicubic_resize(source, refine_img, static_cast<int>(refine_size.width), static_cast<int>(refine_size.height));
        const int tile_h = static_cast<int>(refine_size.height / best_grid->rows);
        const int tile_w = static_cast<int>(refine_size.width / best_grid->cols);
        for (auto& tile : divide_to_patches(refine_img, tile_h, tile_w)) {
            result.crops.push_back(std::move(tile));
            result.crop_grids.push_back(
                {static_cast<size_t>(tile_h) / patch_side, static_cast<size_t>(tile_w) / patch_side}
            );
        }
        result.slices_shape = ov::Shape{best_grid->rows, best_grid->cols};
    }
    return result;
}

// Packs a (grid_h x grid_w patches) crop into NaViT format [1, 3, patch_size, grid_h*grid_w*patch_size],
// applying rescale (1/255) and per-channel normalization. Mirrors MiniCPMV4_6ImageProcessor.reshape_by_patch.
ov::Tensor pack_crop(
    const clip_image_u8& crop,
    size_t grid_h,
    size_t grid_w,
    const std::array<float, 3>& norm_mean,
    const std::array<float, 3>& norm_std,
    size_t patch_size
) {
    const size_t width_px = grid_w * patch_size;
    const size_t row_len = grid_h * grid_w * patch_size;
    ov::Tensor pixel_values{ov::element::f32, {1, 3, patch_size, row_len}};
    float* data = pixel_values.data<float>();
    for (size_t c = 0; c < 3; ++c) {
        // normalized = value / 255 / std - mean / std
        const float scale = 1.0f / (255.0f * norm_std[c]);
        const float shift = norm_mean[c] / norm_std[c];
        for (size_t kh = 0; kh < patch_size; ++kh) {
            for (size_t pr = 0; pr < grid_h; ++pr) {
                const size_t row = pr * patch_size + kh;
                for (size_t pc = 0; pc < grid_w; ++pc) {
                    const size_t patch = pr * grid_w + pc;
                    const size_t src_base = (row * width_px + pc * patch_size) * 3 + c;
                    const size_t dst_base = (c * patch_size + kh) * row_len + patch * patch_size;
                    for (size_t kw = 0; kw < patch_size; ++kw) {
                        data[dst_base + kw] = static_cast<float>(crop.buf[src_base + kw * 3]) * scale - shift;
                    }
                }
            }
        }
    }
    return pixel_values;
}

// Nearest-neighbor position ids into a num_patches_per_side^2 grid, row-major over the crop patches.
// Distinct from the language-model M-RoPE positions built in create_position_ids.
ov::Tensor build_vision_position_ids(size_t grid_h, size_t grid_w, size_t num_patches_per_side) {
    std::vector<double> boundaries(num_patches_per_side - 1);
    for (size_t k = 0; k < boundaries.size(); ++k) {
        boundaries[k] = static_cast<double>(k + 1) / num_patches_per_side;
    }
    auto to_bucket = [&boundaries](double value) -> int64_t {
        return std::upper_bound(boundaries.begin(), boundaries.end(), value) - boundaries.begin();
    };
    std::vector<int64_t> bucket_h(grid_h);
    std::vector<int64_t> bucket_w(grid_w);
    for (size_t i = 0; i < grid_h; ++i) {
        bucket_h[i] = to_bucket(static_cast<double>(i) / grid_h);
    }
    for (size_t j = 0; j < grid_w; ++j) {
        bucket_w[j] = to_bucket(static_cast<double>(j) / grid_w);
    }
    ov::Tensor position_ids{ov::element::i64, {grid_h * grid_w}};
    int64_t* data = position_ids.data<int64_t>();
    for (size_t i = 0; i < grid_h; ++i) {
        for (size_t j = 0; j < grid_w; ++j) {
            data[i * grid_w + j] = bucket_h[i] * static_cast<int64_t>(num_patches_per_side) + bucket_w[j];
        }
    }
    return position_ids;
}

// Window-attention reorder indices (length grid_h*grid_w), grouping patches into window x window blocks.
ov::Tensor build_window_index(size_t grid_h, size_t grid_w, size_t window_kernel_size) {
    const size_t pad_h = window_kernel_size - grid_h % window_kernel_size;
    const size_t pad_w = window_kernel_size - grid_w % window_kernel_size;
    const size_t padded_h = grid_h + pad_h;
    const size_t padded_w = grid_w + pad_w;
    const size_t num_windows_h = padded_h / window_kernel_size;
    const size_t num_windows_w = padded_w / window_kernel_size;

    constexpr int64_t PAD = -100;
    std::vector<int64_t> padded(padded_h * padded_w, PAD);
    for (size_t i = 0; i < grid_h; ++i) {
        for (size_t j = 0; j < grid_w; ++j) {
            padded[i * padded_w + j] = static_cast<int64_t>(i * grid_w + j);
        }
    }

    std::vector<int64_t> window_index;
    window_index.reserve(grid_h * grid_w);
    for (size_t wh = 0; wh < num_windows_h; ++wh) {
        for (size_t ww = 0; ww < num_windows_w; ++ww) {
            for (size_t i = 0; i < window_kernel_size; ++i) {
                for (size_t j = 0; j < window_kernel_size; ++j) {
                    const int64_t value =
                        padded[(wh * window_kernel_size + i) * padded_w + (ww * window_kernel_size + j)];
                    if (value != PAD) {
                        window_index.push_back(value);
                    }
                }
            }
        }
    }
    ov::Tensor result{ov::element::i64, {window_index.size()}};
    std::copy(window_index.begin(), window_index.end(), result.data<int64_t>());
    return result;
}

// Reorders the post-window-merge tokens (grid / window_kernel_size per dimension) into
// merge_kernel_size x merge_kernel_size blocks for the final spatial merger.
ov::Tensor build_merge_index(size_t grid_h, size_t grid_w, size_t window_kernel_size, size_t merge_kernel_size) {
    const size_t post_window_h = grid_h / window_kernel_size;
    const size_t post_window_w = grid_w / window_kernel_size;
    const size_t blocks_h = post_window_h / merge_kernel_size;
    const size_t blocks_w = post_window_w / merge_kernel_size;
    ov::Tensor result{ov::element::i64, {post_window_h * post_window_w}};
    int64_t* data = result.data<int64_t>();
    size_t out = 0;
    for (size_t a = 0; a < blocks_h; ++a) {
        for (size_t c = 0; c < blocks_w; ++c) {
            for (size_t b = 0; b < merge_kernel_size; ++b) {
                for (size_t d = 0; d < merge_kernel_size; ++d) {
                    data[out++] =
                        static_cast<int64_t>((a * merge_kernel_size + b) * post_window_w + (c * merge_kernel_size + d));
                }
            }
        }
    }
    return result;
}

// Number of visual tokens a crop with the given patch grid yields after the downsample.
size_t calc_crop_token_count(const ImageSize& grid, size_t downsample_factor) {
    return (grid.height / downsample_factor) * (grid.width / downsample_factor);
}

// Flattens per-image crop grids into prompt order (each image's thumbnail then its slices).
std::vector<ImageSize>
flatten_crop_grids(const std::vector<EncodedImage>& images, const std::vector<size_t>& images_sequence) {
    std::vector<ImageSize> grids;
    for (size_t image_id : images_sequence) {
        const auto& crop_grids = images.at(image_id).crop_grids;
        grids.insert(grids.end(), crop_grids.begin(), crop_grids.end());
    }
    return grids;
}

// Flattens per-video crops into prompt order: every frame contributes the whole crop layout
// (thumbnail then slices), so a video of F frames with C crops each yields F*C grids.
std::vector<ImageSize>
flatten_video_frame_grids(const std::vector<EncodedVideo>& videos, const std::vector<size_t>& videos_sequence) {
    std::vector<ImageSize> grids;
    for (size_t video_id : videos_sequence) {
        const EncodedVideo& video = videos.at(video_id);
        for (size_t frame = 0; frame < video.frame_num; ++frame) {
            grids.insert(grids.end(), video.crop_grids.begin(), video.crop_grids.end());
        }
    }
    return grids;
}

// Builds [3, 1, seq_len] position ids [T, H, W] and the rope delta (Canvas M-RoPE)
std::pair<ov::Tensor, int64_t> create_position_ids(
    const ov::Tensor& input_ids,
    const std::vector<ImageSize>& image_crop_grids,
    const std::vector<ImageSize>& video_frame_grids,
    int64_t image_token_id,
    int64_t video_token_id,
    size_t downsample_factor
) {
    const size_t seq_len = input_ids.get_size();
    const int64_t* input_ids_data = input_ids.data<int64_t>();

    ov::Tensor position_ids{ov::element::i64, {3, 1, seq_len}};
    int64_t* channel[3] = {
        position_ids.data<int64_t>(),
        position_ids.data<int64_t>() + seq_len,
        position_ids.data<int64_t>() + 2 * seq_len};

    int64_t max_pos = 0;
    auto fill_text = [&](size_t from, size_t to, int64_t start) {
        for (size_t k = from; k < to; ++k) {
            const int64_t value = start + static_cast<int64_t>(k - from);
            channel[0][k] = value;
            channel[1][k] = value;
            channel[2][k] = value;
        }
        if (to > from) {
            max_pos = std::max(max_pos, start + static_cast<int64_t>(to - from - 1));
        }
    };

    int64_t current_pos = 0;
    size_t current_idx = 0;
    size_t image_crop_index = 0;
    size_t video_frame_index = 0;
    size_t i = 0;
    while (i < seq_len) {
        const int64_t token = input_ids_data[i];
        const bool is_image = token == image_token_id;
        if (!is_image && token != video_token_id) {
            ++i;
            continue;
        }
        const size_t run_start = i;
        while (i < seq_len && input_ids_data[i] == token) {
            ++i;
        }
        const size_t run_len = i - run_start;
        const auto [grid_h, grid_w] =
            is_image ? image_crop_grids.at(image_crop_index++) : video_frame_grids.at(video_frame_index++);
        const int64_t canvas_h = static_cast<int64_t>(grid_h / downsample_factor);
        const int64_t canvas_w = static_cast<int64_t>(grid_w / downsample_factor);
        const size_t frame_start = run_start - 1;

        if (frame_start > current_idx) {
            fill_text(current_idx, frame_start, current_pos);
            current_pos += static_cast<int64_t>(frame_start - current_idx);
        }

        const int64_t canvas_origin = current_pos;
        const int64_t halo = std::max<int64_t>(canvas_origin - 1, 0);
        channel[0][frame_start] = canvas_origin;
        channel[1][frame_start] = halo;
        channel[2][frame_start] = halo;

        for (size_t m = 0; m < run_len; ++m) {
            const size_t idx = run_start + m;
            channel[0][idx] = canvas_origin;
            channel[1][idx] = canvas_origin + static_cast<int64_t>(m / static_cast<size_t>(canvas_w));
            channel[2][idx] = canvas_origin + static_cast<int64_t>(m % static_cast<size_t>(canvas_w));
        }
        max_pos = std::max(max_pos, canvas_origin + std::max(canvas_h, canvas_w) - 1);

        current_pos = canvas_origin + std::max(canvas_h, canvas_w) + 1;
        current_idx = i;
    }

    if (current_idx < seq_len) {
        fill_text(current_idx, seq_len, current_pos);
    }

    const int64_t rope_delta = max_pos + 1 - static_cast<int64_t>(seq_len);
    return {position_ids, rope_delta};
}

}  // namespace

VisionEncoderMiniCPMv4_7::VisionEncoderMiniCPMv4_7(
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap properties
)
    : VisionEncoder(model_dir, device, properties) {}

VisionEncoderMiniCPMv4_7::VisionEncoderMiniCPMv4_7(
    const ModelsMap& models_map,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config
)
    : VisionEncoder(models_map, config_dir_path, device, device_config) {}

void VisionEncoderMiniCPMv4_7::update_vision_config(const VLMConfig& vlm_config) {
    for (ProcessorConfig* config :
         {static_cast<ProcessorConfig*>(&m_processor_config),
          static_cast<ProcessorConfig*>(&m_video_processor_config)}) {
        config->image_size = vlm_config.vision_config_image_size;
        config->window_kernel_size = vlm_config.vision_config_window_kernel_size;
        config->merge_kernel_size = vlm_config.merge_kernel_size;
    }
}

EncodedImage VisionEncoderMiniCPMv4_7::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    const ProcessorConfig config = ProcessorConfig::from_any_map(config_map, m_processor_config);

    const clip_image_u8 source = tensor_to_clip_image_u8(image);
    SlicedImage sliced = slice_image(source, config);
    ov::Tensor features = encode_crops(sliced.crops, sliced.crop_grids, config);

    EncodedImage encoded_image;
    encoded_image.num_image_tokens = features.get_shape().at(0);
    encoded_image.resized_source = std::move(features);
    encoded_image.resized_source_size = sliced.crop_grids.front();
    // slices_shape holds the {rows, cols} slice layout (empty when the image is not sliced),
    // used to place newlines between slice rows during prompt normalization.
    encoded_image.slices_shape = std::move(sliced.slices_shape);
    encoded_image.crop_grids = std::move(sliced.crop_grids);
    return encoded_image;
}

ov::Tensor VisionEncoderMiniCPMv4_7::encode_crops(
    const std::vector<clip_image_u8>& crops,
    const std::vector<ImageSize>& crop_grids,
    const ProcessorConfig& config
) {
    // Side length of the vision tower's learned 2D position-embedding grid.
    const size_t num_patches_per_side = config.image_size / config.patch_size;
    const size_t window_kernel_size = config.window_kernel_size;
    const size_t merge_kernel_size = config.merge_kernel_size;
    const size_t downsample_factor = calc_vision_downsample_factor(window_kernel_size, merge_kernel_size);

    size_t total_tokens = 0;
    for (const auto& grid : crop_grids) {
        total_tokens += calc_crop_token_count(grid, downsample_factor);
    }

    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();

    // Encode crops one at a time, copying each output into the concatenated feature tensor
    ov::Tensor features;
    size_t offset = 0;
    for (size_t i = 0; i < crops.size(); ++i) {
        const size_t grid_h = crop_grids[i].height;
        const size_t grid_w = crop_grids[i].width;

        encoder.set_tensor(
            "pixel_values",
            pack_crop(crops[i], grid_h, grid_w, config.image_mean, config.image_std, config.patch_size)
        );
        encoder.set_tensor("position_ids", build_vision_position_ids(grid_h, grid_w, num_patches_per_side));
        encoder.set_tensor("window_index", build_window_index(grid_h, grid_w, window_kernel_size));
        encoder.set_tensor("merge_index", build_merge_index(grid_h, grid_w, window_kernel_size, merge_kernel_size));
        encoder.infer();

        const ov::Tensor& output = encoder.get_output_tensor();
        if (i == 0) {
            features = ov::Tensor{output.get_element_type(), {total_tokens, output.get_shape().at(1)}};
        }
        std::memcpy(static_cast<uint8_t*>(features.data()) + offset, output.data(), output.get_byte_size());
        offset += output.get_byte_size();
    }
    return features;
}

EncodedVideo VisionEncoderMiniCPMv4_7::encode_frames(const std::vector<ov::Tensor>& frames) {
    OPENVINO_ASSERT(!frames.empty(), "Cannot encode an empty list of video frames.");

    // Each frame is sliced as an image (thumbnail + optional detail slices)
    std::vector<clip_image_u8> all_crops;
    std::vector<ImageSize> all_crop_grids;
    std::vector<ImageSize> frame_crop_grids;
    ov::Shape frame_slices_shape;
    for (size_t i = 0; i < frames.size(); ++i) {
        SlicedImage sliced = slice_image(tensor_to_clip_image_u8(frames[i]), m_video_processor_config);
        if (i == 0) {
            frame_crop_grids = sliced.crop_grids;
            frame_slices_shape = sliced.slices_shape;
            all_crops.reserve(frames.size() * sliced.crops.size());
            all_crop_grids.reserve(frames.size() * sliced.crop_grids.size());
        }
        all_crops.insert(
            all_crops.end(),
            std::make_move_iterator(sliced.crops.begin()),
            std::make_move_iterator(sliced.crops.end())
        );
        all_crop_grids.insert(all_crop_grids.end(), sliced.crop_grids.begin(), sliced.crop_grids.end());
    }

    ov::Tensor features = encode_crops(all_crops, all_crop_grids, m_video_processor_config);

    EncodedVideo encoded_video;
    encoded_video.num_video_tokens = features.get_shape().at(0);
    encoded_video.frame_num = frames.size();
    encoded_video.resized_source_size = frame_crop_grids.front();
    encoded_video.crop_grids = std::move(frame_crop_grids);
    encoded_video.slices_shape = std::move(frame_slices_shape);
    // video_features holds every frame's crops concatenated as [total_tokens, hidden_size]
    encoded_video.video_features = std::move(features);
    return encoded_video;
}

InputsEmbedderMiniCPMv4_7::InputsEmbedderMiniCPMv4_7(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap device_config
)
    : IInputsEmbedder(vlm_config, model_dir, tokenizer, device, device_config) {
    static_cast<VisionEncoderMiniCPMv4_7&>(*m_vision_encoder).update_vision_config(m_vlm_config);
}

InputsEmbedderMiniCPMv4_7::InputsEmbedderMiniCPMv4_7(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config
)
    : IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {
    static_cast<VisionEncoderMiniCPMv4_7&>(*m_vision_encoder).update_vision_config(m_vlm_config);
}

void InputsEmbedderMiniCPMv4_7::encode_vision_token_ids() {
    std::call_once(m_vision_token_ids_once_flag, [this]() {
        const auto encoded_vision_tokens = m_tokenizer.encode(
            m_vlm_config.image_pad_token + m_vlm_config.video_pad_token,
            ov::genai::add_special_tokens(false)
        ).input_ids;
        OPENVINO_ASSERT(encoded_vision_tokens.get_size() == 2, "Encoded vision tokens must contain two tokens.");
        m_image_token_id = encoded_vision_tokens.data<int64_t>()[0];
        m_video_token_id = encoded_vision_tokens.data<int64_t>()[1];
    });
}

std::vector<ov::genai::EncodedVideo> InputsEmbedderMiniCPMv4_7::encode_videos(
    const std::vector<ov::Tensor>& videos,
    const std::vector<VideoMetadata>& videos_metadata
) {
    OPENVINO_ASSERT(
        videos.size() == videos_metadata.size() || videos_metadata.empty(),
        "Number of videos and videos metadata must match if metadata provided."
    );

    std::vector<EncodedVideo> encoded_videos;
    encoded_videos.reserve(videos.size());
    for (size_t i = 0; i < videos.size(); ++i) {
        VideoMetadata video_metadata = i < videos_metadata.size() ? videos_metadata[i] : VideoMetadata{};
        const ov::Tensor sampled_video = sample_video_if_needed(videos[i], video_metadata);
        std::vector<ov::Tensor> frames = to_single_image_tensors({sampled_video});
        EncodedVideo encoded_video = m_vision_encoder->encode_frames(frames);
        encoded_video.metadata = std::move(video_metadata);
        encoded_videos.emplace_back(std::move(encoded_video));
    }
    return encoded_videos;
}

NormalizedPrompt InputsEmbedderMiniCPMv4_7::normalize_prompt(
    const std::string& prompt,
    size_t base_image_id,
    size_t base_video_id,
    const std::vector<EncodedImage>& images,
    const std::vector<EncodedVideo>& videos
) const {
    const size_t downsample_factor =
        calc_vision_downsample_factor(m_vlm_config.vision_config_window_kernel_size, m_vlm_config.merge_kernel_size);

    auto repeat = [](const std::string& str, size_t count) {
        std::string result;
        result.reserve(str.size() * count);
        for (size_t i = 0; i < count; ++i) {
            result += str;
        }
        return result;
    };

    // Images - expands to an optional id tag, thumbnail tags and optional slices tags (per row with newline).
    auto [unified_prompt, image_sequence] =
        normalize(prompt, NATIVE_TAG, NATIVE_TAG + '\n', base_image_id, images.size(), VisionType::IMAGE);

    const std::string& image_pad_token = m_vlm_config.image_pad_token;
    for (size_t new_image_id : image_sequence) {
        const EncodedImage& encoded_image = images.at(new_image_id - base_image_id);
        const auto& crop_grids = encoded_image.crop_grids;

        std::string expanded_tag;
        if (m_vlm_config.use_image_id) {
            expanded_tag += m_vlm_config.im_id_start + std::to_string(new_image_id) + m_vlm_config.im_id_end;
        }

        expanded_tag += m_vlm_config.im_start +
                        repeat(image_pad_token, calc_crop_token_count(crop_grids.at(0), downsample_factor)) +
                        m_vlm_config.im_end;

        const ov::Shape& slices_shape = encoded_image.slices_shape;
        if (!slices_shape.empty()) {
            const size_t rows = slices_shape.at(0);
            const size_t cols = slices_shape.at(1);
            for (size_t row = 0; row < rows; ++row) {
                for (size_t col = 0; col < cols; ++col) {
                    const size_t slice_tokens =
                        calc_crop_token_count(crop_grids.at(1 + row * cols + col), downsample_factor);
                    expanded_tag +=
                        m_vlm_config.slice_start + repeat(image_pad_token, slice_tokens) + m_vlm_config.slice_end;
                }
                expanded_tag += '\n';
            }
            expanded_tag.pop_back();
        }

        unified_prompt.replace(unified_prompt.find(NATIVE_TAG), NATIVE_TAG.length(), expanded_tag);
    }

    // Videos - each frame is sliced like an image, per-frame tag (thumbnail + slices) is built once and repeated
    std::vector<size_t> videos_sequence;
    std::tie(unified_prompt, videos_sequence) = normalize(
        unified_prompt,
        NATIVE_VIDEO_TAG,
        NATIVE_VIDEO_TAG + '\n',
        base_video_id,
        videos.size(),
        VisionType::VIDEO
    );

    const std::string& video_pad_token = m_vlm_config.video_pad_token;
    for (size_t new_video_id : videos_sequence) {
        const EncodedVideo& encoded_video = videos.at(new_video_id - base_video_id);
        OPENVINO_ASSERT(encoded_video.frame_num > 0, "Video must contain at least one frame.");
        const auto& crop_grids = encoded_video.crop_grids;

        std::string frame_tag = m_vlm_config.im_start +
                                repeat(video_pad_token, calc_crop_token_count(crop_grids.at(0), downsample_factor)) +
                                m_vlm_config.im_end;

        const ov::Shape& slices_shape = encoded_video.slices_shape;
        if (!slices_shape.empty()) {
            const size_t rows = slices_shape.at(0);
            const size_t cols = slices_shape.at(1);
            for (size_t row = 0; row < rows; ++row) {
                for (size_t col = 0; col < cols; ++col) {
                    const size_t slice_tokens =
                        calc_crop_token_count(crop_grids.at(1 + row * cols + col), downsample_factor);
                    frame_tag +=
                        m_vlm_config.slice_start + repeat(video_pad_token, slice_tokens) + m_vlm_config.slice_end;
                }
                frame_tag += '\n';
            }
            frame_tag.pop_back();
        }

        std::string expanded_tag;
        expanded_tag.reserve(frame_tag.size() * encoded_video.frame_num);
        for (size_t frame = 0; frame < encoded_video.frame_num; ++frame) {
            expanded_tag += frame_tag;
        }

        unified_prompt.replace(unified_prompt.find(NATIVE_VIDEO_TAG), NATIVE_VIDEO_TAG.length(), expanded_tag);
    }

    return {std::move(unified_prompt), std::move(image_sequence), std::move(videos_sequence)};
}

ov::Tensor InputsEmbedderMiniCPMv4_7::get_inputs_embeds(
    const std::string& prompt,
    const std::vector<ov::genai::EncodedImage>& images,
    ov::genai::VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& image_sequence
) {
    return get_inputs_embeds(prompt, images, {}, metrics, recalculate_merged_embeddings, image_sequence, {}, {});
}

ov::Tensor InputsEmbedderMiniCPMv4_7::get_inputs_embeds(
    const std::string& prompt,
    const std::vector<ov::genai::EncodedImage>& images,
    const std::vector<ov::genai::EncodedVideo>& videos,
    ov::genai::VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& image_sequence,
    const std::vector<size_t>& videos_sequence,
    const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count
) {
    encode_vision_token_ids();

    ov::Tensor input_ids = get_encoded_input_ids(prompt, metrics);

    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = get_text_embedding(req, input_ids, metrics);

    const size_t downsample_factor =
        calc_vision_downsample_factor(m_vlm_config.vision_config_window_kernel_size, m_vlm_config.merge_kernel_size);
    const auto image_crop_grids = flatten_crop_grids(images, image_sequence);
    const auto video_frame_grids = flatten_video_frame_grids(videos, videos_sequence);
    std::tie(m_position_ids, m_rope_delta) = create_position_ids(
        input_ids,
        image_crop_grids,
        video_frame_grids,
        m_image_token_id,
        m_video_token_id,
        downsample_factor
    );

    // Non-owning view per crop for image tokens
    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(image_crop_grids.size());
    for (size_t image_id : image_sequence) {
        const ov::Tensor& source = images.at(image_id).resized_source;
        const size_t hidden_size = source.get_shape().at(1);
        float* crop_data = const_cast<float*>(source.data<float>());
        size_t offset = 0;
        for (const auto& grid : images.at(image_id).crop_grids) {
            const size_t tokens = calc_crop_token_count(grid, downsample_factor);
            image_embeds.emplace_back(
                source.get_element_type(),
                ov::Shape{1, tokens, hidden_size},
                crop_data + offset * hidden_size
            );
            offset += tokens;
        }
    }

    // Non-owning view per frame-crop for video tokens
    std::vector<ov::Tensor> video_embeds;
    video_embeds.reserve(video_frame_grids.size());
    for (size_t video_id : videos_sequence) {
        const EncodedVideo& video = videos.at(video_id);
        const ov::Tensor& source = video.video_features;
        const size_t hidden_size = source.get_shape().at(1);
        float* crop_data = const_cast<float*>(source.data<float>());
        size_t offset = 0;
        for (size_t frame = 0; frame < video.frame_num; ++frame) {
            for (const auto& grid : video.crop_grids) {
                const size_t tokens = calc_crop_token_count(grid, downsample_factor);
                video_embeds.emplace_back(
                    source.get_element_type(),
                    ov::Shape{1, tokens, hidden_size},
                    crop_data + offset * hidden_size
                );
                offset += tokens;
            }
        }
    }

    ov::Tensor inputs_embeds = text_embeds;
    if (!image_embeds.empty()) {
        inputs_embeds =
            utils::merge_text_and_image_embeddings_llava(input_ids, inputs_embeds, image_embeds, m_image_token_id);
    }
    if (!video_embeds.empty()) {
        inputs_embeds =
            utils::merge_text_and_image_embeddings_llava(input_ids, inputs_embeds, video_embeds, m_video_token_id);
    }
    return inputs_embeds;
}

std::pair<ov::Tensor, std::optional<int64_t>>
InputsEmbedderMiniCPMv4_7::get_position_ids(const size_t inputs_embeds_size, const size_t history_size) {
    if (history_size != 0) {
        return get_generation_phase_position_ids(inputs_embeds_size, history_size, m_rope_delta);
    }
    return {m_position_ids, m_rope_delta};
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderMiniCPMv4_7::get_generation_phase_position_ids(
    const size_t inputs_embeds_size,
    const size_t history_size,
    int64_t rope_delta
) {
    OPENVINO_ASSERT(
        history_size != 0,
        "get_generation_phase_position_ids() should only be called during the generation phase."
    );
    ov::Tensor position_ids{ov::element::i64, {3, 1, inputs_embeds_size}};
    const int64_t new_pos_id = static_cast<int64_t>(history_size) + rope_delta;
    for (size_t dim = 0; dim < 3; ++dim) {
        int64_t* pos_data = position_ids.data<int64_t>() + dim * inputs_embeds_size;
        std::iota(pos_data, pos_data + inputs_embeds_size, new_pos_id);
    }
    return {position_ids, rope_delta};
}

void InputsEmbedderMiniCPMv4_7::start_chat(const std::string& system_message) {
    IInputsEmbedder::start_chat(system_message);
    m_position_ids = ov::Tensor();
    m_rope_delta = 0;
}

void InputsEmbedderMiniCPMv4_7::finish_chat() {
    IInputsEmbedder::finish_chat();
    m_position_ids = ov::Tensor();
    m_rope_delta = 0;
}

}  // namespace ov::genai
