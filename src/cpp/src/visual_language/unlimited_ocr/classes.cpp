// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/unlimited_ocr/classes.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

#include "openvino/core/type/float16.hpp"
#include "utils.hpp"
#include "visual_language/clip.hpp"

namespace ov::genai {

namespace {

// Constants mirror baidu/Unlimited-OCR remote code (modeling_unlimitedocr.py):
//   patch_size = 16, downsample_ratio = 4, base_size = 1024, image_size = 640.
constexpr int BASE_SIZE = 1024;   // global view resolution
constexpr int TILE_SIZE = 640;    // crop tile resolution
constexpr int DYN_MIN_NUM = 2;    // dynamic_preprocess(min_num=2, ...)
constexpr int DYN_MAX_NUM = 32;   // dynamic_preprocess(..., max_num=32)
// BasicImageTransform(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)).
constexpr std::array<float, 3> IMAGE_MEAN{0.5f, 0.5f, 0.5f};
constexpr std::array<float, 3> IMAGE_STD{0.5f, 0.5f, 0.5f};
// ImageOps.pad fill color = int(0.5 * 255) = 127 per channel.
constexpr std::array<uint8_t, 3> PAD_VALUES{127, 127, 127};

std::pair<int, int> find_closest_aspect_ratio(float aspect_ratio,
                                              const std::vector<std::pair<int, int>>& target_ratios,
                                              int width,
                                              int height,
                                              int image_size) {
    float best_ratio_diff = std::numeric_limits<float>::infinity();
    std::pair<int, int> best_ratio = {1, 1};
    const int area = width * height;

    for (const auto& ratio : target_ratios) {
        const float target_aspect_ratio = static_cast<float>(ratio.first) / static_cast<float>(ratio.second);
        const float ratio_diff = std::abs(aspect_ratio - target_aspect_ratio);
        if (ratio_diff < best_ratio_diff) {
            best_ratio_diff = ratio_diff;
            best_ratio = ratio;
        } else if (ratio_diff == best_ratio_diff) {
            if (area > 0.5f * image_size * image_size * ratio.first * ratio.second) {
                best_ratio = ratio;
            }
        }
    }
    return best_ratio;
}

// Reproduces dynamic_preprocess(image, min_num=2, max_num=32, image_size=640).
// Returns the crop tiles (row-major) and the {width_crop_num, height_crop_num} ratio.
std::pair<std::vector<clip_image_u8>, std::pair<int, int>> dynamic_preprocess(const clip_image_u8& image,
                                                                              int image_size,
                                                                              int min_num,
                                                                              int max_num) {
    const int orig_width = image.nx;
    const int orig_height = image.ny;
    const float aspect_ratio = static_cast<float>(orig_width) / static_cast<float>(orig_height);

    std::vector<std::pair<int, int>> target_ratios;
    for (int n = min_num; n <= max_num; ++n) {
        for (int i = 1; i <= n; ++i) {
            for (int j = 1; j <= n; ++j) {
                const int prod = i * j;
                if (prod <= max_num && prod >= min_num) {
                    target_ratios.emplace_back(i, j);
                }
            }
        }
    }
    std::sort(target_ratios.begin(), target_ratios.end());
    target_ratios.erase(std::unique(target_ratios.begin(), target_ratios.end()), target_ratios.end());
    std::sort(target_ratios.begin(), target_ratios.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.first * lhs.second < rhs.first * rhs.second;
    });

    const auto target_aspect_ratio =
        find_closest_aspect_ratio(aspect_ratio, target_ratios, orig_width, orig_height, image_size);

    const int target_width = image_size * target_aspect_ratio.first;
    const int target_height = image_size * target_aspect_ratio.second;
    const int blocks = target_aspect_ratio.first * target_aspect_ratio.second;

    clip_image_u8 resized_img;
    bicubic_resize(image, resized_img, target_width, target_height);

    std::vector<clip_image_u8> processed_images;
    processed_images.reserve(blocks);
    const int cols = target_width / image_size;
    for (int i = 0; i < blocks; ++i) {
        const int x = (i % cols) * image_size;
        const int y = (i / cols) * image_size;

        clip_image_u8 tile_img;
        tile_img.nx = image_size;
        tile_img.ny = image_size;
        tile_img.buf.resize(3 * image_size * image_size);
        for (int dy = 0; dy < image_size; ++dy) {
            for (int dx = 0; dx < image_size; ++dx) {
                for (int c = 0; c < 3; ++c) {
                    const int src_idx = ((y + dy) * target_width + (x + dx)) * 3 + c;
                    const int dst_idx = (dy * image_size + dx) * 3 + c;
                    tile_img.buf[dst_idx] = resized_img.buf[src_idx];
                }
            }
        }
        processed_images.emplace_back(std::move(tile_img));
    }
    return {std::move(processed_images), target_aspect_ratio};
}

// Reproduces PIL ImageOps.pad(img, (target, target), color): scale keeping aspect
// ratio (contain), then center-pad with the given color.
clip_image_u8 pad_to_square(const clip_image_u8& image, int target, const std::array<uint8_t, 3>& pad_values) {
    const float scale = std::min(static_cast<float>(target) / static_cast<float>(image.nx),
                                 static_cast<float>(target) / static_cast<float>(image.ny));
    const int new_width = std::max(1, static_cast<int>(std::lround(static_cast<float>(image.nx) * scale)));
    const int new_height = std::max(1, static_cast<int>(std::lround(static_cast<float>(image.ny) * scale)));

    clip_image_u8 resized;
    bicubic_resize(image, resized, new_width, new_height);

    clip_image_u8 padded;
    padded.nx = target;
    padded.ny = target;
    padded.buf.resize(3 * target * target);
    for (int i = 0; i < target * target; ++i) {
        padded.buf[3 * i + 0] = pad_values[0];
        padded.buf[3 * i + 1] = pad_values[1];
        padded.buf[3 * i + 2] = pad_values[2];
    }

    const int pad_x = (target - new_width) / 2;
    const int pad_y = (target - new_height) / 2;
    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            for (int c = 0; c < 3; ++c) {
                const int dst_index = 3 * ((y + pad_y) * target + (x + pad_x)) + c;
                const int src_index = 3 * (y * new_width + x) + c;
                padded.buf[dst_index] = resized.buf[src_index];
            }
        }
    }
    return padded;
}

// Normalizes a batch of equally sized RGB images into an (N, 3, H, W) f32 tensor:
// value = pixel / 255; out = (value - mean) / std.
ov::Tensor to_batch_tensor(const std::vector<clip_image_u8>& images,
                           const std::array<float, 3>& mean,
                           const std::array<float, 3>& std) {
    OPENVINO_ASSERT(!images.empty(), "Cannot create a batch tensor from an empty image list");
    const size_t channels = 3;
    const size_t height = static_cast<size_t>(images.front().ny);
    const size_t width = static_cast<size_t>(images.front().nx);

    ov::Tensor batch_tensor(ov::element::f32, {images.size(), channels, height, width});
    float* batch_data = batch_tensor.data<float>();

    for (size_t idx = 0; idx < images.size(); ++idx) {
        const clip_image_u8& img = images[idx];
        OPENVINO_ASSERT(static_cast<size_t>(img.nx) == width && static_cast<size_t>(img.ny) == height,
                        "Inconsistent tile tensor shape during Unlimited-OCR preprocessing");
        float* dst = batch_data + idx * channels * height * width;
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                for (size_t c = 0; c < channels; ++c) {
                    const uint8_t pixel = img.buf[3 * (y * width + x) + c];
                    const float value = static_cast<float>(pixel) / 255.0f;
                    dst[c * height * width + y * width + x] = (value - mean[c]) / std[c];
                }
            }
        }
    }
    return batch_tensor;
}

ov::Tensor infer_and_copy(CircularBufferQueue<ov::InferRequest>* queue, const ov::Tensor& pixel_values) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(queue);
    ov::InferRequest& infer_request = infer_request_guard.get();
    infer_request.set_tensor("pixel_values", pixel_values);
    infer_request.infer();
    const ov::Tensor& output = infer_request.get_output_tensor();
    ov::Tensor copy(output.get_element_type(), output.get_shape());
    output.copy_to(copy);
    return copy;
}

void copy_rows_to_f32(const ov::Tensor& src, size_t src_row, float* dst, size_t rows, size_t hidden) {
    const size_t values = rows * hidden;
    if (src.get_element_type() == ov::element::f32) {
        std::copy_n(src.data<const float>() + src_row * hidden, values, dst);
        return;
    }
    OPENVINO_ASSERT(src.get_element_type() == ov::element::f16,
                    "Unlimited-OCR vision features are expected to be fp16/fp32 but got ",
                    src.get_element_type().to_string());
    const ov::float16* src_data = src.data<const ov::float16>() + src_row * hidden;
    for (size_t i = 0; i < values; ++i) {
        dst[i] = static_cast<float>(src_data[i]);
    }
}

// Assembles the global view: a (grid x grid) feature map, terminating every grid row
// with the learnable ``image_newline`` vector. Returns grid * (grid + 1) rows.
// Mirrors UnlimitedOCRModel: view(h, w, dim) -> cat newline -> view(-1, dim).
void append_global_with_newlines(const ov::Tensor& global_features,
                                 const std::vector<float>& image_newline,
                                 std::vector<float>& out,
                                 size_t hidden) {
    const ov::Shape& shape = global_features.get_shape();  // (1, grid*grid, hidden)
    OPENVINO_ASSERT(shape.size() == 3 && shape.at(0) == 1, "Unexpected Unlimited-OCR global feature shape");
    const size_t hw = shape.at(1);
    const auto grid = static_cast<size_t>(std::lround(std::sqrt(static_cast<double>(hw))));
    OPENVINO_ASSERT(grid * grid == hw, "Unlimited-OCR global feature grid must be square, got ", hw);
    OPENVINO_ASSERT(image_newline.size() == hidden, "Unlimited-OCR image_newline size does not match hidden size");

    for (size_t row = 0; row < grid; ++row) {
        const size_t base = out.size();
        out.resize(base + grid * hidden);
        copy_rows_to_f32(global_features, row * grid, out.data() + base, grid, hidden);
        out.insert(out.end(), image_newline.begin(), image_newline.end());
    }
}

// Assembles the local tiles feature map. The (P) tiles of (tgrid x tgrid) tokens are
// rearranged into a (height_crop_num*tgrid) x (width_crop_num*tgrid) grid, then every
// grid row is terminated with the learnable ``image_newline`` vector. Mirrors
// UnlimitedOCRModel: local.view(hcn, wcn, h2, w2, dim).permute(0,2,1,3,4)
//                    .reshape(hcn*h2, wcn*w2, dim) -> cat newline -> view(-1, dim).
void append_local_with_newlines(const ov::Tensor& tile_features,
                                int width_crop_num,
                                int height_crop_num,
                                const std::vector<float>& image_newline,
                                std::vector<float>& out,
                                size_t hidden) {
    const ov::Shape& shape = tile_features.get_shape();  // (P, tgrid*tgrid, hidden)
    OPENVINO_ASSERT(shape.size() == 3, "Unexpected Unlimited-OCR tile feature shape");
    const size_t num_tiles = shape.at(0);
    const size_t hw2 = shape.at(1);
    OPENVINO_ASSERT(shape.at(2) == hidden, "Unlimited-OCR tile feature hidden size mismatch");
    const auto tgrid = static_cast<size_t>(std::lround(std::sqrt(static_cast<double>(hw2))));
    OPENVINO_ASSERT(tgrid * tgrid == hw2, "Unlimited-OCR tile feature grid must be square, got ", hw2);
    OPENVINO_ASSERT(num_tiles == static_cast<size_t>(width_crop_num) * static_cast<size_t>(height_crop_num),
                    "Unlimited-OCR tile count does not match crop ratio");

    const size_t grid_h = static_cast<size_t>(height_crop_num) * tgrid;
    const size_t grid_w = static_cast<size_t>(width_crop_num) * tgrid;

    // Cache all tile rows as f32 for random access.
    std::vector<float> tiles(num_tiles * hw2 * hidden);
    copy_rows_to_f32(tile_features, 0, tiles.data(), num_tiles * hw2, hidden);

    for (size_t gy = 0; gy < grid_h; ++gy) {
        const size_t tile_row = gy / tgrid;   // crop row index (0..height_crop_num-1)
        const size_t in_row = gy % tgrid;     // token row inside the tile
        for (size_t gx = 0; gx < grid_w; ++gx) {
            const size_t tile_col = gx / tgrid;  // crop col index (0..width_crop_num-1)
            const size_t in_col = gx % tgrid;    // token col inside the tile
            const size_t tile_idx = tile_row * static_cast<size_t>(width_crop_num) + tile_col;
            const size_t token_idx = in_row * tgrid + in_col;
            const float* src = tiles.data() + (tile_idx * hw2 + token_idx) * hidden;
            out.insert(out.end(), src, src + hidden);
        }
        out.insert(out.end(), image_newline.begin(), image_newline.end());
    }
}

}  // namespace

VisionEncoderUnlimitedOCR::VisionEncoderUnlimitedOCR(const std::filesystem::path& model_dir,
                                                     const std::string& device,
                                                     const ov::AnyMap properties)
    : VisionEncoder(model_dir, device, properties) {
    auto compiled_tiles = utils::singleton_core().compile_model(
        model_dir / "openvino_vision_embeddings_tiles_model.xml",
        device,
        utils::get_model_properties(properties, "vision_embeddings_tiles", device));
    m_ireq_queue_vision_encoder_tiles = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_tiles.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_tiles]() -> ov::InferRequest {
            return compiled_tiles.create_infer_request();
        });
    m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(model_dir, "config.json");
}

VisionEncoderUnlimitedOCR::VisionEncoderUnlimitedOCR(const ModelsMap& models_map,
                                                     const std::filesystem::path& config_dir_path,
                                                     const std::string& device,
                                                     const ov::AnyMap properties)
    : VisionEncoder(models_map, config_dir_path, device, properties) {
    const auto& [tiles_model, tiles_weights] = utils::get_model_weights_pair(models_map, "vision_embeddings_tiles");
    auto compiled_tiles = utils::singleton_core().compile_model(
        tiles_model,
        tiles_weights,
        device,
        utils::get_model_properties(properties, "vision_embeddings_tiles", device));
    m_ireq_queue_vision_encoder_tiles = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_tiles.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_tiles]() -> ov::InferRequest {
            return compiled_tiles.create_infer_request();
        });
    m_vlm_config = utils::from_config_json_if_exists<VLMConfig>(config_dir_path, "config.json");
}

EncodedImage VisionEncoderUnlimitedOCR::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    const clip_image_u8 input_image = tensor_to_clip_image_u8(image);
    const size_t hidden = m_vlm_config.image_newline.size();
    OPENVINO_ASSERT(hidden > 0, "Unlimited-OCR requires a non-empty image_newline in config.json");
    OPENVINO_ASSERT(m_vlm_config.view_separator.size() == hidden,
                    "Unlimited-OCR view_separator size does not match image_newline size");

    // Crop mode is enabled when the image exceeds the tile resolution in any dimension.
    std::vector<clip_image_u8> tile_images;
    std::pair<int, int> crop_ratio{1, 1};
    if (input_image.nx > TILE_SIZE || input_image.ny > TILE_SIZE) {
        auto [tiles, ratio] = dynamic_preprocess(input_image, TILE_SIZE, DYN_MIN_NUM, DYN_MAX_NUM);
        crop_ratio = ratio;
        if (ratio.first > 1 || ratio.second > 1) {
            tile_images = std::move(tiles);
        }
    }
    const bool crop_mode = !tile_images.empty();

    // Global view is always processed at BASE_SIZE (1024x1024).
    const clip_image_u8 global_view = pad_to_square(input_image, BASE_SIZE, PAD_VALUES);
    const ov::Tensor global_tensor = to_batch_tensor({global_view}, IMAGE_MEAN, IMAGE_STD);
    const ov::Tensor global_features = infer_and_copy(m_ireq_queue_vision_encoder.get(), global_tensor);

    std::vector<float> feature_values;
    if (crop_mode) {
        // Layout: cat([local_tiles, global, view_separator]).
        const ov::Tensor tile_tensor = to_batch_tensor(tile_images, IMAGE_MEAN, IMAGE_STD);
        const ov::Tensor tile_features = infer_and_copy(m_ireq_queue_vision_encoder_tiles.get(), tile_tensor);
        append_local_with_newlines(tile_features, crop_ratio.first, crop_ratio.second,
                                   m_vlm_config.image_newline, feature_values, hidden);
    }
    append_global_with_newlines(global_features, m_vlm_config.image_newline, feature_values, hidden);
    feature_values.insert(feature_values.end(), m_vlm_config.view_separator.begin(), m_vlm_config.view_separator.end());

    OPENVINO_ASSERT(feature_values.size() % hidden == 0, "Unlimited-OCR assembled feature size is not hidden-aligned");
    const size_t total_tokens = feature_values.size() / hidden;
    ov::Tensor merged_features(ov::element::f32, {1, total_tokens, hidden});
    std::copy_n(feature_values.data(), feature_values.size(), merged_features.data<float>());

    EncodedImage encoded_image;
    encoded_image.resized_source = std::move(merged_features);
    encoded_image.num_image_tokens = total_tokens;
    encoded_image.resized_source_size = ImageSize{static_cast<size_t>(BASE_SIZE), static_cast<size_t>(BASE_SIZE)};
    encoded_image.original_image_size = ImageSize{static_cast<size_t>(input_image.ny), static_cast<size_t>(input_image.nx)};
    return encoded_image;
}

}  // namespace ov::genai
