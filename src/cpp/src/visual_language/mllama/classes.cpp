// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/mllama/classes.hpp"

#include "visual_language/clip.hpp"

#include "utils.hpp"

namespace ov::genai {

namespace {

// ref:
// https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/mllama/image_processing_mllama.py#L53
static inline std::vector<ImageSize> get_all_supported_aspect_ratios(const size_t& max_image_tiles) {
    std::vector<ImageSize> aspect_ratios;
    for (size_t width = 1; width < max_image_tiles + 1; width++) {
        for (size_t height = 1; height < max_image_tiles + 1; height++) {
            if (width * height <= max_image_tiles) {
                // Note: It isn't a mistake that we swap width & height here.
                // Even though ref impl. does 'aspect_ratios.append((width, height))', it seems that consumers always
                // interpret results as (height, width)... so we just perform swap here.
                ImageSize ar;
                ar.width = height;
                ar.height = width;
                aspect_ratios.push_back(ar);
            }
        }
    }
    return aspect_ratios;
}

// ref:
// https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/mllama/image_processing_mllama.py#L134
static inline ImageSize get_optimal_tiled_canvas(const size_t& image_height, const size_t& image_width, const size_t& max_image_tiles, const size_t& tile_size) {
    auto possible_canvas_sizes = get_all_supported_aspect_ratios(max_image_tiles);
    std::vector<size_t> target_widths;
    std::vector<size_t> target_heights;
    for (auto& canvas_size : possible_canvas_sizes) {
        canvas_size.width *= tile_size;
        canvas_size.height *= tile_size;
        target_widths.push_back(canvas_size.width);
        target_heights.push_back(canvas_size.height);
    }

    std::vector<float> scales;
    bool upscaling_options_available = false;
    float min_upscaling_option = std::numeric_limits<float>::max();
    float max_downscaling_option = 0.f;
    for (size_t i = 0; i < possible_canvas_sizes.size(); i++) {
        float scale_h = static_cast<float>(target_heights[i]) / image_height;
        float scale_w = static_cast<float>(target_widths[i]) / image_width;
        auto scale = scale_w > scale_h ? scale_h : scale_w;
        scales.push_back(scale);
        if (scale >= 1.f) {
            upscaling_options_available = true;
            if (scale < min_upscaling_option) {
                min_upscaling_option = scale;
            }
        }

        if (scale < 1) {
            if (scale > max_downscaling_option) {
                max_downscaling_option = scale;
            }
        }
    }

    float selected_scale = upscaling_options_available ? min_upscaling_option : max_downscaling_option;

    ImageSize optimal_canvas;
    size_t min_area = std::numeric_limits<size_t>::max();
    for (size_t i = 0; i < possible_canvas_sizes.size(); i++) {
        if (scales[i] == selected_scale) {
            auto& canvas_size = possible_canvas_sizes[i];
            size_t area = canvas_size.width * canvas_size.height;
            if ( area < min_area) {
                optimal_canvas = canvas_size;
                min_area = area;
            }
        }
    }

    return optimal_canvas;
}

// ref:
// https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/mllama/image_processing_mllama.py#L82
static inline ImageSize get_image_size_fit_to_canvas(const ImageSize& image_size,
    const ImageSize &canvas_size,
    size_t tile_size) {
    size_t target_width = image_size.width;
    size_t target_height = image_size.height;

    // clip to [tile_size, canvas_size]
    target_width = std::min(std::max(target_width, tile_size), canvas_size.width);
    target_height = std::min(std::max(target_height, tile_size), canvas_size.height);

    double scale_h = static_cast<double>(target_height) / image_size.height;
    double scale_w = static_cast<double>(target_width) / image_size.width;

    ImageSize new_size;
    if (scale_w < scale_h) {
        new_size.width = target_width;
        size_t h = static_cast<size_t>(std::floor(static_cast<double>(image_size.height) * scale_w)); 
        new_size.height = std::min(h==0 ? 1 : h, target_height);
    } else {
        new_size.height = target_height;
        size_t w = static_cast<size_t>(std::floor(static_cast<double>(image_size.width) * scale_h));
        new_size.width = std::min(w==0 ? 1 : w, target_width);
    }

    return new_size;
}

// ref:
// https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/mllama/image_processing_mllama.py#L789
// Returns padded_image via reference param.
void pad(const clip_image_u8& image, const ProcessorConfig& config, const ImageSize& aspect_ratio, clip_image_u8& padded_image) {
    auto num_tiles_width = aspect_ratio.width;
    auto num_tiles_height = aspect_ratio.height;
    auto padded_width = num_tiles_width * config.size_width;
    auto padded_height = num_tiles_height * config.size_height;

    OPENVINO_ASSERT(padded_width >= image.nx);
    OPENVINO_ASSERT(padded_height >= image.ny);
    padded_image.nx = padded_width;
    padded_image.ny = padded_height;
    padded_image.buf.resize(3 * padded_width * padded_height);

    size_t npadx = padded_width - image.nx;

    for (int y = 0; y < padded_height; y++) {
        uint8_t* ppadded = &padded_image.buf[y * padded_width * 3];
        if (y < image.ny) {
            const uint8_t* pimage = &image.buf[y * image.nx * 3];
            std::memcpy(ppadded, pimage, image.nx * 3);

            if (npadx) {
                ppadded += image.nx * 3;
                std::memset(ppadded, 0, (npadx) * 3);
            }
        } else {
            std::memset(ppadded, 0, padded_width * 3);
        }
    }
}

// ref:
// https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/mllama/image_processing_mllama.py#L836
// Returns the resizes_image & chosen aspect ratio as reference params.
void resize(const ov::Tensor& image, const ProcessorConfig& config, clip_image_u8 &resized_image, ImageSize &aspect_ratio) {
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);

    ImageSize image_size{input_image.ny, input_image.nx};
    auto image_height = input_image.ny;
    auto image_width = input_image.nx;
    auto tile_size = config.size_height;
    OPENVINO_ASSERT(tile_size > 0);

    auto canvas_size = get_optimal_tiled_canvas(image_height, image_width, config.max_image_tiles, tile_size);

    auto num_tiles_height = canvas_size.height / tile_size;
    auto num_tiles_width = canvas_size.width / tile_size;

    auto new_size = get_image_size_fit_to_canvas(image_size, canvas_size, tile_size);

    bilinear_resize(input_image, resized_image, new_size.width, new_size.height);

    aspect_ratio.width = num_tiles_width;
    aspect_ratio.height = num_tiles_height;
}

// ref:
// https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/mllama/image_processing_mllama.py#L283
static inline ov::Tensor split_to_tiles(const clip_image_f32& image,
                                        const size_t& num_tiles_height,
                                        const size_t& num_tiles_width) {
    OPENVINO_ASSERT(num_tiles_height > 0);
    OPENVINO_ASSERT(num_tiles_width > 0);
    OPENVINO_ASSERT(image.nx > 0);
    OPENVINO_ASSERT(image.ny > 0);

    constexpr size_t C = 3;

    const size_t W = static_cast<size_t>(image.nx);
    const size_t H = static_cast<size_t>(image.ny);

    OPENVINO_ASSERT(W % num_tiles_height == 0);
    OPENVINO_ASSERT(H % num_tiles_width == 0);

    const size_t tile_h = H / num_tiles_height;
    const size_t tile_w = W / num_tiles_width;
    const size_t tiles_h = num_tiles_height;
    const size_t tiles_w = num_tiles_width;
    const size_t N = tiles_h * tiles_w;

    const size_t hw = H * W;
    OPENVINO_ASSERT(image.buf.size() == hw * 3);

    ov::Tensor out(ov::element::f32,
                   ov::Shape{static_cast<std::size_t>(N),
                             static_cast<std::size_t>(C),
                             static_cast<std::size_t>(tile_h),
                             static_cast<std::size_t>(tile_w)});

    // wrap 'image' as an ov::Tensor so that we can use view + copies.
    ov::Tensor in(ov::element::f32, ov::Shape{1, C, H, W}, image.buf.data());

    for (size_t ty = 0; ty < tiles_h; ty++) {
        for (size_t tx = 0; tx < tiles_w; tx++) {
            ov::Coordinate in_coord_begin{0, 0, ty * tile_h, tx * tile_w};
            ov::Coordinate in_coord_end{1, C, ty * tile_h + tile_h, tx * tile_w + tile_w};
            ov::Tensor in_view(in, in_coord_begin, in_coord_end);

            size_t n = ty * tiles_w + tx;
            ov::Coordinate out_coord_begin{n, 0, 0, 0};
            ov::Coordinate out_coord_end{n + 1, C, tile_h, tile_w};
            ov::Tensor out_view(out, out_coord_begin, out_coord_end);

            in_view.copy_to(out_view);
        }
    }

    return out;
}

// TODO: This should get merged with split_to_tiles.. I think.. assuming 'encode' is called for each image.
// This just packs a [num_tiles, 3, tile_height, tile_width] tensor into a new tensor with shape:
// [batch_size, num_images, max_image_tiles, 3, tile_height, tile_width]
// If 'num_tiles' in input image is < 'max_image_tiles', this will add some extra tiles of all 0's.
static inline ov::Tensor pack_images(const ov::Tensor& image, const size_t& max_image_tiles) {
    OPENVINO_ASSERT(image.get_shape().size() == 4);
    size_t num_image_tiles = image.get_shape()[0];
    size_t num_channels = image.get_shape()[1];
    size_t tile_height = image.get_shape()[2];
    size_t tile_width = image.get_shape()[3];

    OPENVINO_ASSERT(num_image_tiles <= max_image_tiles);

    constexpr size_t B = 1;
    constexpr size_t NUM_IMAGES = 1;
    ov::Tensor out(ov::element::f32,
        ov::Shape{B, NUM_IMAGES, static_cast<size_t>(max_image_tiles), num_channels, tile_height, tile_width});

    if (num_image_tiles < max_image_tiles) {
        std::memset(out.data(), 0, out.get_byte_size());
    }

    std::memcpy(out.data(), image.data(), image.get_byte_size());

    return out;
}

// ref:
// https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/mllama/image_processing_mllama.py#L424
static inline ov::Tensor convert_aspect_ratio_to_ids(const ImageSize& aspect_ratio, const size_t& max_image_tiles) {
    auto supported_aspect_ratios = get_all_supported_aspect_ratios(max_image_tiles);

    ov::Tensor aspect_ratio_ids = ov::Tensor(ov::element::i64, {1, 1});
    auto* ids = aspect_ratio_ids.data<int64_t>();
    *ids = 0;

    for (size_t i = 0; i < supported_aspect_ratios.size(); i++) {
        auto& supported_aspect_ratio = supported_aspect_ratios[i];
        if (supported_aspect_ratio.width == aspect_ratio.width
            && supported_aspect_ratio.height == aspect_ratio.height ) {
            *ids = i + 1;
            break;
        }
    }

    return aspect_ratio_ids;
}

// ref:
// https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/mllama/image_processing_mllama.py#L314
static inline ov::Tensor build_aspect_ratio_mask(const ImageSize& aspect_ratio, const size_t& max_image_tiles) {
    // shape is [Batch, Number of Images, Max Image Tiles]
    ov::Tensor aspect_ratio_mask = ov::Tensor(ov::element::i64, {1, 1, max_image_tiles});
    std::memset(aspect_ratio_mask.data(), 0, aspect_ratio_mask.get_byte_size());

    //Note: Since this implementation assumes batch=1, number of images=1, it skips the logic in
    // python which does: aspect_ratio_mask[:, :, 0] = 1. If this is adapted to support batch / num_images > 1,
    // then it should get added back in here.

    auto* mask = aspect_ratio_mask.data<int64_t>();
    for (size_t i = 0; i < aspect_ratio.height * aspect_ratio.width; i++) {
        mask[i] = 1;
    }

    return aspect_ratio_mask;
}

// returns <pixel_values, aspect_ratio_ids, aspect_ratio_mask> ov::Tensors
std::tuple<ov::Tensor, ov::Tensor, ov::Tensor> get_pixel_values_mllama(const ov::Tensor& image,
                                                                       const ProcessorConfig& config) {
    // Resize
    ImageSize aspect_ratio;
    clip_image_u8 resized_image{};
    resize(image, config, resized_image, aspect_ratio);

    // Pad
    clip_image_u8 padded_image{};
    pad(resized_image, config, aspect_ratio, padded_image);

    // Rescale + Normalize
    clip_ctx_double ctx;

    // apply fused normalize and rescale to 1.0/255, by the formula:
    // new_mean = mean * (1.0 / scale), new_std = std * (1.0 / rescale_factor)
    for (size_t c = 0; c < 3; c++) {
        ctx.image_mean[c] = config.image_mean[c] * 255;
        ctx.image_std[c] = config.image_std[c] * 255;
    }

    auto normalized_image = normalize_and_convert_to_chw(padded_image, ctx);

    // Tile the image & produce pixel_values tensor
    auto tiled_image = split_to_tiles(normalized_image, aspect_ratio.height, aspect_ratio.width);
    auto pixel_values = pack_images(tiled_image, config.max_image_tiles);

    // Build aspect ratio ids & mask tensors
    auto aspect_ratio_ids = convert_aspect_ratio_to_ids(aspect_ratio, config.max_image_tiles);
    auto aspect_ratio_mask = build_aspect_ratio_mask(aspect_ratio, config.max_image_tiles);

    return {pixel_values, aspect_ratio_ids, aspect_ratio_mask};
}

} // namespace

EncodedImage VisionEncoderMLlamma::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);

    ov::Tensor pixel_values, aspect_ratio_ids, aspect_ratio_mask;
    std::tie(pixel_values, aspect_ratio_ids, aspect_ratio_mask) = get_pixel_values_mllama(image, config);

    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();

    encoder.set_tensor("pixel_values", pixel_values);
    encoder.set_tensor("aspect_ratio_ids", aspect_ratio_ids);
    encoder.set_tensor("aspect_ratio_mask", aspect_ratio_mask);

    encoder.infer();

    // mllama encoder produces cross-kv states that need to be set as input tensors to the language model.
    // TODO: Figure out best way to make that connection.
    // Note: Another option is to have encoder return hidden states, and set those (similar to whisper encoder + decoder pipeline).
    //       In that case, language model will need to produce initial cross-kv during pre-fill. Need to experiment..
    // We also need to consider that for multi-turn conversations, we need to merge new cross-kv states with old.
    // Also, for text-only cases, we need to produce 'dummy' cross-kv.. and that can't happen here as this function won't
    // be called unless an image is used.

    return {};
}

ov::Tensor InputsEmbedderMLlamma::get_inputs_embeds(const std::string& unified_prompt,
                                                  const std::vector<ov::genai::EncodedImage>& images,
                                                  ov::genai::VLMPerfMetrics& metrics,
                                                  bool recalculate_merged_embeddings,
                                                  const std::vector<size_t>& images_sequence) {
    return {};
}

NormalizedPrompt InputsEmbedderMLlamma::normalize_prompt(const std::string& prompt,
                                                       size_t base_id,
                                                       const std::vector<EncodedImage>& images) const {
    return {};
}

InputsEmbedderMLlamma::InputsEmbedderMLlamma(const VLMConfig& vlm_config,
                                             const std::filesystem::path& model_dir,
                                             const std::string& device,
                                             const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, model_dir, device, device_config) {}

InputsEmbedderMLlamma::InputsEmbedderMLlamma(const VLMConfig& vlm_config,
                                             const ModelsMap& models_map,
                                             const Tokenizer& tokenizer,
                                             const std::filesystem::path& config_dir_path,
                                             const std::string& device,
                                             const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {}

}  // namespace ov::genai