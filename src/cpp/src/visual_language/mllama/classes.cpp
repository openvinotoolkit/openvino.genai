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
    padded_image.nx = static_cast<int>(padded_width);
    padded_image.ny = static_cast<int>(padded_height);
    padded_image.buf.resize(3 * padded_width * padded_height);

    size_t npadx = padded_width - image.nx;

    for (size_t y = 0; y < padded_height; ++y) {
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

    OPENVINO_ASSERT(input_image.ny > 0);
    OPENVINO_ASSERT(input_image.nx > 0);
    auto image_height = static_cast<size_t>(input_image.ny);
    auto image_width = static_cast<size_t>(input_image.nx);
    auto tile_size = config.size_height;
    OPENVINO_ASSERT(tile_size > 0);

    ImageSize image_size{image_height, image_width};

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

    OPENVINO_ASSERT(W % num_tiles_width == 0);
    OPENVINO_ASSERT(H % num_tiles_height == 0);

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

// returns <pixel_values, aspect_ratio_ids, aspect_ratio_mask, num_tiles>
std::tuple<ov::Tensor, ov::Tensor, ov::Tensor, size_t> get_pixel_values_mllama(const ov::Tensor& image,
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
    size_t num_tiles = tiled_image.get_shape()[0];
    auto pixel_values = pack_images(tiled_image, config.max_image_tiles);

    // Build aspect ratio ids & mask tensors
    auto aspect_ratio_ids = convert_aspect_ratio_to_ids(aspect_ratio, config.max_image_tiles);
    auto aspect_ratio_mask = build_aspect_ratio_mask(aspect_ratio, config.max_image_tiles);

    return {pixel_values, aspect_ratio_ids, aspect_ratio_mask, num_tiles};
}

// ref:
// https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/mllama/processing_mllama.py#L42
using MaskPair = std::array<int64_t, 2>;
static inline std::vector<MaskPair> get_cross_attention_token_mask(const ov::Tensor& input_ids,
                                                                   int64_t image_token_id) {
    // Validate dtype
    OPENVINO_ASSERT(input_ids.get_element_type() == ov::element::i64,
                    "input_ids must be an ov::Tensor of type i64 (int64_t).");

    // Validate shape [1, L]
    const ov::Shape shape = input_ids.get_shape();
    OPENVINO_ASSERT(shape.size() == 2 && shape[0] == 1, "input_ids must have shape [1, L].");

    const int64_t L = static_cast<int64_t>(shape[1]);
    const auto* ids = input_ids.data<const int64_t>();
    OPENVINO_ASSERT(ids, "input_ids.data<const int64_t>() returned null.");

    // Collect image token locations
    std::vector<int64_t> image_token_locations;
    image_token_locations.reserve(4);

    for (int64_t i = 0; i < L; ++i) {
        if (ids[i] == image_token_id) {
            image_token_locations.push_back(i);
        }
    }

    if (image_token_locations.empty()) {
        return {};
    }

    // Only one image present: unmask until end of sequence (Python uses -1 here)
    if (image_token_locations.size() == 1) {
        return {MaskPair{image_token_locations[0], -1}};
    }

    // Build initial masks: [[loc0, loc1], [loc1, loc2], ...] and last attends to end
    std::vector<MaskPair> vision_masks;
    vision_masks.reserve(image_token_locations.size());

    for (size_t k = 0; k + 1 < image_token_locations.size(); ++k) {
        vision_masks.push_back(MaskPair{image_token_locations[k], image_token_locations[k + 1]});
    }
    vision_masks.push_back(MaskPair{image_token_locations.back(), L});

    // If there are two or more consecutive vision tokens, they all attend to all subsequent tokens.
    int64_t last_mask_end = vision_masks.back()[1];
    for (size_t idx = vision_masks.size(); idx-- > 0;) {
        auto& m = vision_masks[idx];
        if (m[0] == (m[1] - 1)) {  // consecutive image tokens: start == end-1
            m[1] = last_mask_end;
        }
        last_mask_end = m[1];
    }

    return vision_masks;
}

// ref:
// https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/mllama/processing_mllama.py#L90
// Produces: ov::Tensor i64 of shape [B, length, max_num_images, max_num_tiles]
static inline ov::Tensor convert_sparse_cross_attention_mask_to_dense(
    const std::vector<std::vector<MaskPair>>& cross_attention_token_mask,  // [B][num_images] -> {start,end}
    const std::vector<std::vector<size_t>>& num_tiles,                     // [B][num_images] -> tile_count
    size_t max_num_tiles,
    size_t length) {
    const size_t batch_size = cross_attention_token_mask.size();

    if (num_tiles.size() != batch_size) {
        throw std::invalid_argument("num_tiles must have the same batch size as cross_attention_token_mask.");
    }

    // max_num_images = max(len(masks) for masks in cross_attention_token_mask)
    size_t max_num_images = 0;
    for (const auto& sample_masks : cross_attention_token_mask) {
        max_num_images = std::max(max_num_images, sample_masks.size());
    }

    // Allocate dense mask, initialized to zeros (ov::Tensor default-inits but doesn't guarantee zero,
    // so we explicitly fill).
    ov::Tensor cross_attention_mask(ov::element::i64, ov::Shape{batch_size, length, max_num_images, max_num_tiles});

    auto* out = cross_attention_mask.data<int64_t>();
    if (!out) {
        throw std::runtime_error("Failed to get writable tensor data for cross_attention_mask.");
    }
    std::fill(out, out + cross_attention_mask.get_size(), int64_t{0});

    // Helper for flattening indices [B, L, I, T] in row-major order
    const size_t B = batch_size;
    const size_t L = length;
    const size_t I = max_num_images;
    const size_t T = max_num_tiles;

    auto idx = [L, I, T](size_t b, size_t t, size_t i, size_t tile) -> size_t {
        return (((b * L + t) * I + i) * T + tile);
    };

    for (size_t sample_idx = 0; sample_idx < batch_size; ++sample_idx) {
        const auto& sample_masks = cross_attention_token_mask[sample_idx];
        const auto& sample_num_tiles = num_tiles[sample_idx];

        const size_t num_images_in_sample = std::min(sample_masks.size(), sample_num_tiles.size());

        for (size_t mask_idx = 0; mask_idx < num_images_in_sample; ++mask_idx) {
            const auto& loc = sample_masks[mask_idx];
            size_t mask_num_tiles = sample_num_tiles[mask_idx];

            // Equivalent of:
            // start, end = locations
            int64_t start64 = loc[0];
            int64_t end64 = loc[1];

            // Clamp tiles
            if (mask_num_tiles > max_num_tiles) {
                mask_num_tiles = max_num_tiles;
            }

            // Python:
            // end = min(end, length)
            // if end == -1: end = length
            // (note: min(-1, length) stays -1, so -1 survives into the check)
            end64 = std::min<int64_t>(end64, static_cast<int64_t>(length));
            if (end64 == -1) {
                end64 = static_cast<int64_t>(length);
            }

            // Basic sanity / bounds. (In the typical use-case start/end are non-negative.)
            if (start64 < 0) {
                // If you prefer strict matching of Python negative slicing semantics, you can
                // implement that here; most models never produce negative starts.
                start64 = 0;
            }
            if (end64 < 0) {
                // end < 0 (but not -1) => empty range
                continue;
            }

            const size_t start = static_cast<size_t>(start64);
            const size_t end = static_cast<size_t>(end64);

            if (start >= length) {
                continue;  // empty
            }
            const size_t end_clamped = std::min(end, length);
            if (end_clamped <= start) {
                continue;  // empty
            }

            // cross_attention_mask[sample_idx, start:end, mask_idx, :mask_num_tiles] = 1
            for (size_t t = start; t < end_clamped; ++t) {
                for (size_t tile = 0; tile < mask_num_tiles; ++tile) {
                    out[idx(sample_idx, t, mask_idx, tile)] = 1;
                }
            }
        }
    }

    return cross_attention_mask;
}

} // namespace

EncodedImage VisionEncoderMLlama::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    ProcessorConfig config = utils::from_any_map(config_map, m_processor_config);

    ov::Tensor pixel_values, aspect_ratio_ids, aspect_ratio_mask;
    size_t num_tiles;
    std::tie(pixel_values, aspect_ratio_ids, aspect_ratio_mask, num_tiles) = get_pixel_values_mllama(image, config);

    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();

    encoder.set_tensor("pixel_values", pixel_values);
    encoder.set_tensor("aspect_ratio_ids", aspect_ratio_ids);
    encoder.set_tensor("aspect_ratio_mask", aspect_ratio_mask);

    encoder.infer();

    EncodedImage encoded_image{};
    encoded_image.num_tiles = num_tiles;

    const auto &outputs = encoder.get_compiled_model().outputs();
    for (const auto& output : outputs) {
        const auto& name = output.get_any_name();
        auto tensor = encoder.get_tensor(name);
        // copy_to here so that next encoder infer won't overwrite the data we are storing in encoded_image.
        // For example, when multiple images are used, encode is called separately for each.
        ov::Tensor tensor_copy(tensor.get_element_type(), tensor.get_shape());
        tensor.copy_to(tensor_copy);
        encoded_image.cross_kv_states.push_back({name, tensor_copy});
    }

    return encoded_image;
}

std::vector<std::pair<std::string, ov::Tensor>> InputsEmbedderMLlama::get_language_model_inputs(
    const std::string& unified_prompt,
    const std::vector<ov::genai::EncodedImage>& images,
    const std::vector<ov::genai::EncodedVideo>& videos,
    ov::genai::VLMPerfMetrics& metrics,
    bool recalculate_merged_embeddings,
    const std::vector<size_t>& image_sequence,
    const std::vector<size_t>& videos_sequence,
    const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count) {
    if (videos.size()) {
        OPENVINO_THROW(
            "Current model doesn't support video preprocess currently. Input images are processed as separate images.");
    }

    // mllama always adds special tokens in pytorch
    set_add_special_tokens(true);
    ov::Tensor input_ids = get_encoded_input_ids(unified_prompt, metrics);
    ov::Tensor encoded_image_token = m_tokenizer.encode("<|image|>", ov::genai::add_special_tokens(false)).input_ids;
    int64_t image_token_id = encoded_image_token.data<int64_t>()[encoded_image_token.get_size() - 1];

    // TODO: I believe get_language_model_inputs is called for a given batch, although some of the following
    // helpers (get_cross_attention_token_mask, convert_sparse_cross_attention_mask_to_dense, etc.) are written
    // to handle batch size > 1. Not necessarily an issue, but it's confusing and could be simplified.
    std::vector<std::vector<MaskPair>> cross_attention_token_mask;
    size_t num_tokens = input_ids.get_shape()[1];
    for (size_t b = 0; b < input_ids.get_shape()[0]; b++) {
        ov::Coordinate in_coord_begin{b, 0};
        ov::Coordinate in_coord_end{b + 1, num_tokens};
        ov::Tensor input_ids_slice(input_ids, in_coord_begin, in_coord_end);
        cross_attention_token_mask.emplace_back(get_cross_attention_token_mask(input_ids, image_token_id));
    }

    //[B][image_images]
    std::vector<std::vector<size_t>> num_tiles;
    std::vector<std::pair<std::string, ov::Tensor>> cross_kv_inputs;
    if (!images.empty()) {
        std::vector<size_t> batch_tile_sizes;
        for (size_t imagei = 0; imagei < images.size(); imagei++) {
            if (imagei == 0) {
                cross_kv_inputs = images[imagei].cross_kv_states;
            } else {
                // TODO: This requires us to merge multiple cross_kv_states. Handle this later.
                OPENVINO_THROW("This model doesn't yet have support for multiple images.");
            }

            batch_tile_sizes.push_back(images[imagei].num_tiles);
        }
        num_tiles.push_back(batch_tile_sizes);
    }
    else
    {
        // TODO: If it's the first prompt in a chat session, we need to create dummy kv states.
        // Otherwise, re-use previous KV Cache states (they should already be set).
        OPENVINO_THROW("This model doesn't yet have support for text-only input");
    }

    //TODO: replace '4' here with max_tiles from vision preprocessor config.
    auto cross_attention_mask =
        convert_sparse_cross_attention_mask_to_dense(cross_attention_token_mask, num_tiles, 4, num_tokens);

    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor inputs_embeds = m_embedding->infer(req, input_ids);

    std::vector<std::pair<std::string, ov::Tensor>> inputs = {{"inputs_embeds", inputs_embeds}};
    inputs.push_back({"cross_attention_mask", cross_attention_mask});

    for (auto& cross_kv_input : cross_kv_inputs) {
        inputs.push_back(cross_kv_input);
    }

    return inputs;
}

ov::Tensor InputsEmbedderMLlama::get_inputs_embeds(const std::string& unified_prompt,
                                                    const std::vector<ov::genai::EncodedImage>& images,
                                                    ov::genai::VLMPerfMetrics& metrics,
                                                    bool recalculate_merged_embeddings,
                                                    const std::vector<size_t>& images_sequence){
    OPENVINO_THROW("[InputsEmbedderMLlama] The method get_inputs_embeds is not supported for MLlama models because "
                   "cross-kv states are required to also be returned, and set on the language model. "
                   "Please use get_language_model_inputs instead, which returns both the input embeddings "
                   "and the necessary cross-kv state tensors. ");
}

NormalizedPrompt InputsEmbedderMLlama::normalize_prompt(const std::string& prompt,
                                                       size_t base_id,
                                                       const std::vector<EncodedImage>& images) const {
    std::string image_token = "<|image|>";
    auto [unified_prompt, images_sequence] =
        normalize(prompt, image_token, image_token, base_id, images.size(), VisionType::IMAGE);

    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

InputsEmbedderMLlama::InputsEmbedderMLlama(const VLMConfig& vlm_config,
                                             const std::filesystem::path& model_dir,
                                             const std::string& device,
                                             const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, model_dir, device, device_config) {}

InputsEmbedderMLlama::InputsEmbedderMLlama(const VLMConfig& vlm_config,
                                             const ModelsMap& models_map,
                                             const Tokenizer& tokenizer,
                                             const std::filesystem::path& config_dir_path,
                                             const std::string& device,
                                             const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {}

}  // namespace ov::genai