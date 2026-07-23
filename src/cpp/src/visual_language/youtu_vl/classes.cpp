// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/youtu_vl/classes.hpp"

#include <algorithm>
#include <cmath>

#include "visual_language/clip.hpp"
#include "utils.hpp"

namespace ov::genai {

namespace youtu_vl_utils {

ImageSize get_image_size_for_patches(size_t image_height, size_t image_width, size_t patch_size, size_t max_num_patches) {
    // Mirrors image_processing_siglip2_fast.get_image_size_for_patches:
    // round each dimension up to a multiple of patch_size*2, shrinking scale
    // until the number of patches fits within max_num_patches.
    auto scaled = [patch_size](double scale, size_t size) -> size_t {
        size_t step = patch_size * 2;
        double scaled_size = static_cast<double>(size) * scale;
        size_t rounded = static_cast<size_t>(std::ceil(scaled_size / static_cast<double>(step))) * step;
        return std::max(step, rounded);
    };

    double scale = 1.0;
    size_t target_height = 0;
    size_t target_width = 0;
    while (true) {
        target_height = scaled(scale, image_height);
        target_width = scaled(scale, image_width);
        double num_patches = (static_cast<double>(target_height) / patch_size) *
                             (static_cast<double>(target_width) / patch_size);
        if (num_patches > static_cast<double>(max_num_patches)) {
            scale -= 0.02;
        } else {
            break;
        }
    }
    return ImageSize{target_height, target_width};
}

} // namespace youtu_vl_utils

namespace {

// Build the pre-patchified pixel_values tensor of shape [num_patches, patch*patch*channels].
// Layout follows image_processing_siglip2_fast.convert_image_to_patches with merge_size=2:
//  - rows iterate over 2x2 merged blocks in row-major order, then within-block (mh, mw);
//  - each row packs a single patch flattened as (patch_row, patch_col, channel) (HWC).
ov::Tensor build_pixel_values(
    const clip_image_u8& resized,
    const ProcessorConfig& config
) {
    const size_t patch = config.patch_size;
    const size_t merge = config.merge_size;
    const size_t channels = 3;
    const size_t H = static_cast<size_t>(resized.ny);
    const size_t W = static_cast<size_t>(resized.nx);
    OPENVINO_ASSERT(H % patch == 0 && W % patch == 0, "Youtu-VL: resized image must be a multiple of patch_size.");
    const size_t nh = H / patch;
    const size_t nw = W / patch;
    OPENVINO_ASSERT(nh % merge == 0 && nw % merge == 0, "Youtu-VL: patch grid must be a multiple of merge_size.");

    const size_t num_patches = nh * nw;
    const size_t patch_dim = patch * patch * channels;

    // Precompute normalized f32 pixels in HWC layout: value = (u8/255 - mean)/std.
    std::array<float, 3> mean{config.image_mean[0], config.image_mean[1], config.image_mean[2]};
    std::array<float, 3> std_{config.image_std[0], config.image_std[1], config.image_std[2]};

    ov::Tensor pixel_values(ov::element::f32, ov::Shape{num_patches, patch_dim});
    float* out = pixel_values.data<float>();

    size_t row = 0;
    const size_t bh_count = nh / merge;
    const size_t bw_count = nw / merge;
    for (size_t bh = 0; bh < bh_count; ++bh) {
        for (size_t bw = 0; bw < bw_count; ++bw) {
            for (size_t mh = 0; mh < merge; ++mh) {
                for (size_t mw = 0; mw < merge; ++mw) {
                    const size_t ph = bh * merge + mh; // patch row index
                    const size_t pw = bw * merge + mw; // patch col index
                    float* dst = out + row * patch_dim;
                    size_t k = 0;
                    for (size_t py = 0; py < patch; ++py) {
                        const size_t img_y = ph * patch + py;
                        for (size_t px = 0; px < patch; ++px) {
                            const size_t img_x = pw * patch + px;
                            const size_t base = (img_y * W + img_x) * channels;
                            for (size_t c = 0; c < channels; ++c) {
                                float v = static_cast<float>(resized.buf[base + c]) / 255.0f;
                                dst[k++] = (v - mean[c]) / std_[c];
                            }
                        }
                    }
                    ++row;
                }
            }
        }
    }
    OPENVINO_ASSERT(row == num_patches, "Youtu-VL: patch packing row mismatch.");
    return pixel_values;
}

} // namespace

VisionEncoderYoutuVL::VisionEncoderYoutuVL(const std::filesystem::path& model_dir,
                                           const std::string& device,
                                           const ov::AnyMap properties)
    : VisionEncoder(model_dir, device, properties) {}

VisionEncoderYoutuVL::VisionEncoderYoutuVL(const ModelsMap& models_map,
                                           const std::filesystem::path& config_dir_path,
                                           const std::string& device,
                                           const ov::AnyMap properties)
    : VisionEncoder(models_map, config_dir_path, device, properties) {}

EncodedImage VisionEncoderYoutuVL::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    const ProcessorConfig config = ProcessorConfig::from_any_map(config_map, m_processor_config);

    clip_image_u8 input_image = tensor_to_clip_image_u8(image);

    ImageSize target = youtu_vl_utils::get_image_size_for_patches(
        static_cast<size_t>(input_image.ny),
        static_cast<size_t>(input_image.nx),
        config.patch_size,
        config.max_num_patches
    );

    clip_image_u8 resized;
    bilinear_resize(input_image, resized, static_cast<int>(target.width), static_cast<int>(target.height));

    ov::Tensor pixel_values = build_pixel_values(resized, config);

    const size_t grid_h = target.height / config.patch_size;
    const size_t grid_w = target.width / config.patch_size;

    // vision_embeddings model expects a batched [1, num_patches, patch_dim] input.
    ov::Tensor batched_pixel_values(ov::element::f32,
        ov::Shape{1, pixel_values.get_shape().at(0), pixel_values.get_shape().at(1)});
    std::memcpy(batched_pixel_values.data(), pixel_values.data(), pixel_values.get_byte_size());

    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();
    encoder.set_tensor("pixel_values", batched_pixel_values);
    encoder.infer();
    const ov::Tensor& infer_output = encoder.get_output_tensor();

    EncodedImage encoded_img;
    encoded_img.resized_source = ov::Tensor(infer_output.get_element_type(), infer_output.get_shape());
    std::memcpy(encoded_img.resized_source.data(), infer_output.data(), infer_output.get_byte_size());
    encoded_img.resized_source_size = ImageSize{grid_h, grid_w};
    return encoded_img;
}

InputsEmbedderYoutuVL::InputsEmbedderYoutuVL(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config) :
    InputsEmbedderQwen2_5_VL(vlm_config, model_dir, device, device_config) {}

InputsEmbedderYoutuVL::InputsEmbedderYoutuVL(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config) :
    InputsEmbedderQwen2_5_VL(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {}

std::pair<ov::Tensor, int64_t> InputsEmbedderYoutuVL::create_position_ids(
    const ov::Tensor& input_ids_tensor,
    const std::vector<std::array<size_t, 3>>& /*images_grid_thw*/,
    const std::vector<size_t>& /*images_sequence*/,
    const size_t /*image_id*/,
    const std::vector<std::array<size_t, 3>>& /*videos_grid_thw*/,
    const std::vector<size_t>& /*videos_sequence*/,
    const size_t /*video_id*/,
    const int64_t /*vision_start_token_id*/,
    const std::vector<std::pair<std::size_t, std::size_t>>& /*history_vision_count*/
) {
    // Youtu-VL language model uses plain 1D position ids (no 3D mRoPE).
    // Shape [1, seq_len] matches the exported language_model position_ids input [?, ?].
    const size_t seq_len = input_ids_tensor.get_shape().at(input_ids_tensor.get_shape().size() - 1);
    ov::Tensor position_ids{ov::element::i64, {1, seq_len}};
    int64_t* data = position_ids.data<int64_t>();
    std::iota(data, data + seq_len, int64_t{0});
    // rope_delta is unused for plain 1D position ids; keep it at (max_pos + 1) so the
    // generation phase continues sequentially from history.
    int64_t rope_delta = 0;
    return {position_ids, rope_delta};
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderYoutuVL::get_position_ids(const size_t inputs_embeds_size, const size_t history_size) {
    if (history_size != 0) {
        return get_generation_phase_position_ids(inputs_embeds_size, history_size, m_rope_delta);
    }
    return {m_position_ids, m_rope_delta};
}

std::pair<ov::Tensor, std::optional<int64_t>> InputsEmbedderYoutuVL::get_generation_phase_position_ids(const size_t inputs_embeds_size, const size_t history_size, int64_t rope_delta) {
    OPENVINO_ASSERT(history_size != 0, "get_generation_phase_position_ids() should only be called when history_size is non-zero (generation phase).");
    // Plain 1D position ids continuing from history: [history_size, history_size+1, ...].
    ov::Tensor position_ids{ov::element::i64, {1, inputs_embeds_size}};
    int64_t* data = position_ids.data<int64_t>();
    std::iota(data, data + inputs_embeds_size, static_cast<int64_t>(history_size));
    return {position_ids, rope_delta};
}

} // namespace ov::genai
