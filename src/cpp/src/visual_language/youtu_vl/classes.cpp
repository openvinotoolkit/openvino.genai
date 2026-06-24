// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/youtu_vl/classes.hpp"

#include <cmath>
#include <fstream>
#include <limits>

#include "json_utils.hpp"
#include "utils.hpp"

#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"

namespace ov::genai {

namespace {

constexpr size_t MERGE_SIZE = 2;
constexpr size_t DEFAULT_PATCH_SIZE = 16;
constexpr size_t DEFAULT_MAX_NUM_PATCHES = 256;

struct YoutuVisionParams {
    size_t patch_size = DEFAULT_PATCH_SIZE;
    size_t max_num_patches = DEFAULT_MAX_NUM_PATCHES;
    std::array<float, 3> image_mean{0.5f, 0.5f, 0.5f};
    std::array<float, 3> image_std{0.5f, 0.5f, 0.5f};
};

YoutuVisionParams read_vision_params(const std::filesystem::path& config_dir) {
    YoutuVisionParams params;
    std::ifstream stream(config_dir / "preprocessor_config.json");
    if (stream.is_open()) {
        nlohmann::json parsed = nlohmann::json::parse(stream);
        if (parsed.contains("patch_size") && parsed.at("patch_size").is_number_integer()) {
            params.patch_size = parsed.at("patch_size").get<size_t>();
        }
        if (parsed.contains("max_num_patches") && parsed.at("max_num_patches").is_number_integer()) {
            params.max_num_patches = parsed.at("max_num_patches").get<size_t>();
        }
        if (parsed.contains("image_mean") && parsed.at("image_mean").is_array()) {
            auto m = parsed.at("image_mean").get<std::vector<float>>();
            for (size_t i = 0; i < 3 && i < m.size(); ++i) {
                params.image_mean[i] = m[i];
            }
        }
        if (parsed.contains("image_std") && parsed.at("image_std").is_array()) {
            auto s = parsed.at("image_std").get<std::vector<float>>();
            for (size_t i = 0; i < 3 && i < s.size(); ++i) {
                params.image_std[i] = s[i];
            }
        }
    }
    return params;
}

// Mirrors image_processing_siglip2_fast.get_image_size_for_patches().
std::pair<size_t, size_t> get_image_size_for_patches(size_t image_height,
                                                     size_t image_width,
                                                     size_t patch_size,
                                                     size_t max_num_patches) {
    auto scaled = [patch_size](double scale, size_t size) -> size_t {
        size_t ps2 = patch_size * 2;
        double scaled_size = static_cast<double>(size) * scale;
        size_t v = static_cast<size_t>(std::ceil(scaled_size / ps2)) * ps2;
        return std::max(ps2, v);
    };

    double scale = 1.0;
    size_t target_h = 0;
    size_t target_w = 0;
    while (true) {
        target_h = scaled(scale, image_height);
        target_w = scaled(scale, image_width);
        double num_patches = (static_cast<double>(target_h) / patch_size) * (static_cast<double>(target_w) / patch_size);
        if (num_patches > static_cast<double>(max_num_patches)) {
            scale -= 0.02;
        } else {
            break;
        }
    }
    return {target_h, target_w};
}

// Builds a preprocessing model: raw image (1HWC u8) -> Siglip2 resized/normalized
// window-grouped patches [N, channels * patch_size * patch_size].
std::shared_ptr<ov::Model> build_preprocess_model(const std::array<float, 3>& image_mean,
                                                  const std::array<float, 3>& image_std) {
    auto raw_image = std::make_shared<ov::op::v0::Parameter>(ov::element::u8, ov::PartialShape{1, -1, -1, 3});
    raw_image->set_friendly_name("raw_image");
    raw_image->output(0).get_tensor().set_names({"raw_image"});

    auto target_hw = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{2});
    target_hw->set_friendly_name("target_hw");
    target_hw->output(0).get_tensor().set_names({"target_hw"});

    auto reshape7d = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{7});
    reshape7d->set_friendly_name("reshape7d");
    reshape7d->output(0).get_tensor().set_names({"reshape7d"});

    auto reshape2d = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{2});
    reshape2d->set_friendly_name("reshape2d");
    reshape2d->output(0).get_tensor().set_names({"reshape2d"});

    // 1HWC u8 -> 1CHW f32
    auto img_f32 = std::make_shared<ov::op::v0::Convert>(raw_image, ov::element::f32);
    auto to_nchw = std::make_shared<ov::op::v1::Transpose>(
        img_f32, ov::op::v0::Constant::create(ov::element::i32, ov::Shape{4}, std::vector<int32_t>{0, 3, 1, 2}));

    // Bilinear resize (torchvision F.resize, align_corners=False, antialias=True).
    ov::op::v11::Interpolate::InterpolateAttrs attrs;
    attrs.mode = ov::op::v11::Interpolate::InterpolateMode::LINEAR;
    attrs.shape_calculation_mode = ov::op::v11::Interpolate::ShapeCalcMode::SIZES;
    attrs.coordinate_transformation_mode = ov::op::v11::Interpolate::CoordinateTransformMode::PYTORCH_HALF_PIXEL;
    attrs.nearest_mode = ov::op::v11::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    attrs.pads_begin = {0, 0};
    attrs.pads_end = {0, 0};
    attrs.antialias = true;
    auto resize_axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {2, 3});
    auto resized = std::make_shared<ov::op::v11::Interpolate>(to_nchw, target_hw, resize_axes, attrs);

    // do_rescale (1/255) + do_normalize: (pixel/255 - mean) / std.
    std::vector<float> mean_data{image_mean[0] * 255.0f, image_mean[1] * 255.0f, image_mean[2] * 255.0f};
    std::vector<float> scale_data{1.0f / (image_std[0] * 255.0f),
                                  1.0f / (image_std[1] * 255.0f),
                                  1.0f / (image_std[2] * 255.0f)};
    auto mean = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 3, 1, 1}, mean_data);
    auto scale = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 3, 1, 1}, scale_data);
    auto clamped = std::make_shared<ov::op::v0::Clamp>(resized, 0.0, 255.0);
    auto centered = std::make_shared<ov::op::v1::Subtract>(clamped, mean);
    auto normalized = std::make_shared<ov::op::v1::Multiply>(centered, scale);

    // Window-grouped patchify (convert_image_to_patches): reshape to 7D, permute, flatten to [N, C*ps*ps].
    auto reshaped7 = std::make_shared<ov::op::v1::Reshape>(normalized, reshape7d, false);
    auto transposed = std::make_shared<ov::op::v1::Transpose>(
        reshaped7,
        ov::op::v0::Constant::create(ov::element::i32, ov::Shape{7}, std::vector<int32_t>{1, 4, 2, 5, 3, 6, 0}));
    auto patches = std::make_shared<ov::op::v1::Reshape>(transposed, reshape2d, false);

    auto result = std::make_shared<ov::op::v0::Result>(patches);
    return std::make_shared<ov::Model>(
        ov::ResultVector{result},
        ov::ParameterVector{raw_image, target_hw, reshape7d, reshape2d},
        "youtu_vl_preprocess");
}

ov::Tensor ensure_3_channels(const ov::Tensor& image) {
    auto shape = image.get_shape();
    OPENVINO_ASSERT(shape.size() == 4 && image.get_element_type() == ov::element::u8,
                    "Youtu-VL expects a [1, H, W, C] uint8 image tensor.");
    size_t channels = shape.at(3);
    if (channels == 3) {
        return image;
    }
    ov::Shape new_shape = shape;
    new_shape[3] = 3;
    ov::Tensor out(ov::element::u8, new_shape);
    const uint8_t* in_data = image.data<const uint8_t>();
    uint8_t* out_data = out.data<uint8_t>();
    size_t pixels = shape[0] * shape[1] * shape[2];
    for (size_t i = 0; i < pixels; ++i) {
        if (channels == 1) {
            uint8_t v = in_data[i];
            out_data[i * 3 + 0] = v;
            out_data[i * 3 + 1] = v;
            out_data[i * 3 + 2] = v;
        } else {  // channels == 4 (RGBA) or more: take first 3
            out_data[i * 3 + 0] = in_data[i * channels + 0];
            out_data[i * 3 + 1] = in_data[i * channels + 1];
            out_data[i * 3 + 2] = in_data[i * channels + 2];
        }
    }
    return out;
}

}  // namespace

namespace youtu_vl_utils {

// Mirrors _OVYoutuVLForCausalLM.rot_pos_emb for a single image grid (h, w).
ov::Tensor get_rotary_pos_emb(size_t grid_h, size_t grid_w, size_t merge_size, size_t rope_dim) {
    const size_t num_patches = grid_h * grid_w;
    const size_t half = rope_dim / 2;  // number of inverse frequencies

    std::vector<float> inv_freq(half);
    for (size_t k = 0; k < half; ++k) {
        inv_freq[k] = 1.0f / std::pow(10000.0f, static_cast<float>(2 * k) / static_cast<float>(rope_dim));
    }

    // hpos/wpos ids in the same window-grouped order as convert_image_to_patches.
    std::vector<int64_t> hpos(num_patches);
    std::vector<int64_t> wpos(num_patches);
    size_t idx = 0;
    for (size_t bh = 0; bh < grid_h / merge_size; ++bh) {
        for (size_t bw = 0; bw < grid_w / merge_size; ++bw) {
            for (size_t mh = 0; mh < merge_size; ++mh) {
                for (size_t mw = 0; mw < merge_size; ++mw) {
                    hpos[idx] = static_cast<int64_t>(bh * merge_size + mh);
                    wpos[idx] = static_cast<int64_t>(bw * merge_size + mw);
                    ++idx;
                }
            }
        }
    }

    ov::Tensor rotary_pos_emb{ov::element::f32, ov::Shape{num_patches, rope_dim}};
    float* data = rotary_pos_emb.data<float>();
    for (size_t p = 0; p < num_patches; ++p) {
        for (size_t k = 0; k < half; ++k) {
            data[p * rope_dim + k] = static_cast<float>(hpos[p]) * inv_freq[k];
            data[p * rope_dim + half + k] = static_cast<float>(wpos[p]) * inv_freq[k];
        }
    }
    return rotary_pos_emb;
}

// Mirrors _OVYoutuVLForCausalLM.get_window_index for a single image grid (h, w).
// window_size is derived as patch_size * 2 * 8 (see optimum-intel implementation).
std::pair<ov::Tensor, std::vector<int32_t>> get_window_index(size_t grid_h,
                                                             size_t grid_w,
                                                             size_t merge_size,
                                                             size_t patch_size) {
    const size_t spatial_merge_unit = merge_size * merge_size;
    const size_t window_size = patch_size * 2 * 8;
    const size_t vit_merger_window_size = window_size / merge_size / patch_size;

    const size_t llm_grid_h = grid_h / merge_size;
    const size_t llm_grid_w = grid_w / merge_size;

    const size_t pad_h = (vit_merger_window_size - llm_grid_h % vit_merger_window_size) % vit_merger_window_size;
    const size_t pad_w = (vit_merger_window_size - llm_grid_w % vit_merger_window_size) % vit_merger_window_size;
    const size_t num_windows_h = (llm_grid_h + pad_h) / vit_merger_window_size;
    const size_t num_windows_w = (llm_grid_w + pad_w) / vit_merger_window_size;

    std::vector<int64_t> window_index;
    std::vector<int32_t> cu_window_seqlens = {0};

    for (size_t wh = 0; wh < num_windows_h; ++wh) {
        for (size_t ww = 0; ww < num_windows_w; ++ww) {
            int32_t valid_count = 0;
            for (size_t h = 0; h < vit_merger_window_size; ++h) {
                size_t gh = wh * vit_merger_window_size + h;
                if (gh >= llm_grid_h) {
                    continue;
                }
                for (size_t w = 0; w < vit_merger_window_size; ++w) {
                    size_t gw = ww * vit_merger_window_size + w;
                    if (gw >= llm_grid_w) {
                        continue;
                    }
                    window_index.push_back(static_cast<int64_t>(gh * llm_grid_w + gw));
                    ++valid_count;
                }
            }
            cu_window_seqlens.push_back(cu_window_seqlens.back() +
                                        valid_count * static_cast<int32_t>(spatial_merge_unit));
        }
    }

    ov::Tensor window_index_tensor{ov::element::i64, ov::Shape{window_index.size()}};
    std::memcpy(window_index_tensor.data<int64_t>(), window_index.data(), window_index.size() * sizeof(int64_t));
    return {window_index_tensor, cu_window_seqlens};
}

ov::Tensor make_block_mask(size_t seq_len, const std::vector<int32_t>& cu_seqlens) {
    ov::Tensor mask{ov::element::f32, ov::Shape{1, seq_len, seq_len}};
    float* data = mask.data<float>();
    std::fill_n(data, mask.get_size(), -std::numeric_limits<float>::infinity());
    for (size_t i = 1; i < cu_seqlens.size(); ++i) {
        size_t start = static_cast<size_t>(cu_seqlens[i - 1]);
        size_t end = static_cast<size_t>(cu_seqlens[i]);
        for (size_t row = start; row < end; ++row) {
            for (size_t col = start; col < end; ++col) {
                data[row * seq_len + col] = 0.0f;
            }
        }
    }
    return mask;
}

}  // namespace youtu_vl_utils

// ---------------------------------------------------------------------------
// VisionEncoderYoutuVL
// ---------------------------------------------------------------------------

VisionEncoderYoutuVL::VisionEncoderYoutuVL(const std::filesystem::path& model_dir,
                                           const std::string& device,
                                           const ov::AnyMap properties)
    : VisionEncoder(model_dir, device, properties) {
    auto model = utils::singleton_core().read_model(model_dir / "openvino_vision_embeddings_model.xml");
    auto compiled_model = utils::singleton_core().compile_model(
        model, device, utils::get_model_properties(properties, "vision_embeddings", device));
    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM vision embeddings model");
    m_ireq_queue_vision_encoder = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
    init_preprocess_model(device, properties);
}

VisionEncoderYoutuVL::VisionEncoderYoutuVL(const ModelsMap& models_map,
                                           const std::filesystem::path& config_dir_path,
                                           const std::string& device,
                                           const ov::AnyMap properties)
    : VisionEncoder(models_map, config_dir_path, device, properties) {
    const auto& [vision_encoder_model, vision_encoder_weights] =
        utils::get_model_weights_pair(models_map, "vision_embeddings");
    auto model = utils::singleton_core().read_model(vision_encoder_model, vision_encoder_weights);
    auto compiled_model = utils::singleton_core().compile_model(
        model, device, utils::get_model_properties(properties, "vision_embeddings", device));
    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM vision embeddings model");
    m_ireq_queue_vision_encoder = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
    init_preprocess_model(device, properties);
}

void VisionEncoderYoutuVL::init_preprocess_model(const std::string& device, const ov::AnyMap& properties) {
    auto model = build_preprocess_model(m_processor_config.image_mean, m_processor_config.image_std);
    auto compiled_model = utils::singleton_core().compile_model(
        model, device, utils::get_model_properties(properties, "vision_embeddings", device));
    m_ireq_queue_preprocess = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
}

EncodedImage VisionEncoderYoutuVL::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    ov::Tensor rgb_image = ensure_3_channels(image);
    ov::Shape shape = rgb_image.get_shape();
    size_t height = shape.at(1);
    size_t width = shape.at(2);

    size_t patch_size = m_processor_config.patch_size != 0 ? m_processor_config.patch_size : DEFAULT_PATCH_SIZE;
    size_t max_num_patches = DEFAULT_MAX_NUM_PATCHES;
    if (auto it = config_map.find("max_num_patches"); it != config_map.end()) {
        max_num_patches = it->second.as<size_t>();
    }

    auto [target_h, target_w] = get_image_size_for_patches(height, width, patch_size, max_num_patches);
    size_t grid_h = target_h / patch_size;
    size_t grid_w = target_w / patch_size;
    size_t num_patches = grid_h * grid_w;
    size_t flat_patch_dim = 3 * patch_size * patch_size;

    CircularBufferQueueElementGuard<ov::InferRequest> preprocess_guard(this->m_ireq_queue_preprocess.get());
    ov::InferRequest& preprocess = preprocess_guard.get();

    int64_t target_data[2] = {static_cast<int64_t>(target_h), static_cast<int64_t>(target_w)};
    ov::Tensor target_hw(ov::element::i64, ov::Shape{2}, target_data);
    int64_t reshape7d_data[7] = {3,
                                 static_cast<int64_t>(grid_h / MERGE_SIZE),
                                 static_cast<int64_t>(MERGE_SIZE),
                                 static_cast<int64_t>(patch_size),
                                 static_cast<int64_t>(grid_w / MERGE_SIZE),
                                 static_cast<int64_t>(MERGE_SIZE),
                                 static_cast<int64_t>(patch_size)};
    ov::Tensor reshape7d(ov::element::i64, ov::Shape{7}, reshape7d_data);
    int64_t reshape2d_data[2] = {static_cast<int64_t>(num_patches), static_cast<int64_t>(flat_patch_dim)};
    ov::Tensor reshape2d(ov::element::i64, ov::Shape{2}, reshape2d_data);

    preprocess.set_tensor("raw_image", rgb_image);
    preprocess.set_tensor("target_hw", target_hw);
    preprocess.set_tensor("reshape7d", reshape7d);
    preprocess.set_tensor("reshape2d", reshape2d);
    preprocess.infer();
    const ov::Tensor& patches = preprocess.get_output_tensor();

    // Add batch dim -> pixel_values [1, num_patches, 768].
    ov::Tensor pixel_values(ov::element::f32, ov::Shape{1, num_patches, flat_patch_dim});
    std::memcpy(pixel_values.data(), patches.data(), patches.get_byte_size());

    CircularBufferQueueElementGuard<ov::InferRequest> encoder_guard(this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = encoder_guard.get();
    encoder.set_tensor("pixel_values", pixel_values);
    encoder.infer();
    const ov::Tensor& hidden_states = encoder.get_output_tensor();

    EncodedImage encoded_image;
    encoded_image.resized_source = ov::Tensor(hidden_states.get_element_type(), hidden_states.get_shape());
    std::memcpy(encoded_image.resized_source.data(), hidden_states.data(), hidden_states.get_byte_size());
    encoded_image.resized_source_size = ImageSize{grid_h, grid_w};
    return encoded_image;
}

// ---------------------------------------------------------------------------
// InputsEmbedderYoutuVL
// ---------------------------------------------------------------------------

namespace {

void load_youtu_runtime_params(const std::filesystem::path& config_dir,
                               size_t& rope_dim,
                               size_t& patch_size) {
    std::ifstream stream(config_dir / "config.json");
    if (!stream.is_open()) {
        return;
    }
    nlohmann::json parsed = nlohmann::json::parse(stream);
    if (parsed.contains("vision_config")) {
        const auto& vc = parsed.at("vision_config");
        size_t hidden = vc.value("hidden_size", size_t{0});
        size_t heads = vc.value("num_attention_heads", size_t{0});
        if (hidden != 0 && heads != 0) {
            rope_dim = (hidden / heads) / 2;
        }
        patch_size = vc.value("patch_size", patch_size);
    }
}

}  // namespace

InputsEmbedderYoutuVL::InputsEmbedderYoutuVL(const VLMConfig& vlm_config,
                                             const std::filesystem::path& model_dir,
                                             const std::string& device,
                                             const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, model_dir, device, device_config) {
    auto merger = utils::singleton_core().read_model(model_dir / "openvino_vision_embeddings_merger_model.xml");
    init_merger(merger, device, device_config);
    load_youtu_runtime_params(model_dir, m_rope_dim, m_patch_size);
}

InputsEmbedderYoutuVL::InputsEmbedderYoutuVL(const VLMConfig& vlm_config,
                                             const ModelsMap& models_map,
                                             const Tokenizer& tokenizer,
                                             const std::filesystem::path& config_dir_path,
                                             const std::string& device,
                                             const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {
    const auto& [merger_model, merger_weights] = utils::get_model_weights_pair(models_map, "vision_embeddings_merger");
    auto merger = utils::singleton_core().read_model(merger_model, merger_weights);
    init_merger(merger, device, device_config);
    load_youtu_runtime_params(config_dir_path, m_rope_dim, m_patch_size);
}

void InputsEmbedderYoutuVL::init_merger(const std::shared_ptr<ov::Model>& merger_model,
                                        const std::string& device,
                                        const ov::AnyMap& device_config) {
    auto compiled_model = utils::singleton_core().compile_model(
        merger_model, device, utils::get_model_properties(device_config, "vision_embeddings_merger", device));
    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM vision embeddings merger model");
    m_ireq_queue_vision_embeddings_merger = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });

    auto encoded = m_tokenizer.encode(m_vlm_config.image_pad_token, ov::genai::add_special_tokens(false));
    m_image_pad_token_id = encoded.input_ids.data<int64_t>()[0];
}

std::vector<ov::genai::EncodedImage> InputsEmbedderYoutuVL::encode_images(const std::vector<ov::Tensor>& images) {
    std::vector<EncodedImage> embeds;
    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    embeds.reserve(single_images.size());
    for (const ov::Tensor& image : single_images) {
        embeds.emplace_back(m_vision_encoder->encode(image));
    }
    return embeds;
}

ov::Tensor InputsEmbedderYoutuVL::run_image_embeddings_merger(const ov::Tensor& hidden_states,
                                                              size_t grid_h,
                                                              size_t grid_w) {
    size_t seq_len = grid_h * grid_w;

    ov::Tensor rotary_pos_emb = youtu_vl_utils::get_rotary_pos_emb(grid_h, grid_w, m_merge_size, m_rope_dim);
    auto [window_index, cu_window_seqlens] = youtu_vl_utils::get_window_index(grid_h, grid_w, m_merge_size, m_patch_size);

    // Single image -> full attention within the whole image.
    std::vector<int32_t> cu_seqlens = {0, static_cast<int32_t>(seq_len)};
    ov::Tensor attention_mask = youtu_vl_utils::make_block_mask(seq_len, cu_seqlens);
    ov::Tensor window_attention_mask = youtu_vl_utils::make_block_mask(seq_len, cu_window_seqlens);

    CircularBufferQueueElementGuard<ov::InferRequest> guard(this->m_ireq_queue_vision_embeddings_merger.get());
    ov::InferRequest& merger = guard.get();
    merger.set_tensor("hidden_states", hidden_states);
    merger.set_tensor("attention_mask", attention_mask);
    merger.set_tensor("window_attention_mask", window_attention_mask);
    merger.set_tensor("window_index", window_index);
    merger.set_tensor("rotary_pos_emb", rotary_pos_emb);
    merger.infer();

    const ov::Tensor& out = merger.get_output_tensor();
    ov::Tensor result(out.get_element_type(), out.get_shape());
    std::memcpy(result.data(), out.data(), out.get_byte_size());
    return result;
}

NormalizedPrompt InputsEmbedderYoutuVL::normalize_prompt(const std::string& prompt,
                                                         size_t base_id,
                                                         const std::vector<EncodedImage>& images) const {
    auto [unified_prompt, images_sequence] = normalize(prompt, NATIVE_TAG, NATIVE_TAG, base_id, images.size());

    const size_t merge_length = m_merge_size * m_merge_size;
    for (size_t new_image_id : images_sequence) {
        const auto& image = images.at(new_image_id - base_id);
        size_t grid_h = image.resized_source_size.height;
        size_t grid_w = image.resized_source_size.width;
        size_t num_image_pad_tokens = grid_h * grid_w / merge_length;

        std::string expanded_tag;
        expanded_tag.reserve(m_vlm_config.vision_start_token.length() +
                             m_vlm_config.image_pad_token.length() * num_image_pad_tokens +
                             m_vlm_config.vision_end_token.length());
        expanded_tag.append(m_vlm_config.vision_start_token);
        for (size_t i = 0; i < num_image_pad_tokens; ++i) {
            expanded_tag.append(m_vlm_config.image_pad_token);
        }
        expanded_tag.append(m_vlm_config.vision_end_token);

        auto pos = unified_prompt.find(NATIVE_TAG);
        OPENVINO_ASSERT(pos != std::string::npos, "Expected image tag not found while normalizing Youtu-VL prompt.");
        unified_prompt.replace(pos, NATIVE_TAG.length(), expanded_tag);
    }

    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderYoutuVL::get_inputs_embeds(const std::string& unified_prompt,
                                                    const std::vector<ov::genai::EncodedImage>& images,
                                                    ov::genai::VLMPerfMetrics& metrics,
                                                    bool recalculate_merged_embeddings,
                                                    const std::vector<size_t>& images_sequence) {
    ov::Tensor input_ids = get_encoded_input_ids(unified_prompt, metrics);
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = m_embedding->infer(req, input_ids);

    ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
    std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());

    if (images.empty()) {
        return inputs_embeds;
    }

    // Concatenate merged embeddings for all images in prompt order.
    std::vector<ov::Tensor> merged_per_image;
    merged_per_image.reserve(images_sequence.size());
    size_t total_merged = 0;
    for (size_t new_image_id : images_sequence) {
        const auto& image = images.at(new_image_id);
        size_t grid_h = image.resized_source_size.height;
        size_t grid_w = image.resized_source_size.width;
        ov::Tensor merged = run_image_embeddings_merger(image.resized_source, grid_h, grid_w);
        total_merged += merged.get_shape().at(0);
        merged_per_image.push_back(std::move(merged));
    }

    const ov::Shape embeds_shape = inputs_embeds.get_shape();
    size_t seq_length = embeds_shape.at(1);
    size_t hidden_size = embeds_shape.at(2);
    OPENVINO_ASSERT(!merged_per_image.empty() && merged_per_image.front().get_shape().at(1) == hidden_size,
                    "Youtu-VL merged image embedding size does not match language model hidden size.");

    const int64_t* input_ids_data = input_ids.data<const int64_t>();
    float* embeds_data = inputs_embeds.data<float>();

    size_t image_idx = 0;
    size_t within_image_idx = 0;
    size_t scattered = 0;
    for (size_t seq_idx = 0; seq_idx < seq_length; ++seq_idx) {
        if (input_ids_data[seq_idx] != m_image_pad_token_id) {
            continue;
        }
        OPENVINO_ASSERT(image_idx < merged_per_image.size(),
                        "More image pad tokens than available image embeddings in Youtu-VL prompt.");
        const ov::Tensor& merged = merged_per_image[image_idx];
        const float* merged_data = merged.data<const float>();
        std::copy_n(merged_data + within_image_idx * hidden_size, hidden_size, embeds_data + seq_idx * hidden_size);
        ++within_image_idx;
        ++scattered;
        if (within_image_idx == merged.get_shape().at(0)) {
            within_image_idx = 0;
            ++image_idx;
        }
    }

    OPENVINO_ASSERT(scattered == total_merged && image_idx == merged_per_image.size(),
                    "Youtu-VL image pad token count does not match the number of image embeddings.");

    return inputs_embeds;
}

}  // namespace ov::genai
