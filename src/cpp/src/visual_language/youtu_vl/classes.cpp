// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/youtu_vl/classes.hpp"

#include <cmath>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <fstream>

#include "visual_language/clip.hpp"
#include "continuous_batching/timer.hpp"
#include "json_utils.hpp"
#include "utils.hpp"

namespace ov::genai {

namespace {

std::vector<int64_t> tensor_to_i64_vector(const ov::Tensor& tensor) {
    OPENVINO_ASSERT(tensor.get_element_type() == ov::element::i64,
        "YoutuVL token ids are expected to be i64, got ", tensor.get_element_type());
    const int64_t* data = tensor.data<const int64_t>();
    return {data, data + tensor.get_size()};
}

/// @brief Compute target image dimensions (multiples of patch_size*2) such that
/// the total number of patches does not exceed max_num_patches.
/// Mirrors Siglip2ImageProcessorFast.get_image_size_for_patches() in Python.
std::pair<size_t, size_t> get_youtu_vl_image_size(
    size_t img_height,
    size_t img_width,
    size_t patch_size,
    size_t max_num_patches)
{
    const size_t block = patch_size * 2;  // 32 for patch_size=16

    auto get_scaled_size = [block](float scale, size_t orig_size) -> size_t {
        float scaled = orig_size * scale;
        size_t ceil_val = static_cast<size_t>(std::ceil(scaled / block)) * block;
        return std::max(ceil_val, block);
    };

    float scale = 1.0f;
    size_t target_h, target_w;
    while (true) {
        target_h = get_scaled_size(scale, img_height);
        target_w = get_scaled_size(scale, img_width);
        size_t num_patches = (target_h / patch_size) * (target_w / patch_size);
        if (num_patches <= max_num_patches) {
            break;
        }
        scale -= 0.02f;
        OPENVINO_ASSERT(scale > 0.0f,
            "Cannot fit image into max_num_patches=", max_num_patches,
            " even with scale approaching 0.");
    }
    return {target_h, target_w};
}

/// @brief Convert a normalized CHW image (clip_image_f32) into the flat patch
/// tensor expected by openvino_vision_embeddings_model.
///
/// The Python reference implementation (convert_image_to_patches with merge_size=2):
///   image shape: (C, H, W)
///   reshape -> (C, H/32, 2, 16, W/32, 2, 16)
///   permute(1,4,2,5,3,6,0) -> (H/32, W/32, 2, 2, 16, 16, C)
///   reshape -> (N_ph * N_pw, C*16*16)   where N_ph=H/16, N_pw=W/16
///
/// The effective patch ordering is:
///   for gh in [0, N_ph/2):
///     for gw in [0, N_pw/2):
///       for mh in {0,1}:
///         for mw in {0,1}:
///           patch at image row (gh*2+mh), col (gw*2+mw)
///           packed as (py, px, c) = HWC within the 16x16 block
///
/// @param img         Normalized CHW float image (clip_image_f32.buf layout).
/// @param H           Image height in pixels (multiple of 32).
/// @param W           Image width in pixels (multiple of 32).
/// @param patch_size  Patch size (16).
/// @return            Tensor of shape [1, N_ph * N_pw, patch_size * patch_size * C].
ov::Tensor make_pixel_values(
    const float* chw_data,
    size_t H,
    size_t W,
    size_t patch_size = 16)
{
    const size_t C = 3;
    const size_t N_ph = H / patch_size;
    const size_t N_pw = W / patch_size;
    const size_t N_patches = N_ph * N_pw;
    const size_t patch_elems = patch_size * patch_size * C;  // 768

    // chw_data layout: channel c, row y, col x → chw_data[c*H*W + y*W + x]

    ov::Tensor pixel_values(ov::element::f32, {1, N_patches, patch_elems});
    float* out = pixel_values.data<float>();

    // Iterate in interleaved (group) order to match the Python reshape/permute logic.
    for (size_t gh = 0; gh < N_ph / 2; ++gh) {
        for (size_t gw = 0; gw < N_pw / 2; ++gw) {
            for (size_t mh = 0; mh < 2; ++mh) {
                for (size_t mw = 0; mw < 2; ++mw) {
                    // Patch index in the output tensor (row-major over groups then offsets)
                    size_t patch_idx = gh * (N_pw / 2 * 4) + gw * 4 + mh * 2 + mw;

                    // Top-left pixel of this 16x16 patch in the image
                    size_t row_start = (gh * 2 + mh) * patch_size;
                    size_t col_start = (gw * 2 + mw) * patch_size;

                    float* dst = out + patch_idx * patch_elems;
                    size_t elem = 0;
                    // Pack as (py, px, c) — HWC within the patch
                    for (size_t py = 0; py < patch_size; ++py) {
                        for (size_t px = 0; px < patch_size; ++px) {
                            for (size_t c = 0; c < C; ++c) {
                                size_t img_idx =
                                    c * H * W
                                    + (row_start + py) * W
                                    + (col_start + px);
                                dst[elem++] = chw_data[img_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    // Handle the case where N_ph or N_pw is odd (should not happen given resizing
    // to multiples of patch_size*2, but guard defensively by falling back to
    // raster order for the residual row/column).
    // In practice this branch is never taken.
    if (N_ph % 2 != 0 || N_pw % 2 != 0) {
        OPENVINO_THROW(
            "YoutuVL: expected image dimensions to be multiples of ",
            patch_size * 2, " (got H=", H, ", W=", W, ")");
    }

    return pixel_values;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// VisionEncoderYoutuVL::encode
// ---------------------------------------------------------------------------
EncodedImage VisionEncoderYoutuVL::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    CircularBufferQueueElementGuard<ov::InferRequest> infer_request_guard(
        this->m_ireq_queue_vision_encoder.get());
    ov::InferRequest& encoder = infer_request_guard.get();

    ProcessorConfig config = ProcessorConfig::from_any_map(config_map, m_processor_config);

    const size_t patch_size = config.patch_size;  // 16
    const size_t max_num_patches = config.max_num_patches;

    // 1. Convert input tensor (1CHW or CHW) to clip_image_u8
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);
    const size_t orig_h = static_cast<size_t>(input_image.ny);
    const size_t orig_w = static_cast<size_t>(input_image.nx);

    // 2. Compute target size (multiples of patch_size*2 = 32)
    auto [target_h, target_w] = get_youtu_vl_image_size(orig_h, orig_w, patch_size, max_num_patches);

    // 3. Resize (BILINEAR — corresponds to PIL resample=2)
    clip_image_u8 resized;
    bilinear_resize(input_image, resized, static_cast<int>(target_w), static_cast<int>(target_h));

    // 4. Normalize and convert to CHW float
    //    YoutuVL uses image_mean = image_std = 0.5  →  (pixel/255 - 0.5) / 0.5
    //    clip_ctx_double expects raw-pixel mean/std, so we scale:
    //      raw_mean = 0.5 * 255 = 127.5,  raw_std = 0.5 * 255 = 127.5
    clip_ctx_double norm_ctx;
    norm_ctx.image_mean[0] = norm_ctx.image_mean[1] = norm_ctx.image_mean[2] =
        static_cast<double>(config.image_mean[0]) * 255.0;
    norm_ctx.image_std[0] = norm_ctx.image_std[1] = norm_ctx.image_std[2] =
        static_cast<double>(config.image_std[0]) * 255.0;

    clip_image_f32 normalized = normalize_and_convert_to_chw(resized, norm_ctx);

    // 5. Build pixel_values tensor [1, N_ph*N_pw, 768]
    const size_t N_ph = target_h / patch_size;
    const size_t N_pw = target_w / patch_size;

    ov::Tensor pixel_values = make_pixel_values(normalized.buf.data(), target_h, target_w, patch_size);

    // 6. Build pixel_attention_mask [1, N_ph*N_pw] — all ones (no padding for single image)
    const size_t N_patches = N_ph * N_pw;
    ov::Tensor pixel_attention_mask(ov::element::i64, {1, N_patches});
    std::fill_n(pixel_attention_mask.data<int64_t>(), N_patches, 1);

    // 7. Build spatial_shapes [1, 2] — [[N_ph, N_pw]]
    ov::Tensor spatial_shapes(ov::element::i64, {1, 2});
    spatial_shapes.data<int64_t>()[0] = static_cast<int64_t>(N_ph);
    spatial_shapes.data<int64_t>()[1] = static_cast<int64_t>(N_pw);

    // 8. Run vision model
    encoder.set_tensor("pixel_values", pixel_values);
    encoder.set_tensor("pixel_attention_mask", pixel_attention_mask);
    encoder.set_tensor("spatial_shapes", spatial_shapes);
    encoder.infer();

    // 9. Collect output — shape [N_merged_tokens, hidden_size]
    //    Reshape to [1, N_merged_tokens, hidden_size] for compatibility with
    //    merge_text_and_image_embeddings_llava (which reads shape().at(1) = N_tokens).
    const ov::Tensor& raw_output = encoder.get_output_tensor();
    const ov::Shape raw_shape = raw_output.get_shape();  // [N_merged, hidden]
    OPENVINO_ASSERT(raw_shape.size() == 2,
        "YoutuVL vision encoder: unexpected output rank ", raw_shape.size());

    const size_t N_merged = raw_shape[0];
    const size_t hidden   = raw_shape[1];

    ov::Tensor image_features(raw_output.get_element_type(), {1, N_merged, hidden});
    std::memcpy(image_features.data(), raw_output.data(), raw_output.get_byte_size());

    EncodedImage result;
    result.resized_source      = std::move(image_features);
    result.resized_source_size = {N_ph, N_pw};
    result.num_image_tokens    = N_merged;
    return result;
}

// ---------------------------------------------------------------------------
// InputsEmbedderYoutuVL
// ---------------------------------------------------------------------------

InputsEmbedderYoutuVL::InputsEmbedderYoutuVL(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const std::string& device,
    const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, model_dir, device, device_config) {
    load_special_token_ids(model_dir);
}

InputsEmbedderYoutuVL::InputsEmbedderYoutuVL(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config)
    : IInputsEmbedder(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {
    load_special_token_ids(config_dir_path);
}

void InputsEmbedderYoutuVL::load_special_token_ids(const std::filesystem::path& config_dir_path) {
    std::ifstream stream(config_dir_path / "tokenizer.json");
    OPENVINO_ASSERT(stream.is_open(), "Failed to open tokenizer.json for YoutuVL at ", config_dir_path);
    nlohmann::json tokenizer_json = nlohmann::json::parse(stream);

    const std::vector<std::string> required_tokens = {
        "<|begin_of_text|>",
        "<|end_of_text|>",
        m_vlm_config.vision_start_token,
        m_vlm_config.image_pad_token,
        m_vlm_config.vision_end_token,
    };

    for (const auto& token : required_tokens) {
        for (const auto& added_token : tokenizer_json.at("added_tokens")) {
            if (added_token.at("content").get<std::string>() == token) {
                m_special_token_ids[token] = added_token.at("id").get<int64_t>();
                break;
            }
        }
        OPENVINO_ASSERT(m_special_token_ids.count(token) != 0,
            "YoutuVL: tokenizer.json does not contain added token ", token);
    }
}

ov::Tensor InputsEmbedderYoutuVL::encode_prompt_with_special_token_ids(
    const std::string& prompt,
    ov::genai::VLMPerfMetrics& metrics)
{
    ManualTimer encode_timer("Encode");
    encode_timer.start();

    const std::vector<std::string> special_tokens = {
        "<|begin_of_text|>",
        "<|end_of_text|>",
        m_vlm_config.vision_start_token,
        m_vlm_config.image_pad_token,
        m_vlm_config.vision_end_token,
    };

    auto append_text = [this](std::vector<int64_t>& ids, const std::string& text) {
        if (text.empty()) {
            return;
        }
        ov::Tensor encoded = m_tokenizer.encode(text, ov::genai::add_special_tokens(false)).input_ids;
        std::vector<int64_t> text_ids = tensor_to_i64_vector(encoded);
        ids.insert(ids.end(), text_ids.begin(), text_ids.end());
    };

    std::vector<int64_t> ids;
    size_t pos = 0;
    while (pos < prompt.size()) {
        const std::string* matched_token = nullptr;
        for (const std::string& token : special_tokens) {
            if (prompt.compare(pos, token.size(), token) == 0) {
                matched_token = &token;
                break;
            }
        }

        if (matched_token != nullptr) {
            ids.push_back(m_special_token_ids.at(*matched_token));
            pos += matched_token->size();
            continue;
        }

        size_t next_special = std::string::npos;
        for (const std::string& token : special_tokens) {
            size_t token_pos = prompt.find(token, pos);
            next_special = std::min(next_special, token_pos);
        }
        if (next_special == std::string::npos) {
            append_text(ids, prompt.substr(pos));
            break;
        }
        append_text(ids, prompt.substr(pos, next_special - pos));
        pos = next_special;
    }

    encode_timer.end();
    metrics.raw_metrics.tokenization_durations.emplace_back(encode_timer.get_duration_microsec());

    ov::Tensor result(ov::element::i64, {1, ids.size()});
    std::copy(ids.begin(), ids.end(), result.data<int64_t>());
    return update_history(result);
}

std::vector<ov::genai::EncodedImage> InputsEmbedderYoutuVL::encode_images(
    const std::vector<ov::Tensor>& images)
{
    std::vector<EncodedImage> embeds;
    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    embeds.reserve(single_images.size());
    for (const ov::Tensor& img : single_images) {
        embeds.emplace_back(m_vision_encoder->encode(img, {}));
    }
    return embeds;
}

NormalizedPrompt InputsEmbedderYoutuVL::normalize_prompt(
    const std::string& prompt,
    size_t base_id,
    const std::vector<EncodedImage>& images) const
{
    // The YoutuVL chat template produces exactly one NATIVE_TAG per image:
    //   <|vision_start|><|image_pad|><|vision_end|>
    //
    // We need to expand the single <|image_pad|> to num_image_tokens repetitions.

    auto [unified_prompt, images_sequence] =
        normalize(prompt, NATIVE_TAG, NATIVE_TAG, base_id, images.size());

    for (size_t new_image_id : images_sequence) {
        size_t num_tokens = images.at(new_image_id - base_id).num_image_tokens;

        // Build: <|vision_start|> + N * <|image_pad|> + <|vision_end|>
        std::string expanded;
        expanded.reserve(m_vlm_config.vision_start_token.size()
                         + num_tokens * m_vlm_config.image_pad_token.size()
                         + m_vlm_config.vision_end_token.size());
        expanded += m_vlm_config.vision_start_token;
        for (size_t i = 0; i < num_tokens; ++i) {
            expanded += m_vlm_config.image_pad_token;
        }
        expanded += m_vlm_config.vision_end_token;

        size_t pos = unified_prompt.find(NATIVE_TAG);
        OPENVINO_ASSERT(pos != std::string::npos,
            "YoutuVL: could not find NATIVE_TAG in unified prompt");
        unified_prompt.replace(pos, NATIVE_TAG.size(), expanded);
    }

    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

ov::Tensor InputsEmbedderYoutuVL::get_inputs_embeds(
    const std::string& unified_prompt,
    const std::vector<ov::genai::EncodedImage>& images,
    ov::genai::VLMPerfMetrics& metrics,
    bool /*recalculate_merged_embeddings*/,
    const std::vector<size_t>& images_sequence)
{
    // Collect image embeddings in the order they appear in the sequence
    std::vector<ov::Tensor> image_embeds;
    image_embeds.reserve(images_sequence.size());
    for (size_t img_id : images_sequence) {
        image_embeds.push_back(images.at(img_id).resized_source);
    }

    // Tokenize the expanded prompt
    ov::Tensor input_ids = encode_prompt_with_special_token_ids(unified_prompt, metrics);

    // Get text embeddings
    CircularBufferQueueElementGuard<EmbeddingsRequest> emb_guard(
        m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = emb_guard.get();
    ov::Tensor text_embeds = m_embedding->infer(req, input_ids);

    if (images.empty()) {
        // Text-only path: return a copy (the tensor is owned by the infer request)
        ov::Tensor out(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(out.data(), text_embeds.data(), text_embeds.get_byte_size());
        return out;
    }

    int64_t image_pad_token_id = m_special_token_ids.at(m_vlm_config.image_pad_token);

    // Replace <|image_pad|> tokens with vision embeddings
    return utils::merge_text_and_image_embeddings_llava(
        input_ids, text_embeds, image_embeds, image_pad_token_id);
}

} // namespace ov::genai
