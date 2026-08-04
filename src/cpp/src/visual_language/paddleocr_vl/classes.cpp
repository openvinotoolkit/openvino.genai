// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "visual_language/paddleocr_vl/classes.hpp"

#include <fstream>
#include <cmath>

#include "visual_language/clip.hpp"

#include "utils.hpp"
#include "json_utils.hpp"

namespace ov::genai {

namespace {

// SigLIP-variant vision tower uses a fixed rope theta for the 2D rope over (h, w).
constexpr float VISION_ROPE_THETA = 10000.0f;

// Read a subset of vision_config from config.json needed for preprocessing/rope.
struct PaddleOCRVisionConfig {
    size_t hidden_size = 1152;
    size_t num_attention_heads = 16;
    size_t image_size = 384;
    size_t patch_size = 14;
    size_t spatial_merge_size = 2;
};

PaddleOCRVisionConfig read_vision_config(const std::filesystem::path& config_dir) {
    PaddleOCRVisionConfig cfg;
    auto config_path = config_dir / "config.json";
    std::ifstream stream(config_path);
    if (!stream.is_open()) {
        return cfg;
    }
    nlohmann::json parsed = nlohmann::json::parse(stream);
    using ov::genai::utils::read_json_param;
    read_json_param(parsed, "vision_config.hidden_size", cfg.hidden_size);
    read_json_param(parsed, "vision_config.num_attention_heads", cfg.num_attention_heads);
    read_json_param(parsed, "vision_config.image_size", cfg.image_size);
    read_json_param(parsed, "vision_config.patch_size", cfg.patch_size);
    read_json_param(parsed, "vision_config.spatial_merge_size", cfg.spatial_merge_size);
    return cfg;
}

// smart_resize matching image_processing_paddleocr_vl.smart_resize:
// factor = patch_size * merge_size. Result height/width are multiples of factor,
// with total pixels bounded by [min_pixels, max_pixels], aspect ratio preserved.
ImageSize paddleocr_smart_resize(size_t height, size_t width, size_t factor, size_t min_pixels, size_t max_pixels) {
    OPENVINO_ASSERT(std::max(height, width) / std::min(height, width) <= 200,
                    "PaddleOCR-VL: absolute aspect ratio must be smaller than 200");
    auto round_to = [factor](double v) -> size_t {
        return static_cast<size_t>(std::llround(v / static_cast<double>(factor))) * factor;
    };
    size_t h_bar = round_to(static_cast<double>(height));
    size_t w_bar = round_to(static_cast<double>(width));
    if (h_bar * w_bar > max_pixels) {
        double beta = std::sqrt(static_cast<double>(height) * width / static_cast<double>(max_pixels));
        h_bar = static_cast<size_t>(std::floor(height / beta / factor)) * factor;
        w_bar = static_cast<size_t>(std::floor(width / beta / factor)) * factor;
    } else if (h_bar * w_bar < min_pixels) {
        double beta = std::sqrt(static_cast<double>(min_pixels) / (static_cast<double>(height) * width));
        h_bar = static_cast<size_t>(std::ceil(height * beta / factor)) * factor;
        w_bar = static_cast<size_t>(std::ceil(width * beta / factor)) * factor;
    }
    return ImageSize{h_bar, w_bar};
}

// Bilinear interpolation matrix used to resize the learned position-embedding
// grid from (num_positions_side x num_positions_side) to (out_size). Matches
// _OVPaddleOCRVLForCausalLM._bilinear_matrix (align_corners=False).
ov::Tensor bilinear_matrix(size_t out_size, size_t in_size) {
    ov::Tensor m(ov::element::f32, ov::Shape{out_size, in_size});
    float* data = m.data<float>();
    std::fill_n(data, m.get_size(), 0.0f);
    double scale = static_cast<double>(in_size) / static_cast<double>(out_size);
    for (size_t i = 0; i < out_size; ++i) {
        double src = std::max(0.0, (i + 0.5) * scale - 0.5);
        size_t lo = static_cast<size_t>(std::floor(src));
        size_t hi = std::min(lo + 1, in_size - 1);
        double frac = src - static_cast<double>(lo);
        data[i * in_size + lo] += static_cast<float>(1.0 - frac);
        data[i * in_size + hi] += static_cast<float>(frac);
    }
    return m;
}

// 2D rope cos/sin over (h, w) grid positions for a single image, matching
// _OVPaddleOCRVLForCausalLM._vision_rope (t == 1 for images).
std::pair<ov::Tensor, ov::Tensor> vision_rope(size_t h, size_t w, size_t vision_head_dim) {
    size_t num = h * w;
    size_t dim = vision_head_dim / 2;   // e.g. 36
    size_t half = dim / 2;              // e.g. 18 freqs
    std::vector<float> inv_freq(half);
    for (size_t i = 0; i < half; ++i) {
        inv_freq[i] = 1.0f / std::pow(VISION_ROPE_THETA, static_cast<float>(2 * i) / static_cast<float>(dim));
    }
    size_t max_grid = std::max(h, w);
    // freqs[pos][k] = pos * inv_freq[k]
    // rope_emb per pid = concat(freqs[hpos], freqs[wpos]) -> [half*2 = dim] then repeated x2 -> [vision_head_dim]
    ov::Tensor cos_t(ov::element::f32, ov::Shape{num, vision_head_dim});
    ov::Tensor sin_t(ov::element::f32, ov::Shape{num, vision_head_dim});
    float* cos_data = cos_t.data<float>();
    float* sin_data = sin_t.data<float>();
    for (size_t idx = 0; idx < num; ++idx) {
        size_t hpos = idx / w;
        size_t wpos = idx % w;
        // First `dim` entries: [hpos freqs (half) | wpos freqs (half)], then repeated to fill vision_head_dim.
        std::vector<float> base(dim);
        for (size_t k = 0; k < half; ++k) {
            base[k] = static_cast<float>(hpos) * inv_freq[k];
            base[half + k] = static_cast<float>(wpos) * inv_freq[k];
        }
        for (size_t d = 0; d < vision_head_dim; ++d) {
            float ang = base[d % dim];
            cos_data[idx * vision_head_dim + d] = std::cos(ang);
            sin_data[idx * vision_head_dim + d] = std::sin(ang);
        }
    }
    return {cos_t, sin_t};
}

// 2x2 spatial-merge reorder index matching _OVPaddleOCRVLForCausalLM._merge_index:
// idx = arange(t*h*w).reshape(t, h/m, m, w/m, m).permute(0,1,3,2,4).reshape(-1)
ov::Tensor merge_index(size_t t, size_t h, size_t w, size_t m) {
    size_t n = t * h * w;
    ov::Tensor out(ov::element::i64, ov::Shape{n});
    int64_t* data = out.data<int64_t>();
    size_t hb = h / m;
    size_t wb = w / m;
    size_t pos = 0;
    for (size_t tt = 0; tt < t; ++tt) {
        for (size_t i = 0; i < hb; ++i) {
            for (size_t j = 0; j < wb; ++j) {
                for (size_t p1 = 0; p1 < m; ++p1) {
                    for (size_t p2 = 0; p2 < m; ++p2) {
                        size_t hh = i * m + p1;
                        size_t ww = j * m + p2;
                        data[pos++] = static_cast<int64_t>(tt * h * w + hh * w + ww);
                    }
                }
            }
        }
    }
    return out;
}

} // namespace

VisionEncoderPaddleOCRVL::VisionEncoderPaddleOCRVL(const std::filesystem::path& model_dir,
                                                   const std::string& device,
                                                   const ov::AnyMap properties)
    : VisionEncoder(model_dir, device, properties) {
    auto vcfg = read_vision_config(model_dir);
    m_vision_head_dim = vcfg.hidden_size / vcfg.num_attention_heads;
    m_num_positions_side = vcfg.image_size / vcfg.patch_size;
    m_num_positions = m_num_positions_side * m_num_positions_side;

    auto merger_model = utils::singleton_core().read_model(model_dir / "openvino_vision_embeddings_merger_model.xml");
    init_merger(merger_model, device, properties);
}

VisionEncoderPaddleOCRVL::VisionEncoderPaddleOCRVL(const ModelsMap& models_map,
                                                   const std::filesystem::path& config_dir_path,
                                                   const std::string& device,
                                                   const ov::AnyMap properties)
    : VisionEncoder(models_map, config_dir_path, device, properties) {
    auto vcfg = read_vision_config(config_dir_path);
    m_vision_head_dim = vcfg.hidden_size / vcfg.num_attention_heads;
    m_num_positions_side = vcfg.image_size / vcfg.patch_size;
    m_num_positions = m_num_positions_side * m_num_positions_side;

    const auto& [merger_model_str, merger_weights] = utils::get_model_weights_pair(models_map, "vision_embeddings_merger");
    auto merger_model = utils::singleton_core().read_model(merger_model_str, merger_weights);
    init_merger(merger_model, device, properties);
}

void VisionEncoderPaddleOCRVL::init_merger(const std::shared_ptr<ov::Model>& merger_model,
                                           const std::string& device,
                                           const ov::AnyMap& properties) {
    auto compiled_model = utils::singleton_core().compile_model(
        merger_model, device, utils::get_model_properties(properties, "vision_embeddings_merger", device));
    ov::genai::utils::print_compiled_model_properties(compiled_model, "VLM vision embeddings merger model");
    m_ireq_queue_merger = std::make_unique<CircularBufferQueue<ov::InferRequest>>(
        compiled_model.get_property(ov::optimal_number_of_infer_requests),
        [&compiled_model]() -> ov::InferRequest {
            return compiled_model.create_infer_request();
        });
}

EncodedImage VisionEncoderPaddleOCRVL::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    const ProcessorConfig config = ProcessorConfig::from_any_map(config_map, m_processor_config);

    ov::Shape image_shape = image.get_shape();  // [1, H, W, C]
    OPENVINO_ASSERT(image_shape.size() == 4 && image_shape[3] == 3,
                    "PaddleOCR-VL vision encoder expects a single [1, H, W, 3] image tensor.");
    size_t orig_h = image_shape[1];
    size_t orig_w = image_shape[2];

    const size_t factor = config.patch_size * config.merge_size;
    ImageSize target = paddleocr_smart_resize(orig_h, orig_w, factor, config.min_pixels, config.max_pixels);

    // Resize (bicubic), rescale to [0,1] and normalize with mean/std.
    clip_image_u8 input_image = tensor_to_clip_image_u8(image);
    clip_image_u8 resized_image;
    bicubic_resize(input_image, resized_image, static_cast<int>(target.width), static_cast<int>(target.height));

    clip_ctx ctx;
    std::copy(config.image_mean.begin(), config.image_mean.end(), ctx.image_mean);
    std::copy(config.image_std.begin(), config.image_std.end(), ctx.image_std);
    clip_image_f32 normalized_image = clip_image_preprocess(ctx, resized_image);  // CHW, normalized

    size_t grid_h = target.height / config.patch_size;
    size_t grid_w = target.width / config.patch_size;
    size_t grid_t = 1;
    size_t num_patches = grid_t * grid_h * grid_w;
    size_t patch_size = config.patch_size;

    // Build pixel_values [num_patches, 3, patch_size, patch_size] from the CHW
    // normalized image by extracting patches in row-major (h, w) grid order.
    ov::Tensor pixel_values(ov::element::f32, ov::Shape{num_patches, 3, patch_size, patch_size});
    float* pv = pixel_values.data<float>();
    const float* src = normalized_image.buf.data();  // layout: [C, H, W]
    size_t H = target.height;
    size_t W = target.width;
    size_t patch_idx = 0;
    for (size_t gh = 0; gh < grid_h; ++gh) {
        for (size_t gw = 0; gw < grid_w; ++gw) {
            for (size_t c = 0; c < 3; ++c) {
                for (size_t py = 0; py < patch_size; ++py) {
                    for (size_t px = 0; px < patch_size; ++px) {
                        size_t iy = gh * patch_size + py;
                        size_t ix = gw * patch_size + px;
                        float v = src[c * H * W + iy * W + ix];
                        pv[((patch_idx * 3 + c) * patch_size + py) * patch_size + px] = v;
                    }
                }
            }
            ++patch_idx;
        }
    }

    // interp_h [grid_h, num_positions_side], interp_w [grid_w, num_positions_side]
    ov::Tensor interp_h = bilinear_matrix(grid_h, m_num_positions_side);
    ov::Tensor interp_w = bilinear_matrix(grid_w, m_num_positions_side);

    // Stage 1: vision embeddings -> [num_patches, 1152]
    ov::Tensor patch_hidden;
    {
        CircularBufferQueueElementGuard<ov::InferRequest> guard(m_ireq_queue_vision_encoder.get());
        ov::InferRequest& enc = guard.get();
        enc.set_tensor("pixel_values", pixel_values);
        enc.set_tensor("interp_h", interp_h);
        enc.set_tensor("interp_w", interp_w);
        enc.infer();
        const ov::Tensor& out = enc.get_output_tensor();
        patch_hidden = ov::Tensor(out.get_element_type(), out.get_shape());
        out.copy_to(patch_hidden);
    }

    // Stage 2: merger (SigLIP encoder + 2x2 projector) -> [num_patches / merge^2, 1024]
    auto [rope_cos, rope_sin] = vision_rope(grid_h, grid_w, m_vision_head_dim);
    ov::Tensor attention_mask(ov::element::f32, ov::Shape{1, num_patches, num_patches});
    std::fill_n(attention_mask.data<float>(), attention_mask.get_size(), 0.0f);
    ov::Tensor midx = merge_index(grid_t, grid_h, grid_w, config.merge_size);

    EncodedImage encoded_image;
    {
        CircularBufferQueueElementGuard<ov::InferRequest> guard(m_ireq_queue_merger.get());
        ov::InferRequest& merger = guard.get();
        merger.set_tensor("hidden_states", patch_hidden);
        merger.set_tensor("attention_mask", attention_mask);
        merger.set_tensor("rope_emb_cos", rope_cos);
        merger.set_tensor("rope_emb_sin", rope_sin);
        merger.set_tensor("merge_index", midx);
        merger.infer();
        const ov::Tensor& out = merger.get_output_tensor();  // [num_patches / merge^2, hidden]
        encoded_image.resized_source = ov::Tensor(out.get_element_type(), out.get_shape());
        out.copy_to(encoded_image.resized_source);
    }

    encoded_image.resized_source_size = ImageSize{grid_h, grid_w};
    encoded_image.original_image_size = ImageSize{orig_h, orig_w};
    encoded_image.num_image_tokens = num_patches / (config.merge_size * config.merge_size);
    return encoded_image;
}

// ------------------------------- InputsEmbedder ------------------------------

InputsEmbedderPaddleOCRVL::InputsEmbedderPaddleOCRVL(
    const VLMConfig& vlm_config,
    const std::filesystem::path& model_dir,
    const Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap device_config)
    : InputsEmbedderQwen2VL(vlm_config, model_dir, tokenizer, device, device_config) {
    init_paddleocr_tokens();
}

InputsEmbedderPaddleOCRVL::InputsEmbedderPaddleOCRVL(
    const VLMConfig& vlm_config,
    const ModelsMap& models_map,
    const Tokenizer& tokenizer,
    const std::filesystem::path& config_dir_path,
    const std::string& device,
    const ov::AnyMap device_config)
    : InputsEmbedderQwen2VL(vlm_config, models_map, tokenizer, config_dir_path, device, device_config) {
    init_paddleocr_tokens();
}

void InputsEmbedderPaddleOCRVL::init_paddleocr_tokens() {
    // PaddleOCR-VL placeholder token strings differ from Qwen2-VL. Recover the
    // exact token ids so create_position_ids() (inherited) can locate vision spans.
    auto encoded = m_tokenizer.encode(std::string("<|IMAGE_START|>") + "<|IMAGE_END|>" + "<|IMAGE_PLACEHOLDER|>",
                                      ov::genai::add_special_tokens(false));
    const int64_t* ids = encoded.input_ids.data<int64_t>();
    m_vision_token_ids["vision_start"] = ids[0];
    m_vision_token_ids["vision_end"] = ids[1];
    m_vision_token_ids["image_pad"] = ids[2];
    // No video support: keep video_pad distinct so it never matches a real token.
    m_vision_token_ids["video_pad"] = -1;
    // merge_length = merge_size^2 (inherited m_merge_length already set from processor config).
}

NormalizedPrompt InputsEmbedderPaddleOCRVL::normalize_prompt(
    const std::string& prompt,
    size_t image_base_id,
    size_t /*video_base_id*/,
    const std::vector<EncodedImage>& images,
    const std::vector<EncodedVideo>& /*videos*/) const {
    auto [unified_prompt, images_sequence] =
        normalize(prompt, PADDLEOCR_NATIVE_TAG, PADDLEOCR_NATIVE_TAG, image_base_id, images.size(), VisionType::IMAGE);

    const std::string vision_start = "<|IMAGE_START|>";
    const std::string vision_end = "<|IMAGE_END|>";
    const std::string image_pad = "<|IMAGE_PLACEHOLDER|>";

    for (size_t new_image_id : images_sequence) {
        const size_t num_image_pad_tokens = images.at(new_image_id - image_base_id).num_image_tokens;

        std::string expanded_tag;
        expanded_tag.reserve(vision_start.length() + image_pad.length() * num_image_pad_tokens + vision_end.length());
        expanded_tag.append(vision_start);
        for (size_t i = 0; i < num_image_pad_tokens; ++i) {
            expanded_tag.append(image_pad);
        }
        expanded_tag.append(vision_end);

        unified_prompt.replace(unified_prompt.find(PADDLEOCR_NATIVE_TAG), PADDLEOCR_NATIVE_TAG.length(), expanded_tag);
    }

    return {std::move(unified_prompt), std::move(images_sequence), {}};
}

std::vector<ov::genai::EncodedImage> InputsEmbedderPaddleOCRVL::encode_images(const std::vector<ov::Tensor>& images) {
    std::vector<EncodedImage> embeds;
    std::vector<ov::Tensor> single_images = to_single_image_tensors(images);
    embeds.reserve(single_images.size());
    for (ov::Tensor& image : single_images) {
        cvt_to_3_chn_image(image);
        embeds.emplace_back(m_vision_encoder->encode(image));
    }
    return embeds;
}

ov::Tensor InputsEmbedderPaddleOCRVL::get_inputs_embeds(const std::string& prompt,
                                                        const std::vector<ov::genai::EncodedImage>& images,
                                                        ov::genai::VLMPerfMetrics& metrics,
                                                        bool recalculate_merged_embeddings,
                                                        const std::vector<size_t>& image_sequence) {
    return get_inputs_embeds(prompt, images, {}, metrics, recalculate_merged_embeddings, image_sequence, {});
}

ov::Tensor InputsEmbedderPaddleOCRVL::get_inputs_embeds(const std::string& unified_prompt,
                                                        const std::vector<ov::genai::EncodedImage>& images,
                                                        const std::vector<ov::genai::EncodedVideo>& videos,
                                                        ov::genai::VLMPerfMetrics& metrics,
                                                        bool /*recalculate_merged_embeddings*/,
                                                        const std::vector<size_t>& image_sequence,
                                                        const std::vector<size_t>& /*videos_sequence*/,
                                                        const std::vector<std::pair<std::size_t, std::size_t>>& history_vision_count) {
    OPENVINO_ASSERT(videos.empty(), "PaddleOCR-VL does not support video input.");

    std::vector<std::array<size_t, 3>> images_grid_thw;
    images_grid_thw.reserve(images.size());
    for (const auto& encoded_image : images) {
        size_t grid_t = 1;
        size_t grid_h = encoded_image.resized_source_size.height;
        size_t grid_w = encoded_image.resized_source_size.width;
        images_grid_thw.push_back({grid_t, grid_h, grid_w});
    }

    ov::Tensor input_ids = get_encoded_input_ids(unified_prompt, metrics);
    CircularBufferQueueElementGuard<EmbeddingsRequest> embeddings_request_guard(m_embedding->get_request_queue().get());
    EmbeddingsRequest& req = embeddings_request_guard.get();
    ov::Tensor text_embeds = m_embedding->infer(req, input_ids);

    int64_t vision_start_token_id = m_vision_token_ids["vision_start"];
    int64_t image_pad_token_id = m_vision_token_ids["image_pad"];

    // Reuse Qwen2-VL 3D mrope position id logic (image t=1, second_per_grid_t=0).
    std::tie(m_position_ids, m_rope_delta) = create_position_ids(
        input_ids,
        images_grid_thw,
        image_sequence,
        0,
        /*videos_grid_thw=*/{},
        /*videos_sequence=*/{},
        0,
        vision_start_token_id,
        history_vision_count);

    if (images.empty()) {
        ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
        std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
        return inputs_embeds;
    }

    // Image embeddings are already merged by the vision encoder. Concatenate them
    // in prompt order and splice at <|IMAGE_PLACEHOLDER|> positions.
    size_t total_rows = 0;
    size_t hidden = text_embeds.get_shape().at(2);
    for (size_t idx : image_sequence) {
        const auto& src = images.at(idx).resized_source;
        OPENVINO_ASSERT(src.get_shape().size() == 2 && src.get_shape().at(1) == hidden,
                        "PaddleOCR-VL: unexpected merged image embedding shape.");
        total_rows += src.get_shape().at(0);
    }

    ov::Tensor image_embeds(text_embeds.get_element_type(), ov::Shape{total_rows, hidden});
    {
        uint8_t* dst = reinterpret_cast<uint8_t*>(image_embeds.data());
        size_t offset = 0;
        for (size_t idx : image_sequence) {
            const auto& src = images.at(idx).resized_source;
            std::memcpy(dst + offset, src.data(), src.get_byte_size());
            offset += src.get_byte_size();
        }
    }

    ov::Tensor merged(text_embeds.get_element_type(), text_embeds.get_shape());
    std::memcpy(merged.data(), text_embeds.data(), text_embeds.get_byte_size());

    auto shape = text_embeds.get_shape();
    size_t batch = shape.at(0);
    size_t seq_len = shape.at(1);
    const int64_t* ids_data = input_ids.data<int64_t>();
    float* merged_data = merged.data<float>();
    const float* image_data = image_embeds.data<float>();
    size_t image_row = 0;
    for (size_t b = 0; b < batch; ++b) {
        for (size_t s = 0; s < seq_len; ++s) {
            size_t flat = b * seq_len + s;
            if (ids_data[flat] == image_pad_token_id) {
                OPENVINO_ASSERT(image_row < total_rows,
                                "PaddleOCR-VL: more image placeholder tokens than image embedding rows.");
                std::copy_n(image_data + image_row * hidden, hidden, merged_data + flat * hidden);
                ++image_row;
            }
        }
    }
    OPENVINO_ASSERT(image_row == total_rows,
                    "PaddleOCR-VL: image embedding rows (", total_rows,
                    ") do not match image placeholder tokens (", image_row, ") in the prompt.");

    return merged;
}

} // namespace ov::genai
