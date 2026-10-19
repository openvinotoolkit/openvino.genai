// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "classes.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "utils.hpp"
#include "visual_language/clip.hpp"

namespace ov::genai {
namespace {
constexpr const char* audio_tag = "<audio>";
constexpr const char* placeholder = "<|audio_start|>";
}  // namespace

EncodedImage VisionEncoderLFM2::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    const auto config = ProcessorConfig::from_any_map(config_map, m_processor_config);
    const auto source = tensor_to_clip_image_u8(image);
    const auto factor = config.patch_size * config.merge_size;
    const auto [height, width] = bounded_image_size(source.ny, source.nx, factor, config.min_pixels, config.max_pixels);
    const auto rounded = [&](int value) {
        return std::max(double(config.patch_size), std::nearbyint(double(value) / factor) * factor);
    };
    size_t rows = 0, columns = 0;
    if (rounded(source.ny) * rounded(source.nx) > double(config.max_pixels) * 2.) {
        std::vector<std::pair<size_t, size_t>> grids;
        for (size_t n = config.min_patches; n <= config.max_patches; ++n)
            for (size_t w = 1; w <= n; ++w)
                for (size_t h = 1; h <= n; ++h)
                    if (w * h >= config.min_patches && w * h <= config.max_patches &&
                        std::find(grids.begin(), grids.end(), std::pair<size_t, size_t>{w, h}) == grids.end())
                        grids.emplace_back(w, h);
        std::sort(grids.begin(), grids.end(), [](const auto& a, const auto& b) {
            return a.first * a.second < b.first * b.second;
        });
        float difference = std::numeric_limits<float>::max();
        for (const auto& [w, h] : grids) {
            const float delta = std::abs(float(source.nx) / source.ny - float(w) / h);
            if (delta < difference || (delta == difference && float(source.nx) * source.ny >
                                                                  .5f * config.tile_size * config.tile_size * w * h)) {
                difference = delta;
                columns = w;
                rows = h;
            }
        }
    }
    std::vector<clip_image_u8> crops;
    if (rows) {
        clip_image_u8 canvas;
        canvas = resize_and_pad_image(source, {int(columns * config.tile_size), int(rows * config.tile_size)}, 0, true);
        for (size_t row = 0; row < rows; ++row)
            for (size_t col = 0; col < columns; ++col) {
                clip_image_u8 crop;
                crop.nx = crop.ny = int(config.tile_size);
                crop.buf.resize(config.tile_size * config.tile_size * 3);
                for (size_t y = 0; y < config.tile_size; ++y)
                    std::copy_n(
                        canvas.buf.data() + ((row * config.tile_size + y) * canvas.nx + col * config.tile_size) * 3,
                        config.tile_size * 3,
                        crop.buf.data() + y * config.tile_size * 3);
                crops.push_back(std::move(crop));
            }
    }
    clip_image_u8 overview;
    bilinear_resize(source, overview, int(width), int(height));
    crops.push_back(std::move(overview));
    EncodedImage result;
    result.slices_grid = {rows, columns};
    result.original_image_size = {size_t(source.ny), size_t(source.nx)};
    result.resized_source_size = {height / config.patch_size, width / config.patch_size};
    clip_ctx_double normalization;
    for (size_t i = 0; i < 3; ++i) {
        normalization.image_mean[i] = config.image_mean[i] * 255.;
        normalization.image_std[i] = config.image_std[i] * 255.;
    }
    std::vector<ov::Tensor> outputs;
    CircularBufferQueueElementGuard<ov::InferRequest> guard(m_ireq_queue_vision_encoder.get());
    for (const auto& crop : crops) {
        const auto normalized = normalize_and_convert_to_chw(crop, normalization);
        ov::Tensor pixels(ov::element::f32,
                          {1, 3, size_t(crop.ny), size_t(crop.nx)},
                          const_cast<float*>(normalized.buf.data()));
        const size_t gh = crop.ny / config.patch_size, gw = crop.nx / config.patch_size;
        ov::Tensor indices(ov::element::i32, {1, 1, 1, gh * gw});
        size_t offset = 0;
        for (size_t y = 0; y < gh; y += config.merge_size)
            for (size_t x = 0; x < gw; x += config.merge_size)
                for (size_t dy = 0; dy < config.merge_size; ++dy)
                    for (size_t dx = 0; dx < config.merge_size; ++dx)
                        indices.data<int32_t>()[offset++] = int32_t((y + dy) * gw + x + dx);
        auto& request = guard.get();
        request.set_tensor("pixel_values", pixels);
        request.set_tensor("merge_indices", indices);
        request.infer();
        const auto features = request.get_output_tensor();
        ov::Tensor owned(features.get_element_type(), features.get_shape());
        features.copy_to(owned);
        outputs.push_back(owned);
        result.crop_sizes.push_back({features.get_shape()[1], 1});
        result.num_image_tokens += features.get_shape()[1];
    }
    result.resized_source = ov::Tensor(ov::element::f32, {1, result.num_image_tokens, outputs.front().get_shape()[2]});
    auto* target = result.resized_source.data<float>();
    for (const auto& output : outputs) {
        std::copy_n(output.data<const float>(), output.get_size(), target);
        target += output.get_size();
    }
    return result;
}

InputsEmbedderLFM2Vision::InputsEmbedderLFM2Vision(const VLMModels& models,
                                                   const Tokenizer& tokenizer,
                                                   const std::string& device,
                                                   const ov::AnyMap& properties)
    : IInputsEmbedder(models, tokenizer, device, properties) {
    m_image_placeholder = "<image>";
    auto ids = m_tokenizer.encode(m_image_placeholder, add_special_tokens(false)).input_ids;
    if (ids.get_size() != 1) {
        m_image_placeholder = "<|image|>";
        ids = m_tokenizer.encode(m_image_placeholder, add_special_tokens(false)).input_ids;
    }
    OPENVINO_ASSERT(ids.get_size() == 1, "LFM2 vision requires a single image placeholder");
    m_image_token_id = ids.data<const int64_t>()[0];
}

NormalizedPrompt InputsEmbedderLFM2Vision::normalize_prompt(const std::string& prompt,
                                                            size_t base,
                                                            const std::vector<EncodedImage>& images) const {
    auto [unified, sequence] = normalize(prompt, "<image>", "<image>", base, images.size());
    OPENVINO_ASSERT(images.empty() || prompt.find("<|image|>") == std::string::npos,
                    "LFM2 image placeholder is reserved");
    size_t offset = 0;
    for (const auto id : sequence) {
        const auto& image = images.at(id - base);
        std::string expanded = "<|image_start|>";
        for (size_t i = 0; i < image.crop_sizes.size(); ++i) {
            if (image.slices_grid.has_slices()) {
                if (i + 1 == image.crop_sizes.size())
                    expanded += "<|img_thumbnail|>";
                else
                    expanded += "<|img_row_" + std::to_string(i / image.slices_grid.cols + 1) + "_col_" +
                                std::to_string(i % image.slices_grid.cols + 1) + "|>";
            }
            for (size_t t = 0; t < image.crop_sizes[i].height; ++t)
                expanded += m_image_placeholder;
        }
        expanded += "<|image_end|>";
        const auto pos = unified.find("<image>", offset);
        OPENVINO_ASSERT(pos != std::string::npos, "Missing LFM2 image tag");
        unified.replace(pos, 7, expanded);
        offset = pos + expanded.size();
    }
    return {unified, sequence};
}

ov::Tensor InputsEmbedderLFM2Vision::get_inputs_embeds(const std::string& prompt,
                                                       const std::vector<EncodedImage>& images,
                                                       VLMPerfMetrics& metrics,
                                                       bool,
                                                       const std::vector<size_t>& sequence) {
    const auto ids = get_encoded_input_ids(prompt, metrics);
    CircularBufferQueueElementGuard<EmbeddingsRequest> guard(m_embedding->get_request_queue().get());
    auto text = get_text_embedding(guard.get(), ids, metrics);
    // Each crop owns its own placeholder run, so its features are merged as a separate chunk.
    std::vector<ov::Tensor> crops;
    for (const auto id : sequence) {
        const auto& image = images.at(id);
        auto* source = const_cast<float*>(image.resized_source.data<const float>());
        for (const auto& crop : image.crop_sizes) {
            crops.emplace_back(ov::element::f32, ov::Shape{1, crop.height, m_vlm_config.hidden_size}, source);
            source += crop.height * m_vlm_config.hidden_size;
        }
    }
    size_t expected = 0;
    for (const auto& crop : crops)
        expected += crop.get_shape()[1];
    const auto* tokens = ids.data<const int64_t>();
    OPENVINO_ASSERT(std::count(tokens, tokens + ids.get_size(), m_image_token_id) == expected,
                    "LFM2 image placeholder count differs from encoded features");
    return utils::merge_text_and_image_embeddings_llava(ids, text, crops, m_image_token_id);
}

InputsEmbedderLFM2Audio::InputsEmbedderLFM2Audio(const VLMModels& models,
                                                 const Tokenizer& tokenizer,
                                                 const std::string& device,
                                                 const ov::AnyMap& properties)
    : IInputsEmbedder(models, tokenizer, device, properties) {
    auto token = m_tokenizer.encode(placeholder, add_special_tokens(false)).input_ids;
    OPENVINO_ASSERT(token.get_size() == 1, "LFM2 audio requires a single placeholder token");
    m_audio_token_id = token.data<const int64_t>()[0];
}

std::vector<EncodedAudio> InputsEmbedderLFM2Audio::encode_audios(const std::vector<ov::Tensor>& audios) {
    return encode_audios_with_encoder(audios, false);
}

NormalizedPrompt InputsEmbedderLFM2Audio::normalize_prompt(const std::string& prompt,
                                                           size_t image_base_id,
                                                           size_t video_base_id,
                                                           size_t audio_base_id,
                                                           const std::vector<EncodedImage>& images,
                                                           const std::vector<EncodedVideo>& videos,
                                                           const std::vector<EncodedAudio>& audios) const {
    OPENVINO_ASSERT(images.empty() && videos.empty(), "LFM2 audio model accepts audio and text");
    OPENVINO_ASSERT(audios.empty() || prompt.find(placeholder) == std::string::npos,
                    "LFM2 audio input cannot contain the reserved embedding placeholder ",
                    placeholder);
    NormalizedPrompt result;
    std::tie(result.unified_prompt, result.images_sequence) =
        normalize(prompt, "<image>", "<image>", image_base_id, 0, ModalityType::IMAGE);
    std::tie(result.unified_prompt, result.videos_sequence) =
        normalize(result.unified_prompt, "<video>", "<video>", video_base_id, 0, ModalityType::VIDEO);
    std::tie(result.unified_prompt, result.audios_sequence) =
        normalize(result.unified_prompt, audio_tag, audio_tag, audio_base_id, audios.size(), ModalityType::AUDIO);
    size_t offset = 0;
    for (const auto id : result.audios_sequence) {
        const auto& audio = audios.at(id - audio_base_id);
        std::string expanded;
        for (size_t i = 0; i < audio.num_audio_tokens; ++i)
            expanded += placeholder;
        auto pos = result.unified_prompt.find(audio_tag, offset);
        OPENVINO_ASSERT(pos != std::string::npos, "LFM2 audio tag is missing");
        result.unified_prompt.replace(pos, std::strlen(audio_tag), expanded);
        offset = pos + expanded.size();
    }
    return result;
}

ov::Tensor InputsEmbedderLFM2Audio::get_inputs_embeds(const std::string& prompt,
                                                      const std::vector<EncodedImage>& images,
                                                      VLMPerfMetrics& metrics,
                                                      bool recalculate,
                                                      const std::vector<size_t>& image_sequence) {
    return get_inputs_embeds(prompt, images, {}, {}, metrics, recalculate, image_sequence, {}, {}, {});
}

ov::Tensor InputsEmbedderLFM2Audio::get_inputs_embeds(const std::string& prompt,
                                                      const std::vector<EncodedImage>& images,
                                                      const std::vector<EncodedVideo>& videos,
                                                      const std::vector<EncodedAudio>& audios,
                                                      VLMPerfMetrics& metrics,
                                                      bool,
                                                      const std::vector<size_t>&,
                                                      const std::vector<size_t>&,
                                                      const std::vector<size_t>& audio_sequence,
                                                      const std::vector<std::pair<size_t, size_t>>&) {
    OPENVINO_ASSERT(images.empty() && videos.empty(), "LFM2 audio model accepts audio and text");
    auto ids = get_encoded_input_ids(prompt, metrics);
    CircularBufferQueueElementGuard<EmbeddingsRequest> guard(m_embedding->get_request_queue().get());
    auto text = get_text_embedding(guard.get(), ids, metrics);
    ov::Tensor owned(text.get_element_type(), text.get_shape());
    text.copy_to(owned);
    if (audio_sequence.empty())
        return owned;
    const auto* tokens = ids.data<const int64_t>();
    size_t expected = 0;
    for (const auto id : audio_sequence)
        expected += audios.at(id).num_audio_tokens;
    OPENVINO_ASSERT(std::count(tokens, tokens + ids.get_size(), m_audio_token_id) == expected,
                    "LFM2 audio placeholder count differs from the encoded audio length");
    // Adjacent clips share one placeholder run, so each is written at its own offset.
    size_t offset = 0;
    for (const auto id : audio_sequence) {
        const auto& audio = audios.at(id);
        while (offset < ids.get_size() && tokens[offset] != m_audio_token_id)
            ++offset;
        for (size_t i = 0; i < audio.num_audio_tokens; ++i)
            OPENVINO_ASSERT(offset + i < ids.get_size() && tokens[offset + i] == m_audio_token_id,
                            "LFM2 audio placeholders must be contiguous");
        std::memcpy(owned.data<float>() + offset * m_vlm_config.hidden_size,
                    audio.audio_features.data(),
                    audio.audio_features.get_byte_size());
        offset += audio.num_audio_tokens;
    }
    return owned;
}

}  // namespace ov::genai
