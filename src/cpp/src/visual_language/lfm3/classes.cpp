// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "classes.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <nlohmann/json.hpp>
#include <sstream>

#include "utils.hpp"
#include "visual_language/clip.hpp"

namespace ov::genai {
namespace {
void validate_rgb_image(const ov::Tensor& image) {
    const auto& shape = image.get_shape();
    OPENVINO_ASSERT(image.get_element_type() == ov::element::u8 && shape.size() == 4 && shape[0] == 1 && shape[1] > 0 &&
                        shape[2] > 0 && shape[3] == 3,
                    "LFM3 vision expects a uint8 RGB tensor [1,H,W,3] with positive spatial dimensions");
}

std::string repeat(const std::string& token, size_t count) {
    std::string result;
    for (size_t i = 0; i < count; ++i)
        result += token;
    return result;
}
void expand(std::string& prompt, const std::string& tag, const std::vector<std::string>& replacements) {
    size_t offset = 0;
    for (const auto& replacement : replacements) {
        const auto pos = prompt.find(tag, offset);
        OPENVINO_ASSERT(pos != std::string::npos, "Missing LFM3 media placeholder ", tag);
        prompt.replace(pos, tag.size(), replacement);
        offset = pos + replacement.size();
    }
}
std::pair<size_t, size_t> video_size(size_t frames, size_t height, size_t width, const VideoProcessorConfig& config) {
    constexpr size_t factor = 32;
    OPENVINO_ASSERT(frames >= 1 && height && width, "LFM3 video requires at least two frames");
    if (frames > 1 && (height < factor || width < factor)) {
        const auto scale = std::max(double(factor) / height, double(factor) / width);
        height = size_t(height * scale);
        width = size_t(width * scale);
    }
    OPENVINO_ASSERT(double(std::max(height, width)) / std::min(height, width) <= 200.,
                    "LFM3 video aspect ratio exceeds 200");
    size_t h = size_t(std::nearbyint(double(height) / factor)) * factor;
    size_t w = size_t(std::nearbyint(double(width) / factor)) * factor;
    const auto temporal = frames == 1 ? 1 : size_t(std::nearbyint(double(frames) / 2)) * 2;
    if (double(temporal) * h * w > config.max_pixels) {
        const auto beta = std::sqrt(double(frames) * height * width / config.max_pixels);
        h = std::max(factor, size_t(std::floor(height / beta / factor)) * factor);
        w = std::max(factor, size_t(std::floor(width / beta / factor)) * factor);
    } else if (double(temporal) * h * w < config.min_pixels) {
        const auto beta = std::sqrt(double(config.min_pixels) / (double(frames) * height * width));
        h = size_t(std::ceil(height * beta / factor)) * factor;
        w = size_t(std::ceil(width * beta / factor)) * factor;
    }
    return {h, w};
}
}  // namespace

VLMModels read_lfm3_saved_models(const std::filesystem::path& directory, const ModelsMap& models_map) {
    VLMModels models;
    models.config = VLMConfig(directory / "config.json");
    nlohmann::json image = {{"patch_size", 16},
                            {"temporal_patch_size", 2},
                            {"merge_size", 2},
                            {"image_mean", {.5, .5, .5}},
                            {"image_std", {.5, .5, .5}},
                            {"min_pixels", 65536},
                            {"max_pixels", 16777216}};
    auto video = image;
    video["fps"] = 2.;
    video["do_sample_frames"] = true;
    video["min_frames"] = 4;
    video["max_frames"] = models.config.model_type == VLMModelType::LFM3_OMNI ? 384 : 768;
    video["min_pixels"] = 4096;
    video["max_pixels"] = 25165824;
    const auto read_json = [&](const char* name) {
        std::ifstream stream(directory / name);
        return nlohmann::json::parse(stream);
    };
    if (std::filesystem::exists(directory / "preprocessor_config.json"))
        image = read_json("preprocessor_config.json");
    if (std::filesystem::exists(directory / "video_preprocessor_config.json"))
        video = read_json("video_preprocessor_config.json");
    if (std::filesystem::exists(directory / "processor_config.json")) {
        const auto combined = read_json("processor_config.json");
        if (combined.contains("image_processor"))
            image = combined["image_processor"];
        if (combined.contains("video_processor"))
            video = combined["video_processor"];
    }
    models.processor_config = ProcessorConfig(image);
    models.video_processor_config = VideoProcessorConfig(video);
    for (const auto& name : {"text_embeddings", "vision_embeddings", "audio_embeddings"}) {
        if (const auto it = models_map.find(name); it != models_map.end())
            models.models[name] = utils::singleton_core().read_model(it->second.first, it->second.second);
        else if (const auto path = directory / (std::string("openvino_") + name + "_model.xml");
                 std::filesystem::exists(path))
            models.models[name] = utils::singleton_core().read_model(path);
    }
    if (models.find("audio_embeddings"))
        models.audio_encoder = AudioEncoderConfig{AudioEncoderType::LFM3, 128};
    return models;
}

InputsEmbedderLFM3::InputsEmbedderLFM3(const VLMConfig&,
                                       const std::filesystem::path& directory,
                                       const Tokenizer& tokenizer,
                                       const std::string& device,
                                       const ov::AnyMap& properties)
    : InputsEmbedderLFM3(read_lfm3_saved_models(directory), tokenizer, device, properties) {}

InputsEmbedderLFM3::InputsEmbedderLFM3(const VLMConfig&,
                                       const ModelsMap& models_map,
                                       const Tokenizer& tokenizer,
                                       const std::filesystem::path& directory,
                                       const std::string& device,
                                       const ov::AnyMap& properties)
    : InputsEmbedderLFM3(read_lfm3_saved_models(directory, models_map), tokenizer, device, properties) {}

ov::Tensor VisionEncoderLFM3::encode_pair(const ov::Tensor& first,
                                          const ov::Tensor& second,
                                          size_t height,
                                          size_t width) {
    const auto preprocess = [&](const ov::Tensor& image) {
        validate_rgb_image(image);
        clip_image_u8 resized;
        bicubic_resize(tensor_to_clip_image_u8(image), resized, int(width), int(height));
        clip_ctx_double normalization;
        for (size_t c = 0; c < 3; ++c) {
            normalization.image_mean[c] = .5 * 255.;
            normalization.image_std[c] = .5 * 255.;
        }
        return normalize_and_convert_to_chw(resized, normalization);
    };
    const auto a = preprocess(first), b = preprocess(second);
    ov::Tensor pixels(ov::element::f32, {1, 3, height, width}, const_cast<float*>(a.buf.data()));
    ov::Tensor pixels_second(ov::element::f32, {1, 3, height, width}, const_cast<float*>(b.buf.data()));
    const auto gh = height / 16, gw = width / 16, patches = gh * gw;
    ov::Tensor positions(ov::element::i32, {1, 1, 1, 4 * patches});
    std::fill_n(positions.data<int32_t>(), positions.get_size(), 0);
    for (size_t y = 0; y < gh; ++y)
        for (size_t x = 0; x < gw; ++x) {
            positions.data<int32_t>()[y * gw + x] = int32_t(y);
            positions.data<int32_t>()[patches + y * gw + x] = int32_t(x);
        }
    CircularBufferQueueElementGuard<ov::InferRequest> guard(m_ireq_queue_vision_encoder.get());
    auto& request = guard.get();
    request.set_tensor("pixel_values", pixels);
    request.set_tensor("pixel_values_1", pixels_second);
    request.set_tensor("positions", positions);
    request.infer();
    const auto output = request.get_output_tensor();
    ov::Tensor owned(output.get_element_type(), output.get_shape());
    output.copy_to(owned);
    OPENVINO_ASSERT(owned.get_shape().size() == 3 && owned.get_shape()[0] == 1 && owned.get_shape()[1] == patches / 4,
                    "Invalid LFM3 vision feature layout");
    return owned;
}

EncodedImage VisionEncoderLFM3::encode(const ov::Tensor& image, const ov::AnyMap& config_map) {
    validate_rgb_image(image);
    const auto config = ProcessorConfig::from_any_map(config_map, m_processor_config);
    const auto source = tensor_to_clip_image_u8(image);
    OPENVINO_ASSERT(double(std::max(source.ny, source.nx)) / std::min(source.ny, source.nx) <= 200.,
                    "LFM3 image aspect ratio exceeds 200");
    VideoProcessorConfig size_config;
    size_config.min_pixels = config.min_pixels;
    size_config.max_pixels = config.max_pixels;
    const auto [h, w] = video_size(1, source.ny, source.nx, size_config);
    EncodedImage result;
    result.resized_source = encode_pair(image, image, h, w);
    result.num_image_tokens = result.resized_source.get_shape()[1];
    result.resized_source_size = {h / 16, w / 16};
    result.original_image_size = {size_t(source.ny), size_t(source.nx)};
    return result;
}

EncodedVideo VisionEncoderLFM3::encode_frames(const std::vector<ov::Tensor>& frames) {
    OPENVINO_ASSERT(frames.size() >= 2 && frames.size() <= m_video_processor_config.max_frames,
                    "LFM3 video frame count is outside the configured limits");
    for (const auto& frame : frames)
        validate_rgb_image(frame);
    const auto source = tensor_to_clip_image_u8(frames.front());
    const auto [h, w] = video_size(frames.size(), source.ny, source.nx, m_video_processor_config);
    std::vector<ov::Tensor> chunks;
    for (size_t i = 0; i < frames.size(); i += 2)
        chunks.push_back(encode_pair(frames[i], frames[std::min(i + 1, frames.size() - 1)], h, w));
    EncodedVideo result;
    result.frame_num = chunks.size();
    result.num_video_tokens = chunks.size() * chunks.front().get_shape()[1];
    result.resized_source_size = {h / 16, w / 16};
    result.video_features = ov::Tensor(ov::element::f32, {1, result.num_video_tokens, chunks.front().get_shape()[2]});
    auto* destination = result.video_features.data<float>();
    for (const auto& chunk : chunks) {
        std::copy_n(chunk.data<const float>(), chunk.get_size(), destination);
        destination += chunk.get_size();
    }
    return result;
}

InputsEmbedderLFM3::InputsEmbedderLFM3(const VLMModels& models,
                                       const Tokenizer& tokenizer,
                                       const std::string& device,
                                       const ov::AnyMap& properties)
    : IInputsEmbedder(models, tokenizer, device, properties) {
    const auto token = [&](const std::string& text) {
        const auto ids = m_tokenizer.encode(text, add_special_tokens(false)).input_ids;
        OPENVINO_ASSERT(ids.get_size() == 1, "LFM3 requires a single special token for ", text);
        return ids.data<const int64_t>()[0];
    };
    if (models.config.model_type != VLMModelType::LFM3_AUDIO) {
        m_image_id = token("<|image|>");
        m_video_id = token("<|video|>");
        m_vision_start = models.config.model_type == VLMModelType::LFM3_OMNI ? "<|media_start|>" : "<|vision_start|>";
        m_vision_end = models.config.model_type == VLMModelType::LFM3_OMNI ? "<|media_end|>" : "<|vision_end|>";
        if (m_tokenizer.encode(m_vision_start, add_special_tokens(false)).input_ids.get_size() != 1) {
            m_vision_start = "<|vision_start|>";
            m_vision_end = "<|vision_end|>";
        }
        token(m_vision_start);
        token(m_vision_end);
    }
    if (models.audio_encoder)
        m_audio_id = token("<|audio|>");
}

std::vector<EncodedImage> InputsEmbedderLFM3::encode_images(const std::vector<ov::Tensor>& images) {
    OPENVINO_ASSERT(images.empty() || m_vision_encoder, "LFM3 Audio does not accept image input");
    std::vector<EncodedImage> result;
    for (const auto& image : to_single_image_tensors(images))
        result.push_back(m_vision_encoder->encode(image));
    return result;
}

std::vector<EncodedAudio> InputsEmbedderLFM3::encode_audios(const std::vector<ov::Tensor>& audios) {
    return encode_audios_with_encoder(audios, false);
}

std::vector<EncodedVideo> InputsEmbedderLFM3::encode_videos(const std::vector<ov::Tensor>& videos,
                                                            const std::vector<VideoMetadata>& metadata) {
    OPENVINO_ASSERT(videos.empty() || m_vision_encoder, "LFM3 audio model does not accept video");
    OPENVINO_ASSERT(metadata.empty() || metadata.size() == videos.size(), "Video metadata count differs from videos");
    std::vector<EncodedVideo> result;
    for (size_t i = 0; i < videos.size(); ++i) {
        OPENVINO_ASSERT(videos[i].get_shape().size() == 4, "LFM3 video must have shape [frames, height, width, 3]");
        const auto total = videos[i].get_shape().at(0);
        auto clock = metadata.empty() ? VideoMetadata{} : metadata[i];
        if (m_vlm_config.model_type == VLMModelType::LFM3_OMNI)
            OPENVINO_ASSERT(std::isfinite(clock.fps) && clock.fps > 0., "LFM3 Omni video requires source fps metadata");
        else if (clock.fps == 0.f)
            clock.fps = 24.f;
        OPENVINO_ASSERT(std::isfinite(clock.fps) && clock.fps > 0., "Video fps must be positive");
        const auto config = m_vision_encoder->get_video_processor_config();
        if (clock.frames_indices.empty()) {
            const auto count = std::min(
                {std::max(size_t(double(total) / clock.fps * 2.), config.min_frames), config.max_frames, total});
            OPENVINO_ASSERT(count >= 2, "LFM3 video requires at least two source frames");
            for (size_t j = 0; j < count; ++j)
                clock.frames_indices.push_back(size_t(std::nearbyint(double(j) * (total - 1) / (count - 1))));
        }
        OPENVINO_ASSERT(clock.frames_indices.size() <= config.max_frames, "Video exceeds LFM3 frame limit");
        const auto sampled = sample_video_if_needed(videos[i], clock);
        auto encoded = m_vision_encoder->encode_frames(to_single_image_tensors({sampled}));
        encoded.metadata = std::move(clock);
        result.push_back(std::move(encoded));
    }
    return result;
}

NormalizedPrompt InputsEmbedderLFM3::normalize_prompt(const std::string& prompt,
                                                      size_t base,
                                                      const std::vector<EncodedImage>& images) const {
    return normalize_prompt(prompt, base, 0, 0, images, {}, {});
}

NormalizedPrompt InputsEmbedderLFM3::normalize_prompt(const std::string& prompt,
                                                      size_t image_base,
                                                      size_t video_base,
                                                      size_t audio_base,
                                                      const std::vector<EncodedImage>& images,
                                                      const std::vector<EncodedVideo>& videos,
                                                      const std::vector<EncodedAudio>& audios) const {
    OPENVINO_ASSERT(m_image_id >= 0 || (images.empty() && videos.empty()), "LFM3 audio accepts text and audio");
    OPENVINO_ASSERT(m_audio_id >= 0 || audios.empty(), "LFM3 VL accepts text, image and video");
    NormalizedPrompt result;
    std::tie(result.unified_prompt, result.images_sequence) =
        normalize(prompt, "<image>", "<image>", image_base, images.size(), ModalityType::IMAGE);
    std::tie(result.unified_prompt, result.videos_sequence) =
        normalize(result.unified_prompt, "<video>", "<video>", video_base, videos.size(), ModalityType::VIDEO);
    std::tie(result.unified_prompt, result.audios_sequence) =
        normalize(result.unified_prompt, "<audio>", "<audio>", audio_base, audios.size(), ModalityType::AUDIO);
    std::vector<std::string> replacements;
    for (const auto id : result.images_sequence)
        replacements.push_back(m_vision_start + repeat("<|image|>", images.at(id - image_base).num_image_tokens) +
                               m_vision_end);
    expand(result.unified_prompt, "<image>", replacements);
    replacements.clear();
    for (const auto id : result.videos_sequence) {
        const auto& video = videos.at(id - video_base);
        std::ostringstream groups;
        groups << std::fixed << std::setprecision(1);
        for (size_t t = 0; t < video.frame_num; ++t) {
            const auto& indices = video.metadata.frames_indices;
            const auto first = indices.at(t * 2), second = indices.at(std::min(t * 2 + 1, indices.size() - 1));
            groups << '<' << (double(first) + second) / 2. / video.metadata.fps << " seconds>" << m_vision_start
                   << repeat("<|video|>", video.num_video_tokens / video.frame_num) << m_vision_end;
        }
        replacements.push_back(m_vision_start + groups.str() + m_vision_end);
    }
    expand(result.unified_prompt, "<video>", replacements);
    replacements.clear();
    for (const auto id : result.audios_sequence) {
        const auto features = repeat("<|audio|>", audios.at(id - audio_base).num_audio_tokens);
        replacements.push_back(
            m_vlm_config.model_type == VLMModelType::LFM3_OMNI ? m_vision_start + features + m_vision_end : features);
    }
    expand(result.unified_prompt, "<audio>", replacements);
    return result;
}

ov::Tensor InputsEmbedderLFM3::get_inputs_embeds(const std::string& prompt,
                                                 const std::vector<EncodedImage>& images,
                                                 VLMPerfMetrics& metrics,
                                                 bool recalculate,
                                                 const std::vector<size_t>& sequence) {
    return get_inputs_embeds(prompt, images, {}, {}, metrics, recalculate, sequence, {}, {}, {});
}

ov::Tensor InputsEmbedderLFM3::get_inputs_embeds(const std::string& prompt,
                                                 const std::vector<EncodedImage>& images,
                                                 const std::vector<EncodedVideo>& videos,
                                                 const std::vector<EncodedAudio>& audios,
                                                 VLMPerfMetrics& metrics,
                                                 bool,
                                                 const std::vector<size_t>& image_sequence,
                                                 const std::vector<size_t>& video_sequence,
                                                 const std::vector<size_t>& audio_sequence,
                                                 const std::vector<std::pair<size_t, size_t>>&) {
    auto ids = get_encoded_input_ids(prompt, metrics);
    CircularBufferQueueElementGuard<EmbeddingsRequest> guard(m_embedding->get_request_queue().get());
    const auto text = get_text_embedding(guard.get(), ids, metrics);
    ov::Tensor output(text.get_element_type(), text.get_shape());
    text.copy_to(output);
    const auto merge = [&](int64_t token, const std::vector<ov::Tensor>& features) {
        size_t expected = 0;
        for (const auto& value : features)
            expected += value.get_size() / m_vlm_config.hidden_size;
        const auto* tokens = ids.data<const int64_t>();
        OPENVINO_ASSERT(size_t(std::count(tokens, tokens + ids.get_size(), token)) == expected,
                        "LFM3 placeholder count differs from encoded features");
        size_t position = 0;
        for (const auto& value : features) {
            const auto count = value.get_size() / m_vlm_config.hidden_size;
            size_t source = 0;
            while (source < count) {
                while (position < ids.get_size() && tokens[position] != token)
                    ++position;
                OPENVINO_ASSERT(position < ids.get_size(), "LFM3 media token was truncated");
                std::copy_n(value.data<const float>() + source * m_vlm_config.hidden_size,
                            m_vlm_config.hidden_size,
                            output.data<float>() + position * m_vlm_config.hidden_size);
                ++source;
                ++position;
            }
        }
    };
    std::vector<ov::Tensor> features;
    for (auto id : image_sequence)
        features.push_back(images.at(id).resized_source);
    merge(m_image_id, features);
    features.clear();
    for (auto id : video_sequence)
        features.push_back(videos.at(id).video_features);
    merge(m_video_id, features);
    features.clear();
    for (auto id : audio_sequence)
        features.push_back(audios.at(id).audio_features);
    merge(m_audio_id, features);
    return output;
}
}  // namespace ov::genai
