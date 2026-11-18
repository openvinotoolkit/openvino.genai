// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <string>

#include <nlohmann/json.hpp>

#include "openvino/genai/chat_history.hpp"
#include "openvino/genai/json_container.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"
#include "video_generation/ltx2_conditioning.hpp"
#include "video_generation/ltx2_system_prompts.hpp"

namespace ov::genai {

/**
 * LTX-2.5's prompt enhancer: a Gemma 4 VLM ('prompt_enhancer/') rewrites a short prompt into the detailed description
 * the model was trained on, looking at the conditioning image for image-to-video. Its tokenizer and chat template are
 * the 'processor/' component. Follows diffusers' LTX2Pipeline.enhance_prompt with GEMMA4_PROMPT_ENHANCEMENT_CONFIG:
 * greedy decoding of up to 600 tokens without repeated 5-grams.
 */
class LTX2PromptEnhancer {
public:
    /// Whether an export carries the enhancer: both its model and its processor
    static bool is_available(const std::filesystem::path& root_dir) {
        return std::filesystem::exists(root_dir / "prompt_enhancer" / "openvino_language_model.xml") &&
               std::filesystem::exists(root_dir / "processor" / "openvino_tokenizer.xml");
    }

    /// @param tokenizer The 'processor/' tokenizer. Loaded with the pipeline, where the tokenizers extension is found.
    LTX2PromptEnhancer(const std::filesystem::path& root_dir,
                       const Tokenizer& tokenizer,
                       const std::string& device,
                       const ov::AnyMap& properties)
        : m_pipeline(read_models(root_dir / "prompt_enhancer"),
                     tokenizer,
                     root_dir / "prompt_enhancer",
                     device,
                     properties) {}

    /// @param image A u8 image [H, W, 3] or [1, H, W, 3] to describe, or null for text-to-video
    std::string enhance(const std::string& prompt, const std::optional<std::string>& system_prompt, const ov::Tensor* image) {
        // diffusers' messages: for image-to-video the user turn is the image, then the text. The template renders the
        // image as Gemma 4's native '<|image|>' tag, which the pipeline expands into the image tokens in place.
        nlohmann::json messages = nlohmann::json::array();
        messages.push_back({{"role", "system"},
                            {"content", system_prompt.value_or(image ? ltx2_prompts::I2V_SYSTEM_PROMPT
                                                                     : ltx2_prompts::T2V_SYSTEM_PROMPT)}});
        std::vector<ov::Tensor> images;
        if (image) {
            images.push_back(resize_long_side(ltx2_conditioning::as_video(*image)));
            messages.push_back({{"role", "user"},
                                {"content",
                                 {{{"type", "image"}}, {{"type", "text"}, {"text", "User Raw Input Prompt: " + prompt + "."}}}}});
        } else {
            messages.push_back({{"role", "user"}, {"content", "user prompt: " + prompt}});
        }
        const std::string templated = m_pipeline.get_tokenizer().apply_chat_template(
            ChatHistory(JsonContainer::from_json_string(messages.dump())), true);

        // The prompt as text rather than a ChatHistory: generation from a ChatHistory leaves the prompt tokens out of
        // the sampler's history, where the n-gram ban needs them, as Hugging Face's has them
        GenerationConfig config = m_pipeline.get_generation_config();
        config.max_new_tokens = 600;
        config.do_sample = false;
        config.no_repeat_ngram_size = 5;
        config.apply_chat_template = false;
        const VLMDecodedResults result = m_pipeline.generate(templated, images, config, std::monostate{});
        return clean_response(result.texts.at(0));
    }

    /// Replaces the typographic characters Gemma tends to write and drops what precedes the first letter
    /// (diffusers' clean_response)
    static std::string clean_response(std::string text) {
        static const std::pair<const char*, const char*> replacements[] = {
            {"\xE2\x80\x98", "'"},   // left single quote
            {"\xE2\x80\x99", "'"},   // right single quote
            {"\xE2\x80\x9C", "\""},  // left double quote
            {"\xE2\x80\x9D", "\""},  // right double quote
            {"\xE2\x80\x94", "-"},   // em dash
            {"\xE2\x80\x93", "-"},   // en dash
            {"\xC2\xA0", " "},       // no-break space
            {"\xE2\x80\xB2", "'"},   // prime
            {"\xE2\x88\x92", "-"},   // minus sign
        };
        for (const auto& [from, to] : replacements) {
            const std::string needle(from);
            for (size_t pos = text.find(needle); pos != std::string::npos; pos = text.find(needle, pos + 1)) {
                text.replace(pos, needle.size(), to);
            }
        }
        // A non-ASCII byte starts a multi-byte character, taken as a letter
        for (size_t i = 0; i < text.size(); ++i) {
            const auto c = static_cast<unsigned char>(text[i]);
            if (c >= 0x80 || std::isalpha(c)) {
                return text.substr(i);
            }
        }
        return text;
    }

private:
    // The models of a VLM folder by name: 'openvino_<name>_model.xml' -> '<name>', with memory-mapped weights
    static ModelsMap read_models(const std::filesystem::path& dir) {
        ModelsMap models;
        const std::string prefix = "openvino_", suffix = "_model.xml";
        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            const std::string file = entry.path().filename().string();
            if (file.size() <= prefix.size() + suffix.size() || file.rfind(prefix, 0) != 0 ||
                file.compare(file.size() - suffix.size(), suffix.size(), suffix) != 0) {
                continue;
            }
            std::ifstream xml_file(entry.path());
            std::string xml((std::istreambuf_iterator<char>(xml_file)), std::istreambuf_iterator<char>());
            std::filesystem::path weights = entry.path();
            weights.replace_extension(".bin");
            models[file.substr(prefix.size(), file.size() - prefix.size() - suffix.size())] = {
                std::move(xml), ov::read_tensor_data(weights)};
        }
        return models;
    }

    // diffusers' _prepare_enhance_image: the long side to 896 pixels, bicubic
    static ov::Tensor resize_long_side(const ov::Tensor& image) {
        const ov::Shape shape = image.get_shape();  // [1, H, W, 3]
        const double scale = 896.0 / static_cast<double>(std::max(shape[1], shape[2]));
        const auto height = static_cast<int64_t>(static_cast<double>(shape[1]) * scale);
        const auto width = static_cast<int64_t>(static_cast<double>(shape[2]) * scale);
        if (height == static_cast<int64_t>(shape[1]) && width == static_cast<int64_t>(shape[2])) {
            return image;
        }
        return ltx2_conditioning::pillow_resize(image, height, width, /* bicubic */ true);
    }

    VLMPipeline m_pipeline;
};

}  // namespace ov::genai
