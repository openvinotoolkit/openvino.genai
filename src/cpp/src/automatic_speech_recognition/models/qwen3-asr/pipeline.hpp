// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <openvino/runtime/core.hpp>

#include "audio_chunk.hpp"
#include "automatic_speech_recognition/pipeline_base.hpp"
#include "debug_utils.hpp"
#include "decoder.hpp"
#include "encoder.hpp"
#include "openvino/core/parallel.hpp"
#include "openvino/genai/chat_history.hpp"
#include "whisper/feature_extractor.hpp"

// todo: move to cpp after poc

namespace ov::genai {

class Qwen3ASR : public ASRPipelineImplBase {
public:
    static constexpr size_t MAX_ASR_INPUT_SECONDS = 1200;
    static constexpr size_t MAX_FORCE_ALIGN_INPUT_SECONDS = 180;

    Qwen3ASR(const std::filesystem::path& models_path, const std::string& device, const ov::AnyMap& properties)
        : ASRPipelineImplBase(models_path, properties),
          m_feature_extractor{models_path / "preprocessor_config.json"},
          m_encoder{models_path, device, properties},
          m_decoder{models_path, device, properties} {
        // whisper accepts word_timestamps as a property
        // for qwen3-asr it will fail in not erased
        // todo: review fail vs ingore for qwen3-asr
        // from whisper:
        // ov::AnyMap properties_copy = properties;
        // m_generation_config.update_generation_config(properties_copy);
        // erase_whisper_generation_config_keys(properties_copy);

        // Qwen3-ASR EOS tokens: <|endoftext|>=151643, <|im_end|>=151645
        // Both must be in stop_token_ids to terminate generation.
        // The exported model has no generation_config.json, so set them explicitly.
        // Qwen3-ASR hardcodes them as well
        m_generation_config.set_eos_token_id(151643);
        m_generation_config.stop_token_ids.insert(151645);
        m_decoder.set_seed(m_generation_config.rng_seed);

        // todo: handle load time for perf metrics
    }

    ASRDecodedResults generate(const RawSpeechInput& raw_speech_input,
                               std::optional<ASRGenerationConfig> generation_config,
                               const std::shared_ptr<StreamerBase> streamer) override {
        // original qwen3-asr always forcing greedy decoding
        // public api: def transcribe(self, audio, context="", language=None, return_time_stamps=False)

        auto start_time = std::chrono::steady_clock::now();

        const ASRGenerationConfig config = resolve_generation_config(generation_config);

        ASRDecodedResults results;
        results.perf_metrics.raw_metrics.m_inference_durations = {{MicroSeconds(0.0f)}};

        const std::vector<AudioChunk> chunks =
            split_audio_into_chunks({raw_speech_input}, m_feature_extractor.sampling_rate, MAX_ASR_INPUT_SECONDS);

        const auto infer_results = infer(chunks, config);

        // todo: merge splitted wavs back
        for (const auto& text : infer_results) {
            results.texts.push_back(text);
            results.scores.push_back(0.0f);
        }

        auto stop_time = std::chrono::steady_clock::now();
        return results;
    }

private:
    Qwen3ASREncoder m_encoder;
    Qwen3ASRDecoder m_decoder;
    WhisperFeatureExtractor m_feature_extractor;

    std::vector<std::string> build_text_prompt(const size_t batch_size, const ASRGenerationConfig& config) {
        if (config.initial_prompts.has_value()) {
            OPENVINO_ASSERT(config.initial_prompts->size() == batch_size,
                            "Number of initial prompts must match batch size");
        }
        if (config.languages.has_value()) {
            OPENVINO_ASSERT(config.languages->size() == batch_size, "Number of languages must match batch size");
        }

        // todo: minja prints: Failed to infer a tool call example (possible template bug)
        // consider replace with hardcoded template string
        const JsonContainer audio_payload = {{"type", "audio"}, {"audio", ""}};
        std::vector<std::string> prompts;
        prompts.reserve(batch_size);
        for (size_t batch = 0; batch < batch_size; ++batch) {
            std::string context = "";
            if (config.initial_prompts.has_value()) {
                context = config.initial_prompts.value()[batch];
            } else if (config.initial_prompt.has_value()) {
                context = config.initial_prompt.value();
            }

            ChatHistory history{{{"role", "system"}, {"content", context}},
                                {{"role", "user"}, {"content", audio_payload}}};
            std::string prompt = m_tokenizer.apply_chat_template(history, true);

            std::string language = "";
            if (config.languages.has_value()) {
                language = config.languages.value()[batch];
            } else if (config.language.has_value()) {
                language = config.language.value();
            }

            if (language != "") {
                prompt += "language " + language + "<asr_text>";
            }
            prompts.push_back(prompt);
        }
        return prompts;
    }

    /**
     * @brief Replaces the single audio placeholder token in each prompt with a repeated audio token span.
     *
     * Each prompt is expected to contain exactly one audio placeholder. The function expands it
     * to exactly audio_lengths[i] repeated audio tokens for the i-th prompt.
     *
     * Example:
     * prompts = {"A <|audio_pad|> B"}, audio_lengths = {3}
     * result  = {"A <|audio_pad|><|audio_pad|><|audio_pad|> B"}
     */
    std::vector<std::string> replace_multimodal_special_tokens(const std::vector<std::string>& prompts,
                                                               const std::vector<size_t>& audio_lengths) {
        OPENVINO_ASSERT(prompts.size() == audio_lengths.size(),
                        "replace_multimodal_special_tokens: prompts and audio_lengths must have the same size");

        // TODO(asr): obtain audio token from model config.json
        static const std::string audio_token = "<|audio_pad|>";

        std::vector<std::string> results = prompts;

        for (size_t i = 0; i < prompts.size(); ++i) {
            const size_t token_pos = prompts[i].find(audio_token);
            OPENVINO_ASSERT(token_pos != std::string::npos,
                            "replace_multimodal_special_tokens: audio token not found in prompt");

            std::string replacement;
            replacement.reserve(audio_token.size() * audio_lengths[i]);
            for (size_t j = 0; j < audio_lengths[i]; ++j) {
                replacement += audio_token;
            }

            results[i].replace(token_pos, audio_token.size(), replacement);
        }

        return results;
    }

    std::vector<std::string> infer(std::vector<AudioChunk> chunks, const ASRGenerationConfig& config) {
        // inference batch_size. Can be different to input batch size due to chunking
        const size_t batch_size = chunks.size();
        const std::vector<std::string> prompts = build_text_prompt(batch_size, config);

        std::vector<WhisperFeatures> features;
        features.reserve(batch_size);
        ov::parallel_for(batch_size, [&](size_t i) {
            features.push_back(m_feature_extractor.extract(chunks[i].wav, false));
        });

        std::vector<std::string> results;
        results.reserve(batch_size);

        for (size_t batch = 0; batch < batch_size; ++batch) {
            const std::string prompt = prompts[batch];
            const ov::Tensor encoder_output = m_encoder.encode({features[batch]});
            const size_t audio_token_count = encoder_output.get_shape()[1];

            // Use actual encoder output length for prompt building
            const std::vector<size_t> audio_lengths{audio_token_count};
            const std::vector<std::string> processed_prompts =
                replace_multimodal_special_tokens({prompt}, audio_lengths);

            const auto input_ids = m_tokenizer.encode(processed_prompts).input_ids;

            // Decode
            const auto encoded_results = m_decoder.decode(input_ids, encoder_output, config);

            const auto text = m_tokenizer.decode(encoded_results.tokens[0]);
            results.push_back(text);
        }

        return results;
    }

    ASRGenerationConfig resolve_generation_config(std::optional<ASRGenerationConfig> generation_config) const {
        ASRGenerationConfig config = generation_config.value_or(m_generation_config);
        if (config.stop_token_ids.empty()) {
            config.stop_token_ids = m_generation_config.stop_token_ids;
        }

        if (config.eos_token_id == -1) {
            config.set_eos_token_id(m_generation_config.eos_token_id);
        }

        config.validate();
        return config;
    }
};
}  // namespace ov::genai
