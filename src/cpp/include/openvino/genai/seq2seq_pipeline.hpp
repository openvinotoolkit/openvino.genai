// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <filesystem>
#include <optional>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/common_types.hpp"

namespace ov {
namespace genai {

class Seq2SeqPipelineImplBase;

/**
 * Structure that stores the result from the generate method for Seq2Seq models
 * Includes token IDs, decoded text, confidence scores, and performance metrics
 */
struct Seq2SeqDecodedResults {
    /// Generated token sequences (one per batch element)
    std::vector<std::vector<int64_t>> token_ids;
    /// Decoded text sequences (one per batch element)
    std::vector<std::string> texts;
    /// Sum of log probabilities for each sequence (greedy: zeros, beam search: log probs)
    std::vector<float> scores;
    /// Reason why generation stopped for each sequence
    std::vector<GenerationFinishReason> finish_reasons;
    /// Inference performance metrics
    PerfMetrics perf_metrics;
};

/**
 * @brief Seq2SeqPipeline for encoder-decoder models (e.g., T5, BART, FLAN-T5).
 *
 * This class provides a unified API for seq2seq generation using OpenVINO.
 * It handles the encoder-decoder inference pattern with configurable decoding strategies.
 *
 * MVP Features:
 * - Greedy decoding
 * - Basic batch support
 * - Configurable max token length
 *
 * Future Features:
 * - Beam search decoding
 * - Continuous batching
 * - LoRA adapters
 * - Streaming output
 */
class OPENVINO_GENAI_EXPORTS Seq2SeqPipeline {
public:
    /**
     * @brief Constructs a Seq2SeqPipeline from a directory with model files.
     *
     * Expected directory structure:
     * - encoder.xml / encoder.bin (encoder model)
     * - decoder.xml / decoder.bin (decoder model)
     * - tokenizer.model, tokenizer.json, or vocab.txt (tokenizer files)
     * - generation_config.json (optional generation parameters)
     *
     * @param models_path Path to the directory containing model files and configs
     * @param device Device to compile models on (e.g., "CPU", "GPU", "NPU")
     * @param properties Optional properties for model compilation (compile_config, etc.)
     *
     * @throws std::runtime_error if model files are not found or compilation fails
     */
    Seq2SeqPipeline(
        const std::filesystem::path& models_path,
        const std::string& device,
        const ov::AnyMap& properties = {}
    );

    /**
     * @brief Constructs a Seq2SeqPipeline with a custom tokenizer.
     *
     * @param encoder_path Path to encoder model (xml/bin or gguf)
     * @param decoder_path Path to decoder model (xml/bin or gguf)
     * @param tokenizer Pre-configured tokenizer
     * @param device Device to compile models on
     * @param properties Optional compilation properties
     */
    Seq2SeqPipeline(
        const std::filesystem::path& encoder_path,
        const std::filesystem::path& decoder_path,
        const ov::genai::Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap& properties = {}
    );

    /**
     * @brief Template constructor for variadic properties.
     */
    template <typename... Properties,
              typename std::enable_if<ov::util::StringAny<Properties...>::value, bool>::type = true>
    Seq2SeqPipeline(const std::filesystem::path& models_path, const std::string& device, Properties&&... properties)
        : Seq2SeqPipeline(models_path, device, ov::AnyMap{std::forward<Properties>(properties)...}) {}

    /**
     * @brief Generate text from input sequences.
     *
     * @param input_text Single input text to encode
     * @param properties Optional generation parameters (max_new_tokens, num_beams, etc.)
     * @returns Struct containing generated tokens, decoded texts, scores, and metrics
     */
    Seq2SeqDecodedResults generate(
        const std::string& input_text,
        const ov::AnyMap& properties = {}
    ) {
        return generate(std::vector<std::string>{input_text}, properties);
    }

    /**
     * @brief Generate text from batch of input sequences.
     *
     * @param input_texts Batch of input texts to encode
     * @param properties Optional generation parameters (max_new_tokens, num_beams, etc.)
     * @returns Struct containing generated tokens, decoded texts, scores, and metrics
     *
     * All sequences in the batch are padded to the same length for efficiency.
     */
    Seq2SeqDecodedResults generate(
        const std::vector<std::string>& input_texts,
        const ov::AnyMap& properties = {}
    );

    /**
     * @brief Template method for variadic properties.
     */
    template <typename... Properties>
    Seq2SeqDecodedResults generate(
        const std::vector<std::string>& input_texts,
        Properties&&... properties
    ) {
        return generate(input_texts, ov::AnyMap{std::forward<Properties>(properties)...});
    }

    /**
     * @brief Get the default generation config.
     * @return GenerationConfig with default values
     */
    GenerationConfig get_generation_config() const;

    /**
     * @brief Override default generation config.
     * @param new_config New default configuration
     */
    void set_generation_config(const GenerationConfig& new_config);

    /**
     * @brief Get the tokenizer used by this pipeline.
     * @return Reference to internal tokenizer
     */
    const ov::genai::Tokenizer& get_tokenizer() const;

    /// @brief Get encoder model output name for debugging
    std::string get_encoder_output_name() const;

    /// @brief Get decoder model input/output names for debugging
    std::pair<std::string, std::string> get_decoder_io_names() const;

    ~Seq2SeqPipeline();

private:
    std::shared_ptr<Seq2SeqPipelineImplBase> m_impl;
    GenerationConfig m_generation_config;
};

}  // namespace genai
}  // namespace ov
