// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>
#include <filesystem>

#include "openvino/core/core.hpp"
#include "openvino/genai/tokenizer.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/seq2seq_pipeline.hpp"

namespace ov {
namespace genai {

/**
 * @brief Abstract base class for Seq2Seq pipeline implementations.
 *
 * Allows different backend implementations (e.g., CPU, GPU, NPU) while
 * maintaining a consistent public API.
 */
class Seq2SeqPipelineImplBase {
public:
    virtual ~Seq2SeqPipelineImplBase() = default;

    /**
     * @brief Generate text from batch of tokenized inputs.
     *
     * @param encoded_inputs Pre-tokenized input with input_ids and attention_mask
     * @param generation_config Parameters controlling generation
     * @returns Results including token_ids, texts, scores, and metrics
     */
    virtual Seq2SeqDecodedResults generate(
        const TokenizedInputs& encoded_inputs,
        const GenerationConfig& generation_config
    ) = 0;

    /**
     * @brief Get encoder model output tensor name.
     */
    virtual std::string get_encoder_output_name() const = 0;

    /**
     * @brief Get decoder input tensor name and output tensor name.
     */
    virtual std::pair<std::string, std::string> get_decoder_io_names() const = 0;
};

/**
 * @brief Seq2Seq pipeline implementation using OpenVINO Core.
 *
 * Manages both encoder and decoder models, handles tokenization and inference,
 * and provides generation with greedy decoding.
 */
class Seq2SeqPipelineImpl : public Seq2SeqPipelineImplBase {
public:
    /**
     * @brief Construct from directory with model files.
     */
    Seq2SeqPipelineImpl(
        const std::filesystem::path& models_path,
        const std::string& device,
        const ov::AnyMap& properties = {}
    );

    /**
     * @brief Construct from separate encoder and decoder paths.
     */
    Seq2SeqPipelineImpl(
        const std::filesystem::path& encoder_path,
        const std::filesystem::path& decoder_path,
        const ov::genai::Tokenizer& tokenizer,
        const std::string& device,
        const ov::AnyMap& properties = {}
    );

    ~Seq2SeqPipelineImpl() override = default;

    /**
     * @brief Generate text from batch of tokenized inputs.
     *
     * Process flow:
     * 1. Pass input_ids through encoder to get encoder_hidden_states
     * 2. Initialize decoder with BOS token
     * 3. Loop: decoder(decoder_input_ids, encoder_hidden_states) -> logits
     * 4. Select next token via greedy (argmax) or beam search
     * 5. Stop on EOS token or max_new_tokens
     * 6. Decode tokens back to text
     *
     * @param encoded_inputs Pre-tokenized input
     * @param generation_config Generation parameters
     * @returns Generated sequences
     */
    Seq2SeqDecodedResults generate(
        const TokenizedInputs& encoded_inputs,
        const GenerationConfig& generation_config
    ) override;

    std::string get_encoder_output_name() const override { return m_encoder_output_name; }
    std::pair<std::string, std::string> get_decoder_io_names() const override {
        return {m_decoder_input_name, m_decoder_output_name};
    }

private:
    ov::Core m_core;
    ov::CompiledModel m_encoder;
    ov::CompiledModel m_decoder;
    ov::InferRequest m_encoder_infer_request;
    ov::InferRequest m_decoder_infer_request;

    std::string m_device;
    ov::AnyMap m_compile_config;

    // Model output/input names
    std::string m_encoder_output_name;
    std::string m_decoder_input_name;   // Typically "input_ids"
    std::string m_decoder_output_name;  // Typically "logits"

    // BOS and EOS token IDs (typically 0 and 1, but can be config-dependent)
    int64_t m_bos_token_id = 0;
    int64_t m_eos_token_id = 1;

    /**
     * @brief Load encoder and decoder models from given paths.
     */
    void load_models(
        const std::filesystem::path& encoder_path,
        const std::filesystem::path& decoder_path
    );

    /**
     * @brief Run encoder on input_ids to produce encoder_hidden_states.
     */
    ov::Tensor encode(const ov::Tensor& input_ids, const ov::Tensor& attention_mask);

    /**
     * @brief Single decoder step: given decoder_input_ids and encoder_hidden_states,
     *        return logits for next token prediction.
     */
    ov::Tensor decode_step(
        const ov::Tensor& decoder_input_ids,
        const ov::Tensor& encoder_hidden_states
    );

    /**
     * @brief Greedy decoding loop.
     */
    std::vector<std::vector<int64_t>> greedy_decode(
        const ov::Tensor& encoder_hidden_states,
        size_t batch_size,
        int32_t max_new_tokens
    );

    /**
     * @brief Initialize decoder input tokens (prepend BOS to each sequence).
     */
    ov::Tensor initialize_decoder_input(size_t batch_size);

    /**
     * @brief Check if all sequences in batch have reached EOS.
     */
    bool all_sequences_finished(
        const std::vector<std::vector<int64_t>>& sequences,
        const std::vector<bool>& finished_mask
    ) const;
};

}  // namespace genai
}  // namespace ov
