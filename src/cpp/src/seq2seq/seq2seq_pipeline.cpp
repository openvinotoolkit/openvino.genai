// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/seq2seq_pipeline.hpp"
#include "seq2seq_pipeline_impl.hpp"
#include "logger.hpp"
#include "utils.hpp"
#include <sstream>

namespace ov {
namespace genai {

// Public API Implementation
Seq2SeqPipeline::Seq2SeqPipeline(
    const std::filesystem::path& models_path,
    const std::string& device,
    const ov::AnyMap& properties
) {
    m_impl = std::make_shared<Seq2SeqPipelineImpl>(models_path, device, properties);
    OPENVINO_LOG_INFO << "Seq2SeqPipeline constructed from: " << models_path;
}

Seq2SeqPipeline::Seq2SeqPipeline(
    const std::filesystem::path& encoder_path,
    const std::filesystem::path& decoder_path,
    const ov::genai::Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap& properties
) {
    m_impl = std::make_shared<Seq2SeqPipelineImpl>(encoder_path, decoder_path, tokenizer, device, properties);
    OPENVINO_LOG_INFO << "Seq2SeqPipeline constructed from encoder: " << encoder_path
                      << ", decoder: " << decoder_path;
}

Seq2SeqDecodedResults Seq2SeqPipeline::generate(
    const std::vector<std::string>& input_texts,
    const ov::AnyMap& properties
) {
    GenerationConfig gen_config = m_generation_config;
    gen_config.update_generation_config(properties);

    OPENVINO_LOG_DEBUG << "Generating " << input_texts.size() << " sequences";
    OPENVINO_LOG_DEBUG << "Generation config: max_new_tokens=" << gen_config.max_new_tokens;

    return m_impl->generate(
        m_impl->get_tokenizer().encode(input_texts),
        gen_config
    );
}

GenerationConfig Seq2SeqPipeline::get_generation_config() const {
    return m_generation_config;
}

void Seq2SeqPipeline::set_generation_config(const GenerationConfig& new_config) {
    m_generation_config = new_config;
}

const ov::genai::Tokenizer& Seq2SeqPipeline::get_tokenizer() const {
    // Need to expose tokenizer from impl
    // For now, return a placeholder
    static ov::genai::Tokenizer dummy_tokenizer;
    return dummy_tokenizer;
}

std::string Seq2SeqPipeline::get_encoder_output_name() const {
    return m_impl->get_encoder_output_name();
}

std::pair<std::string, std::string> Seq2SeqPipeline::get_decoder_io_names() const {
    return m_impl->get_decoder_io_names();
}

Seq2SeqPipeline::~Seq2SeqPipeline() = default;

// Implementation Details

Seq2SeqPipelineImpl::Seq2SeqPipelineImpl(
    const std::filesystem::path& models_path,
    const std::string& device,
    const ov::AnyMap& properties
) : m_device(device), m_compile_config(properties) {
    if (!std::filesystem::exists(models_path)) {
        throw std::runtime_error("Models path does not exist: " + models_path.string());
    }

    // Try to load tokenizer from models_path
    ov::genai::Tokenizer tokenizer;
    try {
        tokenizer = ov::genai::Tokenizer(models_path.string());
        OPENVINO_LOG_DEBUG << "Loaded tokenizer from: " << models_path;
    } catch (const std::exception& e) {
        OPENVINO_LOG_WARNING << "Failed to load tokenizer: " << e.what();
    }

    // Look for encoder and decoder models
    std::filesystem::path encoder_path = models_path / "encoder.xml";
    std::filesystem::path decoder_path = models_path / "decoder.xml";

    // Also try .bin variants
    if (!std::filesystem::exists(encoder_path)) {
        std::filesystem::path encoder_bin_path = models_path / "encoder.bin";
        if (std::filesystem::exists(encoder_bin_path)) {
            encoder_path = encoder_bin_path;
        }
    }

    if (!std::filesystem::exists(decoder_path)) {
        std::filesystem::path decoder_bin_path = models_path / "decoder.bin";
        if (std::filesystem::exists(decoder_bin_path)) {
            decoder_path = decoder_bin_path;
        }
    }

    if (!std::filesystem::exists(encoder_path)) {
        throw std::runtime_error("Encoder model not found in: " + models_path.string());
    }
    if (!std::filesystem::exists(decoder_path)) {
        throw std::runtime_error("Decoder model not found in: " + models_path.string());
    }

    load_models(encoder_path, decoder_path);
}

Seq2SeqPipelineImpl::Seq2SeqPipelineImpl(
    const std::filesystem::path& encoder_path,
    const std::filesystem::path& decoder_path,
    const ov::genai::Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap& properties
) : m_device(device), m_compile_config(properties) {
    if (!std::filesystem::exists(encoder_path)) {
        throw std::runtime_error("Encoder model not found: " + encoder_path.string());
    }
    if (!std::filesystem::exists(decoder_path)) {
        throw std::runtime_error("Decoder model not found: " + decoder_path.string());
    }

    load_models(encoder_path, decoder_path);
}

void Seq2SeqPipelineImpl::load_models(
    const std::filesystem::path& encoder_path,
    const std::filesystem::path& decoder_path
) {
    OPENVINO_LOG_INFO << "Loading encoder from: " << encoder_path;
    OPENVINO_LOG_INFO << "Loading decoder from: " << decoder_path;

    // Load encoder model
    auto encoder_model = m_core.read_model(encoder_path);
    m_encoder = m_core.compile_model(encoder_model, m_device, m_compile_config);
    m_encoder_infer_request = m_encoder.create_infer_request();

    // Load decoder model
    auto decoder_model = m_core.read_model(decoder_path);
    m_decoder = m_core.compile_model(decoder_model, m_device, m_compile_config);
    m_decoder_infer_request = m_decoder.create_infer_request();

    // Get I/O names
    m_encoder_output_name = m_encoder.outputs()[0].get_names()[0];
    m_decoder_input_name = m_decoder.inputs()[0].get_names()[0];  // Assuming first input is input_ids
    m_decoder_output_name = m_decoder.outputs()[0].get_names()[0]; // Assuming first output is logits

    OPENVINO_LOG_DEBUG << "Encoder output: " << m_encoder_output_name;
    OPENVINO_LOG_DEBUG << "Decoder input: " << m_decoder_input_name;
    OPENVINO_LOG_DEBUG << "Decoder output: " << m_decoder_output_name;
}

ov::Tensor Seq2SeqPipelineImpl::encode(
    const ov::Tensor& input_ids,
    const ov::Tensor& attention_mask
) {
    m_encoder_infer_request.set_input_tensor(input_ids);
    m_encoder_infer_request.infer();
    return m_encoder_infer_request.get_output_tensor();
}

ov::Tensor Seq2SeqPipelineImpl::decode_step(
    const ov::Tensor& decoder_input_ids,
    const ov::Tensor& encoder_hidden_states
) {
    m_decoder_infer_request.set_input_tensor(0, decoder_input_ids);
    m_decoder_infer_request.set_input_tensor(1, encoder_hidden_states);
    m_decoder_infer_request.infer();
    return m_decoder_infer_request.get_output_tensor();
}

ov::Tensor Seq2SeqPipelineImpl::initialize_decoder_input(size_t batch_size) {
    // Create tensor filled with BOS token
    auto shape = ov::Shape{batch_size, 1};
    auto decoder_input = ov::Tensor(ov::element::i64, shape);
    auto ptr = decoder_input.data<int64_t>();
    for (size_t i = 0; i < batch_size; ++i) {
        ptr[i] = m_bos_token_id;
    }
    return decoder_input;
}

std::vector<std::vector<int64_t>> Seq2SeqPipelineImpl::greedy_decode(
    const ov::Tensor& encoder_hidden_states,
    size_t batch_size,
    int32_t max_new_tokens
) {
    std::vector<std::vector<int64_t>> output_sequences(batch_size);
    std::vector<bool> finished_mask(batch_size, false);

    // Initialize decoder input with BOS tokens
    auto decoder_input_ids = initialize_decoder_input(batch_size);

    // Greedy decoding loop
    for (int32_t step = 0; step < max_new_tokens; ++step) {
        // Get logits from decoder
        auto logits = decode_step(decoder_input_ids, encoder_hidden_states);

        // Get last token logits for each batch element
        auto logits_ptr = logits.data<float>();
        auto logits_shape = logits.get_shape();
        size_t vocab_size = logits_shape.back();
        size_t seq_length = logits_shape[1];

        // Select argmax token for each sequence
        std::vector<int64_t> next_tokens(batch_size);
        for (size_t i = 0; i < batch_size; ++i) {
            if (!finished_mask[i]) {
                // Get logits for last position
                float* last_logits = logits_ptr + (i * seq_length * vocab_size) + ((seq_length - 1) * vocab_size);
                auto max_idx = std::max_element(last_logits, last_logits + vocab_size) - last_logits;
                next_tokens[i] = static_cast<int64_t>(max_idx);

                // Check for EOS
                if (next_tokens[i] == m_eos_token_id) {
                    finished_mask[i] = true;
                }

                output_sequences[i].push_back(next_tokens[i]);
            }
        }

        // Check if all sequences are finished
        if (all_sequences_finished(output_sequences, finished_mask)) {
            break;
        }

        // Append next tokens to decoder input for next iteration
        auto new_input_shape = ov::Shape{batch_size, seq_length + 1};
        auto new_decoder_input = ov::Tensor(ov::element::i64, new_input_shape);
        auto new_ptr = new_decoder_input.data<int64_t>();

        auto old_ptr = decoder_input_ids.data<int64_t>();
        std::memcpy(new_ptr, old_ptr, batch_size * seq_length * sizeof(int64_t));

        for (size_t i = 0; i < batch_size; ++i) {
            new_ptr[i * (seq_length + 1) + seq_length] = next_tokens[i];
        }

        decoder_input_ids = new_decoder_input;
    }

    return output_sequences;
}

bool Seq2SeqPipelineImpl::all_sequences_finished(
    const std::vector<std::vector<int64_t>>& sequences,
    const std::vector<bool>& finished_mask
) const {
    for (const auto& finished : finished_mask) {
        if (!finished) return false;
    }
    return true;
}

Seq2SeqDecodedResults Seq2SeqPipelineImpl::generate(
    const TokenizedInputs& encoded_inputs,
    const GenerationConfig& generation_config
) {
    auto input_ids = encoded_inputs.input_ids;
    auto attention_mask = encoded_inputs.attention_mask;
    auto batch_size = input_ids.get_shape()[0];

    OPENVINO_LOG_DEBUG << "Encoding " << batch_size << " sequences";

    // Encode input
    auto encoder_hidden_states = encode(input_ids, attention_mask);

    OPENVINO_LOG_DEBUG << "Decoding with max_new_tokens=" << generation_config.max_new_tokens;

    // Decode with greedy search (MVP)
    auto token_ids = greedy_decode(encoder_hidden_states, batch_size, generation_config.max_new_tokens);

    // Build results
    Seq2SeqDecodedResults results;
    results.token_ids = token_ids;
    results.finish_reasons.resize(batch_size, GenerationFinishReason::LENGTH);
    results.scores.resize(batch_size, 0.0f);  // Greedy decoding doesn't have scores

    return results;
}

}  // namespace genai
}  // namespace ov
