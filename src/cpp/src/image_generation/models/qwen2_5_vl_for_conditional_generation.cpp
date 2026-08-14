// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/qwen2_5_vl_for_conditional_generation.hpp"

#include <fstream>
#include <cstring>
#include <algorithm>

#include "json_utils.hpp"
#include "lora/helper.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

std::filesystem::path get_tokenizer_path_by_text_encoder(const std::filesystem::path& text_encoder_path);

const std::string Qwen2_5_VLForConditionalGeneration::PROMPT_TEMPLATE =
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, texture, quantity, text, "
    "spatial relationships of the objects and background:<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n"
    "<|im_start|>assistant\n";

Qwen2_5_VLForConditionalGeneration::Config::Config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "hidden_size", hidden_size);
}

Qwen2_5_VLForConditionalGeneration::Qwen2_5_VLForConditionalGeneration(const std::filesystem::path& root_dir)
    : m_config(root_dir / "config.json"),
      m_tokenizer(get_tokenizer_path_by_text_encoder(root_dir)) {
    m_model = utils::singleton_core().read_model(root_dir / "openvino_model.xml");
}

Qwen2_5_VLForConditionalGeneration::Qwen2_5_VLForConditionalGeneration(const std::filesystem::path& root_dir,
                                           const std::string& device,
                                           const ov::AnyMap& properties)
    : Qwen2_5_VLForConditionalGeneration(root_dir) {
    compile(device, properties);
}

Qwen2_5_VLForConditionalGeneration::Qwen2_5_VLForConditionalGeneration(const Qwen2_5_VLForConditionalGeneration&) = default;

std::shared_ptr<Qwen2_5_VLForConditionalGeneration> Qwen2_5_VLForConditionalGeneration::clone() {
    OPENVINO_ASSERT((m_model != nullptr) ^ static_cast<bool>(m_request),
                    "Qwen2_5_VLForConditionalGeneration must have exactly one of m_model or m_request initialized");

    std::shared_ptr<Qwen2_5_VLForConditionalGeneration> cloned = std::make_shared<Qwen2_5_VLForConditionalGeneration>(*this);

    if (m_model) {
        cloned->m_model = m_model->clone();
    } else {
        cloned->m_request = m_request.get_compiled_model().create_infer_request();
    }

    return cloned;
}

Qwen2_5_VLForConditionalGeneration& Qwen2_5_VLForConditionalGeneration::reshape(const int batch_size, const int max_sequence_length) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot reshape already compiled model");

    const int total_max_length = max_sequence_length + static_cast<int>(PROMPT_TEMPLATE_PREFIX_LENGTH);

    std::map<std::string, ov::PartialShape> name_to_shape;
    for (auto&& input : m_model->inputs()) {
        std::string input_name = input.get_any_name();
        name_to_shape[input_name] = input.get_partial_shape();
        if (input_name == "input_ids" || input_name == "attention_mask") {
            name_to_shape[input_name] = {batch_size, total_max_length};
        }
    }

    m_model->reshape(name_to_shape);
    return *this;
}

Qwen2_5_VLForConditionalGeneration& Qwen2_5_VLForConditionalGeneration::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    std::optional<AdapterConfig> adapters;
    auto filtered_properties = extract_adapters_from_properties(properties, &adapters);
    if (adapters) {
        adapters->set_tensor_name_prefix(adapters->get_tensor_name_prefix().value_or("text_encoder"));
        m_adapter_controller = AdapterController(m_model, *adapters, device);
    }
    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_model, device, *filtered_properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "QwenImage text encoder model");
    m_request = compiled_model.create_infer_request();
    // release the original model
    m_model.reset();

    return *this;
}

ov::Tensor Qwen2_5_VLForConditionalGeneration::infer(const std::string& pos_prompt, const std::string& neg_prompt, bool do_classifier_free_guidance, const int max_sequence_length) {
    OPENVINO_ASSERT(m_request, "QwenImage text encoder model must be compiled first. Cannot infer non-compiled model");

    const size_t text_embedding_batch_size = do_classifier_free_guidance ? 2 : 1;
    const size_t total_max_length = static_cast<size_t>(max_sequence_length) + PROMPT_TEMPLATE_PREFIX_LENGTH;

    const ov::element::Type input_type = m_request.get_compiled_model().input("input_ids").get_element_type();
    const int64_t pad_token_id = m_tokenizer.get_pad_token_id();

    auto tokenize_prompt = [&](const std::string& prompt, size_t batch_idx,
                               ov::Tensor& input_ids, ov::Tensor& attention_mask) {
        std::string formatted = PROMPT_TEMPLATE;
        const std::string placeholder = "{}";
        size_t pos = formatted.find(placeholder);
        OPENVINO_ASSERT(pos != std::string::npos, "Prompt template must contain '{}'");
        formatted.replace(pos, placeholder.length(), prompt);

        auto tok_output = m_tokenizer.encode(formatted);
        ov::Tensor ids_tok = tok_output.input_ids;
        ov::Tensor mask_tok = tok_output.attention_mask;

        const size_t tok_len = ids_tok.get_shape()[1];
        const size_t actual_len = std::min(tok_len, total_max_length);

        if (input_type == ov::element::i32) {
            int32_t* ids_row = input_ids.data<int32_t>() + batch_idx * total_max_length;
            int32_t* mask_row = attention_mask.data<int32_t>() + batch_idx * total_max_length;
            std::fill_n(ids_row, total_max_length, static_cast<int32_t>(pad_token_id));
            std::fill_n(mask_row, total_max_length, static_cast<int32_t>(0));
            std::copy_n(ids_tok.data<int64_t>(), actual_len, ids_row);
            std::copy_n(mask_tok.data<int64_t>(), actual_len, mask_row);
        } else {
            int64_t* ids_row = input_ids.data<int64_t>() + batch_idx * total_max_length;
            int64_t* mask_row = attention_mask.data<int64_t>() + batch_idx * total_max_length;
            std::fill_n(ids_row, total_max_length, pad_token_id);
            std::fill_n(mask_row, total_max_length, static_cast<int64_t>(0));
            std::copy_n(ids_tok.data<int64_t>(), actual_len, ids_row);
            std::copy_n(mask_tok.data<int64_t>(), actual_len, mask_row);
        }
    };

    // Prepare batched input tensors
    ov::Tensor input_ids(input_type, {text_embedding_batch_size, total_max_length});
    ov::Tensor attention_mask(input_type, {text_embedding_batch_size, total_max_length});

    size_t current_batch_idx = 0;
    if (do_classifier_free_guidance) {
        tokenize_prompt(neg_prompt, current_batch_idx, input_ids, attention_mask);
        ++current_batch_idx;
    }
    tokenize_prompt(pos_prompt, current_batch_idx, input_ids, attention_mask);

    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("attention_mask", attention_mask);
    m_request.infer();

    // Post-process: drop template prefix, pad to max_sequence_length
    ov::Tensor hidden_states = m_request.get_output_tensor();
    const float* hs_data = hidden_states.data<float>();
    const size_t hidden_size = hidden_states.get_shape()[2];
    const size_t output_seq_len = static_cast<size_t>(max_sequence_length);

    ov::Tensor prompt_embeds(ov::element::f32, {text_embedding_batch_size, output_seq_len, hidden_size});
    m_encoder_attention_mask = ov::Tensor(ov::element::i64, {text_embedding_batch_size, output_seq_len});

    float* embeds_out = prompt_embeds.data<float>();
    int64_t* mask_out = m_encoder_attention_mask.data<int64_t>();

    for (size_t b = 0; b < text_embedding_batch_size; ++b) {
        const float* batch_hs = hs_data + b * total_max_length * hidden_size;

        // Count valid tokens from attention mask
        size_t valid_length = 0;
        if (input_type == ov::element::i32) {
            const int32_t* mask_data = attention_mask.data<int32_t>() + b * total_max_length;
            for (size_t i = 0; i < total_max_length; ++i) {
                if (mask_data[i] != 0) ++valid_length;
            }
        } else {
            const int64_t* mask_data = attention_mask.data<int64_t>() + b * total_max_length;
            for (size_t i = 0; i < total_max_length; ++i) {
                if (mask_data[i] != 0) ++valid_length;
            }
        }

        OPENVINO_ASSERT(valid_length > PROMPT_TEMPLATE_PREFIX_LENGTH,
                        "Token count after encoding must be greater than the template prefix length (",
                        PROMPT_TEMPLATE_PREFIX_LENGTH, "), got ", valid_length);

        const size_t content_length = valid_length - PROMPT_TEMPLATE_PREFIX_LENGTH;
        const size_t content_seq_len = std::min(content_length, output_seq_len);

        float* batch_embeds_out = embeds_out + b * output_seq_len * hidden_size;
        int64_t* batch_mask_out = mask_out + b * output_seq_len;

        // Copy valid content embeddings (skip template prefix)
        std::memcpy(batch_embeds_out,
                    batch_hs + PROMPT_TEMPLATE_PREFIX_LENGTH * hidden_size,
                    content_seq_len * hidden_size * sizeof(float));

        // Zero-pad remaining positions
        std::memset(batch_embeds_out + content_seq_len * hidden_size, 0,
                    (output_seq_len - content_seq_len) * hidden_size * sizeof(float));

        // Mask: 1 for valid content tokens, 0 for padding
        std::fill_n(batch_mask_out, content_seq_len, static_cast<int64_t>(1));
        std::fill_n(batch_mask_out + content_seq_len, output_seq_len - content_seq_len, static_cast<int64_t>(0));
    }

    return prompt_embeds;
}

ov::Tensor Qwen2_5_VLForConditionalGeneration::get_encoder_attention_mask() const {
    OPENVINO_ASSERT(m_encoder_attention_mask,
                    "Encoder attention mask is not available. Run infer() first");
    return m_encoder_attention_mask;
}

void Qwen2_5_VLForConditionalGeneration::set_adapters(const std::optional<AdapterConfig>& adapters) {
    OPENVINO_ASSERT(m_request, "Text encoder model must be compiled first");
    if (adapters) {
        m_adapter_controller.apply(m_request, *adapters);
    }
}

const Qwen2_5_VLForConditionalGeneration::Config& Qwen2_5_VLForConditionalGeneration::get_config() const {
    return m_config;
}

}  // namespace genai
}  // namespace ov
