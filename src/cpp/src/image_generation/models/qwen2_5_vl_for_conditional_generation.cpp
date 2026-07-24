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

std::pair<ov::Tensor, ov::Tensor> Qwen2_5_VLForConditionalGeneration::infer(const std::string& prompt, const int max_sequence_length) {
    OPENVINO_ASSERT(m_request, "QwenImage text encoder model must be compiled first. Cannot infer non-compiled model");

    // 1. Format prompt with template
    std::string formatted_prompt = PROMPT_TEMPLATE;
    const std::string placeholder = "{}";
    size_t pos = formatted_prompt.find(placeholder);
    OPENVINO_ASSERT(pos != std::string::npos, "Prompt template must contain '{}'");
    formatted_prompt.replace(pos, placeholder.length(), prompt);

    // 2. Tokenize
    const size_t total_max_length = static_cast<size_t>(max_sequence_length) + PROMPT_TEMPLATE_PREFIX_LENGTH;
    auto tokenizer_output = m_tokenizer.encode(formatted_prompt);
    ov::Tensor input_ids_token = tokenizer_output.input_ids;
    ov::Tensor attention_mask_token = tokenizer_output.attention_mask;

    const size_t token_len = input_ids_token.get_shape()[1];
    const size_t actual_len = std::min(token_len, total_max_length);

    // Get element type from compiled model
    const ov::element::Type input_type = m_request.get_compiled_model().input("input_ids").get_element_type();
    const int64_t pad_token_id = m_tokenizer.get_pad_token_id();

    // Prepare padded tensors
    ov::Tensor input_ids(input_type, {1, total_max_length});
    ov::Tensor attention_mask(input_type, {1, total_max_length});

    if (input_type == ov::element::i32) {
        int32_t* ids_data = input_ids.data<int32_t>();
        int32_t* mask_data = attention_mask.data<int32_t>();
        std::fill_n(ids_data, total_max_length, static_cast<int32_t>(pad_token_id));
        std::fill_n(mask_data, total_max_length, static_cast<int32_t>(0));
        for (size_t i = 0; i < actual_len; ++i) {
            ids_data[i] = static_cast<int32_t>(input_ids_token.data<int64_t>()[i]);
            mask_data[i] = static_cast<int32_t>(attention_mask_token.data<int64_t>()[i]);
        }
    } else {
        int64_t* ids_data = input_ids.data<int64_t>();
        int64_t* mask_data = attention_mask.data<int64_t>();
        std::fill_n(ids_data, total_max_length, pad_token_id);
        std::fill_n(mask_data, total_max_length, static_cast<int64_t>(0));
        std::copy_n(input_ids_token.data<int64_t>(), actual_len, ids_data);
        std::copy_n(attention_mask_token.data<int64_t>(), actual_len, mask_data);
    }

    // 3. Run inference
    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("attention_mask", attention_mask);
    m_request.infer();

    // 4. Get last_hidden_state output
    ov::Tensor hidden_states = m_request.get_output_tensor();
    const float* hs_data = hidden_states.data<float>();
    const size_t seq_len = hidden_states.get_shape()[1];
    const size_t hidden_size = hidden_states.get_shape()[2];

    // 5. Extract masked hidden states: only tokens where attention_mask == 1
    size_t valid_length = 0;
    if (input_type == ov::element::i32) {
        const int32_t* mask_data = attention_mask.data<int32_t>();
        for (size_t i = 0; i < total_max_length; ++i) {
            if (mask_data[i] != 0) ++valid_length;
        }
    } else {
        const int64_t* mask_data = attention_mask.data<int64_t>();
        for (size_t i = 0; i < total_max_length; ++i) {
            if (mask_data[i] != 0) ++valid_length;
        }
    }

    // 6. Drop first PROMPT_TEMPLATE_PREFIX_LENGTH tokens from valid hidden states
    OPENVINO_ASSERT(valid_length > PROMPT_TEMPLATE_PREFIX_LENGTH,
                    "Token count after encoding must be greater than the template prefix length (",
                    PROMPT_TEMPLATE_PREFIX_LENGTH, "), got ", valid_length);
    const size_t content_length = valid_length - PROMPT_TEMPLATE_PREFIX_LENGTH;

    // 7. Clamp to max_sequence_length
    const size_t output_seq_len = std::min(content_length, static_cast<size_t>(max_sequence_length));

    // 8. Create prompt_embeds and attention mask
    ov::Tensor prompt_embeds(ov::element::f32, {1, output_seq_len, hidden_size});
    ov::Tensor encoder_attention_mask(ov::element::i64, {1, output_seq_len});

    float* embeds_data = prompt_embeds.data<float>();
    int64_t* mask_out_data = encoder_attention_mask.data<int64_t>();

    // Copy from position PROMPT_TEMPLATE_PREFIX_LENGTH within the valid (non-padded) region
    // Since the model pads at the end and valid tokens are at the beginning,
    // the hidden states for valid tokens are at indices [0, valid_length)
    const size_t src_offset = PROMPT_TEMPLATE_PREFIX_LENGTH;
    std::memcpy(embeds_data,
                hs_data + src_offset * hidden_size,
                output_seq_len * hidden_size * sizeof(float));

    // All output tokens are valid (we already trimmed to content length)
    std::fill_n(mask_out_data, output_seq_len, static_cast<int64_t>(1));

    return {prompt_embeds, encoder_attention_mask};
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
