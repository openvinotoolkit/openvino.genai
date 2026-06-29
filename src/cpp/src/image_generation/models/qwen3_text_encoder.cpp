// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/qwen3_text_encoder.hpp"

#include <fstream>
#include <cstring>

#include "json_utils.hpp"
#include "lora/helper.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

std::filesystem::path get_tokenizer_path_by_text_encoder(const std::filesystem::path& text_encoder_path);

Qwen3TextEncoder::Config::Config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "hidden_size", hidden_size);
    read_json_param(data, "num_hidden_layers", num_hidden_layers);
}

Qwen3TextEncoder::Qwen3TextEncoder(const std::filesystem::path& root_dir)
    : m_config(root_dir / "config.json"),
      m_tokenizer(get_tokenizer_path_by_text_encoder(root_dir)) {
    m_model = utils::singleton_core().read_model(root_dir / "openvino_model.xml");
}

Qwen3TextEncoder::Qwen3TextEncoder(const std::filesystem::path& root_dir,
                                   const std::string& device,
                                   const ov::AnyMap& properties)
    : Qwen3TextEncoder(root_dir) {
    compile(device, properties);
}

Qwen3TextEncoder::Qwen3TextEncoder(const Qwen3TextEncoder&) = default;

std::shared_ptr<Qwen3TextEncoder> Qwen3TextEncoder::clone() {
    OPENVINO_ASSERT((m_model != nullptr) ^ static_cast<bool>(m_request),
                    "Qwen3TextEncoder must have exactly one of m_model or m_request initialized");

    std::shared_ptr<Qwen3TextEncoder> cloned = std::make_shared<Qwen3TextEncoder>(*this);

    if (m_model) {
        cloned->m_model = m_model->clone();
    } else {
        cloned->m_request = m_request.get_compiled_model().create_infer_request();
    }

    return cloned;
}

Qwen3TextEncoder& Qwen3TextEncoder::reshape(int batch_size, int max_sequence_length) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot reshape already compiled model");

    std::map<std::string, ov::PartialShape> name_to_shape;
    for (auto&& input : m_model->inputs()) {
        std::string input_name = input.get_any_name();
        name_to_shape[input_name] = input.get_partial_shape();
        if (input_name == "input_ids" || input_name == "attention_mask") {
            name_to_shape[input_name] = {batch_size, max_sequence_length};
        }
    }

    m_model->reshape(name_to_shape);
    return *this;
}

Qwen3TextEncoder& Qwen3TextEncoder::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    std::optional<AdapterConfig> adapters;
    auto filtered_properties = extract_adapters_from_properties(properties, &adapters);
    if (adapters) {
        adapters->set_tensor_name_prefix(adapters->get_tensor_name_prefix().value_or("text_encoder"));
        m_adapter_controller = AdapterController(m_model, *adapters, device);
    }
    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_model, device, *filtered_properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "Qwen3 text encoder model");
    m_request = compiled_model.create_infer_request();
    // release the original model
    m_model.reset();

    return *this;
}

ov::Tensor Qwen3TextEncoder::infer(const std::string& prompt, int max_sequence_length) {
    OPENVINO_ASSERT(m_request, "Qwen3 text encoder model must be compiled first. Cannot infer non-compiled model");

    // Apply chat template as diffusers does:
    // messages = [{"role": "user", "content": prompt}]
    // text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, enable_thinking=False)
    ChatHistory history = {{{"role", "user"}, {"content", prompt}}};
    JsonContainer extra_context = JsonContainer::from_json_string("{\"enable_thinking\": false}");
    std::string formatted_prompt = m_tokenizer.apply_chat_template(history, true, {}, std::nullopt, extra_context);

    // Tokenize the formatted prompt
    auto tokenizer_output = m_tokenizer.encode(formatted_prompt);
    ov::Tensor input_ids_token = tokenizer_output.input_ids;
    ov::Tensor attention_mask_token = tokenizer_output.attention_mask;

    const size_t token_len = input_ids_token.get_shape()[1];
    const size_t seq_len = static_cast<size_t>(max_sequence_length);
    const size_t actual_len = std::min(token_len, seq_len);

    // Prepare padded input_ids and attention_mask
    ov::Tensor input_ids(ov::element::i64, {1, seq_len});
    ov::Tensor attention_mask(ov::element::i64, {1, seq_len});

    int64_t* ids_data = input_ids.data<int64_t>();
    int64_t* mask_data = attention_mask.data<int64_t>();

    // Fill with pad_token_id (e.g. 151643 for Qwen3's <|endoftext|>)
    const int64_t pad_token_id = m_tokenizer.get_pad_token_id();
    std::fill(ids_data, ids_data + seq_len, pad_token_id);
    std::fill(mask_data, mask_data + seq_len, static_cast<int64_t>(0));

    // Copy actual tokens
    const int64_t* src_ids = input_ids_token.data<int64_t>();
    const int64_t* src_mask = attention_mask_token.data<int64_t>();
    std::copy(src_ids, src_ids + actual_len, ids_data);
    std::copy(src_mask, src_mask + actual_len, mask_data);

    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("attention_mask", attention_mask);
    m_request.infer();

    // Gather hidden states from selected layers and concatenate along channel dimension
    // Output names are "hidden_states.9", "hidden_states.18", "hidden_states.27"
    const size_t num_layers = m_config.hidden_states_layers.size();
    const size_t hidden_size = m_config.hidden_size;
    const size_t output_dim = num_layers * hidden_size;

    // Output shape: (1, seq_len, num_layers * hidden_size)
    ov::Tensor result(ov::element::f32, {1, seq_len, output_dim});
    float* result_data = result.data<float>();

    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        const size_t layer_num = m_config.hidden_states_layers[layer_idx];
        const std::string output_name = "hidden_states." + std::to_string(layer_num);

        ov::Tensor hidden_state = m_request.get_tensor(output_name);
        const float* hs_data = hidden_state.data<float>();

        // Copy this layer's hidden states into the concatenated output
        // hidden_state shape: (1, seq_len, hidden_size)
        // result layout: (1, seq_len, [layer0_hidden | layer1_hidden | layer2_hidden])
        for (size_t s = 0; s < seq_len; ++s) {
            std::memcpy(result_data + s * output_dim + layer_idx * hidden_size,
                        hs_data + s * hidden_size,
                        hidden_size * sizeof(float));
        }
    }

    return result;
}

void Qwen3TextEncoder::set_adapters(const std::optional<AdapterConfig>& adapters) {
    OPENVINO_ASSERT(m_request, "Text encoder model must be compiled first");
    if (adapters) {
        m_adapter_controller.apply(m_request, *adapters);
    }
}

const Qwen3TextEncoder::Config& Qwen3TextEncoder::get_config() const {
    return m_config;
}

}  // namespace genai
}  // namespace ov
