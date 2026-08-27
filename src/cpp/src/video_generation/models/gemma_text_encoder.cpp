// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "video_generation/models/gemma_text_encoder.hpp"

#include <algorithm>
#include <cstring>

#include "utils.hpp"

namespace ov {
namespace genai {

std::filesystem::path get_tokenizer_path_by_text_encoder(const std::filesystem::path& text_encoder_path);

namespace {

std::vector<std::string> collect_hidden_state_names(const std::vector<ov::Output<const ov::Node>>& outputs) {
    // Compilation can merge the 'last_hidden_state' port with the final hidden-state port and
    // duplicate its name, so dedupe by layer index
    std::map<int, std::string> indexed;
    for (const auto& output : outputs) {
        if (output.get_names().count("last_hidden_state")) {
            continue;
        }
        for (const std::string& name : output.get_names()) {
            const std::string prefix = "hidden_states.";
            if (name.rfind(prefix, 0) == 0) {
                indexed.emplace(std::stoi(name.substr(prefix.size())), name);
                break;
            }
        }
    }
    OPENVINO_ASSERT(!indexed.empty(), "Text encoder model must expose 'hidden_states.N' outputs");
    std::vector<std::string> names;
    for (const auto& [idx, name] : indexed)
        names.push_back(name);
    return names;
}

}  // namespace

GemmaTextEncoder::GemmaTextEncoder(const std::filesystem::path& root_dir)
    : m_tokenizer(get_tokenizer_path_by_text_encoder(root_dir)) {
    m_model = utils::singleton_core().read_model(root_dir / "openvino_model.xml");
}

GemmaTextEncoder::GemmaTextEncoder(const std::filesystem::path& root_dir,
                                   const std::string& device,
                                   const ov::AnyMap& properties)
    : GemmaTextEncoder(root_dir) {
    compile(device, properties);
}

GemmaTextEncoder::GemmaTextEncoder(const GemmaTextEncoder&) = default;

std::shared_ptr<GemmaTextEncoder> GemmaTextEncoder::clone() {
    OPENVINO_ASSERT((m_model != nullptr) ^ static_cast<bool>(m_request),
                    "GemmaTextEncoder must have exactly one of m_model or m_request initialized");

    std::shared_ptr<GemmaTextEncoder> cloned = std::make_shared<GemmaTextEncoder>(*this);

    if (m_model) {
        cloned->m_model = m_model->clone();
    } else {
        cloned->m_request = m_request.get_compiled_model().create_infer_request();
    }

    return cloned;
}

GemmaTextEncoder& GemmaTextEncoder::reshape(const int batch_size, const int max_sequence_length) {
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

GemmaTextEncoder& GemmaTextEncoder::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_model, device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "Gemma text encoder model");
    m_request = compiled_model.create_infer_request();
    m_hidden_state_names = collect_hidden_state_names(compiled_model.outputs());
    m_model.reset();

    return *this;
}

GemmaTextEncoder::EncodeResult GemmaTextEncoder::infer(const std::string& pos_prompt,
                                                       const std::string& neg_prompt,
                                                       const bool do_classifier_free_guidance,
                                                       const int max_sequence_length) {
    OPENVINO_ASSERT(m_request, "Gemma text encoder model must be compiled first. Cannot infer non-compiled model");

    const size_t batch_size = do_classifier_free_guidance ? 2 : 1;
    const size_t seq_len = static_cast<size_t>(max_sequence_length);

    const ov::element::Type input_type = m_request.get_compiled_model().input("input_ids").get_element_type();
    OPENVINO_ASSERT(input_type == ov::element::i64, "'input_ids' input must be i64");
    int64_t pad_token_id = m_tokenizer.get_pad_token_id();
    if (pad_token_id < 0) {
        pad_token_id = m_tokenizer.get_eos_token_id();
    }

    ov::Tensor input_ids(input_type, {batch_size, seq_len});
    ov::Tensor attention_mask(input_type, {batch_size, seq_len});

    // Left padding matching the reference Gemma tokenization
    auto tokenize_prompt = [&](const std::string& prompt, size_t batch_idx) {
        std::string stripped = prompt;
        stripped.erase(0, stripped.find_first_not_of(" \t\n\r"));
        stripped.erase(stripped.find_last_not_of(" \t\n\r") + 1);

        auto tokenizer_output = m_tokenizer.encode(stripped, ov::genai::add_special_tokens(true));
        const size_t token_len = tokenizer_output.input_ids.get_shape()[1];
        const size_t actual_len = std::min(token_len, seq_len);
        const size_t pad_len = seq_len - actual_len;

        int64_t* ids_row = input_ids.data<int64_t>() + batch_idx * seq_len;
        int64_t* mask_row = attention_mask.data<int64_t>() + batch_idx * seq_len;
        std::fill_n(ids_row, pad_len, pad_token_id);
        std::fill_n(mask_row, pad_len, static_cast<int64_t>(0));
        std::copy_n(tokenizer_output.input_ids.data<int64_t>(), actual_len, ids_row + pad_len);
        std::fill_n(mask_row + pad_len, actual_len, static_cast<int64_t>(1));
    };

    size_t current_batch_idx = 0;
    if (do_classifier_free_guidance) {
        tokenize_prompt(neg_prompt, current_batch_idx);
        ++current_batch_idx;
    }
    tokenize_prompt(pos_prompt, current_batch_idx);

    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("attention_mask", attention_mask);
    m_request.infer();

    // torch.stack(hidden_states, dim=-1).flatten(2, 3): layer index varies fastest in the packed dim
    const size_t num_layers = m_hidden_state_names.size();
    const size_t hidden_size = m_request.get_tensor(m_hidden_state_names.front()).get_shape()[2];
    const size_t packed_dim = hidden_size * num_layers;

    ov::Tensor prompt_embeds(ov::element::f32, {batch_size, seq_len, packed_dim});
    float* embeds_data = prompt_embeds.data<float>();

    for (size_t layer_idx = 0; layer_idx < num_layers; ++layer_idx) {
        const ov::Tensor hidden_state = m_request.get_tensor(m_hidden_state_names[layer_idx]);
        const float* hs_data = hidden_state.data<const float>();
        for (size_t b = 0; b < batch_size; ++b) {
            for (size_t s = 0; s < seq_len; ++s) {
                const float* src = hs_data + (b * seq_len + s) * hidden_size;
                float* dst = embeds_data + (b * seq_len + s) * packed_dim + layer_idx;
                for (size_t h = 0; h < hidden_size; ++h) {
                    dst[h * num_layers] = src[h];
                }
            }
        }
    }

    ov::Tensor mask_f32(ov::element::f32, {batch_size, seq_len});
    const int64_t* mask_src = attention_mask.data<const int64_t>();
    float* mask_dst = mask_f32.data<float>();
    for (size_t i = 0; i < mask_f32.get_size(); ++i) {
        mask_dst[i] = static_cast<float>(mask_src[i]);
    }

    return {prompt_embeds, mask_f32};
}

}  // namespace genai
}  // namespace ov
