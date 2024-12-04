// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/clip_text_model.hpp"

#include <fstream>

#include "json_utils.hpp"
#include "lora_helper.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

std::filesystem::path get_tokenizer_path_by_text_encoder(const std::filesystem::path& text_encoder_path) {
    const std::string to_replace = "text_encoder", replacement = "tokenizer";
    std::string text_encoder_path_str = text_encoder_path.string();

    size_t pos = text_encoder_path_str.find(to_replace);
    OPENVINO_ASSERT(pos != std::string::npos, "Failed to find '", to_replace, "' substring in '", text_encoder_path_str, "'");
    text_encoder_path_str.replace(pos, to_replace.length(), replacement);

    return text_encoder_path_str;
}

CLIPTextModel::Config::Config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "max_position_embeddings", max_position_embeddings);
    read_json_param(data, "num_hidden_layers", num_hidden_layers);
}

CLIPTextModel::CLIPTextModel(const std::filesystem::path& root_dir) :
    m_clip_tokenizer(get_tokenizer_path_by_text_encoder(root_dir)),
    m_config(root_dir / "config.json") {
    ov::Core core = utils::singleton_core();
    m_model = core.read_model((root_dir / "openvino_model.xml").string());
}

CLIPTextModel::CLIPTextModel(const std::filesystem::path& root_dir,
                             const std::string& device,
                             const ov::AnyMap& properties) :
    CLIPTextModel(root_dir) {
    compile(device, properties);
}

CLIPTextModel::CLIPTextModel(const std::string& model,
                             const Tensor& weights,
                             const Config& config,
                             const Tokenizer& clip_tokenizer) :
    m_clip_tokenizer(clip_tokenizer), m_config(config) {
    ov::Core core = utils::singleton_core();
    m_model = core.read_model(model, weights);
}

CLIPTextModel::CLIPTextModel(const std::string& model,
                             const Tensor& weights,
                             const Config& config,
                             const Tokenizer& clip_tokenizer,
                             const std::string& device,
                             const ov::AnyMap& properties) :
    CLIPTextModel(model, weights, config, clip_tokenizer) {
    compile(device, properties);
}

CLIPTextModel::CLIPTextModel(const CLIPTextModel&) = default;

const CLIPTextModel::Config& CLIPTextModel::get_config() const {
    return m_config;
}

CLIPTextModel& CLIPTextModel::reshape(int batch_size) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot reshape already compiled model");

    ov::PartialShape input_shape = m_model->input(0).get_partial_shape();
    input_shape[0] = batch_size;
    input_shape[1] = m_config.max_position_embeddings;
    std::map<size_t, ov::PartialShape> idx_to_shape{{0, input_shape}};
    m_model->reshape(idx_to_shape);

    return *this;
}

CLIPTextModel& CLIPTextModel::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    ov::Core core = utils::singleton_core();
    ov::CompiledModel compiled_model;
    std::optional<AdapterConfig> adapters;
    if (auto filtered_properties = extract_adapters_from_properties(properties, &adapters)) {
        adapters->set_tensor_name_prefix(adapters->get_tensor_name_prefix().value_or("lora_te"));
        m_adapter_controller = AdapterController(m_model, *adapters, device);
        compiled_model = core.compile_model(m_model, device, *filtered_properties);
    } else {
        compiled_model = core.compile_model(m_model, device, properties);
    }
    m_request = compiled_model.create_infer_request();
    // release the original model
    m_model.reset();

    return *this;
}

void CLIPTextModel::set_adapters(const std::optional<AdapterConfig>& adapters) {
    if (adapters) {
        m_adapter_controller.apply(m_request, *adapters);
    }
}

ov::Tensor CLIPTextModel::infer(const std::string& pos_prompt, const std::string& neg_prompt, bool do_classifier_free_guidance) {
    OPENVINO_ASSERT(m_request, "CLIP text encoder model must be compiled first. Cannot infer non-compiled model");

    const int32_t pad_token_id = m_clip_tokenizer.get_pad_token_id();
    const size_t text_embedding_batch_size = do_classifier_free_guidance ? 2 : 1;

    auto perform_tokenization = [&](const std::string& prompt, ov::Tensor input_ids) {
        std::fill_n(input_ids.data<int32_t>(), input_ids.get_size(), pad_token_id);

        ov::Tensor input_ids_token = m_clip_tokenizer.encode(prompt).input_ids;
        std::copy_n(input_ids_token.data<std::int64_t>(), input_ids_token.get_size(), input_ids.data<std::int32_t>());
    };

    ov::Tensor input_ids(ov::element::i32, {text_embedding_batch_size, m_config.max_position_embeddings});
    size_t current_batch_idx = 0;

    if (do_classifier_free_guidance) {
        perform_tokenization(neg_prompt,
                             ov::Tensor(input_ids, {current_batch_idx    , 0},
                                                   {current_batch_idx + 1, m_config.max_position_embeddings}));
        ++current_batch_idx;
    } else {
        // Negative prompt is ignored when --guidanceScale < 1.0
    }

    perform_tokenization(pos_prompt,
                         ov::Tensor(input_ids, {current_batch_idx    , 0},
                                               {current_batch_idx + 1, m_config.max_position_embeddings}));

    // text embeddings
    m_request.set_tensor("input_ids", input_ids);
    m_request.infer();

    return m_request.get_output_tensor(0);
}

ov::Tensor CLIPTextModel::get_output_tensor(const size_t idx) {
    return m_request.get_output_tensor(idx);
}

} // namespace genai
} // namespace ov
