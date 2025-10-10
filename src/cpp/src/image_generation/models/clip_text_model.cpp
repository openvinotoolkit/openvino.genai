// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/clip_text_model.hpp"

#include <iostream>
#include <memory>
#include <fstream>

#include "json_utils.hpp"
#include "lora/helper.hpp"
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
    m_model = utils::singleton_core().read_model(root_dir / "openvino_model.xml");
}

CLIPTextModel::CLIPTextModel(const std::filesystem::path& root_dir,
                             const std::string& device,
                             const ov::AnyMap& properties)
    : m_clip_tokenizer(get_tokenizer_path_by_text_encoder(root_dir)),
      m_config(root_dir / "config.json") {
    const auto [properties_without_blob, blob_path] = utils::extract_export_properties(properties);

    if (blob_path.has_value()) {
        import_model(*blob_path, device, properties_without_blob);
        return;
    }

    m_model = utils::singleton_core().read_model(root_dir / "openvino_model.xml");
    compile(device, properties);
}

CLIPTextModel::CLIPTextModel(const std::string& model,
                             const Tensor& weights,
                             const Config& config,
                             const Tokenizer& clip_tokenizer) :
    m_clip_tokenizer(clip_tokenizer), m_config(config) {
    m_model = utils::singleton_core().read_model(model, weights);
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

std::shared_ptr<CLIPTextModel> CLIPTextModel::clone() {
    OPENVINO_ASSERT((m_model != nullptr) ^ static_cast<bool>(m_request), "CLIPTextModel must have exactly one of m_model or m_request initialized");

    std::shared_ptr<CLIPTextModel> cloned = std::make_shared<CLIPTextModel>(*this);

    if (m_model) {
        cloned->m_model = m_model->clone();
    } else {
        cloned->m_request = m_request.get_compiled_model().create_infer_request();
    }

    return cloned;
}

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
    std::optional<AdapterConfig> adapters;
    auto filtered_properties = extract_adapters_from_properties(properties, &adapters);
    if (adapters) {
        adapters->set_tensor_name_prefix(adapters->get_tensor_name_prefix().value_or("lora_te"));
        m_adapter_controller = AdapterController(m_model, *adapters, device);
    }
    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_model, device, *filtered_properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "Clip Text model");
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
        ov::Tensor input_ids_token = m_clip_tokenizer.encode(prompt).input_ids;

        if (input_ids.get_element_type() == ov::element::i32) {
            std::fill_n(input_ids.data<int32_t>(), input_ids.get_size(), pad_token_id);
            std::copy_n(input_ids_token.data<int32_t>(), std::min(input_ids_token.get_size(), input_ids.get_size()), input_ids.data<int32_t>());
        } else {
            std::fill_n(input_ids.data<int64_t>(), input_ids.get_size(), pad_token_id);
            std::copy_n(input_ids_token.data<int64_t>(), std::min(input_ids_token.get_size(), input_ids.get_size()), input_ids.data<int64_t>());
        }
    };

    ov::PartialShape compiled_input_partial_shape = m_request.get_compiled_model().inputs()[0].get_partial_shape();

    ov::Tensor input_ids = m_request.get_input_tensor();

    if (compiled_input_partial_shape.is_dynamic()) {
        input_ids.set_shape({text_embedding_batch_size, m_config.max_position_embeddings});
    } else {
        auto compiled_input_shape = input_ids.get_shape();
        OPENVINO_ASSERT(compiled_input_shape.size() == 2, "CLIP text encoder model input must have rank of 2");
        OPENVINO_ASSERT(text_embedding_batch_size <= compiled_input_shape[0],
                        "text_embedding_batch_size (", text_embedding_batch_size,
                        ") > CLIP text encoder model batch size (",compiled_input_shape[0], ").");
        OPENVINO_ASSERT(m_config.max_position_embeddings == compiled_input_shape[1],
                        "max_position_embeddings (", m_config.max_position_embeddings,
                        ") != what CLIP text encoder model was compiled for (", compiled_input_shape[1], ").");
    }

    size_t current_batch_idx = 0;

    if (input_ids.get_shape()[0] == 2) {
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
    m_request.infer();

    // This is true when text_embedding_batch_size is 1, but model was reshaped / compiled as batch size 2.
    m_slice_batch1_output = (text_embedding_batch_size != input_ids.get_shape()[0]);

    return get_output_tensor(0);
}

ov::Tensor CLIPTextModel::get_output_tensor(const size_t idx) {
    auto infer_out_tensor = m_request.get_output_tensor(idx);
    if (m_slice_batch1_output) {
        //Slice and return batch index 1 output.
        auto out_shape = infer_out_tensor.get_shape();
        auto begin_coord = ov::Coordinate(out_shape.size(), 0);
        begin_coord[0] = 1;
        auto end_coord = ov::Coordinate(out_shape);
        auto sliced_out_tensor = ov::Tensor(infer_out_tensor, begin_coord, end_coord);
        return sliced_out_tensor;
    } else {
        return infer_out_tensor;
    }
}

void CLIPTextModel::export_model(const std::filesystem::path& blob_path) {
    OPENVINO_ASSERT(m_request, "CLIP text encoder model must be compiled first.");
    auto compiled_model = m_request.get_compiled_model();
    utils::export_model(compiled_model, blob_path / "openvino_model.blob");
}

void CLIPTextModel::import_model(const std::filesystem::path& blob_path,
                                 const std::string& device,
                                 const ov::AnyMap& properties) {
    OPENVINO_ASSERT(!m_request, "Model has been already compiled. Cannot re-compile already compiled model");
    auto compiled_model = utils::import_model(blob_path / "openvino_model.blob", device, properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "Clip Text model");
    m_request = compiled_model.create_infer_request();
}

} // namespace genai
} // namespace ov
