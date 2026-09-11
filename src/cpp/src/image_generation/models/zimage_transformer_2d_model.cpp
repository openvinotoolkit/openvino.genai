// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/zimage_transformer_2d_model.hpp"

#include <fstream>

#include "json_utils.hpp"
#include "utils.hpp"
#include "lora/helper.hpp"

namespace ov {
namespace genai {

constexpr size_t ZIMAGE_SEQUENCE_MULTIPLE = 32;
constexpr size_t ZIMAGE_PATCH_SIZE = 2;

size_t round_sequence_length(const size_t length) {
    return length + (ZIMAGE_SEQUENCE_MULTIPLE - length % ZIMAGE_SEQUENCE_MULTIPLE) % ZIMAGE_SEQUENCE_MULTIPLE;
}

size_t get_vae_scale_factor(const std::filesystem::path& vae_config_path);

ZImageTransformer2DModel::Config::Config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "in_channels", in_channels);
    read_json_param(data, "out_channels", out_channels);
    read_json_param(data, "sample_size", sample_size);
}

ZImageTransformer2DModel::ZImageTransformer2DModel(const std::filesystem::path& root_dir)
    : m_config(root_dir / "config.json") {
    m_model = utils::singleton_core().read_model(root_dir / "openvino_model.xml");
    m_vae_scale_factor = ov::genai::get_vae_scale_factor(root_dir.parent_path() / "vae_decoder" / "config.json");
}

ZImageTransformer2DModel::ZImageTransformer2DModel(const std::filesystem::path& root_dir,
                                                 const std::string& device,
                                                 const ov::AnyMap& properties)
    : ZImageTransformer2DModel(root_dir) {
    compile(device, properties);
}

ZImageTransformer2DModel::ZImageTransformer2DModel(const std::string& model,
                                                  const Tensor& weights,
                                                  const Config& config,
                                                  const size_t vae_scale_factor) :
    m_config(config), m_vae_scale_factor(vae_scale_factor) {
    m_model = utils::singleton_core().read_model(model, weights);
}

ZImageTransformer2DModel::ZImageTransformer2DModel(const std::string& model,
                                                  const Tensor& weights,
                                                  const Config& config,
                                                  const size_t vae_scale_factor,
                                                  const std::string& device,
                                                  const ov::AnyMap& properties) :
    ZImageTransformer2DModel(model, weights, config, vae_scale_factor) {
    compile(device, properties);
}

ZImageTransformer2DModel::ZImageTransformer2DModel(const ZImageTransformer2DModel&) = default;

ZImageTransformer2DModel ZImageTransformer2DModel::clone() {
    OPENVINO_ASSERT((m_model != nullptr) ^ static_cast<bool>(m_request), "ZImageTransformer2DModel must have exactly one of m_model or m_request initialized");

    ZImageTransformer2DModel cloned = *this;

    if (m_model) {
        cloned.m_model = m_model->clone();
    } else {
        cloned.m_request = m_request.get_compiled_model().create_infer_request();
    }

    return cloned;
}

const ZImageTransformer2DModel::Config& ZImageTransformer2DModel::get_config() const {
    return m_config;
}

ZImageTransformer2DModel& ZImageTransformer2DModel::reshape(int batch_size,
                                                           int height,
                                                           int width,
                                                           int tokenizer_model_max_length) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot reshape already compiled model");

    height /= m_vae_scale_factor;
    width /= m_vae_scale_factor;

    std::map<std::string, ov::PartialShape> name_to_shape;

    for (auto&& input : m_model->inputs()) {
        std::string input_name = input.get_any_name();
        name_to_shape[input_name] = input.get_partial_shape();
        if (input_name == "timestep") {
            name_to_shape[input_name][0] = 1;
        } else if (input_name == "hidden_states") {
            name_to_shape[input_name] = {
                batch_size,
                static_cast<int64_t>(m_config.in_channels),
                height,
                width,
            };
        } else if (input_name == "encoder_hidden_states") {
            name_to_shape[input_name] = {batch_size, tokenizer_model_max_length, name_to_shape[input_name][2]};
        } else if (input_name == "encoder_attention_mask") {
            name_to_shape[input_name] = {batch_size, tokenizer_model_max_length};
        } else if (input_name == "txt_ids") {
            name_to_shape[input_name] = {batch_size, tokenizer_model_max_length, 3};
        } else if (input_name == "img_ids") {
            name_to_shape[input_name] = {batch_size, -1, 3};
        }
    }

    m_model->reshape(name_to_shape);

    return *this;
}

ZImageTransformer2DModel& ZImageTransformer2DModel::compile(const std::string& device, const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model, "Model has been already compiled. Cannot re-compile already compiled model");
    std::optional<AdapterConfig> adapters;
    auto filtered_properties = extract_adapters_from_properties(properties, &adapters);
    if (adapters) {
        adapters->set_tensor_name_prefix(adapters->get_tensor_name_prefix().value_or("transformer"));
        m_adapter_controller = AdapterController(m_model, *adapters, device);
    }
    ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_model, device, *filtered_properties);
    ov::genai::utils::print_compiled_model_properties(compiled_model, "ZImage Transformer 2D model");
    m_request = compiled_model.create_infer_request();
    m_model.reset();

    return *this;
}

void ZImageTransformer2DModel::set_hidden_states(const std::string& tensor_name, ov::Tensor encoder_hidden_states) {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first");
    m_request.set_tensor(tensor_name, encoder_hidden_states);
}

void ZImageTransformer2DModel::set_adapters(const std::optional<AdapterConfig>& adapters) {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first");
    if(adapters) {
        m_adapter_controller.apply(m_request, *adapters);
    }
}

ov::Tensor ZImageTransformer2DModel::step(ov::Tensor sample, ov::Tensor timestep, ov::Tensor encoder_hidden_states) {
    OPENVINO_ASSERT(m_request, "Transformer model must be compiled first");

    const ov::Shape& encoder_shape = encoder_hidden_states.get_shape();
    const size_t batch_size = encoder_shape[0], max_caption_length = encoder_shape[1], embedding_size = encoder_shape[2];
    const size_t image_height = sample.get_shape()[2] / ZIMAGE_PATCH_SIZE;
    const size_t image_width = sample.get_shape()[3] / ZIMAGE_PATCH_SIZE;
    const size_t image_length = image_height * image_width;
    const size_t padded_image_length = round_sequence_length(image_length);

    ov::Tensor encoder_attention_mask(ov::element::f32, {batch_size, max_caption_length});
    ov::Tensor txt_ids(ov::element::i32, {batch_size, max_caption_length, 3});
    ov::Tensor img_ids(ov::element::i32, {batch_size, padded_image_length, 3});
    std::fill_n(encoder_attention_mask.data<float>(), encoder_attention_mask.get_size(),
                std::numeric_limits<float>::lowest());
    std::fill_n(txt_ids.data<int32_t>(), txt_ids.get_size(), 0);
    std::fill_n(img_ids.data<int32_t>(), img_ids.get_size(), 0);

    const float* embeddings = encoder_hidden_states.data<const float>();
    float* attention_mask = encoder_attention_mask.data<float>();
    int32_t* text_positions = txt_ids.data<int32_t>();
    int32_t* image_positions = img_ids.data<int32_t>();

    for (size_t batch = 0; batch < batch_size; ++batch) {
        size_t caption_length = max_caption_length;
        while (caption_length > 0 &&
               std::all_of(embeddings + (batch * max_caption_length + caption_length - 1) * embedding_size,
                           embeddings + (batch * max_caption_length + caption_length) * embedding_size,
                           [](const float value) { return value == 0.0f; })) {
            --caption_length;
        }
        const size_t rounded_caption_length = std::min(round_sequence_length(caption_length), max_caption_length);
        std::fill_n(attention_mask + batch * max_caption_length, caption_length, 0.0f);
        for (size_t token = 0; token < rounded_caption_length; ++token) {
            text_positions[(batch * max_caption_length + token) * 3] = static_cast<int32_t>(token + 1);
        }
        for (size_t token = 0; token < image_length; ++token) {
            int32_t* position = image_positions + (batch * padded_image_length + token) * 3;
            position[0] = static_cast<int32_t>(rounded_caption_length + 1);
            position[1] = static_cast<int32_t>(token / image_width);
            position[2] = static_cast<int32_t>(token % image_width);
        }
    }

    m_request.set_tensor("hidden_states", sample);
    m_request.set_tensor("timestep", timestep);
    m_request.set_tensor("encoder_hidden_states", encoder_hidden_states);
    m_request.set_tensor("encoder_attention_mask", encoder_attention_mask);
    m_request.set_tensor("txt_ids", txt_ids);
    m_request.set_tensor("img_ids", img_ids);
    
    m_request.infer();
    
    return m_request.get_output_tensor();
}

}  // namespace genai
}  // namespace ov
