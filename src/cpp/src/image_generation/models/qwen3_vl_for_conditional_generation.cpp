// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "openvino/genai/image_generation/qwen3_vl_for_conditional_generation.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <numeric>
#include <type_traits>

#include "json_utils.hpp"
#include "lora/helper.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

namespace {

constexpr double VISION_ROPE_THETA = 10000.0;

// The exporter stores the Qwen3-VL processor (and its tokenizer) in 'processor', while models converted with
// the diffusers layout keep it in 'tokenizer'.
std::filesystem::path get_qwen_image21_tokenizer_path(const std::filesystem::path& text_encoder_path) {
    const std::filesystem::path root_dir = text_encoder_path.parent_path();
    for (const std::string& subfolder : {"processor", "tokenizer"}) {
        const std::filesystem::path candidate = root_dir / subfolder;
        if (std::filesystem::exists(candidate / "openvino_tokenizer.xml")) {
            return candidate;
        }
    }
    OPENVINO_THROW("Failed to find 'openvino_tokenizer.xml' neither in '",
                   root_dir / "processor", "' nor in '", root_dir / "tokenizer", "'");
}

// Flattens a normalized (1, 3, H, W) image into the (num_patches, channels * temporal_patch_size * patch_size ^ 2)
// layout the vision tower consumes. Tokens are ordered by spatial merge blocks; the single frame is repeated over
// the temporal axis.
ov::Tensor build_pixel_values(const ov::Tensor image,
                              const Qwen3VLForConditionalGeneration::VisionConfig& config,
                              const size_t grid_height,
                              const size_t grid_width) {
    const ov::Shape& shape = image.get_shape();
    const size_t channels = shape[1], height = shape[2], width = shape[3];
    const size_t patch_size = config.patch_size, merge_size = config.spatial_merge_size;
    const size_t patch_dim = channels * config.temporal_patch_size * patch_size * patch_size;

    ov::Tensor pixel_values(ov::element::f32, {grid_height * grid_width, patch_dim});
    const float* src_data = image.data<const float>();
    float* dst_data = pixel_values.data<float>();

    for (size_t merge_row = 0; merge_row < grid_height / merge_size; ++merge_row) {
        for (size_t merge_col = 0; merge_col < grid_width / merge_size; ++merge_col) {
            for (size_t intra_row = 0; intra_row < merge_size; ++intra_row) {
                for (size_t intra_col = 0; intra_col < merge_size; ++intra_col) {
                    const size_t patch_row = merge_row * merge_size + intra_row;
                    const size_t patch_col = merge_col * merge_size + intra_col;
                    float* patch_data = dst_data;
                    dst_data += patch_dim;

                    for (size_t channel = 0; channel < channels; ++channel) {
                        const float* channel_data = src_data + channel * height * width;
                        for (size_t frame = 0; frame < config.temporal_patch_size; ++frame) {
                            for (size_t row = 0; row < patch_size; ++row) {
                                const float* row_data =
                                    channel_data + (patch_row * patch_size + row) * width + patch_col * patch_size;
                                std::copy_n(row_data, patch_size, patch_data);
                                patch_data += patch_size;
                            }
                        }
                    }
                }
            }
        }
    }

    return pixel_values;
}

// Gather indices and weights that bilinearly resample the learned position embedding grid onto the image grid.
// The graph consumes them in spatial merge order, so the merge permutation is applied here.
std::pair<ov::Tensor, ov::Tensor> build_bilinear_indices_and_weights(
        const Qwen3VLForConditionalGeneration::VisionConfig& config,
        const size_t grid_height,
        const size_t grid_width) {
    const size_t num_grid_per_side = static_cast<size_t>(std::sqrt(config.num_position_embeddings));
    OPENVINO_ASSERT(num_grid_per_side * num_grid_per_side == config.num_position_embeddings,
                    "'num_position_embeddings' (", config.num_position_embeddings, ") must be a perfect square");

    const size_t merge_size = config.spatial_merge_size;
    const size_t num_patches = grid_height * grid_width;

    const auto grid_coordinate = [num_grid_per_side](const size_t index, const size_t extent) {
        return extent > 1 ? static_cast<float>(index) * static_cast<float>(num_grid_per_side - 1) / (extent - 1)
                          : 0.0f;
    };

    ov::Tensor indices(ov::element::i32, {4, num_patches});
    ov::Tensor weights(ov::element::f32, {4, num_patches});
    int32_t* indices_data = indices.data<int32_t>();
    float* weights_data = weights.data<float>();

    const int32_t grid_max = static_cast<int32_t>(num_grid_per_side - 1);
    size_t token = 0;
    for (size_t merge_row = 0; merge_row < grid_height / merge_size; ++merge_row) {
        for (size_t merge_col = 0; merge_col < grid_width / merge_size; ++merge_col) {
            for (size_t intra_row = 0; intra_row < merge_size; ++intra_row) {
                for (size_t intra_col = 0; intra_col < merge_size; ++intra_col) {
                    const float row_coordinate = grid_coordinate(merge_row * merge_size + intra_row, grid_height);
                    const float col_coordinate = grid_coordinate(merge_col * merge_size + intra_col, grid_width);

                    const int32_t row_floor = static_cast<int32_t>(row_coordinate);
                    const int32_t col_floor = static_cast<int32_t>(col_coordinate);
                    const int32_t row_ceil = std::min(row_floor + 1, grid_max);
                    const int32_t col_ceil = std::min(col_floor + 1, grid_max);
                    const float row_frac = row_coordinate - static_cast<float>(row_floor);
                    const float col_frac = col_coordinate - static_cast<float>(col_floor);

                    const int32_t floor_base = row_floor * static_cast<int32_t>(num_grid_per_side);
                    const int32_t ceil_base = row_ceil * static_cast<int32_t>(num_grid_per_side);

                    indices_data[0 * num_patches + token] = floor_base + col_floor;
                    indices_data[1 * num_patches + token] = floor_base + col_ceil;
                    indices_data[2 * num_patches + token] = ceil_base + col_floor;
                    indices_data[3 * num_patches + token] = ceil_base + col_ceil;

                    weights_data[0 * num_patches + token] = (1.0f - row_frac) * (1.0f - col_frac);
                    weights_data[1 * num_patches + token] = (1.0f - row_frac) * col_frac;
                    weights_data[2 * num_patches + token] = row_frac * (1.0f - col_frac);
                    weights_data[3 * num_patches + token] = row_frac * col_frac;
                    ++token;
                }
            }
        }
    }

    return {indices, weights};
}

// Vision rotary table over the (row, column) patch coordinates, in spatial merge order.
std::pair<ov::Tensor, ov::Tensor> build_vision_rotary_embeddings(
        const Qwen3VLForConditionalGeneration::VisionConfig& config,
        const size_t grid_height,
        const size_t grid_width) {
    const size_t head_dim = config.hidden_size / config.num_heads;
    const size_t axis_dim = head_dim / 2;
    const size_t num_frequencies = axis_dim / 2;
    const size_t merge_size = config.spatial_merge_size;
    const size_t num_patches = grid_height * grid_width;

    std::vector<double> inverse_frequencies(num_frequencies);
    for (size_t i = 0; i < num_frequencies; ++i) {
        inverse_frequencies[i] =
            1.0 / std::pow(VISION_ROPE_THETA, 2.0 * static_cast<double>(i) / static_cast<double>(axis_dim));
    }

    ov::Tensor cos(ov::element::f32, {num_patches, head_dim});
    ov::Tensor sin(ov::element::f32, {num_patches, head_dim});
    float* cos_data = cos.data<float>();
    float* sin_data = sin.data<float>();

    size_t token = 0;
    for (size_t merge_row = 0; merge_row < grid_height / merge_size; ++merge_row) {
        for (size_t merge_col = 0; merge_col < grid_width / merge_size; ++merge_col) {
            for (size_t intra_row = 0; intra_row < merge_size; ++intra_row) {
                for (size_t intra_col = 0; intra_col < merge_size; ++intra_col) {
                    const std::array<size_t, 2> positions{merge_row * merge_size + intra_row,
                                                          merge_col * merge_size + intra_col};
                    float* token_cos = cos_data + token * head_dim;
                    float* token_sin = sin_data + token * head_dim;

                    for (size_t axis = 0; axis < positions.size(); ++axis) {
                        for (size_t i = 0; i < num_frequencies; ++i) {
                            const double angle = static_cast<double>(positions[axis]) * inverse_frequencies[i];
                            const size_t offset = axis * num_frequencies + i;
                            token_cos[offset] = token_cos[axis_dim + offset] = static_cast<float>(std::cos(angle));
                            token_sin[offset] = token_sin[axis_dim + offset] = static_cast<float>(std::sin(angle));
                        }
                    }
                    ++token;
                }
            }
        }
    }

    return {cos, sin};
}

// 3D M-RoPE position ids. Text tokens advance a shared position on all three axes; the image block lays its
// tokens out on a temporal/height/width grid starting at the position reached by the preceding text.
ov::Tensor build_mrope_position_ids(const ov::Tensor input_ids,
                                    const int64_t image_token_id,
                                    const size_t merged_height,
                                    const size_t merged_width) {
    const size_t sequence_length = input_ids.get_shape()[1];
    const int64_t* ids_data = input_ids.data<const int64_t>();

    ov::Tensor position_ids(ov::element::i64, {3, 1, sequence_length});
    int64_t* position_data = position_ids.data<int64_t>();

    int64_t current_position = 0;
    for (size_t token = 0; token < sequence_length;) {
        if (ids_data[token] != image_token_id) {
            for (size_t axis = 0; axis < 3; ++axis) {
                position_data[axis * sequence_length + token] = current_position;
            }
            ++current_position;
            ++token;
            continue;
        }

        for (size_t row = 0; row < merged_height; ++row) {
            for (size_t col = 0; col < merged_width; ++col, ++token) {
                position_data[0 * sequence_length + token] = current_position;
                position_data[1 * sequence_length + token] = current_position + static_cast<int64_t>(row);
                position_data[2 * sequence_length + token] = current_position + static_cast<int64_t>(col);
            }
        }
        current_position += static_cast<int64_t>(std::max(merged_height, merged_width));
    }

    return position_ids;
}

// Scatters the DeepStack features into the image-pad positions of a dense additive tensor.
ov::Tensor build_deepstack_dense(const std::vector<ov::Tensor>& deepstack_features,
                                 const ov::Tensor input_ids,
                                 const int64_t image_token_id,
                                 const size_t hidden_size) {
    const size_t sequence_length = input_ids.get_shape()[1];
    const int64_t* ids_data = input_ids.data<const int64_t>();

    ov::Tensor dense(ov::element::f32, {deepstack_features.size(), 1, sequence_length, hidden_size});
    float* dense_data = dense.data<float>();
    std::fill_n(dense_data, dense.get_size(), 0.0f);

    for (size_t layer = 0; layer < deepstack_features.size(); ++layer) {
        const float* feature_data = deepstack_features[layer].data<const float>();
        float* layer_data = dense_data + layer * sequence_length * hidden_size;
        size_t image_token = 0;
        for (size_t token = 0; token < sequence_length; ++token) {
            if (ids_data[token] == image_token_id) {
                std::copy_n(feature_data + image_token * hidden_size, hidden_size, layer_data + token * hidden_size);
                ++image_token;
            }
        }
    }

    return dense;
}

}  // namespace

// The prompt is built as a raw template string instead of going through the chat template: the two tokenize
// differently and the checkpoint expects this one.
const std::string Qwen3VLForConditionalGeneration::SYSTEM_PREFIX =
    "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n";

const std::string Qwen3VLForConditionalGeneration::PROMPT_TEMPLATE =
    Qwen3VLForConditionalGeneration::SYSTEM_PREFIX + "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n";

// The vision prefix only appears in the image-conditioned template.
const std::string Qwen3VLForConditionalGeneration::PROMPT_TEMPLATE_WITH_IMAGE =
    Qwen3VLForConditionalGeneration::SYSTEM_PREFIX +
    "<|im_start|>user\nPicture 1: <|vision_start|><|image_pad|><|vision_end|>{}<|im_end|>\n<|im_start|>assistant\n";

Qwen3VLForConditionalGeneration::Config::Config(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "hidden_size", hidden_size);
    read_json_param(data, "_qwenimage21_image_token_id", image_token_id);
}

Qwen3VLForConditionalGeneration::VisionConfig::VisionConfig(const std::filesystem::path& config_path) {
    std::ifstream file(config_path);
    OPENVINO_ASSERT(file.is_open(), "Failed to open ", config_path);

    nlohmann::json data = nlohmann::json::parse(file);
    using utils::read_json_param;

    read_json_param(data, "hidden_size", hidden_size);
    read_json_param(data, "num_heads", num_heads);
    read_json_param(data, "in_channels", in_channels);
    read_json_param(data, "patch_size", patch_size);
    read_json_param(data, "temporal_patch_size", temporal_patch_size);
    read_json_param(data, "spatial_merge_size", spatial_merge_size);
    read_json_param(data, "num_position_embeddings", num_position_embeddings);

    OPENVINO_ASSERT(data.contains("deepstack_visual_indexes"),
                    "Qwen3-VL vision config must contain 'deepstack_visual_indexes'");
    num_deepstack_layers = data["deepstack_visual_indexes"].size();
}

Qwen3VLForConditionalGeneration::ImageSize Qwen3VLForConditionalGeneration::calculate_dimensions(
        const size_t target_area,
        const double aspect_ratio) {
    OPENVINO_ASSERT(aspect_ratio > 0.0, "Image aspect ratio must be positive, got ", aspect_ratio);
    // The models require both sides to be multiples of the VAE scale factor times the transformer's 2x2 grouping.
    constexpr double granularity = 32.0;
    const double width = std::sqrt(static_cast<double>(target_area) * aspect_ratio);
    const double height = width / aspect_ratio;
    return {static_cast<size_t>(std::round(height / granularity)) * static_cast<size_t>(granularity),
            static_cast<size_t>(std::round(width / granularity)) * static_cast<size_t>(granularity)};
}

Qwen3VLForConditionalGeneration::Qwen3VLForConditionalGeneration(const std::filesystem::path& root_dir)
    : m_config(root_dir / "config.json"),
      m_tokenizer(get_qwen_image21_tokenizer_path(root_dir)),
      m_system_prefix_length(
          m_tokenizer.encode(SYSTEM_PREFIX, ov::genai::add_special_tokens(false)).input_ids.get_shape()[1]) {
    m_model = utils::singleton_core().read_model(root_dir / "openvino_model.xml");
}

Qwen3VLForConditionalGeneration::Qwen3VLForConditionalGeneration(const std::filesystem::path& root_dir,
                                                                 const std::string& device,
                                                                 const ov::AnyMap& properties)
    : Qwen3VLForConditionalGeneration(root_dir) {
    compile(device, properties);
}

Qwen3VLForConditionalGeneration::Qwen3VLForConditionalGeneration(const std::filesystem::path& root_dir,
                                                                 const std::filesystem::path& vision_encoder_path,
                                                                 const std::filesystem::path& text_encoder_i2i_path)
    : m_config(root_dir / "config.json"),
      m_vision_config(vision_encoder_path / "config.json"),
      m_tokenizer(get_qwen_image21_tokenizer_path(root_dir)),
      m_system_prefix_length(
          m_tokenizer.encode(SYSTEM_PREFIX, ov::genai::add_special_tokens(false)).input_ids.get_shape()[1]) {
    m_vision_model = utils::singleton_core().read_model(vision_encoder_path / "openvino_model.xml");
    m_i2i_model = utils::singleton_core().read_model(text_encoder_i2i_path / "openvino_model.xml");
}

Qwen3VLForConditionalGeneration::Qwen3VLForConditionalGeneration(const Qwen3VLForConditionalGeneration&) = default;

bool Qwen3VLForConditionalGeneration::has_vision_tower() const {
    return (m_vision_model != nullptr || static_cast<bool>(m_vision_request)) &&
           (m_i2i_model != nullptr || static_cast<bool>(m_i2i_request));
}

std::shared_ptr<Qwen3VLForConditionalGeneration> Qwen3VLForConditionalGeneration::clone() {
    OPENVINO_ASSERT(!(m_model && m_request),
                    "Qwen3VLForConditionalGeneration must not have both m_model and m_request initialized");

    std::shared_ptr<Qwen3VLForConditionalGeneration> cloned =
        std::make_shared<Qwen3VLForConditionalGeneration>(*this);

    if (m_model) {
        cloned->m_model = m_model->clone();
    } else if (m_request) {
        cloned->m_request = m_request.get_compiled_model().create_infer_request();
    }

    if (m_vision_model) {
        cloned->m_vision_model = m_vision_model->clone();
    } else if (m_vision_request) {
        cloned->m_vision_request = m_vision_request.get_compiled_model().create_infer_request();
    }

    if (m_i2i_model) {
        cloned->m_i2i_model = m_i2i_model->clone();
    } else if (m_i2i_request) {
        cloned->m_i2i_request = m_i2i_request.get_compiled_model().create_infer_request();
    }

    return cloned;
}

Qwen3VLForConditionalGeneration& Qwen3VLForConditionalGeneration::compile(const std::string& device,
                                                                          const ov::AnyMap& properties) {
    OPENVINO_ASSERT(m_model || m_vision_model,
                    "Model has been already compiled. Cannot re-compile already compiled model");
    std::optional<AdapterConfig> adapters;
    auto filtered_properties = extract_adapters_from_properties(properties, &adapters);

    if (m_model) {
        if (adapters) {
            adapters->set_tensor_name_prefix(adapters->get_tensor_name_prefix().value_or("text_encoder"));
            m_adapter_controller = AdapterController(m_model, *adapters, device);
        }
        ov::CompiledModel compiled_model = utils::singleton_core().compile_model(m_model, device, *filtered_properties);
        ov::genai::utils::print_compiled_model_properties(compiled_model, "QwenImage 2.1 text encoder model");
        m_request = compiled_model.create_infer_request();
        // release the original model
        m_model.reset();
    }

    if (m_vision_model) {
        ov::CompiledModel compiled_vision =
            utils::singleton_core().compile_model(m_vision_model, device, *filtered_properties);
        ov::genai::utils::print_compiled_model_properties(compiled_vision, "QwenImage 2.1 vision encoder model");
        m_vision_request = compiled_vision.create_infer_request();
        m_vision_model.reset();

        ov::CompiledModel compiled_i2i =
            utils::singleton_core().compile_model(m_i2i_model, device, *filtered_properties);
        ov::genai::utils::print_compiled_model_properties(compiled_i2i,
                                                          "QwenImage 2.1 image conditioned text encoder model");
        m_i2i_request = compiled_i2i.create_infer_request();
        m_i2i_model.reset();
    }

    return *this;
}

ov::Tensor Qwen3VLForConditionalGeneration::drop_system_prefix(const ov::Tensor hidden_states,
                                                               const size_t prompt_length) const {
    const size_t hidden_size = hidden_states.get_shape()[2];
    ov::Tensor prompt_embeds(ov::element::f32, {1, prompt_length, hidden_size});
    std::memcpy(prompt_embeds.data<float>(),
                hidden_states.data<const float>() + m_system_prefix_length * hidden_size,
                prompt_length * hidden_size * sizeof(float));
    return prompt_embeds;
}

ov::Tensor Qwen3VLForConditionalGeneration::infer(const std::string& prompt, const int max_sequence_length) {
    OPENVINO_ASSERT(m_request, "QwenImage 2.1 text encoder model must be compiled first. Cannot infer non-compiled model");
    OPENVINO_ASSERT(max_sequence_length > 0, "'max_sequence_length' must be positive, got ", max_sequence_length);

    std::string formatted_prompt = PROMPT_TEMPLATE;
    const std::string placeholder = "{}";
    const size_t placeholder_pos = formatted_prompt.find(placeholder);
    OPENVINO_ASSERT(placeholder_pos != std::string::npos, "Prompt template must contain '{}'");
    formatted_prompt.replace(placeholder_pos, placeholder.length(), prompt);

    const ov::Tensor token_ids =
        m_tokenizer.encode(formatted_prompt, ov::genai::add_special_tokens(false)).input_ids;
    const size_t token_count = token_ids.get_shape()[1];

    OPENVINO_ASSERT(token_count > m_system_prefix_length,
                    "Tokenized prompt length (", token_count, ") must be greater than the system prefix length (",
                    m_system_prefix_length, ")");
    const size_t prompt_length = token_count - m_system_prefix_length;
    OPENVINO_ASSERT(prompt_length <= static_cast<size_t>(max_sequence_length),
                    "Tokenized prompt length (", prompt_length, ") exceeds 'max_sequence_length' (",
                    max_sequence_length, ")");

    const ov::element::Type input_type = m_request.get_compiled_model().input("input_ids").get_element_type();
    ov::Tensor input_ids(input_type, {1, token_count});
    ov::Tensor attention_mask(input_type, {1, token_count});

    if (input_type == ov::element::i32) {
        std::copy_n(token_ids.data<const int64_t>(), token_count, input_ids.data<int32_t>());
        std::fill_n(attention_mask.data<int32_t>(), token_count, int32_t{1});
    } else {
        std::copy_n(token_ids.data<const int64_t>(), token_count, input_ids.data<int64_t>());
        std::fill_n(attention_mask.data<int64_t>(), token_count, int64_t{1});
    }

    m_request.set_tensor("input_ids", input_ids);
    m_request.set_tensor("attention_mask", attention_mask);
    m_request.infer();

    m_image_pad_mask = ov::Tensor(ov::element::boolean, {1, prompt_length});
    std::fill_n(m_image_pad_mask.data<bool>(), prompt_length, false);

    return drop_system_prefix(m_request.get_output_tensor(), prompt_length);
}

ov::Tensor Qwen3VLForConditionalGeneration::infer_vision_tower(const ov::Tensor condition_image,
                                                               std::vector<ov::Tensor>& deepstack_features) {
    const ov::Shape& image_shape = condition_image.get_shape();
    const size_t patch_size = m_vision_config.patch_size;
    const size_t granularity = patch_size * m_vision_config.spatial_merge_size;
    OPENVINO_ASSERT(image_shape.size() == 4 && image_shape[0] == 1 && image_shape[1] == m_vision_config.in_channels,
                    "Condition image must have shape (1, ", m_vision_config.in_channels, ", height, width), got ",
                    image_shape);
    OPENVINO_ASSERT(image_shape[2] % granularity == 0 && image_shape[3] % granularity == 0,
                    "Condition image height and width must be divisible by ", granularity, ", got ",
                    image_shape[2], "x", image_shape[3]);

    const size_t grid_height = image_shape[2] / patch_size, grid_width = image_shape[3] / patch_size;

    const ov::Tensor pixel_values = build_pixel_values(condition_image, m_vision_config, grid_height, grid_width);
    const auto [bilinear_indices, bilinear_weights] =
        build_bilinear_indices_and_weights(m_vision_config, grid_height, grid_width);
    const auto [cos, sin] = build_vision_rotary_embeddings(m_vision_config, grid_height, grid_width);

    m_vision_request.set_tensor("pixel_values", pixel_values);
    m_vision_request.set_tensor("bilinear_indices", bilinear_indices);
    m_vision_request.set_tensor("bilinear_weights", bilinear_weights);
    m_vision_request.set_tensor("cos", cos);
    m_vision_request.set_tensor("sin", sin);
    m_vision_request.infer();

    deepstack_features.clear();
    for (size_t layer = 0; layer < m_vision_config.num_deepstack_layers; ++layer) {
        deepstack_features.push_back(m_vision_request.get_output_tensor(layer + 1));
    }

    return m_vision_request.get_output_tensor(0);
}

ov::Tensor Qwen3VLForConditionalGeneration::infer(const std::string& prompt,
                                                  const ov::Tensor condition_image,
                                                  const int max_sequence_length) {
    OPENVINO_ASSERT(m_i2i_request && m_vision_request,
                    "QwenImage 2.1 vision tower must be compiled first. Cannot infer non-compiled model");
    OPENVINO_ASSERT(max_sequence_length > 0, "'max_sequence_length' must be positive, got ", max_sequence_length);

    std::vector<ov::Tensor> deepstack_features;
    const ov::Tensor image_embeds = infer_vision_tower(condition_image, deepstack_features);
    const size_t num_image_tokens = image_embeds.get_shape()[0];

    std::string formatted_prompt = PROMPT_TEMPLATE_WITH_IMAGE;
    const std::string placeholder = "{}";
    const size_t placeholder_pos = formatted_prompt.find(placeholder);
    OPENVINO_ASSERT(placeholder_pos != std::string::npos, "Prompt template must contain '{}'");
    formatted_prompt.replace(placeholder_pos, placeholder.length(), prompt);

    // The template carries a single '<|image_pad|>'; the processor expands it into one token per merged vision
    // patch, so the placeholder is repeated here to match the vision tower output length.
    const ov::Tensor template_ids =
        m_tokenizer.encode(formatted_prompt, ov::genai::add_special_tokens(false)).input_ids;
    const size_t template_length = template_ids.get_shape()[1];
    const int64_t* template_data = template_ids.data<const int64_t>();

    const std::ptrdiff_t template_image_tokens =
        std::count(template_data, template_data + template_length, m_config.image_token_id);
    OPENVINO_ASSERT(template_image_tokens == 1,
                    "Image conditioned prompt must contain exactly one '<|image_pad|>' token, got ",
                    template_image_tokens, ". Remove '<|image_pad|>' from the prompt");

    const int64_t* image_token_position =
        std::find(template_data, template_data + template_length, m_config.image_token_id);
    const size_t prefix_length = static_cast<size_t>(image_token_position - template_data);
    const size_t token_count = template_length - 1 + num_image_tokens;

    OPENVINO_ASSERT(token_count > m_system_prefix_length,
                    "Tokenized prompt length (", token_count, ") must be greater than the system prefix length (",
                    m_system_prefix_length, ")");
    const size_t prompt_length = token_count - m_system_prefix_length;
    OPENVINO_ASSERT(prompt_length <= static_cast<size_t>(max_sequence_length),
                    "Tokenized prompt length (", prompt_length, ") exceeds 'max_sequence_length' (",
                    max_sequence_length, ")");

    ov::Tensor input_ids(ov::element::i64, {1, token_count});
    int64_t* ids_data = input_ids.data<int64_t>();
    std::copy_n(template_data, prefix_length, ids_data);
    std::fill_n(ids_data + prefix_length, num_image_tokens, m_config.image_token_id);
    std::copy_n(template_data + prefix_length + 1,
                template_length - prefix_length - 1,
                ids_data + prefix_length + num_image_tokens);

    ov::Tensor attention_mask(ov::element::i64, {1, token_count});
    std::fill_n(attention_mask.data<int64_t>(), token_count, int64_t{1});

    const size_t merge_size = m_vision_config.spatial_merge_size;
    const ov::Tensor position_ids = build_mrope_position_ids(input_ids,
                                                             m_config.image_token_id,
                                                             condition_image.get_shape()[2] / m_vision_config.patch_size / merge_size,
                                                             condition_image.get_shape()[3] / m_vision_config.patch_size / merge_size);
    const ov::Tensor deepstack_dense =
        build_deepstack_dense(deepstack_features, input_ids, m_config.image_token_id, m_config.hidden_size);

    m_i2i_request.set_tensor("input_ids", input_ids);
    m_i2i_request.set_tensor("image_embeds", image_embeds);
    m_i2i_request.set_tensor("attention_mask", attention_mask);
    m_i2i_request.set_tensor("position_ids", position_ids);
    m_i2i_request.set_tensor("deepstack_dense", deepstack_dense);
    m_i2i_request.infer();

    m_image_pad_mask = ov::Tensor(ov::element::boolean, {1, prompt_length});
    bool* mask_data = m_image_pad_mask.data<bool>();
    for (size_t token = 0; token < prompt_length; ++token) {
        mask_data[token] = ids_data[m_system_prefix_length + token] == m_config.image_token_id;
    }

    return drop_system_prefix(m_i2i_request.get_output_tensor(), prompt_length);
}

ov::Tensor Qwen3VLForConditionalGeneration::get_image_pad_mask() const {
    OPENVINO_ASSERT(m_image_pad_mask, "Image pad mask is not available. Run infer() first");
    return m_image_pad_mask;
}

void Qwen3VLForConditionalGeneration::set_adapters(const std::optional<AdapterConfig>& adapters) {
    OPENVINO_ASSERT(m_request, "Text encoder model must be compiled first");
    if (adapters) {
        m_adapter_controller.apply(m_request, *adapters);
    }
}

const Qwen3VLForConditionalGeneration::Config& Qwen3VLForConditionalGeneration::get_config() const {
    return m_config;
}

const Qwen3VLForConditionalGeneration::VisionConfig& Qwen3VLForConditionalGeneration::get_vision_config() const {
    return m_vision_config;
}

}  // namespace genai
}  // namespace ov
