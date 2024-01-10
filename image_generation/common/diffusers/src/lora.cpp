// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "lora.hpp"

#include <algorithm>
#include <set>
#include <map>
#include <string>
#include <vector>
#include <fstream>

#include <Eigen/Dense>

#include "openvino/op/add.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#define SAFETENSORS_IMPLEMENTATION
#include "safetensors.h"

InsertLoRA::InsertLoRA(LoRAMap& lora_map) :
    m_lora_map(&lora_map) {
    OPENVINO_ASSERT(!m_lora_map->empty(), "Map with LoRA weights is empty");

    auto pattern = ov::pass::pattern::wrap_type<ov::op::v0::MatMul, ov::op::v1::Convolution>();

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        auto root = m.get_match_root();
        if (!root) {
            return false;
        }
        std::string root_name = root->get_friendly_name();
        std::replace(root_name.begin(), root_name.end(), '.', '_');

        auto it = m_lora_map->begin();
        while (it != m_lora_map->end()) {
            if (root_name.find(it->first) != std::string::npos) {
                ov::Output<ov::Node> weights_port = root->input_value(1);
                std::set<ov::Input<ov::Node>> consumers = weights_port.get_target_inputs();
                auto reshaped_const = std::make_shared<ov::op::v0::Constant>(*(it->second), weights_port.get_shape());
                auto lora_add = std::make_shared<ov::op::v1::Add>(weights_port, reshaped_const);
                for (auto consumer : consumers) {
                    consumer.replace_source_output(lora_add->output(0));
                }
                register_new_node(lora_add);
                it = m_lora_map->erase(it);
                break;
            } else {
                it++;
            }
        }
        return true;
    };

    // Register pattern with Parameter operation as a pattern root node
    auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern, "InsertLoRA");
    // Register Matcher
    register_matcher(m, callback);
}

namespace {

std::vector<std::uint8_t> read_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    OPENVINO_ASSERT(file.is_open(), "Cannot open file ", filename, " with LoRA weights");

    size_t filesize = file.tellg();
    std::vector<std::uint8_t> buffer;
    buffer.reserve(filesize);

    file.seekg(0, std::ios::beg);
    std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), std::back_inserter(buffer));

    return buffer;
}

std::vector<float> convert_to_float(const safetensors_TensorDescriptor& tensor) {
    std::vector<float> data;
    size_t tensor_size = (tensor.end_offset_bytes - tensor.begin_offset_bytes) / sizeof(ov::float16);

    const ov::float16* ptr = static_cast<const ov::float16*>(tensor.ptr);
    for (size_t i = 0; i < tensor_size; ++i) {
        data.push_back(ptr[i]);
    }

    return data;
}

} // namespace

std::map<std::string, InsertLoRA::LoRAMap>
read_lora_adapters(const std::string& filename, const float alpha) {
    std::vector<std::uint8_t> file_buffer = read_file(filename);
    void* buffer_ptr = file_buffer.data();

    safetensors_File safe_tensors_file = {0};
    OPENVINO_ASSERT(safetensors_file_init(buffer_ptr, file_buffer.size(), &safe_tensors_file) == NULL, "Cannot parse ", filename, " using safetensors");

    using FloatMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    using FloatMatrixMap = Eigen::Map<FloatMatrix>;

    // modify the layer name
    std::map<std::string, InsertLoRA::LoRAMap> lora_constants;

    std::set<std::string> visited;
    const std::string LORA_PREFIX_UNET = "lora_unet";
    const std::string LORA_PREFIX_TEXT_ENCODER = "lora_te";

    // loading safetensor
    for (int i = 0; i < safe_tensors_file.num_tensors; i++) {
        std::map<std::string, std::string> lora_map;

        safetensors_TensorDescriptor tensor = safe_tensors_file.tensors[i];
        std::string tensor_name(tensor.name.ptr, tensor.name.ptr + tensor.name.len);

        const bool tensor_visited = std::find(visited.begin(), visited.end(), tensor_name) != visited.end();
        // alpha tensors are overriden by users' alpha
        bool alpha_tensor = tensor_name.find(".alpha") != std::string::npos;
        if (alpha_tensor || tensor_visited)
            continue;

        const bool is_text_lora = tensor_name.find("text") != std::string::npos;
        const std::string lora_prefix = is_text_lora ? LORA_PREFIX_TEXT_ENCODER : LORA_PREFIX_UNET;
        std::string layer_infos = tensor_name.substr(tensor_name.find(lora_prefix) + lora_prefix.length() + 1);
        // drop LoRA name suffixes which comes after '.'
        std::string layer_name_str = layer_infos.substr(0, layer_infos.find("."));
        // Create C++ lora_map instead of Python lora_dict
        lora_map["name"] = layer_name_str;
        lora_map["type"] = is_text_lora ? "text_encoder" : "unet";

        // update value of weights
        std::vector<safetensors_TensorDescriptor> pair_tensor;

        // up at first, down at second
        if (tensor_name.find("lora_down") != std::string::npos) {
            pair_tensor.push_back(safe_tensors_file.tensors[i + 1]);
            pair_tensor.push_back(safe_tensors_file.tensors[i]);
        } else {
            pair_tensor.push_back(safe_tensors_file.tensors[i]);
            pair_tensor.push_back(safe_tensors_file.tensors[i + 1]);
        }

        for (auto p_t : pair_tensor) {
            safetensors_Str key_st = p_t.name;
            std::string k_s(key_st.ptr, key_st.ptr + key_st.len);
            visited.insert(k_s);
        }

        ov::Shape shape_vec_0(pair_tensor[0].shape, pair_tensor[0].shape + pair_tensor[0].n_dimensions);
        ov::Shape shape_vec_1(pair_tensor[1].shape, pair_tensor[1].shape + pair_tensor[1].n_dimensions);

        // matmul with floats
        std::vector<float> float_data_0 = convert_to_float(pair_tensor[0]);
        std::vector<float> float_data_1 = convert_to_float(pair_tensor[1]);

        // RowMajor
        FloatMatrixMap mat2d_f_0(float_data_0.data(), shape_vec_0[0], shape_vec_0[1]);
        FloatMatrixMap mat2d_f_1(float_data_1.data(), shape_vec_1[0], shape_vec_1[1]);
        FloatMatrix matmul_f = alpha * mat2d_f_0 * mat2d_f_1;

        lora_constants[is_text_lora ? "text_encoder" : "unet"][layer_name_str] =
            ov::op::v0::Constant::create(ov::element::f32, {static_cast<size_t>(matmul_f.rows() * matmul_f.cols())}, matmul_f.data());
    }

    free(safe_tensors_file.tensors);
    free(safe_tensors_file.metadata);

    return lora_constants;
}
