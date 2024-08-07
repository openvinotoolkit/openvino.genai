#include "lora.hpp"


#if GENAI_NEW_LORA

#include "../../../../src/cpp/src/lora.cpp"

#else

// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <set>
#include <map>
#include <string>
#include <vector>
#include <fstream>

#include <Eigen/Dense>

#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/manager.hpp"

#define SAFETENSORS_IMPLEMENTATION
#include "safetensors.h"

using NodePtr = std::shared_ptr<ov::Node>;

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
        //DEBUG_PRINT(root->get_type_info().name);

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
        ++applied;
        return true;
    };

    // Register pattern with Parameter operation as a pattern root node
    auto m = std::make_shared<ov::pass::pattern::Matcher>(pattern, "InsertLoRA");
    // Register Matcher
    register_matcher(m, callback);
}

namespace {

// FIXME: Use ov::AlignedBuffer instead of std::vector. ov::AlignedBuffer is not available in public OV API
using Buffer = std::vector<std::uint8_t>;
using BufferPtr = std::shared_ptr<Buffer>;

BufferPtr read_file_helper(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    OPENVINO_ASSERT(file.is_open(), "Cannot open file ", filename, " with LoRA weights");

    size_t filesize = file.tellg();
    auto buffer = std::make_shared<std::vector<std::uint8_t>>();
    buffer->reserve(filesize);
    file.seekg(0, std::ios::beg);
    // FIXME: Use mmapped AlignedBuffer as ov::Core::read_model can do, necessary functionality is not available in public OV API
    std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), std::back_inserter(*buffer));

    return buffer;
}

// FIXME: Remove this legacy trampoline
Buffer read_file(const std::string& filename) {
    return std::move(*read_file_helper(filename));
}

ov::element::Type safetensors_to_ov_element_type (int dtype) {
    switch(dtype) {
        case SAFETENSORS_F32:
            return ov::element::f32;
        case SAFETENSORS_F16:
            return ov::element::f16;
        case SAFETENSORS_BF16:
            return ov::element::bf16;
        default:
            OPENVINO_THROW("Not supported safetensors dtype: ", dtype);
    }
}

using ConstantMap = std::map<std::string, std::shared_ptr<ov::op::v0::Constant>>;
ConstantMap read_safetensors(const std::string& filename) {
    ConstantMap tensors;
    auto buffer = read_file_helper(filename);
    safetensors_File safe_tensors_file = {0};
    OPENVINO_ASSERT(safetensors_file_init(&(*buffer)[0], buffer->size(), &safe_tensors_file) == nullptr, "Cannot parse ", filename, " using safetensors");
    DEBUG_PRINT("Opened " << filename << " as safetensors file format, it contains " << safe_tensors_file.num_tensors << " tensors");
    for (int i = 0; i < safe_tensors_file.num_tensors; i++) {
        safetensors_TensorDescriptor tensor = safe_tensors_file.tensors[i];
        std::string name(tensor.name.ptr, tensor.name.ptr + tensor.name.len);
        ov::Shape shape(tensor.shape, tensor.shape + tensor.n_dimensions);
        void* ptr = tensor.ptr;     // FIXME: needs a non-constant pointer because Tensor doesn't accept a constant pointer
        OPENVINO_ASSERT(ov::shape_size(shape) <= tensor.end_offset_bytes - tensor.begin_offset_bytes, " ", ov::shape_size(shape), " ", tensor.end_offset_bytes - tensor.begin_offset_bytes);
        auto type = safetensors_to_ov_element_type(tensor.dtype);
        // FIXME: Extend OV with a new Constant ctor that shares memory to avoid two stage Tensor->Constant initialization
        ov::Tensor wrapper(type, shape, ptr);  // wraps existing memory, no ownership
        auto constant = std::make_shared<ov::op::v0::Constant>(wrapper);    // wraps existing memory, no ownership
        constant->get_rt_info()["__safetensors_buffer_holder"] = buffer;    // to automatically deallocate underlying memory buffer when last constant holding it is destoyed
        //DEBUG_PRINT("Tensor with name " << name << ", shape " << shape << " and type " << type << " was allocated.");
        tensors[name] = constant;
    }
    free(safe_tensors_file.tensors);
    free(safe_tensors_file.metadata);
    return std::move(tensors);
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



#define OPENVINO_REGISTER_MATCHER(PATTERN, CALLBACK) do register_matcher(std::make_shared<ov::pass::pattern::Matcher>(PATTERN, this->get_type_info().name), CALLBACK); while(false)

// Squeeze all dimensions from right to 2D shape
NodePtr squeeze_2d (NodePtr node) {
    // auto rank = node->get_output_partial_shape(0).rank().get_length();
    // std::vector<unsigned int> dims(2);
    //auto squeeze_num = rank - 2;
    // std::fill_n(dims.begin() + 2, dims.end(), 1);
    auto shape = ov::op::v0::Constant::create(ov::element::i32, {2}, std::vector<unsigned int>{0, 0});
    auto reshape = std::make_shared<ov::op::v1::Reshape>(node->output(0), shape->output(0), true);
    return reshape;
}

// Unsqueeze shape to add dimensions to the right to have `rank`-D tensor
NodePtr unsqueeze (NodePtr node, unsigned int rank) {
    auto src_rank = node->get_output_partial_shape(0).rank().get_length();
    std::vector<unsigned int> dims(rank);
    std::fill(dims.begin() + src_rank, dims.end(), 1);
    auto shape = ov::op::v0::Constant::create(ov::element::i32, {rank}, dims);
    auto reshape = std::make_shared<ov::op::v1::Reshape>(node->output(0), shape->output(0), true);
    return reshape;
}

class ApplyLoRA : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("ApplyLoRA");
    ApplyLoRA(const AdapterMap& adapter_map) {
        OPENVINO_REGISTER_MATCHER(
            (ov::pass::pattern::wrap_type<ov::op::v0::MatMul, ov::op::v1::Convolution>()),
            ([=, this](ov::pass::pattern::Matcher& m) {
                auto node = m.get_match_root();
                auto name = node->get_friendly_name();
                try{
                std::replace(name.begin(), name.end(), '.', '_');   // FIXME: Customize mapping or change PT FE to produce correct weight names
                auto adapter_iter = std::find_if(adapter_map.begin(), adapter_map.end(), [name](const AdapterMap::value_type& pair){
                    return name.find(pair.first) != std::string::npos;  // FIXME: Should it be an exact match instead of substring taking into account that we should provide custom mapper for names?
                });

                if(adapter_iter == adapter_map.end()) {
                    return false;
                }

                ov::Output<ov::Node> weights = node->input_value(1);
                auto weights_type = weights.get_element_type();
                auto adapter = adapter_iter->second;
                NodePtr add_term = nullptr;
                bool normalize_shape = false;
                for(auto multiplier : adapter) {
                    NodePtr normalized = multiplier;
                    if(normalized->get_element_type() != weights_type) {
                        normalized = std::make_shared<ov::op::v0::Convert>(normalized, weights_type);
                    }
                    if(normalized->get_output_partial_shape(0).rank().get_length() > 2) {
                        // FIXME: Any other shape patterns possible?
                        normalized = squeeze_2d(normalized);
                    }
                    if(add_term) {
                        // FIXME: Apply alpha multiplication separately
                        if(add_term->get_output_partial_shape(0).rank().get_length() == 0) {
                            add_term = std::make_shared<ov::op::v1::Multiply>(add_term, normalized);
                        } else {
                            add_term = std::make_shared<ov::op::v0::MatMul>(add_term, normalized);
                        }
                    } else {
                        add_term = multiplier;
                    }
                }

                auto weights_rank =  weights.get_partial_shape().rank();
                if(add_term->get_output_partial_shape(0).rank() != weights_rank) {
                    // FIXME: Make sure that this is always unsqueeze of the same kind
                    add_term = unsqueeze(add_term, weights_rank.get_length());
                }

                auto consumers = weights.get_target_inputs();
                auto add = std::make_shared<ov::op::v1::Add>(weights, add_term);
                for (auto consumer : consumers) {
                    consumer.replace_source_output(add->output(0));
                }
                ++applied;
                return true;
                } catch(...) {
                    DEBUG_PRINT("Exception happens on layer: " << name);
                    throw;
                }
            })
        );
    }

    ~ApplyLoRA () { 
        DEBUG_PRINT("LoRA Applied: " << applied);
    }

private:
    size_t applied = 0;
};



} // namespace


std::map<std::string, AdapterMap> load_lora_adapter(const std::string& adapter_file_path, const float alpha, const LoRAPrefixes& prefixes) {
    auto adapter_tensors = read_safetensors(adapter_file_path);
    std::map<std::string, AdapterMap> result;
    auto alpha_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape(), {alpha});
    for(const auto& named_tensor: adapter_tensors) {
        if(named_tensor.first.find(".alpha") != std::string::npos) {
            //DEBUG_PRINT("Alpha tensor was ignored: " << named_tensor.first);
            continue;
        }

        auto prefix_it = std::find_if(prefixes.begin(), prefixes.end(), [&named_tensor](const LoRAPrefixes::value_type& pair) {
            // FIXME: Make sure there is no other matches
            return named_tensor.first.find(pair.first) != std::string::npos;
        });

        if(prefix_it == prefixes.end()) {
            DEBUG_PRINT("Ignored LoRA tensor " << named_tensor.first << " as there is are no matches with any of given prefixes." );
            continue;
        }

        auto name = named_tensor.first.substr(named_tensor.first.find(prefix_it->first) + prefix_it->first.length() + 1);
        auto delimiter = name.find('.');
        auto layer_name = name.substr(0, delimiter);
        auto suffix = name.substr(delimiter);

        auto& adapter = result[prefix_it->second][layer_name];
        if(adapter.empty()) {
            adapter.push_back(alpha_const);
        }
        switch(adapter.size()) {
            case 1:
                adapter.push_back(named_tensor.second);
                break;
            case 2:
                if(suffix.find("lora_down") != std::string::npos) {
                    adapter.push_back(named_tensor.second);
                } else {
                    adapter.insert(adapter.begin() + 1, named_tensor.second);
                }
                break;
            default:
                OPENVINO_THROW("More than two adapter tensors appers for the same layer: ", layer_name, ", started with tensor: ", named_tensor.first);
        }
        //DEBUG_PRINT("Size for tensor layer " << layer_name << ": " << adapter.size());
    }
    return result;
}


void apply_lora_adapter(std::shared_ptr<ov::Model> model, const AdapterMap& adapter_map) {
    ov::pass::Manager pm;
    pm.register_pass<ApplyLoRA>(adapter_map);
    pm.run_passes(model);
}

std::map<std::string, InsertLoRA::LoRAMap>
read_lora_adapters(const std::string& filename, const float alpha) {
    //read_safetensors(filename);
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
        if (alpha_tensor || tensor_visited) {
            //DEBUG_PRINT((alpha_tensor ? "Alpha tensor was ignored: " : "Tensor was visited: ") << tensor_name);
            continue;
        }

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

#endif