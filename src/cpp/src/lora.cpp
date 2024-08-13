// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "lora.hpp"

#include <algorithm>
#include <set>
#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <regex>

#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/core.hpp"

#define SAFETENSORS_IMPLEMENTATION
#include "safetensors.h"

using NodePtr = std::shared_ptr<ov::Node>;
using ov::NodeVector;
using namespace ov::op;

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
        auto constant = std::make_shared<v0::Constant>(wrapper);    // wraps existing memory, no ownership
        constant->get_rt_info()["__safetensors_buffer_holder"] = buffer;    // to automatically deallocate underlying memory buffer when last constant holding it is destoyed
        //DEBUG_PRINT("Tensor with name " << name << ", shape " << shape << " and type " << type << " was allocated.");
        tensors[name] = constant;
    }
    free(safe_tensors_file.tensors);
    free(safe_tensors_file.metadata);
    return std::move(tensors);
}


#define OPENVINO_REGISTER_MATCHER(PATTERN, CALLBACK) do register_matcher(std::make_shared<ov::pass::pattern::Matcher>(PATTERN, this->get_type_info().name), CALLBACK); while(false)

// Squeeze all dimensions from right to 2D shape
NodePtr squeeze_2d (NodePtr node) {
    // auto rank = node->get_output_partial_shape(0).rank().get_length();
    // std::vector<unsigned int> dims(2);
    //auto squeeze_num = rank - 2;
    // std::fill_n(dims.begin() + 2, dims.end(), 1);
    auto shape = v0::Constant::create(ov::element::i32, {2}, std::vector<int>{0, 0});
    auto reshape = std::make_shared<v1::Reshape>(node->output(0), shape->output(0), true);
    return reshape;
}

// Unsqueeze shape to add dimensions to the right to have `rank`-D tensor
NodePtr unsqueeze (NodePtr node, unsigned int rank) {
    auto src_rank = node->get_output_partial_shape(0).rank().get_length();
    std::vector<unsigned int> dims(rank);
    std::fill(dims.begin() + src_rank, dims.end(), 1);
    auto shape = v0::Constant::create(ov::element::i32, {rank}, dims);
    auto reshape = std::make_shared<v1::Reshape>(node->output(0), shape->output(0), true);
    return reshape;
}

AdapterMap::const_iterator find_adapter(const std::string& name, const AdapterMap& adapter_map) {
    auto name_with_underscores = name;
    std::replace(name_with_underscores.begin(), name_with_underscores.end(), '.', '_');   // FIXME: Customize mapping or change PT FE to produce correct weight names
    return std::find_if(adapter_map.begin(), adapter_map.end(), [name, name_with_underscores](const AdapterMap::value_type& pair){
        return name.find(pair.first) != std::string::npos || name_with_underscores.find(pair.first) != std::string::npos;  // FIXME: Should it be an exact match instead of substring taking into account that we should provide custom mapper for names?
    });
}

#define FAST_WEIGHTS_FUSE 0
#define SEPARATE_TERM 1
#define SEPARATE_TERM_STATE 0


class ApplyLoRA : public ov::pass::MatcherPass {
    using Signature = std::string;
    std::map<Signature, ov::InferRequest> compiled_weight_models;
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    ConstantMap variable_map;
    //ov::SinkVector& assigns;
    //ov::op::util::VariableVector& variables;

    void signature_push_back(Signature& signature, ov::Output<ov::Node> input) const {
        // FIXME: Define hash function on vector<tuple<element_type, PartialShape>> to make it C++ish
        signature += "(el: " + input.get_element_type().get_type_name() + ", shape: " + input.get_partial_shape().to_string() + ")";
    }

    NodePtr lora_multiplication(NodePtr input, const NodeVector multipliers, ov::Output<ov::Node> target, bool transpose_weights) {
        const auto target_type = target.get_element_type();
        const auto target_shape = target.get_partial_shape();
        const auto target_rank = target_shape.rank().get_length();
        for(auto multiplier : multipliers) {
            NodePtr normalized = multiplier;
            if(normalized->get_output_element_type(0) != target_type) {
                normalized = std::make_shared<v0::Convert>(normalized, target_type);
            }
            if(normalized->get_output_partial_shape(0).rank().get_length() > 2) {
                // FIXME: Any other shape patterns possible?
                normalized = squeeze_2d(normalized);
            }
            if(input) {
                if(input->get_output_partial_shape(0).rank().get_length() == 0 || normalized->get_output_partial_shape(0).rank().get_length() == 0) {
                    // FIXME: Apply alpha multiplication separately
                    input = std::make_shared<v1::Multiply>(input, normalized);
                } else {
                    input = std::make_shared<v0::MatMul>(input, normalized, /*transpose_a = */false, transpose_weights);  // FIXME: verify transpose_a == true
                }
            } else {
                input = multiplier;
            }
        }

        if(target_rank == 4 && target_shape[-1].is_static() && target_shape[-1].get_length() > 1) {  // FIXME: Check all potentially permuted dimensions, not only the last one
            // FIXME: Check the dimensions we really need to move, currently it is hardcoded 2 + 2 dimensions
            // FIXME: Stash transposition constant to reuse
            auto transposition = v0::Constant::create(ov::element::i32, ov::Shape{4}, std::vector<int>{2, 3, 0, 1});
            input = std::make_shared<v1::Transpose>(input, transposition);
        } else if(input->get_output_partial_shape(0).rank().get_length() != target_rank) {
            // FIXME: Make sure that this is always unsqueeze of the same kind
            input = unsqueeze(input, target_rank);
        }

        input = std::make_shared<v1::Add>(target, input);

        return input;
    }

    NodePtr decompression_convert (NodePtr node) {
        auto convert = std::dynamic_pointer_cast<v0::Convert>(node);
        if(convert) {
            node = convert->get_input_node_shared_ptr(0);
        }
        OPENVINO_ASSERT(std::dynamic_pointer_cast<v0::Constant>(node), "Not supported decompression pattern at the weight input (low bit compression?). Use f32/f16/bf16 weights only.");
        return convert;
    }

public:
    OPENVINO_RTTI("ApplyLoRA");
        ApplyLoRA(
            const AdapterMap& adapter_map,
            std::shared_ptr<ov::Model> model,
            ConstantMap& variable_map
            /*ov::SinkVector& assigns, ov::op::util::VariableVector& variables*/) :
            /*assigns(assigns), variables(variables),*/ model(model), variable_map(variable_map) {
        OPENVINO_REGISTER_MATCHER(
            (ov::pass::pattern::wrap_type<v0::MatMul, v1::Convolution>()),
            ([&, this](ov::pass::pattern::Matcher& m) {
                auto node = m.get_match_root();
                auto name = node->get_friendly_name();
                try{
                    auto adapter_iter = find_adapter(name, adapter_map);
                    if(adapter_iter == adapter_map.end()) {
                        return false;
                    }

                    auto activations = node->input_value(0);    // FIXME: consider MatMul.transpose_a
                    auto weights_input = node->input_value(1);
                    auto weights_input_type = weights_input.get_element_type();
                    DEBUG_PRINT("WEIGHTS SHAPE: " << weights_input.get_partial_shape());
                    #if FAST_WEIGHTS_FUSE
                    auto weights_convert = decompression_convert(weights_input.get_node_shared_ptr());
                    auto weights_constant = weights_convert ? weights_convert->input_value(0) : weights_input;
                    #endif
                    auto adapter = adapter_iter->second; // FIXME: a copy
                    NodePtr add_term = nullptr;
                    Signature signature;
                    #if SEPARATE_TERM
                    adapter = Adapter{adapter[0], adapter[2], adapter[1]};
                    #endif
                    signature_push_back(signature, weights_input);
                    for(auto multiplier : adapter) {
                        signature_push_back(signature, multiplier);
                    }
                    NodePtr replacement = nullptr;

                    #if FAST_WEIGHTS_FUSE

                        auto consumers = weights_input.get_target_inputs();    // replace constant with decompression pattern, TODO: Consider to compress weights in weights_model to save memory (in this case chage weights_input to weights_constant)

                        if(!compiled_weight_models.count(signature)) {
                            ov::ParameterVector parameters;
                            auto target_parameter = std::make_shared<v0::Parameter>(weights_constant.get_element_type(), weights_constant.get_partial_shape());
                            parameters.push_back(target_parameter);   // original weights input is one of the parameters
                            // FIXME: Convert is not counted in the signature but in general it doesn't have to appear at every weights across the entire model, should be a part of the signature
                            // Support only a single convert as a decompression pattern
                            ov::Output<ov::Node> target = weights_convert ? weights_convert->clone_with_new_inputs({target_parameter}) : target_parameter;
                            for(auto multiplier : adapter) {
                                parameters.push_back(std::make_shared<v0::Parameter>(multiplier->get_output_element_type(0), multiplier->get_output_partial_shape(0)));
                            }
                            auto result = std::make_shared<v0::Result>(lora_multiplication(nullptr, NodeVector{parameters.begin() + 1, parameters.end()}, target, false));
                            auto weights_model = std::make_shared<ov::Model>(ov::ResultVector{result}, parameters);
                            auto compiled_model = core.compile_model(weights_model, "CPU");
                            compiled_weight_models[signature] = compiled_model.create_infer_request();
                        }
                        auto request = compiled_weight_models.at(signature);  // FIXME: use .find instead of .count and .at
                        auto output = request.get_compiled_model().output(0);
                        // FIXME: The following constant are big for LLMs, they will eventually replicate all big weights from the model, and it is not mmaped unlike original weights
                        auto replacement_const = std::make_shared<v0::Constant>(output.get_element_type(), output.get_shape());  // TODO: why there is no Constant::create with this signature?
                        replacement = replacement_const;
                        request.set_output_tensor(replacement_const->get_tensor_view());
                        OPENVINO_ASSERT(adapter.size() + 1 == request.get_compiled_model().inputs().size());
                        // set input constants
                        request.set_input_tensor(0, std::dynamic_pointer_cast<v0::Constant>(weights_constant.get_node_shared_ptr())->get_tensor_view());
                        for(size_t i = 0; i < adapter.size(); ++i) {
                            request.set_input_tensor(i+1, adapter[i]->get_tensor_view());
                        }
                        request.infer();
                        // `replacement` contains recomputed weights

                    #else

                    #if SEPARATE_TERM
                    auto target = node->output(0);
                    #else
                    auto target = weights_input;
                    #endif

                    auto target_rank = target.get_partial_shape().rank().get_length();
                    auto consumers = target.get_target_inputs();

                    #if 1
                    // FIXME: Should check rank of activations instead of target
                    if(target_rank == 4 && target.get_partial_shape()[target_rank - 3].get_length() > 1) {
                        //DEBUG_PRINT("Skipping unspported model tensor with shape: " << target.get_partial_shape());
                        // FIXME: Check the dimensions we really need to move, currently it is hardcoded 2 + 2 dimensions
                        // FIXME: Stash transposition constant to reuse
                        auto transposition = v0::Constant::create(ov::element::i32, ov::Shape{4}, std::vector<int>{2, 3, 0, 1});
                        auto transpose = register_new_node<v1::Transpose>(activations, transposition);
                        activations = transpose;
                    }
                    #endif

                    #if SEPARATE_TERM_STATE
                    NodeVector input_constants;
                    for(size_t i = 0; i < adapter.size(); ++i) {
                        std::string variable_id = "lora_state_" + std::to_string(model->get_sinks().size());
                        auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{
                            adapter[i]->get_output_partial_shape(0),
                            adapter[i]->get_output_element_type(0),
                            variable_id});
                        /*variables.push_back*/model->add_variables({variable});
                        variable_map[variable_id] = adapter[i];
                        auto read_value = register_new_node<v6::ReadValue>(variable);
                        input_constants.push_back(read_value);
                        //input_constants.push_back(adapter[i]);
                        /*assigns.push_back*/model->add_sinks({register_new_node<v6::Assign>(read_value, variable)});
                    }
                    #else
                    NodeVector input_constants{adapter.begin(), adapter.end()};
                    #endif
                    replacement = lora_multiplication(
                        #if SEPARATE_TERM
                        activations.get_node_shared_ptr()
                        #else
                        nullptr
                        #endif
                        , input_constants, target,
                        #if SEPARATE_TERM
                                true
                                #else
                                false
                                #endif
                        );
                    #if 0
                    {
                    for(auto multiplier : adapter) {
                        NodePtr normalized = multiplier;
                        if(normalized->get_element_type() != weights_type) {
                            normalized = std::make_shared<v0::Convert>(normalized, weights_type);
                        }
                        if(normalized->get_output_partial_shape(0).rank().get_length() > 2) {
                            // FIXME: Any other shape patterns possible?
                            normalized = squeeze_2d(normalized);
                        }
                        if(add_term) {
                            // FIXME: Apply alpha multiplication separately
                            if(add_term->get_output_partial_shape(0).rank().get_length() == 0) {
                                add_term = std::make_shared<v1::Multiply>(add_term, normalized);
                            } else {
                                #if SEPARATE_TERM
                                bool transpose_b = true;
                                #else
                                bool transpose_b = false;
                                #endif
                                add_term = std::make_shared<v0::MatMul>(add_term, normalized, /*transpose_a = */false, transpose_b);  // FIXME: verify transpose_a == true
                            }
                        } else {
                            #if SEPARATE_TERM
                            add_term = register_new_node<v1::Multiply>(activations, multiplier);
                            #else
                            add_term = multiplier;
                            #endif
                        }
                    }

                    if(add_term->get_output_partial_shape(0).rank() != target_rank) {
                        // FIXME: Make sure that this is always unsqueeze of the same kind
                        add_term = unsqueeze(add_term, target_rank.get_length());
                    }

                    replacement = register_new_node<v1::Add>(target, add_term);

                    #endif
                    #endif

                    for (auto consumer : consumers) {
                        consumer.replace_source_output(replacement->output(0));
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
        for(auto signature: compiled_weight_models) {
            DEBUG_PRINT(signature.first);
        }
    }

private:
    size_t applied = 0;
};



} // namespace


std::map<std::string, AdapterMap> load_lora_adapter(const std::string& adapter_file_path, const float alpha, const LoRAPrefixes& prefixes) {
    auto adapter_tensors = read_safetensors(adapter_file_path);
    std::map<std::string, AdapterMap> result;
    auto alpha_const = v0::Constant::create(ov::element::f32, ov::Shape(), {alpha});
    const std::regex name_pattern = std::regex(R"((.*)\.(lora_(A|B|up|down)\.weight))");
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
        //auto delimiter = name.find('.');
        //DEBUG_PRINT("Tensor name before patching " << name);
        std::smatch match;
        if(!std::regex_match(name, match, name_pattern)) {
            DEBUG_PRINT("Skipping lora tensor: " << name << " because it does't match the expected pattern");
            continue;
        }
        auto layer_name = match[1].str(); // name.substr(0, delimiter);
        auto suffix = match[2].str(); //name.substr(delimiter);
        //DEBUG_PRINT("layer_name " << layer_name);
        //DEBUG_PRINT("suffix " << suffix);

        auto& adapter = result[prefix_it->second][layer_name];
        if(adapter.empty()) {
            adapter.push_back(alpha_const);
        }
        switch(adapter.size()) {
            case 1:
                adapter.push_back(named_tensor.second);
                break;
            case 2:
                if(suffix.find("lora_down") != std::string::npos || suffix.find("lora_A") != std::string::npos) {
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

void apply_lora_adapter(std::shared_ptr<ov::Model> model, const AdapterMap& adapter_map, ConstantMap& variables) {
    //ov::save_model(model, "before_lora.xml", false);
    //std::cout.flush();
    const auto start{std::chrono::steady_clock::now()};
    ov::pass::Manager pm;
    //ov::op::util::VariableVector variables;
    //ov::SinkVector assigns; // new assigns will be collected in a certain mode of LoRA fusion
    pm.register_pass<ApplyLoRA>(adapter_map, model, variables);
    pm.run_passes(model);
    //ov::serialize(model, "after_lora.xml");
    //model->add_variables(variables);
    //model->add_sinks(assigns);
    const auto end{std::chrono::steady_clock::now()};
    DEBUG_PRINT("Fusing LoRA adapter: " << std::chrono::duration<float>(end - start).count());
    //std::cout.flush();
    //ov::save_model(model, "after_lora.xml", false);
}


void connect_lora_adapter(ov::InferRequest infer_request, const ConstantMap& variables) {
    for(auto& state: infer_request.query_state()) {
        auto name = state.get_name();
        auto it = variables.find(name);
        if(it != variables.end()) {
            state.set_state(it->second->get_tensor_view());
        }
    }
}