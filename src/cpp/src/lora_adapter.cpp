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

namespace {

using NodePtr = std::shared_ptr<ov::Node>;
using ov::NodeVector;
using namespace ov::op;

// FIXME: Use ov::AlignedBuffer instead of std::vector. ov::AlignedBuffer is not available in public OV API
using Buffer = std::vector<std::uint8_t>;
using BufferPtr = std::shared_ptr<Buffer>;

tempalate <typename T>
struct LoRAParts {
    LoRAParts(const T& alpha, const T& A, const T& B) : alpha(alpha), A(A), B(B)
    T alpha, A, B;
};

using LoRAWeights = LoRAParts<std::shared_ptr<v0::Constant>>;
using LoRAPartsParser = LoRAParts<std::function<std::optional<std::string>(const std::string&)>>;
using LoRATensors = std::map<std::string, LoRAWeights>;

struct RegexParser {
    std::regex pattern;
    size_t capture_index;
    RegexParser (const std::string& pattern, size_t capture_index) : pattern(pattern), capture_index(capture_index) {}
    std::optional<std::string> operator() (const std::string& name) {
        std::smatch match;
        if(std::regex_match(name, match, pattern)) {
            return match[capture_index];
        }
        return std::nullopt;
    }
};

LoRAPartsParser default_lora_patterns () {
    return LoRAPartsParser(
        RegexParser(R"((.*)\.(lora_(A|down)\.weight))", 1),
        RegexParser(R"((.*)\.(lora_(B|up)\.weight))", 1),
        RegexParser(R"((.*)\.alpha))", 1),
    )
}


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

using LoRAWeightGetter = std::function< std::optional<LoRAWeight>(NodePtr node)>;

struct LoRAWeightGetterDefault {
    LoRATensors& lora_tensors;
    LoRAWeightGetterDefault (LoRATensors& lora_tensors) : lora_tensors(lora_tensors) {}

    std::optional<LoRAWeight> operator() (NodePtr node) const {
        std::string name = node->get_friendly_name();
        std::string name_with_underscores;
        std::replace(name_with_underscores.begin(), name_with_underscores.end(), '.', '_');   // FIXME: Customize mapping or change PT FE to produce correct weight names
        auto it = std::find_if(lora_tensors.begin(), lora_tensors.end(), [name, name_with_underscores](const AdapterMap::value_type& pair){
            return name.find(pair.first) != std::string::npos || name_with_underscores.find(pair.first) != std::string::npos;  // FIXME: Should it be an exact match instead of substring taking into account that we should provide custom mapper for names?
        });
        if(it != lora_tensors.end()) {
            return it->second;
        }
        return std::nullopt;
    }
}

#define FAST_WEIGHTS_FUSE 0
#define SEPARATE_TERM 1
#define SEPARATE_TERM_STATE 0


class LoRATransformBase : public ov::pass::MatcherPass {
public:

    OPENVINO_RTTI("ApplyLoRA");
   
    LoRATransformBase(const LoRAWeightGetter& lora_weight_getter) : lora_weight_getter(lora_weight_getter) {
        OPENVINO_REGISTER_MATCHER(
            (ov::pass::pattern::wrap_type<v0::MatMul, v1::Convolution>()),
            ([&, this](ov::pass::pattern::Matcher& m) {
                try {
                    auto node = m.get_match_root();
                    if(auto lora_weight = lora_weight_getter(node)) {
                        return apply(node, *lora_weight);
                    }
                    return false;
                } catch(const std::exception& exception) {
                    DEBUG_PRINT("Exception happens on layer: " << name << " with exception message: " << exception.what());
                    throw;
                } catch(...) {
                    DEBUG_PRINT("Unknown exception happens on layer: " << name);
                    throw;
                }
            })
        );
    }

    ~LoRATransformBase () {
        DEBUG_PRINT("LoRA applied for " << applied << " layers");
    }

protected:

    virtual bool apply(NodePtr node) = 0;

private:

    size_t applied = 0; // FIXME: For debug statistics only

};


NodePtr tensors_multiplication(NodePtr input, const NodeVector multipliers, ov::Output<ov::Node> target, bool transpose_weights) {
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



class LoRAFuseTransform : public LoRATransformBase {
    using Signature = std::string;
    ov::Core core;
    std::map<Signature, ov::InferRequest> compiled_weight_models;
    //std::shared_ptr<ov::Model> model;
    ConstantMap variable_map;
    //ov::SinkVector& assigns;
    //ov::op::util::VariableVector& variables;

    void signature_push_back(Signature& signature, ov::Output<ov::Node> input) const {
        // FIXME: Define hash function on vector<tuple<element_type, PartialShape>> to make it C++ish
        signature += "(el: " + input.get_element_type().get_type_name() + ", shape: " + input.get_partial_shape().to_string() + ")";
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
        for(auto signature: compiled_weight_models) {
            DEBUG_PRINT(signature.first);
        }
    }

private:
    size_t applied = 0;
};


class LoRASeparateTransform : public LoRATransformBase {
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


LoRATensors group_lora_tensors(const ConstantMap& tensors, const LoRAPartsParser& parts_parser) {
    LoRATensors result;
    for(const auto& named_tensor: adapter.tensors) {
        if(auto parsed = parts_parser.A(named_tensor.first)) {
            result[*parsed].A = named_tensor.second;
        } else if(auto parsed = parts_parser.B(named_tensor.first)) {
            result[*parsed].B = named_tensor.second;
        } else if(auto parsed = parts_parser.alpha(named_tensor.first)) {
            result[*parsed].alpha = named_tensor.second;
        } else {
            DEBUG_PRINT("Ignored LoRA tensor " << named_tensor.first << " as was not able to recognize expected name pattern." );
        }
    }

    // Check that A and B exist for eacch LoRA entry
    for(const auto& lora_tensor: result)
        OPENVINO_ASSERT(lora_tensor.second.A && lora_tensor.second.B, "Either A, B or both matrices are missing in LoRA tensors for layer: ", lora_tensor.first);
    }
    return result;
}


} // namespace


namespace ov {
namespace genai {

class Adapter::Impl {
public:
    Impl(const std::string& path, float default_alpha = std::nullopt) :  // FIXME: Pass lora patterns
        tensors(group_lora_tensors(read_safetensors(path), default_lora_patterns())),  // FIXME: Accept directory with the config as well
        default_alpha(default_alpha) {
    }

    LoRATensors tensors;
    std::optional<float> default_alpha;
};

Adapter::Adapter(const std::string& path, float default_alpha) :
    m_pimpl(std::make_shared<Adapter::Impl>(path, default_alpha)) {
}

explicit Adapter(const std::string& path) :
    m_impl(std::make_shared<Adapter::Impl>(path)) {
}


}  // namespace genai
}  // namespace ov




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