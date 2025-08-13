// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <set>
#include <map>
#include <string>
#include <vector>
#include <fstream>
#include <optional>
#include <numeric>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include <memory>
#include <cmath>

#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/if.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"

#include "openvino/genai/lora_adapter.hpp"

#include "utils.hpp"
#include "lora/common.hpp"
#include "lora/names_mapping.hpp"

extern "C" {
    #include "safetensors.h"
}

// FIXME: Remove or move to a dedicated common header
#ifdef NDEBUG
    #define DEBUG_PRINT(X) do {} while(false)
#else
    #define DEBUG_PRINT(X) do { std::cerr << "[ DEBUG ] " << X << "\n"; } while(false)
#endif

namespace {

using NodePtr = std::shared_ptr<ov::Node>;
using ov::NodeVector;
using namespace ov::op;
using namespace ov::genai::utils;

// FIXME: Use ov::AlignedBuffer instead of std::vector. ov::AlignedBuffer is not available in public OV API
using ConstantVector = std::vector<std::shared_ptr<v0::Constant>>;


// Holds usual LoRA parameters alpha, A and B of a given type.
using LoRANode = LoRAParts<std::shared_ptr<ov::Node>>;
using LoRAPartsParser = LoRAParts<std::function<std::optional<std::string>(const std::string& name)>>;

// Converts Safetensors element type to OV element type. Only part of the types are supported.
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

// Safetensor file parser that deallocates temporary buffers automatically.
// Drop-in replacement for the third party safetensors_File struct.
struct AutoSafetensor: public safetensors_File {
    ~AutoSafetensor () {
        std::free(tensors);
        std::free(metadata);
    }
};

// The key in the map is a tensor name and the Constant uses a region of memory from the memory block.
// Each Constant holds a shared pointer to the block in the runtime info.
// The memory block will be deallocated when the last Constant is destroyed.
ConstantMap safetensor_to_constant_map(const ov::Tensor& safetensor) {
    AutoSafetensor safe_tensors_file{};

    OPENVINO_ASSERT(safetensors_file_init(safetensor.data<char>(), safetensor.get_byte_size(), &safe_tensors_file) == nullptr,
        "Cannot parse safetensor as a Safetensors file format. Safetensors file format is supported only"
    );

    ConstantMap tensors;
    for (int i = 0; i < safe_tensors_file.num_tensors; i++) {
        safetensors_TensorDescriptor tensor = safe_tensors_file.tensors[i];
        std::string name(tensor.name.ptr, tensor.name.ptr + tensor.name.len);
        ov::Shape shape(tensor.shape, tensor.shape + tensor.n_dimensions);
        void* ptr = tensor.ptr;     // FIXME: needs a non-constant pointer because Tensor doesn't accept a constant pointer

        auto type = safetensors_to_ov_element_type(tensor.dtype);
        auto constant =
            std::make_shared<v0::Constant>(type, shape, ptr, nullptr);      // wraps existing memory, no ownership
        constant->get_rt_info()["__safetensors_buffer_holder"] = safetensor;    // to automatically deallocate underlying memory buffer when last constant that holds it is destroyed
        tensors[name] = constant;
    }
    return tensors;
}

// Reads a file with a given filename expecting Safetensors file format.
// The file data is mmaped to tensor.
ConstantMap read_safetensors(const std::filesystem::path& filename) {
    auto safetensor = ov::read_tensor_data(filename);

    return safetensor_to_constant_map(safetensor);
}

// Default LoRA tensor name patterns observed in the existing LoRA adapters, captures the prefix that should correspond
// to a layer name in the base model
LoRAPartsParser default_lora_patterns () {
    return LoRAPartsParser(
        RegexParser("(.*)\\.alpha", 1),
        RegexParser("((.*)[_.](lora[_.](A|down)\\.weight))|((.*lora[12])\\.(down\\.weight))", {2, 6}),
        RegexParser("((.*)[_.](lora[_.](B|up)\\.weight))|((.*lora[12])\\.(up\\.weight))", {2, 6})
    );
}

// Default LoRA tensor name patterns observed in the existing LoRA weights, captures the prefix that should correspond
// to a layer name in the base model. Example: https://hf-mirror.com/hfl/llama-3-chinese-8b-lora
std::vector<RegexParser> default_lora_constant_patterns () {
    return {
        RegexParser("(.*)\\.lm_head\\.weight", 0),
        RegexParser("(.*)\\.embed_tokens\\.weight", 0),
    };
}


// Group tensors loaded from LoRA adapter file into triads A, B and alpha grouped by layer names.
LoRATensors group_lora_tensors(const ConstantMap& tensors, const LoRAPartsParser& parts_parser) {
    LoRATensors result;
    for(const auto& named_tensor: tensors) {
        if(auto parsed = parts_parser.A(named_tensor.first)) {
            result[*parsed].A = named_tensor.second;
        } else if(auto parsed = parts_parser.B(named_tensor.first)) {
            result[*parsed].B = named_tensor.second;
        } else if(auto parsed = parts_parser.alpha(named_tensor.first)) {
            result[*parsed].alpha = named_tensor.second;
        } else {
            DEBUG_PRINT("Ignored LoRA tensor \"" << named_tensor.first << "\" because couldn't recognize expected name pattern." );
        }
    }

    // Check that A and B exist for each LoRA entry
    for(const auto& lora_tensor: result) {
        OPENVINO_ASSERT(lora_tensor.second.A && lora_tensor.second.B, "Either A, B or both matrices are missing in LoRA tensors for layer: ", lora_tensor.first);
    }
    return result;
}

using LoRAConstantTensors = std::map<std::string, NodePtr>;

// Group constant tensors loaded from LoRA adapter file into constants
LoRAConstantTensors group_lora_constant_tensors(const ConstantMap& tensors, const std::vector<RegexParser>& const_parsers) {
    LoRAConstantTensors  result;
    for(const auto& named_tensor: tensors) {
        for (const auto& const_parser : const_parsers) {
            if(auto parsed = const_parser(named_tensor.first)) {
                result[*parsed] = named_tensor.second;
                break;
            }
        }
    }

    // Check that all LoRA constant exists
    for (const auto& lora_tensor : result) {
        OPENVINO_ASSERT(lora_tensor.second, "Weight matrix is missing in LoRA tensors for layer: ", lora_tensor.first);
    }
    return result;
}


// Squeeze all dimensions from the right of the shape producing a tensor of 2D shape.
NodePtr squeeze_2d (const ov::Output<ov::Node>& input) {
    auto shape = v0::Constant::create(ov::element::i32, {2}, std::vector<int>{0, 0});
    auto dims = static_cast<std::vector<ov::Dimension>>(input.get_partial_shape());
    OPENVINO_ASSERT(
        dims.end() == std::find_if(dims.begin() + 2, dims.end(), [](const ov::Dimension& d) { return d.get_max_length() > 1; }),
        "LoRA adapter with not pointwise Convolutional kernel is not supported."
    );
    auto reshape = std::make_shared<v1::Reshape>(input, shape->output(0), true);
    return reshape;
}


// Unsqueeze shape to add dimensions to the right of the shape to have a tensor of a given rank.
NodePtr unsqueeze (const ov::Output<ov::Node>& input, unsigned int rank) {
    auto src_rank = input.get_partial_shape().rank().get_length();
    std::vector<unsigned int> dims(rank);
    std::fill(dims.begin() + src_rank, dims.end(), 1);
    auto shape = v0::Constant::create(ov::element::i32, {rank}, dims);
    auto reshape = std::make_shared<v1::Reshape>(input, shape->output(0), true);
    return reshape;
}


using LoRAWeightGetter = std::function<std::optional<LoRANode>(const std::string&)>;
using LoRAConstantGetter = std::function<std::optional<NodePtr>(const std::string&)>;
using LoRAWeightByNodeGetter = std::function<std::optional<LoRANode>(NodePtr)>;
using LoRAConstantByNodeGetter = std::function<std::optional<NodePtr>(NodePtr)>;


// LoRA adapter parameters applied to a specific place in the model.
// Depending on LoRA mode can have static or dynamic LoRA rank that accumulates
// the ranks from all applicable LoRA tensors (if there are multiple LoRA adapters).
struct LoRAParameters {
    ov::Dimension rank;         // accumulated LoRA rank, could be dynamic if rank is not known or DYNAMIC mode is applied
    ov::element::Type type;     // element type of a tensor that will be applied to the model, negotiated based on multiple LoRA adapters
    bool fine_grained_alpha;    // use 1D tensor of the same rank for alpha instead of a scalar to blend multiple weighted LoRAs
    // TODO: flag to have various alphas over the batch
};

using LoRAParametersGetter = std::function<std::optional<LoRAParameters>(NodePtr node)>;

// Maps a given layer name to corresponding LoRA tensors based on the default name mapping schema.
// Layer name should start with a given prefix that is eliminated from the name before search for matching LoRA tensor.
// It works for a single LoRA adapter.
// Returns std::nullopt, if there is no LoRA adapter for a given layer name.
template<typename TENSOR_TYPE, typename NODE_TYPE>
struct LoRAWeightGetterDefault {
    const std::map<std::string, TENSOR_TYPE>* lora_tensors = nullptr;
    const std::string prefix;
    mutable std::set<std::string> used_tensors;
    mutable bool active = false;    // true if operator() was called at least once to filter out the case when this object is temporary object that is not used for tensor queries

    LoRAWeightGetterDefault (const std::map<std::string, TENSOR_TYPE>* lora_tensors, const std::string& prefix) : lora_tensors(lora_tensors), prefix(prefix) {}

    std::optional<NODE_TYPE> operator() (const std::string& name) const {
        active = true;
        std::string name_with_underscores = name;
        // TODO: Investigate what is the root cause for this replacement in the name. Customize mapping or change PT FE to produce correct weight names.
        std::replace(name_with_underscores.begin(), name_with_underscores.end(), '.', '_');
        std::vector<std::string> variants{name, name_with_underscores};
        auto it = std::find_if(lora_tensors->begin(), lora_tensors->end(), [this, variants](const std::pair<std::string, TENSOR_TYPE>& pair) {
            std::string lora_name = pair.first;
            // TODO: Make this filtering for prefix once in ctor as a more efficient solution
            if(lora_name.find(prefix) == 0) {
                lora_name = lora_name.substr(prefix.length());
            } else {
                return false;
            }
            // TODO: Should it be an exact match instead of substring taking into account that we should provide custom mapper for names?
            return variants.end() != std::find_if(variants.begin(), variants.end(), [lora_name](const std::string& name) { return name.find(lora_name) != std::string::npos; });
        });

        if(it != lora_tensors->end()) {
            used_tensors.insert(it->first);
            return it->second;
        }
        return std::nullopt;
    }

    // Return list of LoRA tensors that are dedicated for the model but left unused
    std::list<std::string> get_unused_tensors() const {
        std::list<std::string> unused;
        for(auto const& tensor: *lora_tensors) {
            if(tensor.first.find(prefix) == 0) {
                if(used_tensors.find(tensor.first) == used_tensors.end()) {
                    unused.push_back(tensor.first);
                }
            }
        }
        return unused;
    }

    ~LoRAWeightGetterDefault() {
        if(!active || !lora_tensors) {
            return;
        }

        auto unused = get_unused_tensors();
        if(unused.empty()) {
            return;
        }

        std::cerr << "[ WARNING ] There unused LoRA tensors. The result of generation can be not accurate. Check if a given adapter file is compatible with the base model.\n";

        for(const auto& unused_name: unused) {
            std::cerr << "    Unused LoRA tensor: " << unused_name << "\n";
        }
    }
};


// Maps a node in the base model to LoRA parameters object that describes how the LoRA tensors should be injected for that node.
// Works with multiple LoRAs accumulating their properties into a single LoRAParameter instance.
// Returns std::nullopt, if there is no LoRA adapter for a given node.
struct LoRAParametersByWeightGetter {
    std::vector<LoRAWeightGetter> weight_getter;
    bool dynamic_lora_rank = true;
    bool fine_grained_alpha = true;
    ov::element::Type type;

    std::optional<LoRAParameters> operator() (NodePtr node) const {
        // If at least one weight_getter gives the weight for the node, then this node should be processed.

        ov::Dimension rank = ov::Dimension::dynamic();
        if(dynamic_lora_rank) {
            // Leave rank dynamic if at least one adapter exist for a give node.
            // It is important to go over all weight_getter's because they record used LoRA tensors to
            // be able to report unused tensors later.
            // Hence, avoid find_if here and use an std algorithm that goes over all elements in a sequence.
            if(!std::count_if(weight_getter.begin(), weight_getter.end(), [node](const LoRAWeightGetter& getter) {
                    return bool(getter(node->get_friendly_name()));
            })) {
                return std::nullopt;
            }
        } else {
            // Accumulates all ranks from all adapters applicable for a given node.
            auto size = std::accumulate(weight_getter.begin(), weight_getter.end(), 0u, [node](unsigned int acc, const LoRAWeightGetter& getter) {
                if(auto nodes = getter(node->get_friendly_name())) {
                    return static_cast<unsigned int>(acc + nodes->A->get_output_partial_shape(0)[0].get_length());
                } else {
                    return acc;
                }
            });
            if(size == 0) {
                // as LoRA adapters with 0 rank cannot exist, 0 means there are no adapters for a given node
                return std::nullopt;
            }
            rank = size;
        }

        LoRAParameters result;
        result.rank = rank;
        result.type = type;
        result.fine_grained_alpha = fine_grained_alpha;
        return result;
    }
};


using LoRAIndices = LoRAParts<size_t>;
using LoRAVarIDs = LoRAParts<ov::op::util::VariableInfo>;


// Deduce expected LoRA input and output static dimensions based on a given node where LoRA is applied
// A given node should be MatMul or Convolution
void deduce_input_output_dims(NodePtr node, ov::Dimension& input_dim, ov::Dimension& output_dim) {
    if(std::dynamic_pointer_cast<v1::Convolution>(node)) {
        input_dim = node->get_input_partial_shape(1)[1];
        output_dim = node->get_input_partial_shape(1)[0];
    } else if(auto matmul = std::dynamic_pointer_cast<v0::MatMul>(node)) {
        input_dim = node->get_input_partial_shape(1)[matmul->get_transpose_b()];
        output_dim = node->get_input_partial_shape(1)[!matmul->get_transpose_b()];
    } else {
        OPENVINO_THROW(
            "deduce_input_output_dims expects MatMul or Convolution, but got ", node,
            ". Given LoRA adapter is unsupported."
        );
    }
}


using LoRAVarMap = std::map<std::string, LoRAVarIDs>;

namespace {
        NodePtr add_variable(const ov::op::util::VariableInfo& variable_info, std::shared_ptr<ov::Model> model) {
        auto variable = std::make_shared<ov::op::util::Variable>(variable_info);
        model->add_variables({variable});
        #if 0
        // Attempt to pre-build initialization expression with empty tensors that should discard LoRA effect by default
        // FIXME: CPU plugin fails when there is no initialization expression is given and type is not fp32
        ov::Shape init_shape(shape.rank().get_length());
        for(size_t i = 0; i < shape.size(); ++i) {
            init_shape[i] = shape[i].get_min_length();
        }
        auto init = v0::Constant::create(type, init_shape, std::vector<float>(ov::shape_size(init_shape), 0));
        auto read_value = std::make_shared<v6::ReadValue>(init, variable);
        #else
        auto read_value = std::make_shared<v6::ReadValue>(variable);
        #endif
        model->add_sinks({std::make_shared<v6::Assign>(read_value, variable)});  // FIXME: Required? -- Yes, create ticket against CPU
        return read_value;
    }
}

// Creates ReadValue and Assign nodes to inject LoRA tensors as variables for a given node but
// doesn't connect them to the model returning as LoRANode instance.
struct LoRAWeightStateGetter {
    std::shared_ptr<ov::Model> model;
    LoRAParametersGetter params_getter;
    LoRAVarMap& variable_ids;
    // TODO: Use variable indices instead of variable_id for faster search for a state tensor

    LoRAWeightStateGetter(const LoRAParametersGetter& params_getter,
                          std::shared_ptr<ov::Model> model,
                          LoRAVarMap& variable_ids)
        : model(model),
          params_getter(params_getter),
          variable_ids(variable_ids) {}

    std::optional<LoRANode> operator() (NodePtr node) const {
        if(auto params = params_getter(node)) {
            ov::Dimension input_dim, output_dim;
            deduce_input_output_dims(node, input_dim, output_dim);

            std::string name = node->get_friendly_name();
            // FIXME: Potential name conflict if LoRA is applied multiple times by using this infrastructure independently each time (not a recommended approach).
            // TODO: Check for name collisions searching for existing variables with the same names.
            std::string variable_id_prefix = "lora_state_" + std::to_string(model->get_sinks().size()) + name;
            LoRANode result;
            LoRAVarIDs var_ids;

            // FIXME: No guarantees on ordering of state in InferRequest makes impossible using indices of variables later, forced to use variable_id instead
            //indices.A = model->get_variables().size();
            var_ids.A = ov::op::util::VariableInfo{
                ov::PartialShape{params->rank, input_dim},  // Will be used with transpose_b == true
                params->type,
                variable_id_prefix + ".A"
            };
            result.A = add_variable(var_ids.A, model);
            // FIXME: No guarantees on ordering of state in InferRequest makes impossible using indices of variables later, forced to use variable_id instead
            //indices.A = model->get_variables().size();
            var_ids.alpha = ov::op::util::VariableInfo{
                params->fine_grained_alpha ? ov::PartialShape{1, params->rank} : ov::PartialShape{},
                ov::element::f32,   // alpha is always f32 because it is set from host as float data type
                variable_id_prefix + ".alpha"
            };
            result.alpha = add_variable(var_ids.alpha, model);
            // FIXME: No guarantees on ordering of state in InferRequest makes impossible using indices of variables later, forced to use variable_id instead
            //indices.B = model->get_variables().size();
            var_ids.B = ov::op::util::VariableInfo{
                ov::PartialShape{output_dim, params->rank},  // Will be used with transpose_b == true
                params->type,
                variable_id_prefix + ".B"
            };
            result.B = add_variable(var_ids.B, model);
            variable_ids.emplace(name, var_ids);
            return result;
        } else {
            return std::nullopt;
        }
    }
};

struct LoRAStateGetterForConst {
    std::shared_ptr<ov::Model> model;
    LoRAConstantGetter getter;
    std::map<std::string, ov::op::util::VariableInfo>& variable_ids;

    LoRAStateGetterForConst(const LoRAConstantGetter& getter,
                            std::shared_ptr<ov::Model> model,
                            std::map<std::string, ov::op::util::VariableInfo>& variable_ids)
        : model(model),
          getter(getter),
          variable_ids(variable_ids) {}

    std::optional<NodePtr> operator() (NodePtr node) const {
        std::string name = node->get_friendly_name();
        if (auto params = getter(name)) {
            // FIXME: Potential name conflict if LoRA is applied multiple times by using this infrastructure independently each time (not a recommended approach).
            // TODO: Check for name collisions searching for existing variables with the same names.
            std::string variable_id_name = "lora_constant_" + std::to_string(model->get_sinks().size()) + "_" + name;
            NodePtr result;

            // No guarantees on ordering of state in InferRequest makes impossible using indices of variables later, forced to use variable_id instead
            ov::op::util::VariableInfo variable_info = ov::op::util::VariableInfo {
                ov::Output<ov::Node>(*params, 0).get_shape(),
                ov::Output<ov::Node>(*params, 0).get_element_type(),
                variable_id_name
            };
            result = add_variable(variable_info, model);
            variable_ids.emplace(name, variable_info);

            return result;
        } else {
            return std::nullopt;
        }
    }

    NodePtr create_if_input() {
        const std::string if_variable_id = "lora_state_0_replace_orig_constant";
        auto variable_info = ov::op::util::VariableInfo {
            ov::Shape{1},
            ov::element::Type_t::boolean,
            if_variable_id
        };
        variable_ids.emplace(if_variable_id, variable_info);
        auto result_tensor = add_variable(variable_info, model);
        return result_tensor;
    }
};


// Transformation that injects LoRA tensors or tensors entry points into the base model.
// The exact form of injection is implemented in the derived classes via overriding `apply` method
// that is called for each applicable node in the base model.
// Detects if a given node requires adaptation based on LoRAWeightByNodeGetter object which maps
// a node to LoRA parameters object.
// Applies only for MatMul and Convolution nodes.
class LoRATransformBase : public ov::pass::MatcherPass {
public:

    OPENVINO_MATCHER_PASS_RTTI("LoRATransformBase");

    LoRATransformBase(const LoRAWeightByNodeGetter& lora_weight_getter) {
        register_matcher(
            std::make_shared<ov::pass::pattern::Matcher>(ov::pass::pattern::wrap_type<v0::MatMul, v1::Convolution>(), this->get_type_info().name),
            ([lora_weight_getter, this](ov::pass::pattern::Matcher& m) {
                auto node = m.get_match_root();
                try {
                    if(auto lora_weight = lora_weight_getter(node)) {
                        if(apply(node, *lora_weight)) {
                            ++applied; // FIXME: For debugging purposes only
                            return true;
                        }
                    }
                    return false;
                } catch(const std::exception& exception) {
                    DEBUG_PRINT("Exception happens on layer: " << node << " with exception message: " << exception.what());
                    throw;
                } catch(...) {
                    DEBUG_PRINT("Unknown exception happens on layer: " << node);
                    throw;
                }
            })
        );
    }

    ~LoRATransformBase () {
        DEBUG_PRINT("LoRA applied for " << applied << " layers");
    }

protected:

    virtual bool apply(NodePtr node, const LoRANode& lora_weight) = 0;
    size_t applied = 0; // For debug statistics only
};

class LoRAReplaceConstantTransform : public ov::pass::MatcherPass {
public:
    LoRAReplaceConstantTransform(const LoRAConstantByNodeGetter& getter) {
        register_matcher(
            std::make_shared<ov::pass::pattern::Matcher>(ov::pass::pattern::wrap_type<v0::Constant>(), this->get_type_info().name),
            ([getter, this](ov::pass::pattern::Matcher& m) {
                auto src_node = m.get_match_root();
                try {
                    if (auto lora_weight = getter(src_node)) {
                        if(apply(src_node, *lora_weight)) {
                            ++applied; // FIXME: For debugging purposes only
                            return true;
                        }
                    }
                    return false;
                } catch(const std::exception& exception) {
                    DEBUG_PRINT("Exception happens on layer: " << src_node << " with exception message: " << exception.what());
                    throw;
                } catch(...) {
                    DEBUG_PRINT("Unknown exception happens on layer: " << src_node);
                    throw;
                }
            })
        );
    }

    ~LoRAReplaceConstantTransform () {
        DEBUG_PRINT("LoRA Constant replacement is applied for " << applied << " layers");
    }

protected:
    size_t applied = 0; // For debug statistics only
    virtual bool apply(NodePtr node, const NodePtr& lora_weight) = 0;
};

class LoRAReplaceConstantTransformStatic : public LoRAReplaceConstantTransform {
public:
    LoRAReplaceConstantTransformStatic(const LoRAConstantByNodeGetter& getter) :
        LoRAReplaceConstantTransform(getter) {}

protected:
    bool apply(NodePtr node, const NodePtr& lora_weight) override {
        ov::replace_node(node, lora_weight);
        return true;
    }
};

class LoRAReplaceConstantTransformDynamic : public LoRAReplaceConstantTransform {
public:
    LoRAReplaceConstantTransformDynamic(const LoRAConstantByNodeGetter& getter,
                                        const NodePtr if_input) :
    LoRAReplaceConstantTransform(getter), if_input(if_input) {}

protected:
    NodePtr if_input;

    bool apply(NodePtr node, const NodePtr& lora_weight) override {
        auto consumers = node->get_output_target_inputs(0);
        const auto node_type = node->get_element_type();
    
        // cast to node type
        auto lora_output = lora_weight;
        if (lora_weight->get_element_type() != node_type) {
            lora_output = std::make_shared<ov::op::v0::Convert>(lora_weight, node_type);
        }

        auto if_node = std::make_shared<ov::op::v8::If>(if_input);

        // IF branch: there is the constant weight, replace with ReadValue
        auto then_param = std::make_shared<ov::op::v0::Parameter>(lora_output->get_element_type(), lora_output->get_output_partial_shape(0));
        auto then_result = std::make_shared<ov::op::v0::Result>(then_param);
        auto then_body = std::make_shared<ov::Model>(ov::ResultVector{then_result}, ov::ParameterVector{then_param});
        if_node->set_then_body(then_body);

        // ELSE branch: use original weight
        auto else_param = std::make_shared<ov::op::v0::Parameter>(node->get_element_type(), node->get_output_partial_shape(0));
        auto else_result = std::make_shared<ov::op::v0::Result>(else_param);
        auto else_body = std::make_shared<ov::Model>(ov::ResultVector{else_result}, ov::ParameterVector{else_param});
        if_node->set_else_body(else_body);

        // set if_node inputs
        if_node->set_input(lora_output, then_param, nullptr);  // put LoRA tensor to then
        if_node->set_input(node, nullptr, else_param);         // put original Constant to else

        // set if_node output
        if_node->set_output(then_result, else_result);

        for (auto& consumer : consumers) {
            consumer.replace_source_output(if_node->output(0));
        }

        return true;
    }
};

// TODO [ticket 171466]: LoRA should support two separate parameters: alpha (defined in the config or .safetensors), weight (set by the user)
// Currently, the implementation merges them into a single effective scaling factor: alpha = (alpha_from_file / rank) * weight
// Check whether `weight` should be supported for text generation

// Builds LoRA subgraph that consists of several matrix and element-wise multiplications with optional data type
// conversions and reshapes to build a consistent graph.
NodePtr tensors_multiplication(NodePtr input,
                               const NodeVector multipliers,
                               ov::Output<ov::Node> target,
                               bool transpose_weights,
                               size_t alpha_pos,
                               size_t A_pos,
                               bool transpose_in_end) {
    const auto target_type = target.get_element_type();
    const auto target_shape = target.get_partial_shape();
    const auto target_rank = target_shape.rank().get_length();

    for (size_t i = 0; i < multipliers.size(); ++i) {
        NodePtr normalized = multipliers[i];
        if (normalized->get_output_element_type(0) != target_type) {
            normalized = std::make_shared<v0::Convert>(normalized, target_type);
            if (std::dynamic_pointer_cast<v0::Constant>(normalized)) {
                input->get_rt_info()["decompression"];
            }
        }
        if (normalized->get_output_partial_shape(0).rank().get_length() > 2) {
            // FIXME: Any other shape patterns possible?
            normalized = squeeze_2d(normalized);
        }
        if (input) {
            if (i == alpha_pos) {  // Multiply for alpha
                // TODO: Apply alpha multiplication separately
                input = std::make_shared<v1::Multiply>(input, normalized);
            } else {  // MatMul for A and B
                input = std::make_shared<v0::MatMul>(input,
                                                     normalized,
                                                     /*transpose_a = */ false,
                                                     transpose_weights);  // FIXME: verify transpose_a == true
            }
        } else {  // used in case of MODE_FUSE
            input = normalized;
        }
    }

    if (transpose_in_end) {
        // FIXME: Check the dimensions we really need to move, currently it is hardcoded 2 + 2 dimensions that usually
        // appears in 2D Convolution case where we need to apply LoRA for the first two dimensions (channels) while
        // interpreting two last dimensions (spatial )
        // TODO: Stash transposition constant to reuse
        auto transposition = v0::Constant::create(ov::element::i32, ov::Shape{4}, std::vector<int>{2, 3, 0, 1});
        input = std::make_shared<v1::Transpose>(input, transposition);
    } else if (input->get_output_partial_shape(0).rank().get_length() != target_rank) {
        input = unsqueeze(input, target_rank);
    }

    input = std::make_shared<v1::Add>(target, input);

    return input;
}

// Taking a node detects an optional weight decompression pattern Constant -> Convert.
// Returns a pointer to Convert node if it exists, or nullptr if there is no Convert.
// If unsupported decompression pattern is used, throws an exception.
NodePtr decompression_convert (NodePtr node) {
    auto convert = std::dynamic_pointer_cast<v0::Convert>(node);
    if(convert) {
        node = convert->get_input_node_shared_ptr(0);
    }
    OPENVINO_ASSERT(
        std::dynamic_pointer_cast<v0::Constant>(node),
        "LoRA adapter application: not supported decompression pattern at the weight input (presumably low-bit compression). Use f32/f16/bf16 weights only if MODE_FUSE is used."
    );
    return convert;
}


// Cache of infer request for on-demand build and compiled helper models for weight modification.
// It maps a model signature which is an arbitrary string to OpenVINO infer request.
// Defines `evaluate` method that compute a model by a given signature and input tensors.
class InferRequestSignatureCache {

    // Infer request with additional input-output pairs that are bypassed from input to output to eliminate Parameter -> Result pairs from the OV model
    struct RequestWithBypass {
        ov::InferRequest request;
        std::vector<std::pair<size_t, size_t>> bypass; // a set of index pairs [j, k], where j is an index of input tensor to be forwarded to k-th output tensor
        std::vector<size_t> inputs; // inputs[i] gives an index in the original input tensor vector to be set to i-th input of the request
        std::vector<size_t> outputs;  // outputs[i] gives an index in the original output tensor vector to be set as an i-th output of the request
    };

public:
    using Signature = std::string;

    InferRequestSignatureCache (const std::string& device) : device(device) {}

    bool exist (const Signature& signature) {
        return requests.count(signature);
    }

    void insert (const Signature& signature, ov::ResultVector& results, ov::ParameterVector& parameters) {
        // Detect Parameter -> Result patterns and do not allow them to be included into compiled model to avoid unnecessary overheads, and handle them via a bypass.
        // Assume that each parameter from parameters vector do not have other consumers outside model formed by parameters -> ... -> results.
        // That allows filter out those parameters that are consumed by Result operations only detecting them by count of consumers instead of
        // tracing dependencies inside the model.

        ov::ResultVector request_results;
        request_results.reserve(results.size());
        ov::ParameterVector request_parameters;
        request_parameters.reserve(parameters.size());
        RequestWithBypass rwb;

        for(size_t result_index = 0; result_index < results.size(); ++result_index) {
            auto& result = results[result_index];
            auto parameter = std::dynamic_pointer_cast<v0::Parameter>(result->get_input_node_shared_ptr(0));
            if(parameter) {
                // Bypass result
                size_t parameter_index = std::distance(parameters.begin(), std::find(parameters.begin(), parameters.end(), parameter));
                rwb.bypass.emplace_back(parameter_index, result_index);
                result.reset();     // enough under the assumption there are no other refernces to that result
            } else {
                // Normal output
                request_results.push_back(result);
                rwb.outputs.push_back(result_index);
            }
        }

        for(size_t parameter_index = 0; parameter_index < parameters.size(); ++parameter_index) {
            auto& parameter = parameters[parameter_index];
            if(!parameter->get_output_target_inputs(0).empty()) {
                request_parameters.push_back(parameter);
                rwb.inputs.push_back(parameter_index);
            } else {
                parameter.reset();
            }
        }

        ov::Core core = ov::genai::utils::singleton_core();
        auto model = std::make_shared<ov::Model>(request_results, request_parameters);
        auto compiled_model = core.compile_model(model, device);
        ov::genai::utils::print_compiled_model_properties(compiled_model, "Infer Request Signature Cache");
        rwb.request = compiled_model.create_infer_request();
        requests.emplace(signature, rwb);
    }

    void evaluate(const Signature& signature, const ov::TensorVector& inputs, ov::TensorVector& outputs) {
        auto& rwb = at(signature);
        auto request = rwb.request;
        auto compiled_model = request.get_compiled_model();
        for(size_t i = 0; i < rwb.inputs.size(); ++i) {
            request.set_input_tensor(i, inputs[rwb.inputs[i]]);
        }
        for(size_t i = 0; i < rwb.outputs.size(); ++i) {
            auto target_shape = request.get_compiled_model().output(i).get_partial_shape();
            auto& output_tensor = outputs[rwb.outputs[i]];
            if(target_shape != output_tensor.get_shape() && target_shape.is_static()) {
                // do it for static case only, because if target shape is dynamic, the plugin is allowed to set shape on its own
                output_tensor.set_shape(target_shape.get_shape());
            }
            request.set_output_tensor(i, output_tensor);
        }
        for(auto bypass: rwb.bypass) {
            outputs[bypass.second] = inputs[bypass.first];
        }
        request.infer();    // TODO: Consider using async to increase throughput, requires more complicated archestration
    }

private:

    RequestWithBypass& at(const Signature& signature) {
        return requests.at(signature);
    }

    std::unordered_map<Signature, RequestWithBypass> requests;
    std::string device;
};


// Transformation that modifies existing weights in the base model fusing an arbitrary number of LoRA adapters.
// This is one-way LoRA fusion that cannot be undone.
// By default it uses CPU plugin to modify the base model weights.
// TODO: This transformation unpacks potentially compressed to f16/bf16 weights to f32,
// we should pack it back into the original precision to maintain the same weight size.
// But it will work well if all plugins equally support fp-compressed weights and can unpack them on-line.
class LoRAFuseTransform : public LoRATransformBase {

    InferRequestSignatureCache fusers;

    void signature_push_back(InferRequestSignatureCache::Signature& signature, ov::Output<ov::Node> input) const {
        // TODO: Define hash function on vector<tuple<element_type, PartialShape>> to make it C++ish
        signature += "(el: " + input.get_element_type().get_type_name() + ", shape: " + input.get_partial_shape().to_string() + ")";
    }

public:

    OPENVINO_RTTI("LoRAFuseTransform", "genai", LoRATransformBase);

    LoRAFuseTransform(const LoRAWeightByNodeGetter& lora_weight_getter, const std::string& device_for_fusion = "CPU") :
        LoRATransformBase(lora_weight_getter),
        fusers(device_for_fusion)
    {}

    bool apply (NodePtr node, const LoRANode& lora_weight) override {
        auto weights_input = node->input_value(1);
        auto weights_input_type = weights_input.get_element_type();
        auto weights_convert = decompression_convert(weights_input.get_node_shared_ptr());
        auto weights_constant = weights_convert ? weights_convert->input_value(0) : weights_input;
        ConstantVector adapter = {
            std::dynamic_pointer_cast<v0::Constant>(lora_weight.alpha),
            std::dynamic_pointer_cast<v0::Constant>(lora_weight.B),
            std::dynamic_pointer_cast<v0::Constant>(lora_weight.A)};
        InferRequestSignatureCache::Signature signature;
        signature_push_back(signature, weights_input);
        for(auto multiplier : adapter) {
            signature_push_back(signature, multiplier);
        }

        // TODO: In case when compressed repacking of newly created weights is retained,
        // replace weights_input by weigths_constant to keep decompression Convert in the model.
        auto consumers = weights_input.get_target_inputs();

        if(!fusers.exist(signature)) {
            // Build a small model for weight and LoRA fusion, and stash it into `fusers` cache.
            ov::ParameterVector parameters;
            auto target_parameter = std::make_shared<v0::Parameter>(weights_constant.get_element_type(), weights_constant.get_partial_shape());
            parameters.push_back(target_parameter);   // original weights input is one of the parameters
            ov::Output<ov::Node> target = weights_convert ? weights_convert->clone_with_new_inputs({target_parameter}) : target_parameter;
            for(auto multiplier : adapter) {
                parameters.push_back(std::make_shared<v0::Parameter>(multiplier->get_output_element_type(0), multiplier->get_output_partial_shape(0)));
            }

            auto result = std::make_shared<v0::Result>(
                tensors_multiplication(nullptr,
                                       NodeVector{parameters.begin() + 1, parameters.end()},
                                       target,
                                       false,
                                       1, // alpha idx
                                       2, // A idx
                                       false));

            ov::ResultVector results{result};
            fusers.insert(signature, results, parameters);
        }

        // Newly created constants in the next line are not mmaped unlike original weights, so it will inflate required memory
        // eventually allocating up to 2x of the base model size.
        // 2X is due to usually applied compression in the base model that is not retained in the current version of this code.
        // But even if the compression is used, then still a copy of all weights that affected by the LoRA adapters are allocated in memory.
        // FIXME: Provide a way for postponed weight repacking that will be triggered by the plugin in compile_model call for the base model.
        // Constant sub-expression can be a solution, but it requires improvements inside plugins, because currently it works extremely slow.
        auto replacement_const = std::make_shared<v0::Constant>(weights_input.get_element_type(), weights_input.get_shape());

        ov::TensorVector outputs{replacement_const->get_tensor_view()};
        // set input constants
        ov::TensorVector inputs;
        inputs.reserve(1 + adapter.size());
        inputs.push_back(std::dynamic_pointer_cast<v0::Constant>(weights_constant.get_node_shared_ptr())->get_tensor_view());
        for(size_t i = 0; i < adapter.size(); ++i) {
            inputs.push_back(adapter[i]->get_tensor_view());
        }
        fusers.evaluate(signature, inputs, outputs);

        for (auto consumer : consumers) {
            consumer.replace_source_output(replacement_const->output(0));
        }
        return true;
    }
};


// Transformation that modifies the base model inserting new nodes that do LoRA matrix multiplications alongside with the original MatMul/Convolution.
class LoRASeparateTransform : public LoRATransformBase {
public:

    OPENVINO_RTTI("LoRASeparateTransform", "genai", LoRATransformBase);

    LoRASeparateTransform(const LoRAWeightByNodeGetter& lora_getter) : LoRATransformBase(lora_getter) {}

    bool apply (NodePtr node, const LoRANode& lora_weight) override {
        auto activations = node->input_value(0);    // FIXME: consider MatMul.transpose_a
        auto weights_input = node->input_value(1);
        auto weights_input_type = weights_input.get_element_type();
        NodePtr add_term = nullptr;
        NodePtr replacement = nullptr;

        auto target = node->output(0);

        auto target_rank = target.get_partial_shape().rank().get_length();
        auto consumers = target.get_target_inputs();
        bool transpose_in_end = false;

        // FIXME: Should check rank of activations instead of target rank
        if(target_rank == 4 && target.get_partial_shape()[target_rank - 3].get_length() > 1) {
            // FIXME: Check the dimensions we really need to move, currently it is hardcoded 2 + 2 dimensions
            // FIXME: Stash transposition constant to reuse
            auto transposition = v0::Constant::create(ov::element::i32, ov::Shape{4}, std::vector<int>{2, 3, 0, 1});
            auto transpose = register_new_node<v1::Transpose>(activations, transposition);
            activations = transpose;
            transpose_in_end = true;
        }

        NodeVector lora_variables{lora_weight.A, lora_weight.alpha, lora_weight.B};

        replacement = tensors_multiplication(activations.get_node_shared_ptr(),
                                             lora_variables,
                                             target,
                                             true,
                                             1, // alpha idx
                                             0, // A idx
                                             transpose_in_end);

        replacement->get_output_tensor(0).add_names(target.get_names());
        for (auto consumer : consumers) {
            consumer.replace_source_output(replacement->output(0));
        }

        return true;
    }
};


std::shared_ptr<v0::Constant> alpha_as_constant(float alpha) {
    return v0::Constant::create(ov::element::f32, ov::Shape{1}, {alpha});
}


LoRATensors flux_normalization(const LoRATensors& tensors) {
    // Check for specific substrings to improve performance, equivalent to chaining the preprocessors
    // apply flux_xlabs_lora_preprocessing if at least one tensor in tensors has "processor" substring in its name
    for(const auto& src_tensor: tensors) {
        if(src_tensor.first.find("processor") != std::string::npos) {
            return flux_xlabs_lora_preprocessing(tensors);
        }
    }
    // apply flux_kohya_lora_preprocessing if at least one tensor in tensors has "lora_unet" substring in its name
    for(const auto& src_tensor: tensors) {
        if(src_tensor.first.find("lora_unet") != std::string::npos) {
            return flux_kohya_lora_preprocessing(tensors);
        }
    }
    return tensors;
}

LoRATensors diffusers_normalization (const LoRATensors& tensors) {
    std::set<std::string> keys;
    for(const auto& kv: tensors) {
        keys.insert(kv.first);
    }
    auto mapping = maybe_map_non_diffusers_lora_to_diffusers(keys);
    if(!mapping.empty()) {
        LoRATensors new_tensors;
        for(const auto& kv: tensors) {
            auto it = mapping.find(kv.first);
            if(it == mapping.end()) {
                // pass
                new_tensors[kv.first] = kv.second;
            } else {
                // replace key
                new_tensors[it->second] = kv.second;
            }
        }
        return new_tensors;
    } else {
        return tensors;
    }
}

} // namespace


namespace ov {
namespace genai {


class AdapterImpl {
public:

    virtual const LoRAConstantTensors& get_constant_tensors() const = 0;
    virtual const LoRATensors& get_tensors() const = 0;
    virtual bool eq(const AdapterImpl* other) const = 0;
};

class SafetensorsAdapterImpl : public AdapterImpl {
public:

    SafetensorsAdapterImpl(const std::filesystem::path& path) {
        auto safetensor_content = read_safetensors(path);
        constant_tensors = group_lora_constant_tensors(safetensor_content, default_lora_constant_patterns());
        for (const auto& constant_tensor : constant_tensors) {
            safetensor_content.erase(constant_tensor.first);
        }
        tensors = group_lora_tensors(safetensor_content, default_lora_patterns());
    }

    SafetensorsAdapterImpl(const ov::Tensor& safetensor)
        : tensors(group_lora_tensors(safetensor_to_constant_map(safetensor), default_lora_patterns())) {}

    const LoRATensors& get_tensors() const override {
        return tensors;
    }

    const LoRAConstantTensors& get_constant_tensors() const override {
        return constant_tensors;
    }

    bool eq(const AdapterImpl* other) const override {
        if(auto other_casted = dynamic_cast<const SafetensorsAdapterImpl*>(other)) {
            return other == this;
        }
        return false;
    }

private:

    LoRATensors tensors;
    LoRAConstantTensors constant_tensors;
};


/// @brief Adapter that derived from another adapter by applying Derivation function.
/// Two objects instanciated from the same Derivation type are equal when both origins and derivations are equal (while comparing with operator==).
/// The derivation is postponed to the first call of get_tensors(), giving a way to compare Adapters without applying the derivation.
/// It is supposed that Derivation works always in the same way and don't have a side effect.
template <typename Derivation>
class DerivedAdapterImpl : public AdapterImpl {
public:

    DerivedAdapterImpl(const std::shared_ptr<AdapterImpl>& origin, const Derivation& derivation) : origin(origin), derivation(derivation) {}

    const LoRATensors& get_tensors() const override {
        if(!tensors) {
            tensors = derivation(origin->get_tensors());
        }
        return *tensors;
    }

    const LoRAConstantTensors& get_constant_tensors() const override {
        return origin->get_constant_tensors();
    }

    bool eq(const AdapterImpl* other) const override {
        if(auto other_casted = dynamic_cast<const DerivedAdapterImpl<Derivation>*>(other)) {
            return origin.get() == other_casted->origin.get() && derivation == other_casted->derivation;
        }
        return false;
    }

private:

    std::shared_ptr<AdapterImpl> origin;
    Derivation derivation;
    mutable std::optional<LoRATensors> tensors;
    mutable std::optional<LoRAConstantTensors> constant_tensors;
};


Adapter diffusers_adapter_normalization(const Adapter& adapter) {
    auto origin = adapter.m_pimpl;
    using DiffusersDerivedAdapter = DerivedAdapterImpl<decltype(&diffusers_normalization)>;
    if(std::dynamic_pointer_cast<DiffusersDerivedAdapter>(origin)) {
        return adapter; // it is already derived adapter, skipping
    }
    return Adapter(std::make_shared<DiffusersDerivedAdapter>(origin, diffusers_normalization));
}

Adapter flux_adapter_normalization(const Adapter& adapter) {
    auto origin = adapter.m_pimpl;
    using FluxDerivedAdapter = DerivedAdapterImpl<decltype(&flux_normalization)>;
    if(std::dynamic_pointer_cast<FluxDerivedAdapter>(origin)) {
        return adapter; // it is already derived adapter, skipping
    }
    return Adapter(std::make_shared<FluxDerivedAdapter>(origin, flux_normalization));
}


Adapter::Adapter(const std::shared_ptr<AdapterImpl>& pimpl) : m_pimpl(pimpl) {}


Adapter::Adapter(const std::filesystem::path& path) :
    m_pimpl(std::make_shared<SafetensorsAdapterImpl>(path)) {
}


Adapter::Adapter(const ov::Tensor& safetensor) :
    m_pimpl(std::make_shared<SafetensorsAdapterImpl>(safetensor)) {
}

bool operator== (const Adapter& a, const Adapter& b) {
    return a.m_pimpl->eq(b.m_pimpl.get());
}


struct AdapterControllerImpl {
    LoRAVarMap variable_ids;
    std::map<std::string, ov::op::util::VariableInfo> constant_variable_ids;
    std::unordered_set<std::string> variable_names;
    AdapterConfig current_config;
    bool need_full_apply = true;
    InferRequestSignatureCache lora_state_evaluators;

    // Stores the actual LoRA weight getter used for Constant tensor replacement
    // Needed to track which LoRA tensors were actually applied to suppress unused tensor warnings
    std::shared_ptr<LoRAWeightGetterDefault<NodePtr, NodePtr>> const_getter_impl;

    AdapterControllerImpl(std::shared_ptr<ov::Model> model, const AdapterConfig& config) :
        current_config(config),  // FIXME: Compare current and passed configs and change incrementally
        lora_state_evaluators("CPU")    // FIXME: Try to run on the same device that is used for model inference
    {
        LoRAConstantGetter const_getter;
        LoRAParametersByWeightGetter params_getter;
        params_getter.type = ov::element::dynamic;

        for(auto const& adapter : current_config.get_adapters()) {
            auto adapter_impl = get_adapter_impl(adapter);
            if (!adapter_impl->get_constant_tensors().empty()) {
                OPENVINO_ASSERT(!const_getter, "OpenVINO.GenAI does not support several LoRA adapters with constants!");
                const_getter_impl = std::make_shared<LoRAWeightGetterDefault<NodePtr, NodePtr>>(
                    &adapter_impl->get_constant_tensors(),
                    config.get_tensor_name_prefix().value_or(""));

                const_getter = [const_getter_impl = this->const_getter_impl](const std::string& name) {
                    return (*const_getter_impl)(name);
                };
            }
            params_getter.weight_getter.push_back(LoRAWeightGetterDefault<LoRAWeight, LoRANode>(&adapter_impl->get_tensors(), config.get_tensor_name_prefix().value_or("")));
            if(params_getter.type != ov::element::f32) {
                for(auto const& tensor : adapter_impl->get_tensors()) {
                    auto lora_tensor_type = tensor.second.A->get_output_element_type(0);
                    OPENVINO_ASSERT(lora_tensor_type == tensor.second.B->get_output_element_type(0));
                    if(params_getter.type == ov::element::dynamic) {
                        params_getter.type = lora_tensor_type;
                    } else if(params_getter.type != lora_tensor_type) {
                        // If types are not match among multiple LoRA tensos then fall back to f32
                        // TODO: Provide a more smart negotiation between multiple LoRAs: check ranges, try to pack to f16
                        //       Make decision on precision per node in LoRAWeightStateGetter instead of setting this global precision
                        params_getter.type = ov::element::f32;
                        break;
                    }
                }
            }
        }

        auto weight_as_constant = [&, this](NodePtr node) -> std::optional<LoRANode> {
            // FIXME: lora_placeholder is for passing element type only
            LoRAParts<ov::Tensor> lora_placeholder{
                ov::Tensor(ov::element::f32, Shape{0}),
                ov::Tensor(params_getter.type, ov::Shape{0}),
                ov::Tensor(params_getter.type, ov::Shape{0})
            };
            auto name = node->get_friendly_name();
            auto lora_weight = prepare_lora_tensors(name, params_getter.weight_getter, lora_placeholder, /*set_empty_tensors=*/false, /*alpha_only=*/false);
            if(lora_weight.alpha) {
                return LoRANode(
                    // TODO: Make sure that tensors will not be disposed during constant life time
                    std::make_shared<v0::Constant>(lora_weight.alpha),
                    std::make_shared<v0::Constant>(lora_weight.A),
                    std::make_shared<v0::Constant>(lora_weight.B)
                );
            } else {
                return std::nullopt;
            }
        };

        auto const_replacement_getter = [const_getter](NodePtr node) -> std::optional<NodePtr> {
            if (const_getter) {
                return const_getter(node->get_friendly_name());
            } else {
                return std::nullopt;
            }
        };

        ov::pass::Manager pm;
        auto mode = current_config.get_mode();
        if(mode == AdapterConfig::MODE_DYNAMIC || mode == AdapterConfig::MODE_STATIC_RANK || mode == AdapterConfig::MODE_AUTO) {
            // State mode
            params_getter.dynamic_lora_rank = (mode != AdapterConfig::MODE_STATIC_RANK);
            pm.register_pass<LoRASeparateTransform>(LoRAWeightStateGetter(params_getter, model, variable_ids));
            if (const_getter) {
                LoRAStateGetterForConst getter = LoRAStateGetterForConst(const_getter, model, constant_variable_ids);
                pm.register_pass<LoRAReplaceConstantTransformDynamic>(getter, getter.create_if_input());
            }
        } else if(mode == AdapterConfig::MODE_STATIC) {
            // Separate constant mode
            pm.register_pass<LoRASeparateTransform>(weight_as_constant);
            pm.register_pass<LoRAReplaceConstantTransformStatic>(const_replacement_getter);
            
        } else if(mode == AdapterConfig::MODE_FUSE) {
            // Fuse mode
            pm.register_pass<LoRAFuseTransform>(weight_as_constant);
            pm.register_pass<LoRAReplaceConstantTransformStatic>(const_replacement_getter);
        } else {
            OPENVINO_THROW("Unrecognized AdapterConfig::Mode was used: ", mode);
        }

        pm.run_passes(model);

        // Collect all variable names to quickly detect which state tensor belongs to this adapter controller later
        for(const auto& var: variable_ids) {
            variable_names.insert(var.second.A.variable_id);
            variable_names.insert(var.second.B.variable_id);
            variable_names.insert(var.second.alpha.variable_id);
        }
        for(const auto& var: constant_variable_ids) {
            variable_names.insert(var.second.variable_id);
        }
    }

    static std::shared_ptr<AdapterImpl> get_adapter_impl(const Adapter& adapter) {
        return adapter.m_pimpl;
    }

    struct ConfigChanged {
        bool mode = false;
        bool alpha = false;
        bool adapter = false;

        operator bool() const {
            return mode || alpha || adapter;
        }
    };

    ConfigChanged compare_configs(const AdapterConfig& config1, const AdapterConfig& config2) {
        ConfigChanged diff;
        diff.mode = config1.get_mode() != config2.get_mode();
        // TODO: Use `set` from this commented block when the config change tracking is implemented at adapter granularity and will track order of adapters correctly
        // std::set<Adapter>
        //     adapters1(config1.adapters.begin(), config1.adapters.end()),
        //     adapters2(config2.adapters.begin(), config2.adapters.end());
        const auto& adapters1 = config1.get_adapters(), adapters2 = config2.get_adapters();

        if(adapters1 != adapters2) {
            diff.adapter = true;
            diff.alpha = true;
        } else {
            for(auto const& adapter: adapters1) {
                diff.alpha = config1.get_alpha(adapter) != config2.get_alpha(adapter);
            }
        }
        return diff;
    }

    void apply (ov::InferRequest& infer_request, std::optional<AdapterConfig> config) {
        // FIXME: If a part of LoRA state tensors are not set here, then need to carefully reset state in LLMPipeline where global reset is called after the generation
        ConfigChanged diff;
        if(config) {
            diff = compare_configs(current_config, *config);
            OPENVINO_ASSERT(
                !diff.mode || config->get_mode() == AdapterConfig::MODE_AUTO,  // MODE_AUTO in this call means that mode is not changed
                "AdapterConfig::mode cannot be changed and should be configured once for a model at the initialization");
            OPENVINO_ASSERT(
                config->get_mode() == AdapterConfig::MODE_AUTO || config->get_mode() == AdapterConfig::MODE_DYNAMIC || config->get_mode() == AdapterConfig::MODE_STATIC_RANK || (!diff.alpha && !diff.adapter),
                "Cannot change adapters and/or the alphas when not one of the dynamic modes are used.");
            current_config.update(*config);
        }
        if(need_full_apply) {
            need_full_apply = false;
            set_new_adapter_tensors(infer_request);
        } else if(diff) {
            if(diff.adapter) {
                set_new_adapter_tensors(infer_request);
            } else if(diff.alpha)  {
                set_new_adapter_alphas(infer_request);
            }
        }
    }

    bool has_state_name(const std::string& name) {
        return variable_names.count(name);
    }

    void set_new_adapter_alphas (ov::InferRequest& infer_request) {
        set_new_adapter_tensors(infer_request, /*alpha_only=*/true);
    }

    void set_new_adapter_tensors(ov::InferRequest& infer_request, bool alpha_only = false) {        
        if (current_config.get_mode() != AdapterConfig::MODE_AUTO && 
            current_config.get_mode() != AdapterConfig::MODE_DYNAMIC &&
            current_config.get_mode() != AdapterConfig::MODE_STATIC_RANK ) {
            return;
        }

        std::vector<LoRAWeightGetter> weight_getters;
        LoRAConstantGetter const_getter;
        const auto& adapters = current_config.get_adapters();
        weight_getters.reserve(adapters.size());
        for (const auto& adapter : adapters) {
            auto adapter_impl = get_adapter_impl(adapter);
            if (!adapter_impl->get_constant_tensors().empty()) {
                OPENVINO_ASSERT(!const_getter,
                                "OpenVINO.GenAI does not support several LoRA adapters with constants!");
                const_getter = LoRAWeightGetterDefault<NodePtr, NodePtr>(
                    &adapter_impl->get_constant_tensors(),
                    current_config.get_tensor_name_prefix().value_or(""));
            }
            weight_getters.emplace_back(
                LoRAWeightGetterDefault<LoRAWeight, LoRANode>(&adapter_impl->get_tensors(),
                                                              current_config.get_tensor_name_prefix().value_or("")));
        }

        auto state = infer_request.query_state();
        // TODO: Forced to use variable_id instead of index to address the state tensors, require the same order for state as for variables from plugins

        // Convert LoRAVarIDs to LoRAIndices to speedup search for state with a given name
        // TODO: If state order is stable, then the mapping should be done once for a given infer request, TODO: cache it based on the infer request
        std::map<std::string, size_t> state_name_to_index;
        for(size_t i = 0; i < state.size(); ++i) {
            auto name = state[i].get_name();
            state_name_to_index[name] = i;
        }

        for(const auto& lora_var_ids : variable_ids) {
            // FIXME: Remove this mapping when the order of state will be the same as the order of variables
            LoRAIndices lora_indices;
            lora_indices.alpha = state_name_to_index.at(lora_var_ids.second.alpha.variable_id);
            lora_indices.A = state_name_to_index.at(lora_var_ids.second.A.variable_id);
            lora_indices.B = state_name_to_index.at(lora_var_ids.second.B.variable_id);
            set_lora_tensors(state, lora_var_ids.first, lora_var_ids.second, lora_indices, weight_getters, alpha_only);
        }

        for (const auto& [const_name, var_info] : constant_variable_ids) {

            size_t const_lora_index = state_name_to_index.at(var_info.variable_id);

            if (const_name.find("const") != std::string::npos) {  // if constant flag
                ov::Tensor const_tensor(var_info.data_type, dynamic_to_static(var_info.data_shape));
                const_tensor.data<bool>()[0] = static_cast<bool>(const_getter);
                state[const_lora_index].set_state(const_tensor);

            } else if (const_getter) {
                auto opt_lora_const = const_getter(const_name);

                if (opt_lora_const && const_getter_impl) {
                    const_getter_impl->used_tensors.insert(const_name);
                }

                auto constant_node = std::dynamic_pointer_cast<v0::Constant>(*opt_lora_const);
                OPENVINO_ASSERT(constant_node, "Expected ov::op::v0::Constant for ", const_name);

                ov::Tensor const_tensor = ov::Tensor(constant_node->get_element_type(), constant_node->get_shape());
                std::memcpy(const_tensor.data(), constant_node->get_data_ptr(), const_tensor.get_byte_size());
                state[const_lora_index].set_state(const_tensor);
            }
        }

    }

    std::vector<LoRAWeight> collect_applicable_tensors (const std::string& lora_name, const std::vector<LoRAWeightGetter>& weight_getters) {
        const auto& adapters = current_config.get_adapters();
        OPENVINO_ASSERT(weight_getters.size() == adapters.size());
        std::vector<LoRAWeight> result;
        result.reserve(weight_getters.size());
        for(size_t i = 0; i < adapters.size(); ++i) {
            if(auto lora_tensors = weight_getters[i](lora_name)) {
                // TODO: Is it practical to use alpha from the adapter file itself. In the current code it is ignored and only alpha from config is used.
                OPENVINO_ASSERT(lora_tensors->A);
                OPENVINO_ASSERT(lora_tensors->B);
                lora_tensors->alpha = alpha_as_constant(current_config.get_alpha(adapters[i]));
                result.push_back(LoRAWeight(
                    std::dynamic_pointer_cast<v0::Constant>(lora_tensors->alpha),
                    std::dynamic_pointer_cast<v0::Constant>(lora_tensors->A),
                    std::dynamic_pointer_cast<v0::Constant>(lora_tensors->B)
                ));
            }
        }
        return result;
    }

    using Signature = InferRequestSignatureCache::Signature;

    Signature get_tensor_signature(const ov::element::Type& type, const ov::PartialShape& shape) {
        return '(' + type.get_type_name() + shape.to_string() + ')';
    }

    Signature get_tensor_signature(const std::shared_ptr<v0::Constant>& constant) {
        return get_tensor_signature(constant->get_element_type(), constant->get_shape());
    }

    Signature get_tensor_signature(const ov::Tensor& tensor, const PartialShape& overridden_shape) {
        return tensor ? get_tensor_signature(tensor.get_element_type(), overridden_shape) : Signature();
    }

    Signature get_lora_signature(const std::vector<LoRAWeight>& inputs, const LoRAParts<ov::Tensor>& outputs) {
        Signature signature;
        for(const auto& input: inputs) {
            signature +=
                std::string("(") +
                    get_tensor_signature(input.alpha) +
                    get_tensor_signature(input.A) +  // TODO: Adjust shape to have a dynamic low-rank LoRA dimension in case of fully static shape doesn't have significant speedup
                    get_tensor_signature(input.B) +  // TODO: Adjust shape to have a dynamic low-rank LoRA dimension in case of fully static shape doesn't have significant speedup
                ")";
        }
        signature +=
            std::string("(") +
                // Shape is set to be dynamic because it doesn't matter for signature as it is completely determined by the corresponding model
                // The corresponding model hasn't been created at the moment when this function is got called, so to avoid duplicated shape propagation logic for
                // outputs we just use the target rank of output tensors that is enough to distinguish output signatures.
                get_tensor_signature(outputs.alpha, ov::PartialShape::dynamic(1)) +
                get_tensor_signature(outputs.A, ov::PartialShape::dynamic(2)) +
                get_tensor_signature(outputs.B, ov::PartialShape::dynamic(2)) +
            ")";
        return signature;
    }

    ov::TensorVector to_tensor_vector(const std::vector<LoRAWeight>& v, bool alpha_only) {
        ov::TensorVector result;
        result.reserve(v.size()*(alpha_only ? 1 : 3));
        for(auto const& lora_weights: v) {
            result.push_back(lora_weights.alpha->get_tensor_view());
            if(!alpha_only) {
                result.push_back(lora_weights.A->get_tensor_view());
                result.push_back(lora_weights.B->get_tensor_view());
            }
        }
        return result;
    }

    LoRAParts<ov::Tensor> from_tensor_vector(const ov::TensorVector& v, bool alpha_only) {
        OPENVINO_ASSERT(v.size() == (alpha_only ? 1 : 3));
        return LoRAParts<ov::Tensor>(v[0], alpha_only ? ov::Tensor() : v[1], alpha_only ? ov::Tensor() : v[2]);
    }

    ov::TensorVector to_tensor_vector(const LoRAParts<ov::Tensor>& lora_tensors, bool alpha_only) {
        ov::TensorVector result;
        result.reserve((alpha_only ? 1 : 3));
        result.push_back(lora_tensors.alpha);
        if(!alpha_only) {
            result.push_back(lora_tensors.A);
            result.push_back(lora_tensors.B);
        }
        return result;
    }

    void build_concat_model(
        ov::ParameterVector& parameters,
        ov::ResultVector& results,
        const std::vector<LoRAWeight>& inputs,
        ov::Tensor output,
        size_t offset,
        size_t concat_axis,
        bool alpha_only,
        std::function<std::shared_ptr<v0::Parameter>(const LoRAWeight&)> input_accessor,
        std::function<NodePtr(const LoRAWeight&, NodePtr)> parameter_postprocessing = [](const LoRAWeight&, NodePtr node) { return node; }
    ) {
        ov::OutputVector concat_inputs;
        concat_inputs.reserve(inputs.size());
        for(size_t i = 0; i < inputs.size(); ++i) {
            NodePtr input = parameters[(alpha_only ? 1 : 3)*i + offset] = input_accessor(inputs[i]);
            if(input->get_output_element_type(0) != output.get_element_type()) {
                input = std::make_shared<v0::Convert>(input, output.get_element_type());
            }
            if(input->get_output_partial_shape(0).rank().get_length() > 2) {
                input = squeeze_2d(input);
            }
            input = parameter_postprocessing(inputs[i], input);
            concat_inputs.push_back(input);
        }

        NodePtr result;
        if(concat_inputs.size() > 1) {
            result = std::make_shared<v0::Concat>(concat_inputs, concat_axis);
        } else {
            result = concat_inputs.front().get_node_shared_ptr();
        }

        results[offset] = std::make_shared<v0::Result>(result);
    }

    LoRAParts<ov::Tensor> empty_adapters(const std::vector<LoRAWeight>& inputs, LoRAParts<ov::Tensor>& outputs) {
        outputs.alpha.set_shape({1, 0});
        outputs.A.set_shape({0, outputs.A.get_shape()[1]});
        outputs.B.set_shape({outputs.B.get_shape()[0], 0});
        return outputs;
    }

    LoRAParts<ov::Tensor> concat_adapters(const std::vector<LoRAWeight>& inputs, LoRAParts<ov::Tensor>& outputs, bool alpha_only) {
        auto signature = get_lora_signature(inputs, outputs);
        size_t inputs_per_adapter = alpha_only ? 1 : 3;
        if(!lora_state_evaluators.exist(signature)) {
            // Prepare LoRA state evaluate model
            ov::ParameterVector parameters(inputs_per_adapter*inputs.size());
            ov::ResultVector results(inputs_per_adapter);

            build_concat_model(parameters, results, inputs, outputs.alpha, 0, 1,
                alpha_only,
                [](const LoRAWeight& lora_weight) {
                    return std::make_shared<v0::Parameter>(
                        lora_weight.alpha->get_output_element_type(0),
                        lora_weight.alpha->get_output_partial_shape(0));    // TODO: Consider using dynamic LoRA rank dimension instead of static dimension
                },
                [](const LoRAWeight& lora_weight, NodePtr parameter) {
                    // TODO: This code should be modified if dynamic LoRA rank is used in the evaluator
                    auto lora_rank = lora_weight.A->get_output_partial_shape(0)[0].get_length();
                    // Broadcast a single alpha element to shape [lora_rank]
                    auto lora_rank_constant = v0::Constant::create(ov::element::u32, Shape{2}, std::vector<decltype(lora_rank)>{1, lora_rank});
                    return std::make_shared<v3::Broadcast>(parameter, lora_rank_constant);
                });

            if(!alpha_only) {
                build_concat_model(parameters, results, inputs, outputs.A, 1, 0,
                    alpha_only,
                    [](const LoRAWeight& lora_weight) {
                        return std::make_shared<v0::Parameter>(
                            lora_weight.A->get_output_element_type(0),
                            lora_weight.A->get_output_partial_shape(0));    // TODO: Consider using dynamic LoRA rank dimension instead of static dimension
                    }
                );

                build_concat_model(parameters, results, inputs, outputs.B, 2, 1,
                    alpha_only,
                    [](const LoRAWeight& lora_weight) {
                        return std::make_shared<v0::Parameter>(
                            lora_weight.B->get_output_element_type(0),
                            lora_weight.B->get_output_partial_shape(0));    // TODO: Consider using dynamic LoRA rank dimension instead of static dimension
                    }
                );
            }

            lora_state_evaluators.insert(signature, results, parameters);
        }
        auto output_tensors = to_tensor_vector(outputs, alpha_only);
        lora_state_evaluators.evaluate(signature, to_tensor_vector(inputs, alpha_only), output_tensors);
        return from_tensor_vector(output_tensors, alpha_only);
    }

    ov::Shape dynamic_to_static(const ov::PartialShape& pshape) {
        ov::Shape shape(pshape.rank().get_length());
        for(size_t i = 0; i < pshape.rank().get_length(); ++i) {
            shape[i] = pshape[i].is_dynamic() ? 0 : pshape[i].get_length();
        }
        return shape;
    }

    void set_lora_tensors(
        std::vector<VariableState>& state,
        const std::string& name,
        const LoRAVarIDs& lora_var_ids,
        const LoRAIndices& lora_indices,
        const std::vector<LoRAWeightGetter>& weight_getters,
        bool alpha_only
    ) {
        LoRAParts<ov::Tensor> lora_state_tensors{
            ov::Tensor(lora_var_ids.alpha.data_type, dynamic_to_static(lora_var_ids.alpha.data_shape)),
            alpha_only ? ov::Tensor() : ov::Tensor(lora_var_ids.A.data_type, dynamic_to_static(lora_var_ids.A.data_shape)),
            alpha_only ? ov::Tensor() : ov::Tensor(lora_var_ids.B.data_type, dynamic_to_static(lora_var_ids.B.data_shape))
        };
        auto new_tensors = prepare_lora_tensors(name, weight_getters, lora_state_tensors, /*set_empty_adapters=*/true, alpha_only);
        state[lora_indices.alpha].set_state(new_tensors.alpha);
        if(!alpha_only) {
            state[lora_indices.A].set_state(new_tensors.A);
            state[lora_indices.B].set_state(new_tensors.B);
        }
    }

    LoRAParts<ov::Tensor> prepare_lora_tensors (
        const std::string& name,
        const std::vector<LoRAWeightGetter>& weight_getters,
        LoRAParts<ov::Tensor>& output,
        bool set_empty_adapters,
        bool alpha_only
    ) {
        auto lora_tensors = collect_applicable_tensors(name, weight_getters);  // request A and B regardless of alpha_only, because it is a way to get lora_rank later when alpha is broadcasted
        LoRAParts<ov::Tensor> new_tensors;
        if(!lora_tensors.empty()) {
            new_tensors = concat_adapters(lora_tensors, output, alpha_only);
        } else if(set_empty_adapters) {
            new_tensors = empty_adapters(lora_tensors, output);
        }
        return new_tensors;
    }
};


AdapterController::AdapterController(std::shared_ptr<ov::Model> model, const AdapterConfig& config, std::string device)
{
    // If AdapterConfig::MODE_AUTO is used, then set real mode depending on the device capabilities
    // TODO: Remove this code when devices become aligned on their capabilities for LoRA adapters
    if (config.get_mode() == AdapterConfig::MODE_AUTO) {
        static const std::map<std::string, AdapterConfig::Mode> default_modes {
            {"CPU", AdapterConfig::MODE_DYNAMIC},
            {"GPU", AdapterConfig::MODE_DYNAMIC},
            {"NPU", AdapterConfig::MODE_DYNAMIC},
        };
        if(device.find("GPU") != std::string::npos) {  // to handle GPU device variants which doesn't matter for adapter mode
            device = "GPU";
        }
        auto default_mode = default_modes.find(device);
        if(default_mode != default_modes.end()) {
            AdapterConfig updated_config = config;
            updated_config.set_mode(default_mode->second);
            m_pimpl = std::make_shared<AdapterControllerImpl>(model, updated_config);
            return;
        } else {
            std::string device_msg;
            if(device.empty()) {
                device_msg = "No device set";
            } else {
                device_msg = "Device \"" + device + "\" is unrecognized";
            }
            std::cout
                << "[ WARNING ] " << device_msg << " to deduce default device-dependent LoRA application mode.\n"
                << "This warning appears because no specific LoRA mode was set in AdapterConfig or MODE_AUTO was used explicitly.\n"
                << "To avoid this warning set one of the AdapterConfig::Mode values except MODE_AUTO.";
        }
    }
    m_pimpl = std::make_shared<AdapterControllerImpl>(model, config);
}


// Call it every time when adapter config is changed; if adapter was configured as a static one, this call is not required
void AdapterController::apply(ov::InferRequest request, const std::optional<AdapterConfig>& config) {
    OPENVINO_ASSERT(m_pimpl || !config || !*config,
        "Adapters are passed to AdapterController but it was not configured to use adapters. "
        "Enable using adapters by pass them in the constructor first.");
    if (m_pimpl) {
        m_pimpl->apply(request, config);
    }
}

bool AdapterController::has_state_name(const std::string& name) {
    return m_pimpl->has_state_name(name);
}


void AdapterConfig::set_mode(Mode _mode) {
    mode = _mode;
}


AdapterConfig::AdapterConfig (const std::vector<Adapter>& adapters, Mode mode) : mode(mode), adapters(adapters) {
    alphas.reserve(adapters.size());
    for(const auto& adapter: adapters) {
        auto const alpha = 1;
        alphas.push_back(alpha);
    }
}


AdapterConfig::AdapterConfig (const std::vector<std::pair<Adapter, float>>& _adapters, Mode mode) : mode(mode) {
    adapters.reserve(_adapters.size());
    alphas.reserve(_adapters.size());
    for(auto const& adapter_and_alpha: _adapters) {
        adapters.push_back(adapter_and_alpha.first);
        alphas.push_back(adapter_and_alpha.second);
    }
}


AdapterConfig::AdapterConfig(Mode mode) : mode(mode) {}


AdapterConfig& AdapterConfig::add(const Adapter& adapter, float alpha) {
    OPENVINO_ASSERT(adapters.size() == alphas.size());
    OPENVINO_ASSERT(adapters.end() == std::find(adapters.begin(), adapters.end(), adapter), "Adapter object passed to AdapterConfig::add was already registered");
    adapters.push_back(adapter);
    alphas.push_back(alpha);
    return *this;
}


AdapterConfig& AdapterConfig::add(const Adapter& adapter) {
    return add(adapter, 1);
}


AdapterConfig& AdapterConfig::set_alpha(const Adapter& adapter, float alpha) {
    OPENVINO_ASSERT(adapters.size() == alphas.size());
    auto it = std::find(adapters.begin(), adapters.end(), adapter);
    OPENVINO_ASSERT(adapters.end() != it, "Unknown adapter object passed to AdapterConfig::set_alpha, register adapter object first with AdapterConfig::add");
    auto index = it - adapters.begin();
    alphas[index] = alpha;
    return *this;
}


float AdapterConfig::get_alpha(const Adapter& adapter) const {
    OPENVINO_ASSERT(adapters.size() == alphas.size());
    auto it = std::find(adapters.begin(), adapters.end(), adapter);
    OPENVINO_ASSERT(adapters.end() != it, "Unknown adapter object passed to AdapterConfig::get_alpha, alpha can be retrieved for previously registered adapters only");
    return alphas[it - adapters.begin()];
}


AdapterConfig& AdapterConfig::remove(const Adapter& adapter) {
    OPENVINO_ASSERT(adapters.size() == alphas.size());
    auto it = std::find(adapters.begin(), adapters.end(), adapter);
    OPENVINO_ASSERT(adapters.end() != it, "Unknown adapter object passed to AdapterConfig::remove, you can remove previously registered adapters only");
    auto index = it - adapters.begin();
    alphas.erase(alphas.begin() + index);
    adapters.erase(it);
    return *this;
}


void AdapterConfig::update (const AdapterConfig& other) {
    adapters = other.adapters;
    alphas = other.alphas;
    if(other.mode != MODE_AUTO) {
        mode = other.mode;
    }
    if(other.tensor_name_prefix) {
        tensor_name_prefix = other.tensor_name_prefix;
    }
}

std::vector<std::pair<Adapter, float>> AdapterConfig::get_adapters_and_alphas() const {
    OPENVINO_ASSERT(adapters.size() == alphas.size());
    std::vector<std::pair<Adapter, float>> result;
    for(size_t i = 0; i < adapters.size(); ++i) {
        result.emplace_back(adapters[i], alphas[i]);
    }
    return result;
}

void AdapterConfig::set_adapters_and_alphas(const std::vector<std::pair<Adapter, float>>& _adapters) {
    adapters.clear();
    alphas.clear();
    for(auto const& adapter_and_alpha: _adapters) {
        adapters.push_back(adapter_and_alpha.first);
        alphas.push_back(adapter_and_alpha.second);
    }
}


}  // namespace genai
}  // namespace ov
