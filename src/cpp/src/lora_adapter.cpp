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
#include <optional>

#include "openvino/op/add.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/read_value.hpp"
#include "openvino/op/assign.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/util/variable.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/runtime/core.hpp"

#include "openvino/genai/lora_adapter.hpp"

#define SAFETENSORS_IMPLEMENTATION
#include "safetensors.h"

namespace {

using NodePtr = std::shared_ptr<ov::Node>;
using ov::NodeVector;
using namespace ov::op;

// FIXME: Use ov::AlignedBuffer instead of std::vector. ov::AlignedBuffer is not available in public OV API
using Buffer = std::vector<char>;
using BufferPtr = std::shared_ptr<Buffer>;
using ConstantVector = std::vector<std::shared_ptr<v0::Constant>>;

std::ostream& print_tensor (std::ostream& out, const ov::Tensor& tensor) {
    out << tensor.get_element_type().get_type_name() + tensor.get_shape().to_string() << ":\n";
    auto constant = std::make_shared<v0::Constant>(tensor);
    auto elements = constant->get_value_strings();
    for(size_t i = 0; i < elements.size(); ++i) {
        out << elements[i] << ", ";
    }
    out << "\n";
}

template <typename T>
struct LoRAParts {
    T alpha, A, B;

    LoRAParts() = default;
    LoRAParts(const T& alpha, const T& A, const T& B) : alpha(alpha), A(A), B(B) {}

    template <typename Other>
    LoRAParts(const LoRAParts<Other>& other) : alpha(other.alpha), A(other.A), B(other.B) {}
};

using LoRAWeight = LoRAParts<std::shared_ptr<v0::Constant>>;
using LoRANode = LoRAParts<std::shared_ptr<ov::Node>>;
using LoRAPartsParser = LoRAParts<std::function<std::optional<std::string>(const std::string& name)>>;
using LoRATensors = std::map<std::string, LoRAWeight>;

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
        RegexParser(R"((.*)\.alpha)", 1),
        RegexParser(R"((.*)\.(lora_(A|down)\.weight))", 1),
        RegexParser(R"((.*)\.(lora_(B|up)\.weight))", 1)
    );
}


BufferPtr read_file_helper(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    OPENVINO_ASSERT(file.is_open(), "Cannot open file with LoRA weights: ", filename);

    size_t filesize = file.tellg();
    auto buffer = std::make_shared<Buffer>(filesize);
    //buffer->reserve(filesize);
    file.seekg(0, std::ios::beg);
    // FIXME: Use mmapped AlignedBuffer as ov::Core::read_model can do, necessary functionality is not available in public OV API
    //std::copy(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>(), std::back_inserter(*buffer));
    file.read(&(*buffer)[0], filesize);

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
NodePtr squeeze_2d (const ov::Output<ov::Node>& input) {
    // auto rank = node->get_output_partial_shape(0).rank().get_length();
    // std::vector<unsigned int> dims(2);
    //auto squeeze_num = rank - 2;
    // std::fill_n(dims.begin() + 2, dims.end(), 1);
    auto shape = v0::Constant::create(ov::element::i32, {2}, std::vector<int>{0, 0});
    auto reshape = std::make_shared<v1::Reshape>(input, shape->output(0), true);
    return reshape;
}

// Unsqueeze shape to add dimensions to the right to have `rank`-D tensor
NodePtr unsqueeze (const ov::Output<ov::Node>& input, unsigned int rank) {
    auto src_rank = input.get_partial_shape().rank().get_length();
    std::vector<unsigned int> dims(rank);
    std::fill(dims.begin() + src_rank, dims.end(), 1);
    auto shape = v0::Constant::create(ov::element::i32, {rank}, dims);
    auto reshape = std::make_shared<v1::Reshape>(input, shape->output(0), true);
    return reshape;
}

using LoRAWeightGetter = std::function<std::optional<LoRANode>(const std::string&)>;
using LoRAWeightByNodeGetter = std::function<std::optional<LoRANode>(NodePtr)>;

struct LoRAParameters {
    ov::Dimension rank;
    ov::element::Type type;
    bool fine_grained_alpha;    // use 1D tensor of the same rank for alpha instead of a scalar to blend multiple weighted LoRAs
    // TODO: alpha different over the batch?
};

using LoRAParametersGetter = std::function<std::optional<LoRAParameters>(NodePtr node)>;

struct LoRAWeightGetterDefault {
    // TODO: Add filtering by tensor name prefix
    const LoRATensors* lora_tensors;
    const std::string prefix;
    mutable std::set<std::string> used_tensors;
    LoRAWeightGetterDefault (const LoRATensors* lora_tensors, const std::string& prefix) : lora_tensors(lora_tensors), prefix(prefix) {}

    // std::optional<LoRANode> operator() (NodePtr node) const {
    //     return operator()(node->get_friendly_name());
    // }

    std::optional<LoRANode> operator() (const std::string& name) const {
        std::string name_with_underscores = name;
        std::replace(name_with_underscores.begin(), name_with_underscores.end(), '.', '_');   // FIXME: Customize mapping or change PT FE to produce correct weight names
        //DEBUG_PRINT("Layer candidate: " << name << " ---OR--- " << name_with_underscores);
        auto it = std::find_if(lora_tensors->begin(), lora_tensors->end(), [this, name, name_with_underscores](const LoRATensors::value_type& pair){
            std::string lora_name = pair.first;
            // TODO: Make this filtering for prefix once in ctor
            if(lora_name.find(prefix) == 0) {
                lora_name = lora_name.substr(prefix.length());
            } else {
                return false;
            }
            return name.find(lora_name) != std::string::npos || name_with_underscores.find(lora_name) != std::string::npos;  // FIXME: Should it be an exact match instead of substring taking into account that we should provide custom mapper for names?
        });
        if(it != lora_tensors->end()) {
            used_tensors.insert(it->first);
            return it->second;
        }
        return std::nullopt;
    }

    ~LoRAWeightGetterDefault () {
        //DEBUG_PRINT("Used LoRA tensors: " << used_tensors.size());
    }
};


struct LoRAParametersByWeightGetter {
    // TODO: Consider passing AdapterConfig instead of decomposed separate parameters
    std::vector<LoRAWeightGetter> weight_getter;
    bool dynamic_lora_rank = true;
    bool fine_grained_alpha = true;
    ov::element::Type type;

    std::optional<LoRAParameters> operator() (NodePtr node) const {
        // If at least one weight_getter gives the weight for the node, then this node should be processed
        OPENVINO_ASSERT(dynamic_lora_rank, "LoRAParametersByWeightGetter doesn't support static LoRA rank, use dynamic");
        // TODO: To implement known static LoRA rank, need to accumulate ranks from all found weights instead of searching for at least one match:

        auto it = std::find_if(weight_getter.begin(), weight_getter.end(), [node](const LoRAWeightGetter& getter) {
            return bool(getter(node->get_friendly_name()));
        });
        if(weight_getter.end() != it) {
            LoRAParameters result;
            result.rank = ov::Dimension();      // FIXME: uncomment dynamic dimension
            result.type = type;
            result.fine_grained_alpha = fine_grained_alpha;
            return result;
        } else {
            return std::nullopt;
        }
    }
};

// FIXME: Move name from LoRAVarIDs to to LoRAIndices when the order of state will be the same as the order of variables, remove LoRAVarsIDs
struct LoRAIndices : public LoRAParts<size_t> {
    std::string name;
};

struct LoRAVarIDs : public LoRAParts<std::string> {
    std::string name;
};

void deduce_input_output_dims(NodePtr node, ov::Dimension& input_dim, ov::Dimension& output_dim) {
    if(std::dynamic_pointer_cast<v1::Convolution>(node)) {
        input_dim = node->get_input_partial_shape(1)[1];
        output_dim = node->get_input_partial_shape(1)[0];
    } else if(auto matmul = std::dynamic_pointer_cast<v0::MatMul>(node)) {
        input_dim = node->get_input_partial_shape(1)[matmul->get_transpose_b()];
        output_dim = node->get_input_partial_shape(1)[!matmul->get_transpose_b()];
    } else {
        OPENVINO_THROW("deduce_input_output_dims expects MatMul or Convolution, but get ", node);
    }
}

struct LoRAWeightStateGetter {
    LoRAParametersGetter params_getter;
    std::shared_ptr<ov::Model> model;
    std::vector<LoRAVarIDs>& variable_ids;
    // TODO: Use variable indices instead of variable_id for faster search for a state tensor

    LoRAWeightStateGetter (const LoRAParametersGetter& params_getter, std::shared_ptr<ov::Model> model, std::vector<LoRAVarIDs>& variable_ids) :
        params_getter(params_getter), model(model), variable_ids(variable_ids) {}

    std::optional<LoRANode> operator() (NodePtr node) const {
        if(auto params = params_getter(node)) {
            ov::Dimension input_dim, output_dim;
            deduce_input_output_dims(node, input_dim, output_dim);

            std::string name = node->get_friendly_name();
            std::string variable_id_prefix = "lora_state_" + std::to_string(model->get_sinks().size()) + name;
            LoRANode result;
            LoRAVarIDs var_ids;
            var_ids.name = name;

            // FIXME: No guarantees on ordering of state in InferRequest makes impossible using indices of variables later, forced to use variable_id instead
            //indices.A = model->get_variables().size();
            var_ids.A = variable_id_prefix + ".A";
            result.A = add_variable(
                ov::PartialShape{params->rank, input_dim},  // Will be used with transpose_b == true
                params->type,
                var_ids.A
            );
            // FIXME: No guarantees on ordering of state in InferRequest makes impossible using indices of variables later, forced to use variable_id instead
            //indices.A = model->get_variables().size();
            var_ids.alpha = variable_id_prefix + ".alpha";
            result.alpha = add_variable(
                params->fine_grained_alpha ? ov::PartialShape{1, params->rank} : ov::PartialShape{},
                ov::element::f32,
                var_ids.alpha
            );
            // FIXME: No guarantees on ordering of state in InferRequest makes impossible using indices of variables later, forced to use variable_id instead
            //indices.B = model->get_variables().size();
            var_ids.B = variable_id_prefix + ".B";
            result.B = add_variable(
                ov::PartialShape{output_dim, params->rank},  // Will be used with transpose_b == true
                params->type,
                var_ids.B
            );
            variable_ids.emplace_back(var_ids);
            return result;
        } else {
            return std::nullopt;
        }
    }

    NodePtr add_variable(const ov::PartialShape& shape, const ov::element::Type& type, const std::string& variable_id) const {
        auto variable = std::make_shared<ov::op::util::Variable>(ov::op::util::VariableInfo{
            shape, ov::element::f32, variable_id
        });
        model->add_variables({variable});
        #if 0
        // FIXME: CPU plugin fails when there is no initialization expression is given and type is not fp32
        ov::Shape init_shape(shape.rank().get_length());
        for(size_t i = 0; i < shape.size(); ++i) {
            init_shape[i] = shape[i].get_min_length();
        }
        DEBUG_PRINT("Workaround for init, shape: " << init_shape);
        auto init = v0::Constant::create(type, init_shape, std::vector<float>(ov::shape_size(init_shape), 0));
        auto read_value = std::make_shared<v6::ReadValue>(init, variable);
        #else
        auto read_value = std::make_shared<v6::ReadValue>(variable);
        #endif
        model->add_sinks({std::make_shared<v6::Assign>(read_value, variable)});  // FIXME: Required? -- Yes, create ticket agains CPU
        //return std::make_shared<v1::Add>(read_value, v0::Constant::create(type, ov::Shape{}, {1e-5}));    // FIXME: Workaround for bug in CPU plugin
        return read_value;
    }
};


class LoRATransformBase : public ov::pass::MatcherPass {
public:

    OPENVINO_RTTI("LoRATransformBase");
   
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
        DEBUG_PRINT("LoRA applied for " << applied << " layers"); // FIXME: For debugging purposes only
    }

protected:

    virtual bool apply(NodePtr node, const LoRANode& lora_weight) = 0;

private:

    size_t applied = 0; // FIXME: For debug statistics only
    //LoRAWeightByNodeGetter lora_weight_getter;

};


NodePtr tensors_multiplication(NodePtr input, const NodeVector multipliers, ov::Output<ov::Node> target, bool transpose_weights, size_t alpha_pos) {
    const auto target_type = target.get_element_type();
    const auto target_shape = target.get_partial_shape();
    const auto target_rank = target_shape.rank().get_length();
    for(size_t i = 0; i < multipliers.size(); ++i) {
        NodePtr normalized = multipliers[i];
        if(normalized->get_output_element_type(0) != target_type) {
            normalized = std::make_shared<v0::Convert>(normalized, target_type);
        }
        if(normalized->get_output_partial_shape(0).rank().get_length() > 2) {
            // FIXME: Any other shape patterns possible?
            normalized = squeeze_2d(normalized);
        }
        if(input) {
            if(i == alpha_pos) {
                // FIXME: Apply alpha multiplication separately
                input = std::make_shared<v1::Multiply>(input, normalized);
            } else {
                input = std::make_shared<v0::MatMul>(input, normalized, /*transpose_a = */false, transpose_weights);  // FIXME: verify transpose_a == true
            }
        } else {
            input = normalized;
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


class InferRequestSignatureCache {
public:
    using Signature = std::string;

    InferRequestSignatureCache (const std::string& device) : device(device) {}

    bool exist (const Signature& signature) {
        return requests.count(signature);
    }

    void insert (const Signature& signature, std::shared_ptr<ov::Model> model) {
        //DEBUG_PRINT(signature);
        requests[signature] = core.compile_model(model, device).create_infer_request();
    }

    ov::InferRequest& at(const Signature& signature) {
        return requests.at(signature);
    }

    void evaluate(const Signature& signature, const std::vector<ov::Tensor>& inputs, std::vector<ov::Tensor>& outputs) {
        auto& request = at(signature);
        for(size_t i = 0; i < inputs.size(); ++i) {
            request.set_input_tensor(i, inputs[i]);
        }
        for(size_t i = 0; i < outputs.size(); ++i) {
            auto target_shape = request.get_compiled_model().output(i).get_partial_shape();
            if(target_shape.is_static()) {  // TODO: Can we avoid this corner case for static shapes? Make it tensor-type dependent.
                outputs[i].set_shape(target_shape.get_shape());
            }
            request.set_output_tensor(i, outputs[i]);
        }
        request.infer();
    }

private:

    ov::Core core;
    std::unordered_map<Signature, ov::InferRequest> requests;
    std::string device;
};


class LoRAFuseTransform : public LoRATransformBase {
    // TODO: Migrate to InferReuestSignatureCache
    using Signature = std::string;
    ov::Core core;
    std::map<Signature, ov::InferRequest> compiled_weight_models;

    void signature_push_back(Signature& signature, ov::Output<ov::Node> input) const {
        // FIXME: Define hash function on vector<tuple<element_type, PartialShape>> to make it C++ish
        signature += "(el: " + input.get_element_type().get_type_name() + ", shape: " + input.get_partial_shape().to_string() + ")";
    }

public:
    
    OPENVINO_RTTI("LoRAFuseTransform");
    
    LoRAFuseTransform(const LoRAWeightByNodeGetter& lora_weight_getter) : LoRATransformBase(lora_weight_getter) {}

    bool apply (NodePtr node, const LoRANode& lora_weight) override {
        auto activations = node->input_value(0);    // FIXME: consider MatMul.transpose_a
        auto weights_input = node->input_value(1);
        auto weights_input_type = weights_input.get_element_type();
        //DEBUG_PRINT("WEIGHTS SHAPE: " << weights_input.get_partial_shape());
        auto weights_convert = decompression_convert(weights_input.get_node_shared_ptr());
        auto weights_constant = weights_convert ? weights_convert->input_value(0) : weights_input;
        ConstantVector adapter = {
            std::dynamic_pointer_cast<v0::Constant>(lora_weight.alpha),
            std::dynamic_pointer_cast<v0::Constant>(lora_weight.B),
            std::dynamic_pointer_cast<v0::Constant>(lora_weight.A)};
        NodePtr add_term = nullptr;
        Signature signature;
        signature_push_back(signature, weights_input);
        for(auto multiplier : adapter) {
            signature_push_back(signature, multiplier);
        }
        NodePtr replacement = nullptr;

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
            auto result = std::make_shared<v0::Result>(tensors_multiplication(nullptr, NodeVector{parameters.begin() + 1, parameters.end()}, target, false, 1));
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

        for (auto consumer : consumers) {
            consumer.replace_source_output(replacement->output(0));
        }
        return true;
    }

    ~LoRAFuseTransform () {
        for(auto signature: compiled_weight_models) {
            DEBUG_PRINT(signature.first);
        }
    }
};


class LoRASeparateTransform : public LoRATransformBase {
public:

    OPENVINO_RTTI("LoRASeparateTransform");

    LoRASeparateTransform(const LoRAWeightByNodeGetter& lora_getter) : LoRATransformBase(lora_getter) {}

    bool apply (NodePtr node, const LoRANode& lora_weight) override {
        auto activations = node->input_value(0);    // FIXME: consider MatMul.transpose_a
        auto weights_input = node->input_value(1);
        auto weights_input_type = weights_input.get_element_type();
        //DEBUG_PRINT("WEIGHTS SHAPE: " << weights_input.get_partial_shape());
        NodePtr add_term = nullptr;
        NodePtr replacement = nullptr;

        auto target = node->output(0);

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

        NodeVector lora_variables{lora_weight.A, lora_weight.alpha, lora_weight.B};
        replacement = tensors_multiplication(activations.get_node_shared_ptr(), lora_variables, target, true, 1);

        for (auto consumer : consumers) {
            consumer.replace_source_output(replacement->output(0));
        }

        return true;
    }
};


LoRATensors group_lora_tensors(const ConstantMap& tensors, const LoRAPartsParser& parts_parser) {
    LoRATensors result;
    for(const auto& named_tensor: tensors) {
        //DEBUG_PRINT(named_tensor.first);
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

    // Check that A and B exist for each LoRA entry
    for(const auto& lora_tensor: result) {
        // DEBUG_PRINT(lora_tensor.first);
        // if(lora_tensor.second.A) {
        //     DEBUG_PRINT(lora_tensor.second.A);
        // } else {
        //     DEBUG_PRINT("nullptr");
        // }
        // if(lora_tensor.second.B) {
        //     DEBUG_PRINT(lora_tensor.second.B);
        // } else {
        //     DEBUG_PRINT("nullptr");
        // }
        // if(lora_tensor.second.alpha) {
        //     DEBUG_PRINT(lora_tensor.second.alpha);
        // } else {
        //     DEBUG_PRINT("nullptr");
        // }
        OPENVINO_ASSERT(lora_tensor.second.A && lora_tensor.second.B, "Either A, B or both matrices are missing in LoRA tensors for layer: ", lora_tensor.first);
    }
    return result;
}


std::shared_ptr<v0::Constant> alpha_as_constant(float alpha) {
    return v0::Constant::create(ov::element::f32, ov::Shape{1}, {alpha});
}


} // namespace


namespace ov {
namespace genai {

class Adapter::Impl {
public:
    Impl(const std::string& path, std::optional<float> default_alpha = std::nullopt) :  // FIXME: Pass lora patterns
        tensors(group_lora_tensors(read_safetensors(path), default_lora_patterns())),  // FIXME: Accept directory with the config as well
        default_alpha(default_alpha) {
    }

    std::optional<float> get_default_alpha () const {
        return default_alpha;
    }

    LoRATensors tensors;
    std::optional<float> default_alpha;
};

Adapter::Adapter(const std::string& path, float default_alpha) :
    m_pimpl(std::make_shared<Adapter::Impl>(path, default_alpha)) {
}

Adapter::Adapter(const std::string& path) :
    m_pimpl(std::make_shared<Adapter::Impl>(path)) {
}

std::optional<float> Adapter::get_default_alpha() const {
    return m_pimpl->get_default_alpha();
}


struct AdapterControllerImpl {
    virtual void apply (ov::InferRequest& infer_request, const AdapterConfig& config) = 0;
    virtual void apply (ov::InferRequest& infer_request) = 0;
    static std::shared_ptr<Adapter::Impl> get_adapter_impl(const Adapter& adapter) {
        return adapter.m_pimpl;
    }

    virtual void force_full_apply(bool full_apply) = 0;
};

bool operator== (const Adapter& a, const Adapter& b) {
    return a.m_pimpl == b.m_pimpl;
}

bool operator< (const Adapter& a, const Adapter& b) {
    return a.m_pimpl < b.m_pimpl;
}

struct AdapterControllerImplSeparateState : public AdapterControllerImpl {
    std::vector<LoRAVarIDs> variable_ids;
    const std::string prefix;
    AdapterConfig current_config;
    bool need_full_apply = true;
    InferRequestSignatureCache lora_state_evaluators;

    AdapterControllerImplSeparateState(std::shared_ptr<ov::Model> model, const AdapterConfig& config, const std::string& prefix) :
        prefix(prefix),
        current_config(config),  // FIXME: Compare current and passed configs and change incrementally
        lora_state_evaluators("CPU")    // FIXME: Try to run on the same device that is used for model inference
    {
        OPENVINO_ASSERT(!config.is_dynamic || !config.fuse, "Both is_dynamic and fuse modes cannot be set simultaniously in adapter config");
        LoRAParametersByWeightGetter params_getter;
        params_getter.type = ov::element::dynamic;

        for(auto const& adapter : current_config.adapters) {
            auto adapter_impl = get_adapter_impl(adapter);
            params_getter.weight_getter.push_back(LoRAWeightGetterDefault(&adapter_impl->tensors, prefix));
            // TODO: Instead of aggregating types over all tensors in each adapter, make decision per node in LoRAWeightStateGetter
            /*if(params_getter.type != ov::element::f32)*/ {  // FIXME: Implement element_type tolerant code when state is set and uncomment this condition
                for(auto const& tensor : adapter_impl->tensors) {
                    auto lora_tensor_type = tensor.second.A->get_output_element_type(0);
                    OPENVINO_ASSERT(lora_tensor_type == tensor.second.B->get_output_element_type(0));
                    if(params_getter.type == ov::element::dynamic) {
                        params_getter.type = lora_tensor_type;
                    } else if(params_getter.type != lora_tensor_type) {
                        // if types are not match among multiple LoRA tensos then fall back to f32
                        params_getter.type = ov::element::f32;
                        break;
                    }
                }
            }
        }

        auto weight_as_constant = [&, this](NodePtr node) -> std::optional<LoRANode> {
            // FIXME: lora_placeholder is for passing element type only
            LoRAParts<ov::Tensor> lora_placeholder{ov::Tensor(ov::element::f32, Shape{0}),  ov::Tensor(params_getter.type, ov::Shape{0}),  ov::Tensor(params_getter.type, ov::Shape{0})};                
            auto name = node->get_friendly_name();
            auto lora_weight = prepare_lora_tensors(name, params_getter.weight_getter, lora_placeholder, false);
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

        ov::pass::Manager pm;
        if(current_config.is_dynamic) {
            // State mode
            pm.register_pass<LoRASeparateTransform>(LoRAWeightStateGetter(params_getter, model, variable_ids));
            //ov::serialize(model, "after_lora.xml");
            // auto variables = model->get_variables();
            // for(size_t i = 0; i < variables.size(); ++i) {
            //     DEBUG_PRINT("Variable: " << variables[i]->get_info().variable_id);
            // }(
        } else if(!current_config.fuse) {
            // Separate constant mode
            pm.register_pass<LoRASeparateTransform>(weight_as_constant);
        } else {
            // Fuse mode
            pm.register_pass<LoRAFuseTransform>(weight_as_constant);
        }

        pm.run_passes(model);
        model->validate_nodes_and_infer_types();    // FIXME: For debugging purposes only
    }

    struct ConfigChanged {
        bool is_dynamic = false;
        bool fuse = false;
        bool alpha = false;
        bool adapter = false;

        operator bool() const {
            return is_dynamic || fuse || alpha || adapter;
        }
    };

    ConfigChanged compare_configs(const AdapterConfig& config1, const AdapterConfig& config2) {
        ConfigChanged diff;
        diff.is_dynamic = config1.is_dynamic != config2.is_dynamic;
        diff.fuse = config1.fuse != config2.fuse;
        // TODO: Use `set` from this commented block when the config change tracking is implemented at adapter granularity and will track order of adapters correctly
        // std::set<Adapter>
        //     adapters1(config1.adapters.begin(), config1.adapters.end()),
        //     adapters2(config2.adapters.begin(), config2.adapters.end());
        const auto& adapters1 = config1.adapters, adapters2 = config2.adapters;

        DEBUG_PRINT(config1.adapters.size());
        DEBUG_PRINT(config2.adapters.size());
        DEBUG_PRINT("adapters1 != adapters2: " << (adapters1 != adapters2));
        
        if(adapters1 != adapters2) {
            diff.adapter = true;
            diff.alpha = true;
        } else {
            OPENVINO_ASSERT(config1.adapters.size() == config2.adapters.size());
            OPENVINO_ASSERT(config1.alphas.size() == config2.alphas.size());
            OPENVINO_ASSERT(config1.adapters.size() == config1.alphas.size());
            for(size_t i = 0; i < config1.adapters.size() && !diff.alpha; ++i) {
                const auto& adapter = config1.adapters[i];
                diff.alpha = config1.get_alpha(adapter) != config2.get_alpha(adapter);
            }
        }
        return diff;
    }

    void apply (ov::InferRequest& infer_request, const AdapterConfig& config) override {
        // FIXME: If a part of LoRA state tensors are not set here, then need to carefully reset state in LLMPipeline where global reset is called after the generation
        OPENVINO_ASSERT(!config.is_dynamic || !config.fuse, "Both is_dynamic and fuse modes cannot be set simultaniously in adapter config");

        const auto diff = compare_configs(current_config, config);
        DEBUG_PRINT("config.is_dynamic: " << config.is_dynamic);
        DEBUG_PRINT("config.fuse: " << config.fuse);
        DEBUG_PRINT("diff.adapter: " << diff.adapter);
        OPENVINO_ASSERT(!diff.is_dynamic, "AdapterConfig::is_dynamic cannot be changed and should be configured once for a model at the initialization");
        OPENVINO_ASSERT(!diff.fuse, "AdapterConfig::fuse cannot be changed and should be configured once for a model at the initialization");
        OPENVINO_ASSERT(config.is_dynamic || (!diff.alpha && !diff.adapter), "Cannot change adapters and/or the alphas when is_dynamic = false.");
        if(need_full_apply) {
            need_full_apply = false;
            DEBUG_PRINT("force apply");
            set_new_adapter_tensors(infer_request, config);
        } else if(diff) {
            if(diff.adapter) {
                set_new_adapter_tensors(infer_request, config);
            } else {
                OPENVINO_ASSERT(diff.alpha);
                set_new_adapter_alphas(infer_request, config);
            }
        }
    }

    void force_full_apply(bool full_apply) override {
        need_full_apply = full_apply;
    }

    void set_new_adapter_alphas (ov::InferRequest& infer_request, const AdapterConfig& config) {
        // FIXME: Provide more economical way to update only alphas
        set_new_adapter_tensors(infer_request, config);
    }

    void set_new_adapter_tensors (ov::InferRequest& infer_request, const AdapterConfig& config) {
        current_config = config;       // FIXME: Keep the old config to map to cached LoRA state tensors instead of the current approach where we start from scratch each time
        if(!current_config.is_dynamic) {
            return;
        }

        std::vector<LoRAWeightGetter> weight_getters;
        weight_getters.reserve(current_config.adapters.size());
        for(const auto& adapter: current_config.adapters) {
            weight_getters.emplace_back(LoRAWeightGetterDefault(&get_adapter_impl(adapter)->tensors, prefix));
        }

        auto state = infer_request.query_state();
        // for(size_t i = 0; i < state.size(); ++i) {
        //     DEBUG_PRINT("State [" << i << "].name: " << state[i].get_name());
        // }
        //std::map<std::string, size_t> var_id_to_index;

        // FIXME: Forced to use variable_id instead of index to address the state tensors, require the same order for state as for variables from plugins

        // Convert LoRAVarIDs to LoRAIndices to speedup search for state with a given name
        // FIXME: If state order is stable, then this should be done this mapping once for a given infer request, TODO: cache it based on the infer request
        std::map<std::string, size_t> state_name_to_index;
        for(size_t i = 0; i < state.size(); ++i) {
            auto name = state[i].get_name();
            state_name_to_index[name] = i;
        }

        for(const auto& lora_var_ids : variable_ids) {
            // FIXME: Remove this mapping when the order of state will be the same as the order of variables
            LoRAIndices lora_indices;
            lora_indices.alpha = state_name_to_index.at(lora_var_ids.alpha);
            lora_indices.A = state_name_to_index.at(lora_var_ids.A);
            lora_indices.B = state_name_to_index.at(lora_var_ids.B);
            lora_indices.name = lora_var_ids.name;

            set_lora_tensors(state, lora_indices, weight_getters);
        }
    }

     std::vector<LoRAWeight> collect_applicable_tensors (const std::string& lora_name, const std::vector<LoRAWeightGetter>& weight_getters) {
        OPENVINO_ASSERT(weight_getters.size() == current_config.adapters.size());
        std::vector<LoRAWeight> result;
        result.reserve(weight_getters.size());
        for(size_t i = 0; i < current_config.adapters.size(); ++i) {
            if(auto lora_tensors = weight_getters[i](lora_name)) {
                // FIXME: Introduce more flexible logic of setting alpha based on alpha set in the adapter file itself, now it is ignored and only alpha from config is used
                OPENVINO_ASSERT(lora_tensors->A);
                OPENVINO_ASSERT(lora_tensors->B);
                lora_tensors->alpha = current_config.alpha_constants[i];
                result.push_back(LoRAWeight(
                    std::dynamic_pointer_cast<v0::Constant>(lora_tensors->alpha),
                    std::dynamic_pointer_cast<v0::Constant>(lora_tensors->A),
                    std::dynamic_pointer_cast<v0::Constant>(lora_tensors->B)
                ));
                
                // TODO: Use the following block of code to adjust alpha in a more flexible way
                // float alpha = 1;
                // if(config.alphas.size()) {
                //     // Override alpha
                //     alpha = config.alphas.front();   // FIXME: Only first alpha is used
                // } else if (adapter->default_alpha) {
                //     alpha = *adapter->default_alpha;
                // } else {
                //     if(lora_tensors->alpha) {
                //         OPENVINO_ASSERT(lora_tensors->alpha->get_output_element_type(0) == alpha_state.get_element_type());
                //         auto const_alpha = std::dynamic_pointer_cast<v0::Constant>(lora_tensors->alpha);
                //         OPENVINO_ASSERT(ov::shape_size(const_alpha->get_shape()) == 1);
                //         alpha = const_alpha->cast_vector<float>()[0];
                //     }
                // }
            }
        }
        return result;
    }

    InferRequestSignatureCache::Signature get_tensor_signature(const ov::Output<ov::Node>& output) {
        return get_tensor_signature(output.get_element_type(), output.get_partial_shape());
    }

    InferRequestSignatureCache::Signature get_tensor_signature(const ov::element::Type& type, const ov::PartialShape& shape) {
        return type.get_type_name() + shape.to_string();
    }

    InferRequestSignatureCache::Signature get_lora_signature(const std::vector<LoRAWeight>& inputs, const LoRAParts<ov::Tensor>& outputs) {
        InferRequestSignatureCache::Signature signature;
        for(const auto& input: inputs) {
            signature +=
                std::string("(") +
                    "(" + get_tensor_signature(input.alpha) + ")" +
                    "(" + get_tensor_signature(input.A) + ")" +  // TODO: Adjust shape to have a dynamic low-rank LoRA dimension in case of fully static shape doesn't have significant speedup
                    "(" + get_tensor_signature(input.B) + ")" +  // TODO: Adjust shape to have a dynamic low-rank LoRA dimension in case of fully static shape doesn't have significant speedup
                ")";
        }
        for(const auto& input: inputs) {
            signature +=
                std::string("(") +
                    // Shape is set to be dynamic because it doesn't mater for signature as it is completelly determined by the corresponding model
                    "(" + get_tensor_signature(outputs.alpha.get_element_type(), ov::PartialShape::dynamic(1)) + ")" +
                    "(" + get_tensor_signature(outputs.A.get_element_type(), ov::PartialShape::dynamic(2)) + ")" +
                    "(" + get_tensor_signature(outputs.B.get_element_type(), ov::PartialShape::dynamic(2)) + ")" +
                ")";
        }
        return signature;
    }

    std::vector<ov::Tensor> to_tensor_vector(const std::vector<LoRAWeight>& v) {
        std::vector<ov::Tensor> result;
        result.reserve(v.size()*3);
        for(auto const& lora_weights: v) {
            result.push_back(lora_weights.alpha->get_tensor_view());
            result.push_back(lora_weights.A->get_tensor_view());
            result.push_back(lora_weights.B->get_tensor_view());
        }
        return result;
    }

    std::vector<ov::Tensor> to_tensor_vector(const LoRAParts<ov::Tensor>& lora_tensors) {
        std::vector<ov::Tensor> result;
        result.reserve(3);
        result.push_back(lora_tensors.alpha);
        result.push_back(lora_tensors.A);
        result.push_back(lora_tensors.B);
        return result;
    }

    // ov::ParameterVector to_parameter_vector(const std::vector<LoRAWeight>& v) {
    //     ov::ParameterVector{
    //         std::make_shared<v0::Parameter>(v);
    //     };
    // }

    void build_concat_model(
        ov::ParameterVector& parameters,
        ov::ResultVector& results,
        const std::vector<LoRAWeight>& inputs,
        ov::Tensor output,
        size_t offset,
        size_t concat_axis,
        std::function<std::shared_ptr<v0::Parameter>(const LoRAWeight&)> input_accessor,
        std::function<NodePtr(const LoRAWeight&, NodePtr)> parameter_postprocessing = [](const LoRAWeight&, NodePtr node) { return node; }
    ) {
        ov::OutputVector concat_inputs;
        concat_inputs.reserve(inputs.size());
        for(size_t i = 0; i < inputs.size(); ++i) {
            NodePtr input = parameters[3*i + offset] = input_accessor(inputs[i]);
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
            // FIXME: Workaround CPU plugin bug with Parameter -> Result models: add a small constant to force copying input to output
            // FIXME: Do it differently: not use model-based evaluation in this case but just pass lora tensor directly as a new state value
            if(result == parameters[offset]) {
                result = std::make_shared<v1::Add>(result, v0::Constant::create(result->get_output_element_type(0), Shape{}, {1e-37f}));
            }
        }

        results[offset] = std::make_shared<v0::Result>(result);

        // TODO: Optimize trivial Parameter->Result cases
    }

    LoRAParts<ov::Tensor> empty_adapters(const std::vector<LoRAWeight>& inputs, LoRAParts<ov::Tensor>& outputs) {
        //DEBUG_PRINT("empty_adapter");
        #define EMPTY_TENSORS_SUPPORTED_IN_MATMUL 0
        #if EMPTY_TENSORS_SUPPORTED_IN_MATMUL

        outputs.alpha.set_shape({1, 0});
        outputs.A.set_shape({0, outputs.A.get_shape()[1]});
        outputs.B.set_shape({outputs.B.get_shape()[0], 0});

        #else

        // TODO: As ov::Tensor lacks a convenient constructor to fill all elements with the same scalar value, do it via Constant that has such constructor
        // FIXME: It's a huge overhead for setting just a scalar 0
        
        std::make_shared<v0::Constant>(outputs.alpha.get_element_type(), ov::Shape{1, 1}, 0)->get_tensor_view().copy_to(outputs.alpha);
        // Element values for A and B don't matter as we are multiplying by 0 in alpha anyway
        std::make_shared<v0::Constant>(outputs.A.get_element_type(), ov::Shape{1, outputs.A.get_shape()[1]}, 0)->get_tensor_view().copy_to(outputs.A);
        std::make_shared<v0::Constant>(outputs.B.get_element_type(), ov::Shape{outputs.B.get_shape()[0], 1}, 0)->get_tensor_view().copy_to(outputs.B);
        //outputs.A.set_shape({0, outputs.A.get_shape()[1]});
        //outputs.B.set_shape({outputs.B.get_shape()[0], 0});

        #endif

        return outputs;
    }

    LoRAParts<ov::Tensor> concat_adapters(const std::vector<LoRAWeight>& inputs, LoRAParts<ov::Tensor>& outputs) {
        auto signature = get_lora_signature(inputs, outputs);
        if(!lora_state_evaluators.exist(signature)) {
            // Prepare LoRA state evaluate model
            ov::ParameterVector parameters(3*inputs.size());
            ov::ResultVector results(3);

            build_concat_model(parameters, results, inputs, outputs.alpha, 0, 1,
                [](const LoRAWeight& lora_weight) {
                    return std::make_shared<v0::Parameter>(
                        lora_weight.alpha->get_output_element_type(0),
                        lora_weight.alpha->get_output_partial_shape(0));    // TODO: Consider using dynamic LoRA rank dimension instead of static dimension
                },
                [](const LoRAWeight& lora_weight, NodePtr parameter) {
                    // TODO: This code should be modified if dynamic LoRA rank is used in the evaluator
                    auto lora_rank = lora_weight.A->get_output_partial_shape(0)[0].get_length();
                    // Broadcast a single alpha element to shape [lora_rank]
                    auto lora_rank_constant = v0::Constant::create(ov::element::u32, Shape{2}, std::vector<unsigned int>{1, lora_rank});
                    return std::make_shared<v3::Broadcast>(parameter, lora_rank_constant);
                });

            build_concat_model(parameters, results, inputs, outputs.A, 1, 0,
                [](const LoRAWeight& lora_weight) {
                    return std::make_shared<v0::Parameter>(
                        lora_weight.A->get_output_element_type(0),
                        lora_weight.A->get_output_partial_shape(0));    // TODO: Consider using dynamic LoRA rank dimension instead of static dimension
                }
            );

            build_concat_model(parameters, results, inputs, outputs.B, 2, 1,
                [](const LoRAWeight& lora_weight) {
                    return std::make_shared<v0::Parameter>(
                        lora_weight.B->get_output_element_type(0),
                        lora_weight.B->get_output_partial_shape(0));    // TODO: Consider using dynamic LoRA rank dimension instead of static dimension
                }
            );

            lora_state_evaluators.insert(signature, std::make_shared<ov::Model>(results, parameters));
        }
        auto output_tensors = to_tensor_vector(outputs);
        //std::vector<ov::Tensor> output_tensors{ov::Tensor(outputs.alpha.get_element_type(), {}), ov::Tensor(outputs.A.get_element_type(), {}), ov::Tensor(outputs.B.get_element_type(), {})};
        lora_state_evaluators.evaluate(signature, to_tensor_vector(inputs), output_tensors);
        //return LoRAParts<ov::Tensor>(output_tensors[0], output_tensors[1], output_tensors[2]);
        return outputs;
    }

    void set_lora_tensors(std::vector<VariableState>& state, const LoRAIndices& lora_indices, const std::vector<LoRAWeightGetter>& weight_getters) {
        LoRAParts<ov::Tensor> lora_state_tensors(
            state[lora_indices.alpha].get_state(),  
            state[lora_indices.A].get_state(),
            state[lora_indices.B].get_state()
        );
        auto new_tensors = prepare_lora_tensors(lora_indices.name, weight_getters, lora_state_tensors);
        state[lora_indices.alpha].set_state(new_tensors.alpha);
        state[lora_indices.A].set_state(new_tensors.A);
        state[lora_indices.B].set_state(new_tensors.B);
        //print_tensor(std::cout, new_tensors.alpha);
    }

    LoRAParts<ov::Tensor> prepare_lora_tensors (
        const std::string& name,
        const std::vector<LoRAWeightGetter>& weight_getters,
        LoRAParts<ov::Tensor>& output,
        bool set_empty_adapters = true
    ) {
        auto lora_tensors = collect_applicable_tensors(name, weight_getters);
        LoRAParts<ov::Tensor> new_tensors;
        if(!lora_tensors.empty()) {
            new_tensors = concat_adapters(lora_tensors, output);
        } else if(set_empty_adapters) {  // FIXME: Make it as a separate step outside of this function
            new_tensors = empty_adapters(lora_tensors, output);
        }
        return new_tensors;
    }

    void apply (ov::InferRequest& infer_request) override {
        return apply(infer_request, current_config);
    }
};

AdapterController::AdapterController(std::shared_ptr<ov::Model> model, const AdapterConfig& config, const std::string& prefix) :
    m_pimpl(std::make_shared<AdapterControllerImplSeparateState>(model, config, prefix)) {
    // FIXME: Create Static lora if user requested, now always dynamic LoRA is created
}

// Call it every time when adapter config is changed; if adapter was configured as a static one, this call is not required
void AdapterController::apply(ov::InferRequest& request, const AdapterConfig& config) {
    return m_pimpl->apply(request, config);
}

void AdapterController::apply(ov::InferRequest& request){
    return m_pimpl->apply(request);
}

void AdapterController::force_full_apply(bool full_apply) {
    return m_pimpl->force_full_apply(full_apply);
}

AdapterConfig::AdapterConfig (const std::vector<Adapter>& adapters, bool is_dynamic) : is_dynamic(is_dynamic), adapters(adapters) {
    alphas.reserve(adapters.size());
    for(const auto& adapter: adapters) {
        auto const alpha = adapter.get_default_alpha().value_or(1);
        alphas.push_back(alpha);
        alpha_constants.push_back(alpha_as_constant(alpha));
    }
}

AdapterConfig::AdapterConfig (const std::vector<std::pair<Adapter, float>>& _adapters, bool is_dynamic) : is_dynamic(is_dynamic) {
    adapters.reserve(_adapters.size());
    alphas.reserve(_adapters.size());
    for(auto const& adapter_and_alpha: _adapters) {
        adapters.push_back(adapter_and_alpha.first);
        alphas.push_back(adapter_and_alpha.second);
        alpha_constants.push_back(alpha_as_constant(adapter_and_alpha.second));
    }
}

//AdapterConfig::AdapterConfig (const Adapter& adapter, float alpha, bool is_dynamic) : AdapterConfig(std::vector<std::pair<Adapter, float>>{{adapter, alpha}}, is_dynamic) {}

AdapterConfig& AdapterConfig::add(const Adapter& adapter, float alpha) {
    OPENVINO_ASSERT(adapters.size() == alphas.size());
    OPENVINO_ASSERT(adapters.end() == std::find(adapters.begin(), adapters.end(), adapter), "Adapter object passed to AdapterConfig::add was already registered");
    adapters.push_back(adapter);
    alphas.push_back(alpha);
    alpha_constants.push_back(alpha_as_constant(alpha));
    return *this;
}

AdapterConfig& AdapterConfig::add(const Adapter& adapter) {
    return add(adapter, adapter.get_default_alpha().value_or(1));
}

AdapterConfig& AdapterConfig::set_alpha(const Adapter& adapter, float alpha) {
    OPENVINO_ASSERT(adapters.size() == alphas.size());
    auto it = std::find(adapters.begin(), adapters.end(), adapter);
    OPENVINO_ASSERT(adapters.end() != it, "Unknown adapter object passed to AdapterConfig::set_alpha, register adapter object first with AdapterConfig::add");
    auto index = it - adapters.begin();
    alphas[index] = alpha;
    alpha_constants[index] = alpha_as_constant(alpha);
    return *this;
}

float AdapterConfig::get_alpha(const Adapter& adapter) const {
    OPENVINO_ASSERT(adapters.size() == alphas.size());
    auto it = std::find(adapters.begin(), adapters.end(), adapter);
    OPENVINO_ASSERT(adapters.end() != it, "Unknown adapter object passed to AdapterConfig::get_alpha, alpha can be retrieved for previously registered adatpers only");  
    return alphas[it - adapters.begin()];
}

AdapterConfig& AdapterConfig::remove(const Adapter& adapter) {
    OPENVINO_ASSERT(adapters.size() == alphas.size());
    auto it = std::find(adapters.begin(), adapters.end(), adapter);
    OPENVINO_ASSERT(adapters.end() != it, "Unknown adapter object passed to AdapterConfig::remove, you can remove previously registered adapters only");
    auto index = it - adapters.begin();
    alphas.erase(alphas.begin() + index);
    alpha_constants.erase(alpha_constants.begin() + index);
    adapters.erase(it);
    return *this;
}


}  // namespace genai
}  // namespace ov
