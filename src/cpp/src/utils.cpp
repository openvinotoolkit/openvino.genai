// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "utils.hpp"

#include <variant>
#include <fstream>
#include <memory>

#include "openvino/op/add.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/tanh.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/genai/text_streamer.hpp"
#include "gguf_utils/gguf_modeling.hpp"


#include "sampling/sampler.hpp"

namespace ov {

namespace genai {
const std::string PA_BACKEND = "PA";
const std::string SDPA_BACKEND = "SDPA";
}
}

namespace {

void update_config(ov::AnyMap& config, const std::pair<std::string, ov::Any>& pair) {
    if (config.count(pair.first) == 0) {
        config.insert(pair);
    }
}

void rename_key(ov::AnyMap& config, const std::string& old_key, const std::string& new_key) {
    if (config.count(old_key) != 0) {
        auto opt_value = ov::genai::utils::pop_option(config, old_key);
        config[new_key] = opt_value.value();
    }
}

template <typename T>
std::optional<T> get_option(const ov::AnyMap& config, const std::string& option_name) {
    if (auto it = config.find(option_name); it != config.end()) {
        return std::make_optional(it->second.as<T>());
    }
    return std::nullopt;
}

std::optional<uint32_t> pop_int_and_cast(ov::AnyMap& config, const std::string& key) {
    auto anyopt = ov::genai::utils::pop_option(config, key);
    if (anyopt.has_value()) {
        const auto any = anyopt.value();
        int64_t value;
        // NB: Integer value coming from python has int64_t datatype
        if (any.is<int64_t>()) {
            value = any.as<int64_t>();
        } else if (any.is<int>()) {
            value = any.as<int>();
        } else {
            OPENVINO_THROW("Failed to extract " + key + ". Type mismatch: expected types: int or int64_t");
        }
        if (value < 0) {
            OPENVINO_THROW(key + " cannot be negative!");
        }
        return std::make_optional(static_cast<uint32_t>(value));
    }
    return std::nullopt;
}

void update_npu_config(ov::AnyMap& config,
                       const std::shared_ptr<ov::Model>& model,
                       const ov::genai::utils::KVAxesPosition& kv_pos,
                       const ov::genai::utils::KVDesc& kv_desc) {
    update_config(config, {"NPU_USE_NPUW", "YES"});
    update_config(config, {"NPUW_LLM", "YES"});

    update_config(config, {"NPUW_LLM_BATCH_DIM", kv_pos.batch});
    update_config(config, {"NPUW_LLM_SEQ_LEN_DIM", kv_pos.seq_len});

    update_config(config, {"NPUW_LLM_MAX_PROMPT_LEN", kv_desc.max_prompt_len});
    update_config(config, {"NPUW_LLM_MIN_RESPONSE_LEN", kv_desc.min_response_len});

    rename_key(config, "PREFILL_CONFIG", "NPUW_LLM_PREFILL_CONFIG");
    rename_key(config, "PREFILL_HINT", "NPUW_LLM_PREFILL_HINT");
    rename_key(config, "GENERATE_CONFIG", "NPUW_LLM_GENERATE_CONFIG");
    rename_key(config, "GENERATE_HINT", "NPUW_LLM_GENERATE_HINT");
    rename_key(config, "SHARED_HEAD_CONFIG", "NPUW_LLM_SHARED_HEAD_CONFIG"); 
    
    rename_key(config, "++PREFILL_CONFIG", "++NPUW_LLM_PREFILL_CONFIG");
    rename_key(config, "++GENERATE_CONFIG", "++NPUW_LLM_GENERATE_CONFIG");
    rename_key(config, "++SHARED_HEAD_CONFIG", "++NPUW_LLM_SHARED_HEAD_CONFIG");
}

inline bool is_paged_attention_available() {
#if defined(OPENVINO_ARCH_X86_64) || defined(OPENVINO_ARCH_ARM64)
    return true;
#else
    return false;
#endif
}

} // anonymous

namespace ov {
namespace genai {
namespace utils {

Tensor init_attention_mask(const Tensor& input_ids) {
    auto shape = input_ids.get_shape();
    auto attention_mask = ov::Tensor{input_ids.get_element_type(), shape};
    std::fill_n(attention_mask.data<int64_t>(), shape[0] * shape[1], 1);
    return attention_mask;
}

/**
 * Initializes position ids based on attention mask and starting position
 */
void initialize_position_ids(ov::Tensor& position_ids, const ov::Tensor& attention_mask, int64_t start_pos) {
    OPENVINO_ASSERT(position_ids.get_element_type() == ov::element::i64,
                    "position_ids tensor element type should be an i64");
    OPENVINO_ASSERT(position_ids.get_shape().size() == 2,
                    "position_ids tensor should of rank 2 with shape [batch_size, seq_len]");
    OPENVINO_ASSERT(attention_mask.get_element_type() == ov::element::i64,
                    "attention_mask tensor element type should be an i64");
    OPENVINO_ASSERT(attention_mask.get_shape().size() == 2,
                    "attention_mask tensor should of rank 2 with shape [batch_size, seq_len]");

    const size_t batch_size = attention_mask.get_shape()[0];
    const size_t seq_length = attention_mask.get_shape()[1];

    const int64_t* attention_mask_data = attention_mask.data<int64_t>();
    int64_t* position_ids_data = position_ids.data<int64_t>();

    for (size_t batch = 0; batch < batch_size; batch++) {
        size_t sum = start_pos;
        for (size_t i = 0; i < seq_length; i++) {
            const size_t element_offset = batch * seq_length + i;
            position_ids_data[element_offset] = sum;
            if (attention_mask_data[element_offset] == 1) {
                sum += 1;
            }
        }
    }
}

ov::genai::StreamerVariant get_streamer_from_map(const ov::AnyMap& config_map) {
    ov::genai::StreamerVariant streamer = std::monostate();

    if (config_map.count(STREAMER_ARG_NAME)) {
        auto any_val = config_map.at(STREAMER_ARG_NAME);
        if (any_val.is<std::shared_ptr<ov::genai::StreamerBase>>()) {
            streamer = any_val.as<std::shared_ptr<ov::genai::StreamerBase>>();
        } else if (any_val.is<std::function<bool(std::string)>>()) {
            streamer = any_val.as<std::function<bool(std::string)>>();
        } else if (any_val.is<std::function<StreamingStatus(std::string)>>()) {
            streamer = any_val.as<std::function<StreamingStatus(std::string)>>();
        }
    }
    return streamer;
}

std::shared_ptr<StreamerBase> create_streamer(StreamerVariant streamer, Tokenizer tokenizer) {
    return std::visit(overloaded{
        [](std::monostate) -> std::shared_ptr<StreamerBase> {
            return nullptr;
        },
        [](const std::shared_ptr<StreamerBase>& streamer) {
            return streamer;
        },
        [&tokenizer = tokenizer](const std::function<bool(std::string)>& streamer) -> std::shared_ptr<StreamerBase> {
            return std::make_unique<TextStreamer>(tokenizer, streamer);
        },
        [&tokenizer = tokenizer](const std::function<ov::genai::StreamingStatus(std::string)>& streamer) -> std::shared_ptr<StreamerBase> {
            return std::make_unique<TextStreamer>(tokenizer, streamer);
        }
    }, streamer);
}

ov::genai::OptionalGenerationConfig get_config_from_map(const ov::AnyMap& config_map) {
    if (config_map.count(CONFIG_ARG_NAME))
        return config_map.at(CONFIG_ARG_NAME).as<ov::genai::GenerationConfig>();
    else
        return std::nullopt;
}

ProcessorConfig from_any_map(
    const ov::AnyMap& config_map,
    const ProcessorConfig& initial
) {
    auto iter = config_map.find("processor_config");
    ProcessorConfig extracted_config = config_map.end() != iter ?
        iter->second.as<ProcessorConfig>() : initial;
    using utils::read_anymap_param;
    read_anymap_param(config_map, "patch_size", extracted_config.patch_size);
    read_anymap_param(config_map, "scale_resolution", extracted_config.scale_resolution);
    read_anymap_param(config_map, "max_slice_nums", extracted_config.max_slice_nums);
    read_anymap_param(config_map, "norm_mean", extracted_config.norm_mean);
    read_anymap_param(config_map, "norm_std", extracted_config.norm_std);
    return extracted_config;
}

ov::genai::ModelDesc get_draft_model_from_config(const ov::AnyMap& config) {
    ov::genai::ModelDesc draft_model;
    if (config.find(utils::DRAFT_MODEL_ARG_NAME) != config.end()) {
        draft_model = config.at(utils::DRAFT_MODEL_ARG_NAME).as<ov::genai::ModelDesc>();
    }
    return draft_model;
}

ov::genai::ModelDesc extract_draft_model_from_config(ov::AnyMap& config) {
    ov::genai::ModelDesc draft_model;
    if (config.find(ov::genai::utils::DRAFT_MODEL_ARG_NAME) != config.end()) {
        draft_model = config.at(ov::genai::utils::DRAFT_MODEL_ARG_NAME).as<ov::genai::ModelDesc>();
        config.erase(ov::genai::utils::DRAFT_MODEL_ARG_NAME);
    }
    return draft_model;
}

bool is_npu_requested(const std::string& device, const ov::AnyMap& properties) {
    if (device == "NPU") {
        return true;
    }

    auto draft_model_descr = get_draft_model_from_config(properties);
    if (draft_model_descr.model != nullptr) {
        return draft_model_descr.device == "NPU";
    }

    return false;
}

ov::genai::TokenizedInputs subtract_chat_tokenized_inputs(const ov::genai::TokenizedInputs& minuend, const ov::genai::TokenizedInputs& subtrahend) {
    auto minuend_size = minuend.input_ids.get_size();
    auto subtrahend_size = subtrahend.input_ids.get_size();
    ov::Shape new_shape{1, minuend_size - subtrahend_size};

    ov::Tensor new_input_ids(ov::element::i64, new_shape);
    auto data_ptr = minuend.input_ids.data<int64_t>();
    std::copy(data_ptr + subtrahend_size, data_ptr + minuend_size, new_input_ids.data<int64_t>());

    ov::Tensor new_attention_mask(ov::element::i64, new_shape);
    std::fill_n(new_attention_mask.data<int64_t>(), new_shape[1], 1);

    return {new_input_ids, new_attention_mask};
}

namespace {

bool has_op_with_type(const std::shared_ptr<const ov::Model>& function, const std::string& type_name) {
    for (const auto& op : function->get_ops()) {
        if (op->get_type_name() == type_name) {
            return true;
        }
    }
    return false;
}

std::tuple<std::shared_ptr<ov::Node>, int64_t> find_llm_matmul(const std::shared_ptr<ov::Model>& model) {
    auto last_node = model->output(0).get_node()->input_value(0).get_node_shared_ptr();
    std::shared_ptr<ov::Node> matmul = ov::as_type_ptr<ov::op::v0::MatMul>(last_node);

    // in case of PA all tokens are moved to batch dimension and we have to slice / gather accordingly
    const bool pa_based_model = has_op_with_type(model, "PagedAttentionExtension");
    int64_t slice_gather_dim = pa_based_model ? 0 : 1;

    // There are several patterns for matmul we are looking for:
    // Matmul -> Result
    // Matmul -> Add -> Result
    // Matmul -> Transpose -> Result
    // MatMul -> Divide -> Tanh -> Multiply -> Result
    if (!matmul) {
        if (auto add = ov::as_type_ptr<ov::op::v1::Add>(last_node)) {
            matmul = ov::as_type_ptr<ov::op::v0::MatMul>(add->input_value(0).get_node_shared_ptr());
        } else if (auto transpose = ov::as_type_ptr<ov::op::v1::Transpose>(last_node)) {
            matmul = ov::as_type_ptr<ov::op::v0::MatMul>(transpose->input_value(0).get_node_shared_ptr());
            auto order = ov::as_type_ptr<ov::op::v0::Constant>(transpose->input_value(1).get_node_shared_ptr())->get_axis_vector_val();
            slice_gather_dim = order[slice_gather_dim];
        } else if (auto multiply = ov::as_type_ptr<ov::op::v1::Multiply>(last_node)) {
            if (auto tanh = ov::as_type_ptr<ov::op::v0::Tanh>(multiply->input_value(0).get_node_shared_ptr())) {
                if (auto divide = ov::as_type_ptr<ov::op::v1::Divide>(tanh->input_value(0).get_node_shared_ptr())) {
                    matmul = as_type_ptr<ov::op::v0::MatMul>(divide->input_value(0).get_node_shared_ptr());
                }
            }
        }
    }
    return std::make_tuple(matmul, slice_gather_dim);
}

} // namespace

void apply_slice_before_matmul_transformation(std::shared_ptr<ov::Model> model) {
    std::shared_ptr<ov::Node> matmul = nullptr;
    int64_t slice_gather_dim = -1;
    std::tie(matmul, slice_gather_dim) = find_llm_matmul(model);

    if (matmul && matmul->input(0).get_partial_shape().rank().get_length() == 3) {
        auto start = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
        auto stop = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-2});
        auto step = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
        auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{slice_gather_dim});
        auto slice = std::make_shared<ov::op::v8::Slice>(matmul->input_value(0), start, stop, step, axis);
        matmul->input(0).replace_source_output(slice);
    }
}

void apply_gather_before_matmul_transformation(std::shared_ptr<ov::Model> model) {
    std::shared_ptr<ov::Node> matmul = nullptr;
    int64_t slice_gather_dim = -1;
    std::tie(matmul, slice_gather_dim) = find_llm_matmul(model);

    if (matmul && matmul->input(0).get_partial_shape().rank().get_length() == 3) {
        auto indices = std::make_shared<ov::op::v0::Parameter>(ov::element::i64, ov::PartialShape{-1});
        indices->set_friendly_name("sampled_tokens_indices");
        indices->output(0).get_tensor().set_names({"sampled_tokens_indices"});
        auto axis = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{slice_gather_dim});
        auto gather = std::make_shared<ov::op::v8::Gather>(matmul->input_value(0), indices, axis);
        matmul->input(0).replace_source_output(gather);
        model->add_parameters({indices});
    }
}

ov::Core& singleton_core() {
    static ov::Core core;
    return core;
}


namespace {
bool is_gguf_model(const std::filesystem::path& file_path) {
    return file_path.extension() == ".gguf";
}

} // namespace

std::pair<ov::AnyMap, bool> extract_gguf_properties(const ov::AnyMap& external_properties) {
    bool enable_save_ov_model = false;
    ov::AnyMap properties = external_properties;

    auto it = properties.find(ov::genai::enable_save_ov_model.name());
    if (it != properties.end()) {
        enable_save_ov_model = it->second.as<bool>();
        properties.erase(it);
    }

    return {properties, enable_save_ov_model};
}

void save_openvino_model(const std::shared_ptr<ov::Model>& model, const std::string& save_path, bool compress_to_fp16) {
    try {
        auto serialize_start_time = std::chrono::high_resolution_clock::now();
        ov::save_model(model, save_path, compress_to_fp16);
        auto serialize_finish_time = std::chrono::high_resolution_clock::now();
        auto serialize_duration = std::chrono::duration_cast<std::chrono::milliseconds>(serialize_finish_time - serialize_start_time).count();
        std::stringstream ss;
        ss << "Save generated OpenVINO model to: " << save_path << " done. Time: " << serialize_duration << " ms";
        ov::genai::utils::print_gguf_debug_info(ss.str());
    }
    catch (const ov::Exception& e) {
        OPENVINO_THROW("Exception during model serialization ", e.what(), ", user can disable it by setting 'ov::genai::enable_save_ov_model' property to false");
    }
}

std::shared_ptr<ov::Model> read_model(const std::filesystem::path& model_dir,  const ov::AnyMap& properties) {
    auto [filtered_properties, enable_save_ov_model] = extract_gguf_properties(properties);
    if (is_gguf_model(model_dir)) {
#ifdef ENABLE_GGUF
        return create_from_gguf(model_dir.string(), enable_save_ov_model);
#else
        OPENVINO_ASSERT("GGUF support is switched off. Please, recompile with 'cmake -DENABLE_GGUF=ON'");
#endif
    } else {
        std::filesystem::path model_path = model_dir;

        if (std::filesystem::exists(model_dir / "openvino_model.xml")) {
            model_path = model_dir / "openvino_model.xml";
        } else if (std::filesystem::exists(model_dir / "openvino_language_model.xml")) {
            model_path = model_path / "openvino_language_model.xml";
        } else {
            OPENVINO_THROW("Could not find a model in the directory '", model_dir, "'");
        }

        return singleton_core().read_model(model_path, {}, filtered_properties);
    }
}

size_t get_first_history_difference(const ov::Tensor& encoded_history, const std::vector<int64_t> tokenized_history) {
    size_t idx = 0;
    auto encoded_history_data = encoded_history.data<int64_t>();
    while(idx < encoded_history.get_size() && idx < tokenized_history.size()) {
        if (encoded_history_data[idx] != tokenized_history[idx])
            break;
        idx++;
    }

    return idx;
}

KVAxesPosition get_kv_axes_pos(std::shared_ptr<const ov::Model> model) {
    // sequence length axis in key/values tensors, for most cases [BATCH_SIZE, num_kv_heads, seq_len, head_size],
    // therefore usually seq_length_axis = 2 and batch = 0
    KVAxesPosition kv_pos { 0u, 2u };

    // "ReadValue" node is KV cache representation in stateful model
    std::string kv_node_type_name = std::string(ov::op::v6::ReadValue::get_type_info_static().name);

    for (const auto op : model->get_ops()) {
        // check input size, as in LoRA adapters case it could be 0
        if (op->get_type_name() != kv_node_type_name || op->get_input_size() < 1) {
            continue;
        }

        // Shape example: [-1,4,0,64]
        auto shape = op->get_input_partial_shape(0);

        for (size_t i = 0; i < shape.rank().get_length(); i++) {
            // Find axis = 0. This would be sequence length axis.
            if (shape[i] == 0) {
                kv_pos.seq_len = i;
            } else if (shape[i].is_dynamic()) {
                // Dynamic axis is a batch
                kv_pos.batch = i;
            }
        }
        break;
    }

    return kv_pos;
}

void trim_kv_cache(ov::InferRequest request, KVCacheState& kv_cache_state, std::optional<AdapterController> adapter_controller) {
    if (kv_cache_state.reset_mem_state) {
        if (adapter_controller) {
            for(auto& state: request.query_state()) {
                if(!adapter_controller->has_state_name(state.get_name())) {
                    state.reset();
                }
            }
        } else {
            request.reset_state();
        }

        return;
    }

    // nothing to trim in this case
    if (kv_cache_state.num_tokens_to_trim == 0)
        return;

    auto states = request.query_state();

    OPENVINO_ASSERT(states.size() > 0, "Request contains no states.");

    for (auto& state : states) {
        if(adapter_controller && adapter_controller->has_state_name(state.get_name()))
            continue;

        ov::Tensor old_tensor = state.get_state();
        // [BATCH_SIZE, num_kv_heads, seq_len, head_size]
        auto shape = old_tensor.get_shape();
        shape[kv_cache_state.seq_length_axis] -= kv_cache_state.num_tokens_to_trim;

        ov::Coordinate new_shape_begin{0, 0, 0, 0};
        ov::Coordinate new_shape_end{shape};

        auto trimmed_tensor = ov::Tensor(old_tensor, new_shape_begin, new_shape_end);

        ov::Tensor new_tensor(old_tensor.get_element_type(), shape);
        trimmed_tensor.copy_to(new_tensor);

        state.set_state(new_tensor);
    }
}

ov::Tensor push_front_inputs(const ov::Tensor& base_tensor, int64_t add_to_front) {
    ov::Tensor new_tensor = ov::Tensor{ov::element::i64, {base_tensor.get_shape().at(0), base_tensor.get_shape().at(1) + 1}};
    auto new_tensor_data = new_tensor.data<int64_t>();
    new_tensor_data[0] = add_to_front;
    std::copy_n(base_tensor.data<int64_t>(), base_tensor.get_size(), new_tensor_data + 1);
    return new_tensor;
}

bool env_setup_for_print_debug_info() {
    // Specify the name of the environment variable
    const char* env_var_name = "OPENVINO_LOG_LEVEL";
    const char* env_var_value = std::getenv(env_var_name);
    // Check if the environment variable was found
    return (env_var_value != nullptr && atoi(env_var_value) > static_cast<int>(ov::log::Level::WARNING));
}

void print_compiled_model_properties(ov::CompiledModel& compiled_Model, const char* model_title) {
    if (!env_setup_for_print_debug_info()) {
        return;
    }
    // output of the actual settings that the device selected
    auto supported_properties = compiled_Model.get_property(ov::supported_properties);
    std::cout << "Model: " << model_title << std::endl;
    for (const auto& cfg : supported_properties) {
        if (cfg == ov::supported_properties)
            continue;
        ov::Any prop;
        try {
            prop = compiled_Model.get_property(cfg);
        } catch (const ov::Exception&) {
            continue;  // NPU: Unsupported configuration key: EXECUTION_MODE_HINT. Ticket 172485
        }
        if (cfg == ov::device::properties) {
            auto devices_properties = prop.as<ov::AnyMap>();
            for (auto& item : devices_properties) {
                std::cout << "  " << item.first << ": " << std::endl;
                for (auto& item2 : item.second.as<ov::AnyMap>()) {
                    std::cout << "    " << item2.first << ": " << item2.second.as<std::string>() << std::endl;
                }
            }
        } else {
            std::cout << "  " << cfg << ": " << prop.as<std::string>() << std::endl;
        }
    }

    std::vector<std::string> exeTargets;
    exeTargets = compiled_Model.get_property(ov::execution_devices);
    std::cout << "EXECUTION_DEVICES:" << std::endl;
    for (const auto& device : exeTargets) {
        std::string full_name;
        try {
            full_name = singleton_core().get_property(device, ov::device::full_name);
        } catch (const ov::Exception&) {
            continue;  // NPU: No available devices. Ticket 172485
        }
        std::cout << " " << device << ": " << full_name << std::endl;
    }
}

void print_gguf_debug_info(const std::string &debug_info) {
    if (!env_setup_for_print_debug_info()) {
        return;
    }
    std::cout << "[GGUF Reader]: " << debug_info << std::endl;
}

void print_scheduler_config_info(const SchedulerConfig &scheduler_config) {
    if (!env_setup_for_print_debug_info()) {
        return;
    }

    std::cout << scheduler_config.to_string() << std::endl;
}

std::pair<ov::CompiledModel, KVDesc>
compile_decoder_for_npu(const std::shared_ptr<ov::Model>& model,
                        const ov::AnyMap& config,
                        const KVAxesPosition& kv_pos) {
    ov::CompiledModel compiled;
    ov::AnyMap properties = config;
    KVDesc kv_desc;

    auto blob_path = pop_or_default(properties, "BLOB_PATH", std::string{});
    const auto export_blob = pop_or_default(properties, "EXPORT_BLOB", false);
    const bool do_import = (!blob_path.empty() && !export_blob);

    if (do_import) {
        if (!std::filesystem::exists(blob_path)) {
            OPENVINO_THROW("Blob file is not found at: " + blob_path);
        }
        std::ifstream fin(blob_path, std::ios::in | std::ios::binary);
        if (!fin.is_open()) {
            OPENVINO_THROW("Blob file can't be opened: " + blob_path);
        }
        compiled = ov::genai::utils::singleton_core().import_model(fin, "NPU", config);
        kv_desc.max_prompt_len = compiled.get_property("NPUW_LLM_MAX_PROMPT_LEN").as<uint32_t>();
        kv_desc.min_response_len = compiled.get_property("NPUW_LLM_MIN_RESPONSE_LEN").as<uint32_t>();
    } else {
        kv_desc.max_prompt_len = pop_int_and_cast(properties, "MAX_PROMPT_LEN").value_or(1024u);
        kv_desc.min_response_len = pop_int_and_cast(properties, "MIN_RESPONSE_LEN").value_or(128u);
        update_npu_config(properties, model, kv_pos, kv_desc);
        compiled = ov::genai::utils::singleton_core().compile_model(model, "NPU", properties);
        // Also export compiled model if required
        if (export_blob) {
            if (blob_path.empty()) {
                blob_path = "openvino_model.blob";
            }
            // Check the path is full
            const int EXT_SIZE = 5; // ".blob"
            if (blob_path.size() < EXT_SIZE) {
                OPENVINO_THROW("Please provide a full path to blob file in BLOB_PATH: " + blob_path);
            }
            if (strncmp(".blob", &blob_path[blob_path.size() - EXT_SIZE], EXT_SIZE) != 0) {
                OPENVINO_THROW("Please provide a full path to blob file in BLOB_PATH: " + blob_path);
            }
            std::ofstream fout(blob_path, std::ios::out | std::ios::binary);
            if (!fout.is_open()) {
                OPENVINO_THROW("Blob file can't be exported to: " + blob_path);
            }
            compiled.export_model(fout);
        }
    }
    return { compiled, kv_desc };
}

std::optional<ov::Any> pop_option(ov::AnyMap& config, const std::string& option_name) {
    if (auto it = config.find(option_name); it != config.end()) {
        std::optional<ov::Any> found = std::make_optional(it->second);
        config.erase(it);
        return found;
    }
    return std::nullopt;
}

const ModelsMap::mapped_type& get_model_weights_pair(const ModelsMap& models_map, const std::string& key) {
    auto it = models_map.find(key);
    if (it != models_map.end()) {
        return it->second;
    }
    OPENVINO_THROW("Model with key '", key, "' not found in models map.");
}

std::pair<ov::AnyMap, SchedulerConfig> extract_scheduler_config(const ov::AnyMap& properties, std::optional<SchedulerConfig> default_config) {
    ov::AnyMap plugin_config = properties;
    auto it = plugin_config.find(ov::genai::scheduler_config.name());
    SchedulerConfig scheduler_config;
    if (it != plugin_config.end()) {
        scheduler_config = it->second.as<SchedulerConfig>();
        plugin_config.erase(it);
    } else if (default_config.has_value()) {
        scheduler_config = *default_config;
    }
    return {plugin_config, scheduler_config};
};

SchedulerConfig get_latency_oriented_scheduler_config() {
    SchedulerConfig default_config;
    default_config.max_num_batched_tokens = std::numeric_limits<size_t>::max(); // don't limit total batch size
    default_config.enable_prefix_caching = true; // for better TTFT in chat scenarios
    return default_config;
}

bool explicitly_requires_paged_attention(const ov::AnyMap& properties, bool is_npu_requested) {
    auto attention_backend_it = properties.find("ATTENTION_BACKEND");

    if (properties.find(ov::genai::scheduler_config.name()) != properties.end() ||
        (attention_backend_it != properties.end() && attention_backend_it->second.as<std::string>() == PA_BACKEND)) {
        if (is_paged_attention_available()) {
            return true;
        } else {
            OPENVINO_THROW("Continuous batching backend requires PagedAttention operation support, which is available on x86_64 or ARM64 platforms only");
        }
    }

    if (properties.find(utils::DRAFT_MODEL_ARG_NAME) != properties.end() && !is_npu_requested) {
        if (is_paged_attention_available()) {
            return true;
        } else {
            OPENVINO_THROW("Speculative decoding requires PagedAttention operation support on non-NPU devices, which is available on x86_64 or ARM64 platforms only");
        }
    }

    auto prompt_lookup_prop = properties.find("prompt_lookup");
    if (prompt_lookup_prop != properties.end() && prompt_lookup_prop->second.as<bool>() == true) {
        if (is_paged_attention_available()) {
            return true;
        } else {
            OPENVINO_THROW("Prompt lookup decoding requires PagedAttention operation support, which is available on x86_64 or ARM64 platforms only");
        }
    }
    return false;
}

std::pair<ov::AnyMap, std::string> extract_attention_backend(const ov::AnyMap& external_properties,
                                                             bool is_npu_requested) {
    std::string attention_backend = PA_BACKEND;
    ov::AnyMap properties = external_properties;

    auto it = properties.find("ATTENTION_BACKEND");
    if (it != properties.end()) {
        attention_backend = it->second.as<std::string>();
        OPENVINO_ASSERT(attention_backend == PA_BACKEND || attention_backend == SDPA_BACKEND,
            "Attention backend must be either '", PA_BACKEND, "' or '", SDPA_BACKEND, "', got '", attention_backend, "'");
        properties.erase(it);
    }

    if (explicitly_requires_paged_attention(external_properties, is_npu_requested)) {
        OPENVINO_ASSERT(attention_backend == PA_BACKEND,
            "User properties are conflicting: some of them requires PagedAttention backend, while 'ATTENTION_BACKEND' is set to 'SDPA'");
    }

    return {properties, attention_backend};
};

void release_core_plugin(const std::string& device) {
    try {
        singleton_core().unload_plugin(device);
    } catch (const ov::Exception&) {
        // Note: in a theory it can throw an exception when 2 different pipelines are created from
        // different threads and then both of them unload plugin for 'device' from ov::Core
    }
}

ov::Tensor merge_text_and_image_embeddings_llava(const ov::Tensor& input_ids, ov::Tensor& text_embeds, const std::vector<ov::Tensor>& image_embeds, int64_t image_token_id) {
    auto text_embeds_shape = text_embeds.get_shape();
    size_t text_embeds_seq_length = text_embeds_shape[1];
    size_t hidden_size = text_embeds_shape[2];

    const int64_t* input_ids_data = input_ids.data<const int64_t>();
    int token_offset = text_embeds_seq_length - 1;
    auto text_embeds_data = text_embeds.data<float>();
    const float* text_embeds_end = text_embeds_data + text_embeds_seq_length * hidden_size;

    // Copy in reversed order because a tokenizer may truncate the input removing the prefix.
    for (auto image_embed_it = image_embeds.rbegin(); image_embed_it != image_embeds.rend(); ++image_embed_it) {
        for (; token_offset != -1; --token_offset) {
            if (input_ids_data[token_offset] == image_token_id) {
                break;
            }
        }
        if (token_offset == -1) {
            break;
        }
        int changed_token_offset = token_offset;
        for (; changed_token_offset != -1; --changed_token_offset) {
            if (input_ids_data[changed_token_offset] != image_token_id) {
                break;
            }
        }
        size_t n_tokens = std::min(image_embed_it->get_shape().at(1), size_t(token_offset - changed_token_offset));
        size_t n_floats = n_tokens * hidden_size;
        auto text_embeds_idx = text_embeds_data + (changed_token_offset + 1) * hidden_size;
        OPENVINO_ASSERT(text_embeds_idx + n_floats <= text_embeds_end);
        std::copy_n(
            image_embed_it->data<const float>() + image_embed_it->get_size() - n_floats,
            n_floats,
            text_embeds_idx
        );
        token_offset -= n_tokens + 1;
    }
    // text_embeds is bound to infer request that can be used by another thread after leaving embeddings calculation scope
    // so we need to return a copy to make sure data does not get corrupted
    ov::Tensor inputs_embeds(text_embeds.get_element_type(), text_embeds.get_shape());
    std::memcpy(inputs_embeds.data(), text_embeds.data(), text_embeds.get_byte_size());
    return inputs_embeds;
}

size_t get_available_gpu_memory(const std::string& device, size_t num_decoder_layers) {
    OPENVINO_ASSERT(device.find("GPU") != std::string::npos, "get_available_gpu_memory() is applicable for GPU only.");

    ov::Core core = utils::singleton_core();
    auto memory_statistics = core.get_property(device, ov::intel_gpu::memory_statistics);
    auto device_type = core.get_property(device, ov::device::type);

    // sum up all used device memory
    std::vector<std::string> device_memory_types = {"cl_mem", "usm_device"};
    size_t used_device_mem = 0;
    for (auto mem_type: device_memory_types) {
        used_device_mem += memory_statistics[mem_type];
    }

    if (device_type == ov::device::Type::INTEGRATED) {
        used_device_mem += memory_statistics["usm_host"];
    }

    // there could be unaccounted extra memory reserved by kernels, kept
    // in memory pools, etc
    // therefore, add a threshold to account for this
    float used_memory_threshold = 1.1;
    used_device_mem *= used_memory_threshold;

    // total device memory in bytes
    auto total_device_memory = core.get_property(device, ov::intel_gpu::device_total_mem_size);

    // max allocatable memory size on GPU
    auto max_alloc_memory_size = core.get_property(device, ov::intel_gpu::device_max_alloc_mem_size);

    // Total KV-cache size if a single tensor is limited by 'device_max_alloc_mem_size' property
    auto max_allocatable_kv_cache = max_alloc_memory_size * num_decoder_layers * 2;

    return std::min(total_device_memory - used_device_mem, max_allocatable_kv_cache);
}

std::pair<ov::AnyMap, std::optional<std::filesystem::path>> extract_export_properties(const ov::AnyMap& external_properties) {
    ov::AnyMap properties = external_properties;
    std::optional<std::filesystem::path> blob_path;

    auto blob_path_it = properties.find(ov::genai::blob_path.name());
    if (blob_path_it != properties.end()) {
        blob_path = blob_path_it->second.as<std::filesystem::path>();
        OPENVINO_ASSERT(!blob_path->empty(), ov::genai::blob_path.name(), " property is empty");
        properties.erase(blob_path_it);
    }

    return {properties, blob_path};
}

ov::CompiledModel import_model(const std::filesystem::path& blob_path,
                               const std::string& device,
                               const ov::AnyMap& properties) {
    OPENVINO_ASSERT(!blob_path.empty(), "blob path is empty");
    ov::Tensor blob_tensor = ov::read_tensor_data(blob_path);
    return ov::genai::utils::singleton_core().import_model(blob_tensor, device, properties);
}

void export_model(ov::CompiledModel& compiled_model, const std::filesystem::path& blob_path) {
    OPENVINO_ASSERT(!blob_path.empty(), "blob path is empty");

    std::filesystem::create_directories(blob_path.parent_path());

    std::ofstream out(blob_path, std::ios::out | std::ios::binary);
    OPENVINO_ASSERT(out.is_open(), "Cannot open file to write: " + blob_path.string());
    compiled_model.export_model(out);
    out.close();
}

}  // namespace utils
}  // namespace genai
}  // namespace ov
