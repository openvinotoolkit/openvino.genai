// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "llm_pipeline_static.hpp"

#include <fstream>

#include "openvino/pass/stateful_to_stateless.hpp"
#include "openvino/runtime/core.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"
#include "openvino/runtime/properties.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

#include <jinja2cpp/user_callable.h>

#include "text_callback_streamer.hpp"
#include "json_utils.hpp"
#include "utils.hpp"

namespace {

std::shared_ptr<ov::Model> cvt_kvcache_to_fp16(const std::shared_ptr<ov::Model>& model) {
    ov::preprocess::PrePostProcessor ppp(model);

    for (auto tensor : model->inputs()) {
        if (tensor.get_any_name().find("past_key") != std::string::npos) {
            ppp.input(tensor.get_any_name()).tensor().set_element_type(ov::element::Type_t::f16);
        }
    }

    for (auto tensor : model->outputs()) {
        if (tensor.get_any_name().find("present") != std::string::npos) {
            ppp.output(tensor.get_any_name()).tensor().set_element_type(ov::element::Type_t::f16);
        }
    }

    return ppp.build();
}

void align_u4_zp_constants(const std::shared_ptr<ov::Model>& model) {
    for (auto op : model->get_ops()) {
        if (ov::op::util::is_constant(op)) {
            auto cst_op = std::dynamic_pointer_cast<ov::op::v0::Constant>(op);
            const auto cst_op_out = cst_op->output(0);
            if (cst_op_out.get_element_type() == ov::element::u4 && ov::shape_size(cst_op_out.get_shape()) == 1u) {
                ov::Tensor cst_tensor(ov::element::u4, cst_op_out.get_shape());
                *static_cast<uint8_t*>(cst_tensor.data()) = cst_op->get_vector<uint8_t>()[0] & 0x0f;
                auto new_cst_op = std::make_shared<ov::op::v0::Constant>(cst_tensor);
                for (auto target_input : cst_op_out.get_target_inputs()) {
                    target_input.replace_source_output(new_cst_op);
                }
            }
        }
    }
}

bool is_cw_compressed(const std::shared_ptr<ov::Model>& model) {
    std::vector<std::string> rt_info_path = {"nncf", "weight_compression", "group_size"};
    if (!model->has_rt_info(rt_info_path)) {
        // NB: Model isn't compressed by NNCF - skip
        return false;
    }
    auto group_size = model->get_rt_info<int>(rt_info_path);
    if (group_size == -1) {
        // NB: Enable DQ for CW quantized models
        return true;
    }
    return false;
}

std::optional<ov::Any> pop_option(ov::AnyMap& config, const std::string& option_name) {
    if (auto it = config.find(option_name); it != config.end()) {
        std::optional<ov::Any> found = std::make_optional(it->second);
        config.erase(it);
        return found;
    }
    return std::nullopt;
}

template <typename T>
std::optional<T> get_option(ov::AnyMap& config, const std::string& option_name) {
    if (auto it = config.find(option_name); it != config.end()) {
        return std::make_optional(it->second.as<T>());
    }
    return std::nullopt;
}

std::shared_ptr<ov::Model> redirect_new_kv_to_output(const std::shared_ptr<ov::Model>& model) {
    const auto kStartOutputKVCacheLayers = 1u;
    for (int i = kStartOutputKVCacheLayers; i < model->outputs().size(); ++i) {
        auto kvout  = model->output(i);
        auto kvrslt = kvout.get_node();
        auto kvcat  = kvrslt->inputs()[0].get_source_output().get_node();
        auto kvval  = kvcat->inputs()[1].get_source_output();
        kvval.set_names({kvout.get_any_name()});
        kvrslt->inputs()[0].replace_source_output(kvval);
    }
    model->validate_nodes_and_infer_types();
    return model;
}

std::shared_ptr<ov::Model> add_slices_to_kvcache_inputs(const std::shared_ptr<ov::Model>& model) {
    const auto kvcache_name_pattern = "past_key_values";
    std::vector<std::shared_ptr<ov::opset13::Parameter>> new_params;
    for (auto param : model->get_parameters()) {
        auto tensor_name = param->get_output_tensor(0).get_any_name();
        if (tensor_name.find(kvcache_name_pattern) == std::string::npos) {
            new_params.push_back(param);
            continue;
        }
        auto shape = param->get_output_shape(0);
        shape[2] += 1;

        auto new_param = std::make_shared<ov::opset13::Parameter>(param->get_element_type(), shape);
        new_param->set_friendly_name(tensor_name);
        new_param->outputs().begin()->get_tensor().set_names(param->outputs().begin()->get_tensor().get_names());

        auto slice_start = std::make_shared<ov::opset13::Constant>(
            ov::element::Type_t::i32, ov::Shape{1}, std::vector<int32_t>{1}
        );
        auto slice_stop = std::make_shared<ov::opset13::Constant>(
            ov::element::Type_t::i32, ov::Shape{1}, std::vector<int32_t>{static_cast<int32_t>(shape[2])}
        );
        auto slice_step = std::make_shared<ov::opset13::Constant>(
            ov::element::Type_t::i32, ov::Shape{1}, std::vector<int32_t>{1}
        );
        auto slice_axes = std::make_shared<ov::opset13::Constant>(
            ov::element::Type_t::i32, ov::Shape{1}, std::vector<int32_t>{2}
        );
        auto slice_node = std::make_shared<ov::opset13::Slice>(
            new_param, slice_start->output(0), slice_stop->output(0), slice_step->output(0), slice_axes->output(0)
        );
        slice_node->set_friendly_name(tensor_name + "_Slice");
        for (auto target_input : param->output(0).get_target_inputs()) {
            target_input.replace_source_output(slice_node->output(0));
        }
        new_params.push_back(new_param);
    }
    return std::make_shared<ov::Model>(model->get_results(), ov::SinkVector{}, new_params);
}

struct KVAxesPosition {
    uint32_t batch;
    uint32_t seq_len;
};

KVAxesPosition get_kv_axes(const std::string& model_type) {
    KVAxesPosition axes;
    if (model_type == "chatglm") {
        axes.batch = 1u;
        axes.seq_len = 0u;
    } else if (model_type == "qwen") {
        // Note, qwen2 does not fall into this category and conforms to default layout
        axes.batch = 0u;
        axes.seq_len = 1u;
    } else {
        axes.batch = 0u;
        axes.seq_len = 2u;
    }
    return axes;
}

std::string get_model_type_from_json(const std::filesystem::path& filepath) {
    std::ifstream file(filepath);
    OPENVINO_ASSERT(file.is_open(), "Could not open file: " + filepath.string());
    nlohmann::json config_data = nlohmann::json::parse(file);
    std::string model_type = config_data["model_type"].get<std::string>();
    return model_type;
}

void reshape_to_static(std::shared_ptr<ov::Model> model,
                       const uint32_t input_size,
                       const uint32_t kvcache_size,
                       const KVAxesPosition& kv_axes_position) {
    std::map<std::string, ov::PartialShape> new_shapes;
    for (auto input : model->inputs()) {
        const auto& input_name = input.get_any_name();
        ov::PartialShape new_shape;
        if (input_name.find("input_ids") != std::string::npos) {
            new_shape = ov::PartialShape({1, input_size});
        } else if (input_name.find("attention_mask") != std::string::npos) {
            new_shape = ov::PartialShape({1, kvcache_size});
        } else if (input_name.find("position_ids") != std::string::npos) {
            new_shape = ov::PartialShape({1, input_size});
        } else {
            const auto& partial_shape = input.get_partial_shape();
            new_shape = partial_shape;
            new_shape[kv_axes_position.batch] = 1;
            new_shape[kv_axes_position.seq_len] = kvcache_size - input_size;
        }
        new_shapes.emplace(input_name, new_shape);
    }
    model->reshape(new_shapes);
}

template <typename T>
void fill_tensor(ov::Tensor tensor, T fill_val, size_t offset = 0u) {
    T* tensor_data = tensor.data<T>();
    std::fill(tensor_data + offset, tensor_data + tensor.get_size(), fill_val);
}

void copy_with_offset(const ov::Tensor& orig, const std::size_t offset, ov::Tensor& padded) {
    int64_t* orig_data = orig.data<int64_t>();
    int64_t* padded_data = padded.data<int64_t>();
    std::copy(orig_data, orig_data + orig.get_size(), padded_data + offset);
}

void merge_config_with(ov::AnyMap& lhs, const ov::AnyMap& rhs) {
    for (const auto& [key, value] : rhs) {
        // NB: Overwrite the value if key already exists
        if (auto it = lhs.find(key); it != lhs.end()) {
            it->second = value;
        } else {
            lhs.emplace(key, value);
        }
    }
}

struct NPUDesc {
    std::string arch;
    int64_t max_tiles;
};

std::optional<NPUDesc> extract_npu_descriptor(ov::Core& core) {
    const auto all_devices = core.get_available_devices();
    if (std::find(all_devices.begin(), all_devices.end(), "NPU") == all_devices.end()) {
        return std::nullopt;
    }
    const auto arch = core.get_property("NPU", ov::device::architecture);
    const auto max_tiles = core.get_property("NPU", ov::intel_npu::max_tiles);
    return std::make_optional(NPUDesc{arch, max_tiles});
}

ov::AnyMap get_baseline_common_config() {
    ov::AnyMap config = {
        { "NPU_COMPILATION_MODE_PARAMS", "compute-layers-with-higher-precision=Sqrt,Power,ReduceMean,Add_RMSNorm" },
        { "NPUW_DEVICES", "NPU" },
        { "NPU_USE_NPUW",  "YES" },
        { "NPUW_FOLD", "YES" },
        { "NPUW_DCOFF_TYPE", "f16" },
        { "NPUW_DCOFF_SCALE", "YES"},
        { "NPUW_WEIGHTS_BANK", "shared" },
        { "NPUW_SLICE_OUT", "YES" },
        { "NPUW_FUNCALL_ASYNC", "YES" }
    };
    return config;
}

ov::AnyMap get_default_common_config(const std::shared_ptr<ov::Model>& model) {
    auto config = get_baseline_common_config();
    const char* npu_l0 = std::getenv("DISABLE_OPENVINO_GENAI_NPU_L0");
    if (npu_l0 && std::atoi(npu_l0) == 1) {
        config.emplace("NPUW_WEIGHTS_BANK_ALLOC", "CPU");
    } else {
        config.emplace("NPUW_FUNCALL_FOR_ALL", "YES");
    }
    return config;
}

ov::AnyMap get_default_prefill_config(const std::shared_ptr<ov::Model>& model,
                                      const std::optional<NPUDesc>& npudesc) {
    auto config = get_default_common_config(model);
    if (is_cw_compressed(model)) {
        config.emplace("NPUW_DQ", "YES");
    } else {
        config.emplace("NPUW_PMM", "NO");
    }
    if (npudesc.has_value() &&
        npudesc->arch == "4000" &&
        npudesc->max_tiles != -1) {
        config.emplace("NPU_DPU_GROUPS", npudesc->max_tiles);
    }
    return config;
}

ov::AnyMap get_default_generate_config(const std::shared_ptr<ov::Model>& model,
                                       const std::optional<NPUDesc>& npudesc) {
    auto config = get_default_common_config(model);
    // NB: Unconditionally set for generation model
    config.emplace("NPUW_DQ", "YES");
    if (npudesc.has_value() && npudesc->arch == "4000") {
        config.emplace("NPU_DPU_GROUPS", 4);
    }
    return config;
}

template <typename T>
T pop_or_default(ov::AnyMap& config, const std::string& key, const T& default_value) {
    auto anyopt = pop_option(config, key);
    if (anyopt.has_value()) {
        return anyopt.value().as<T>();
    }
    return default_value;
}

std::optional<uint32_t> pop_int_and_cast(ov::AnyMap& config, const std::string& key) {
    auto anyopt = pop_option(config, key);
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

ov::Tensor make_tensor_slice(ov::Tensor tensor, size_t dim, size_t start_pos, size_t end_pos) {
    ov::Shape start_shape(std::vector<size_t>(tensor.get_shape().size(), 0u));
    start_shape[dim] = start_pos;
    ov::Shape end_shape = tensor.get_shape();
    end_shape[dim] = end_pos;
    return ov::Tensor(tensor, start_shape, end_shape);
}

void drop_cache_dir(ov::AnyMap& config) {
    if (config.count("NPU_USE_NPUW") != 0u) {
        pop_option(config, "CACHE_DIR");
    }
}

} // anonymous namespace

namespace ov {
namespace genai {

StaticLLMPipeline::StaticLLMPipeline(
    const std::filesystem::path& models_path,
    const ov::genai::Tokenizer& tokenizer,
    const std::string& device,
    const ov::AnyMap& config
) : LLMPipelineImplBase(tokenizer,
                        utils::from_config_json_if_exists(models_path)) {
    auto properties = config;
    /* NB: Static LLM pipeline consists of two models,
       first to process the input prompt (prefill),
       second to use in generation loop (kvcache)

       There are two ways of how these models can be created
       and user chooses one or another via configuration option
       "USE_BLOBS":
        1. When both models are created from the provided .xml one,
           that is "USE_BLOBS=NO" default way.
        2. When both models are directly imported from provided prefill
           and generation precompiled blobs, that is "USE_BLOBS=YES" way.
    */
    const auto use_blobs = pop_or_default(properties, "USE_BLOBS", false);
    if (!use_blobs) {
        setupAndCompileModels(models_path, device, properties);
    } else {
        setupAndImportModels(models_path, device, properties);
    }
    // Initialize tensors
    prepare_for_new_conversation();
};

StaticLLMPipeline::StaticLLMPipeline(
    const std::filesystem::path& models_path,
    const std::string& device,
    const ov::AnyMap& properties
) : StaticLLMPipeline(models_path, Tokenizer(models_path), device, properties) {
}

void StaticLLMPipeline::setupAndCompileModels(
    const std::filesystem::path& models_path,
    const std::string& device,
    ov::AnyMap& properties) {
    /* Initialization assumes multiple steps if user passes "USE_BLOBS=NO":
        1) Read the template model - this will be kvcache model
        2) Expose KV-cache input and output layers from kvcache model
        3) Align u4 ZP constants - TODO: get rid of this step in future
        4) Replace KV-cache tensors for the entire cache to tensors only for new token (before concat)
        5) Clone the model - this will be prefill
        6) Reshape both models to static shape
        7) Compile both models
    */

    ov::Core core;

    // NB: Get information about NPU if available
    auto npudesc = extract_npu_descriptor(core);

    // (1) Read the template model - this will be kvcache model
    m_kvcache_model = core.read_model((models_path / "openvino_model.xml").string());
    // (2) Expose KV-cache input and output layers from kvcache model
    ov::pass::StatefulToStateless().run_on_model(m_kvcache_model);
    // (3) Align u4 ZP constants
    align_u4_zp_constants(m_kvcache_model);
    // (4) Replace KV-tensors for the entire cache to tensors only for new token
    m_kvcache_model = redirect_new_kv_to_output(m_kvcache_model);
    // (5) Convert kvcache tensors to fp16 precision
    m_kvcache_model = cvt_kvcache_to_fp16(m_kvcache_model);
    // (6) Clone the model - this will be prefill
    m_prefill_model = m_kvcache_model->clone();
    m_prefill_model->set_friendly_name(m_kvcache_model->get_friendly_name() + "_prefill");
    // (7) Reshape both models to static shape
    const uint32_t kMaxPromptLen = pop_int_and_cast(properties, "MAX_PROMPT_LEN").value_or(1024u);
    const uint32_t kMinResponseLen = pop_int_and_cast(properties, "MIN_RESPONSE_LEN").value_or(128u);
    KVAxesPosition axes = get_kv_axes(get_model_type_from_json(models_path / "config.json"));
    m_kvcache_desc = KVCacheDesc { kMaxPromptLen, kMaxPromptLen + kMinResponseLen, 0u, axes.seq_len };
    reshape_to_static(m_prefill_model, m_kvcache_desc.max_prompt_size, m_kvcache_desc.max_prompt_size, axes);
    reshape_to_static(m_kvcache_model, 1u, m_kvcache_desc.total_size, axes);
    // (8) Compile both model
    auto prefill_config = pop_or_default(
        properties, "PREFILL_CONFIG", get_default_prefill_config(m_prefill_model, npudesc)
    );
    auto generate_config = pop_or_default(
        properties, "GENERATE_CONFIG", get_default_generate_config(m_kvcache_model, npudesc)
    );
    merge_config_with(prefill_config, properties);
    merge_config_with(generate_config, properties);
    // FIXME: Drop CACHE_DIR option if NPUW is enabled
    drop_cache_dir(prefill_config);
    drop_cache_dir(generate_config);

    m_kvcache_request = core.compile_model(
        m_kvcache_model, device, generate_config
    ).create_infer_request();
    m_prefill_request = core.compile_model(
        m_prefill_model, device, prefill_config
    ).create_infer_request();
}

void StaticLLMPipeline::setupAndImportModels(
    const std::filesystem::path& models_path,
    const std::string& device,
    ov::AnyMap& properties) {
    /* To initialize pipeline in case when user passes "USE_BLOBS=YES",
       next steps are required:
        1) Check that neither MAX_PROMPT_LEN nor MIN_RESPONSE_LEN is
           exposed in the config. These parameters will be retrieved
           from blobs
        2) Import prefill model from model directory or specified path
        3) Import generate model from model directory or specified path
        4) Fill in m_kvcache_desc
    */
    ov::Core core;

    auto import_blob = [this,
                        &models_path,
                        &properties,
                        &core,
                        &device](const std::string& model_name,
                                 ov::AnyMap& model_config) {
        auto blob_path = pop_or_default(model_config, "BLOB_PATH", std::string{});

        if (blob_path.empty()) {
            blob_path = (models_path /
                (std::string("openvino_") + model_name + ".blob")).string();
        }

        if (!std::filesystem::exists(blob_path)) {
            OPENVINO_THROW("Blob for " + model_name + " model is not found at: "
                + blob_path);
        }

        merge_config_with(model_config, properties);

        std::fstream fs(blob_path, std::ios::in | std::ios::binary);

        return core.import_model(
            fs, device, model_config);

    };

    auto get_kvcache_size = [](ov::CompiledModel& model) {
        for (auto input : model.inputs()) {
            const auto& input_name = input.get_any_name();
            if (input_name.find("attention_mask") != std::string::npos) {
                return static_cast<uint32_t>(input.get_shape()[1]);
            }
        }
        OPENVINO_THROW("No attention_mask input is found! Such model isn't supported.");
    };

    // (1) Check that neither MAX_PROMPT_LEN nor MIN_RESPONSE_LEN is
    //     exposed in the config
    if (properties.count("MAX_PROMPT_LEN") ||
        properties.count("MIN_RESPONSE_LEN")) {
        OPENVINO_THROW("Neither \"MAX_PROMPT_LEN\" nor \"MIN_RESPONSE_LEN\""
           " can be specified in \"USE_BLOBS=YES\" configuration!");
    }
    // (2) Import prefill model from model directory or specified path
    auto prefill_config = pop_or_default(properties, "PREFILL_CONFIG", ov::AnyMap());
    auto prefill_model = import_blob("prefill", prefill_config);
    m_prefill_request = prefill_model.create_infer_request();
    // (3) Import generate model from model directory or specified path
    auto generate_config = pop_or_default(properties, "GENERATE_CONFIG", ov::AnyMap());
    auto generate_model = import_blob("generate", generate_config);
    m_kvcache_request = generate_model.create_infer_request();
    // (4) Fill in m_kvcache_desc
    const uint32_t kMaxPromptLen = get_kvcache_size(prefill_model);
    const uint32_t kMinResponseLen = get_kvcache_size(generate_model) - kMaxPromptLen;
    // FIXME For some models KV-cache dim != 2u
    m_kvcache_desc = KVCacheDesc { kMaxPromptLen, kMaxPromptLen + kMinResponseLen, 0u, 2u };
}

void StaticLLMPipeline::start_chat(const std::string& system_message) {
    if (!system_message.empty()) {
        m_history.push_back({{"role", "system"}, {"content", system_message}});
    }
    m_is_chat_conversation = true;
};

void StaticLLMPipeline::finish_chat() {
    m_is_chat_conversation = false;
    m_history.clear();
};

void StaticLLMPipeline::prepare_for_new_conversation() {
    fill_tensor<int64_t>(m_prefill_request.get_tensor("input_ids"), m_tokenizer.get_pad_token_id());
    fill_tensor<int64_t>(m_prefill_request.get_tensor("position_ids"), 0u);
    fill_tensor<int64_t>(m_prefill_request.get_tensor("attention_mask"), 0u);
    fill_tensor<int64_t>(m_kvcache_request.get_tensor("attention_mask"), 0u);
    m_kvcache_desc.num_stored_tokens = 0u;
}

DecodedResults StaticLLMPipeline::generate(
    StringInputs inputs,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer
) {
    auto start_time = std::chrono::steady_clock::now();

    GenerationConfig config = (generation_config.has_value()) ? *generation_config : m_generation_config;
    std::string prompt;
    if (auto input_vector = std::get_if<std::vector<std::string>>(&inputs)) {
        if (input_vector->size() > 1u) {
            OPENVINO_THROW("Currently only batch size=1 is supported");
        }
        OPENVINO_ASSERT(!input_vector->empty());
        prompt = std::move(input_vector->front());
    } else {
        OPENVINO_ASSERT(std::holds_alternative<std::string>(inputs));
        prompt = std::get<std::string>(inputs);
    }

    ov::genai::TokenizedInputs tokenized_input;
    if (m_is_chat_conversation) {
        m_history.push_back({{"role", "user"}, {"content", prompt}});
        constexpr bool add_generation_prompt = true;
        prompt = m_tokenizer.apply_chat_template(m_history, add_generation_prompt);
        // for chat ov::genai::add_special_tokens(false) is aligned with stateful pipeline and HF
        tokenized_input = m_tokenizer.encode(prompt, ov::genai::add_special_tokens(false));
    } else {
        tokenized_input = m_tokenizer.encode(prompt);
    }

    auto encode_stop_time =  std::chrono::steady_clock::now();
    auto encoded_results = generate(tokenized_input, config, streamer);

    auto decode_start_time =  std::chrono::steady_clock::now();
    DecodedResults decoded_results = {m_tokenizer.decode(encoded_results.tokens), encoded_results.scores};
    auto decode_stop_time =  std::chrono::steady_clock::now();

    if (m_is_chat_conversation) {
        auto answer = decoded_results.texts[0];
        m_history.push_back({{"role", "assistant"}, {"content", answer}});
    }
    // generate_durations
    decoded_results.perf_metrics = encoded_results.perf_metrics;
    auto& raw_counters = decoded_results.perf_metrics.raw_metrics;
    auto stop_time = std::chrono::steady_clock::now();
    raw_counters.generate_durations = std::vector<MicroSecond>();
    raw_counters.generate_durations.emplace_back(PerfMetrics::get_microsec(stop_time - start_time));
    raw_counters.tokenization_durations.emplace_back(PerfMetrics::get_microsec(encode_stop_time - start_time));
    raw_counters.detokenization_durations.emplace_back(PerfMetrics::get_microsec(decode_stop_time - decode_start_time));
    decoded_results.perf_metrics.m_evaluated = false;
    decoded_results.perf_metrics.evaluate_statistics(start_time);
    return decoded_results;
}

EncodedResults StaticLLMPipeline::generate(
    const EncodedInputs& inputs,
    OptionalGenerationConfig generation_config,
    StreamerVariant streamer
) {
    auto start_time = std::chrono::steady_clock::now();
    ov::Tensor input_ids;
    ov::Tensor attention_mask;

    if (auto data = std::get_if<ov::Tensor>(&inputs)) {
        input_ids = *data;
        attention_mask = ov::genai::utils::init_attention_mask(input_ids);
    } else if (auto data = std::get_if<TokenizedInputs>(&inputs)) {
        input_ids = data->input_ids;
        attention_mask = data->attention_mask;
    }

    if (input_ids.get_shape().at(0) > 1u) {
        OPENVINO_THROW("Currently only batch size=1 is supported");
    }

    GenerationConfig config = (generation_config.has_value()) ? *generation_config : m_generation_config;
    // If eos_token_id was not provided, take value from default m_generation_config
    if (config.eos_token_id == -1)
        config.set_eos_token_id(m_generation_config.eos_token_id);
    config.validate();

    std::shared_ptr<StreamerBase> streamer_ptr;
    if (auto streamer_obj = std::get_if<std::monostate>(&streamer)) {
        streamer_ptr = nullptr;
    } else if (auto streamer_obj = std::get_if<std::shared_ptr<StreamerBase>>(&streamer)) {
        streamer_ptr = *streamer_obj;
    } else if (auto callback = std::get_if<std::function<bool(std::string)>>(&streamer)) {
        streamer_ptr = std::make_shared<TextCallbackStreamer>(m_tokenizer, *callback);
    }

    if (!config.is_greedy_decoding()) {
        OPENVINO_THROW("Currently only greedy decoding is supported");
    }

    ov::Shape prompts_shape = input_ids.get_shape();
    const size_t batch_size = prompts_shape[0];
    ov::genai::EncodedResults results;
    auto& raw_perf_counters = results.perf_metrics.raw_metrics;
    // NB: Only batch=1 is supported now
    results.scores.resize(1u);
    results.scores[0] = 0u;
    results.tokens.resize(1u);

    // NB: Check if there is enough space in KV-cache to process input prompt
    auto prompt_len = input_ids.get_size();
    if (prompt_len > m_kvcache_desc.max_prompt_size) {
        OPENVINO_THROW("Static LLM pipeline may only process prompts up to "
                       + std::to_string(m_kvcache_desc.max_prompt_size) + " tokens. "
                       + "Set the \"MAX_PROMPT_LEN\" config option to increase the limit.");
    }

    // NB: From the "generate" perspective, every call is treated as start of new conversation,
    // but if continuation is needed, prompt contains information about the entire conversation.
    prepare_for_new_conversation();

    auto padded_input_ids = m_prefill_request.get_tensor("input_ids");
    const size_t offset = padded_input_ids.get_size() - input_ids.get_size();
    copy_with_offset(input_ids, offset, padded_input_ids);

    auto padded_attention_mask = m_prefill_request.get_tensor("attention_mask");
    fill_tensor<int64_t>(padded_attention_mask, 1u, offset);

    auto padded_position_ids = m_prefill_request.get_tensor("position_ids");
    auto* padded_pos_data = padded_position_ids.data<int64_t>();
    std::iota(padded_pos_data + offset, padded_pos_data + padded_position_ids.get_size(), 0u);

    m_prefill_request.infer();
    raw_perf_counters.m_new_token_times.emplace_back(std::chrono::steady_clock::now());
    raw_perf_counters.m_batch_sizes.emplace_back(batch_size);

    // NB: Now there are prompt_len tokens in KV-cache
    m_kvcache_desc.num_stored_tokens += static_cast<uint32_t>(prompt_len);
    int64_t last_token = utils::argmax(m_prefill_request.get_tensor("logits"), 0);
    results.tokens[0].push_back(last_token);
    if (streamer_ptr && streamer_ptr->put(last_token)) {
        return results;
    }

    // Inputs: input_ids, attention_mask, position_ids, ...
    // Outputs: logits, ...
    const auto kStartInputKVCacheLayers = 3u;
    const auto kStartOutputKVCacheLayers = 1u;

    // NB: Copy KV-cache tensors from prefill model to kvcache model
    const auto& kvcache_compiled = m_kvcache_request.get_compiled_model();
    for (int i = 0; i < kvcache_compiled.outputs().size() - 1; ++i) {

        const auto& output_name = kvcache_compiled.outputs()[kStartOutputKVCacheLayers + i].get_any_name();
        auto prefill_out_tensor = m_prefill_request.get_tensor(output_name);
        auto prefill_out_slice = make_tensor_slice(
            prefill_out_tensor, m_kvcache_desc.dim, m_kvcache_desc.max_prompt_size - m_kvcache_desc.num_stored_tokens, m_kvcache_desc.max_prompt_size
        );

        const auto& input_name = kvcache_compiled.inputs()[kStartInputKVCacheLayers + i].get_any_name();
        auto kvcache_in_tensor = m_kvcache_request.get_tensor(input_name);
        fill_tensor<ov::float16>(kvcache_in_tensor, 0);

        auto kvcache_in_slice = make_tensor_slice(
            kvcache_in_tensor, m_kvcache_desc.dim, 0u, m_kvcache_desc.num_stored_tokens
        );

        prefill_out_slice.copy_to(kvcache_in_slice);
    }

    auto* input_ids_data = m_kvcache_request.get_tensor("input_ids").data<int64_t>();
    auto* position_ids_data = m_kvcache_request.get_tensor("position_ids").data<int64_t>();
    auto* attention_mask_data = m_kvcache_request.get_tensor("attention_mask").data<int64_t>();

    // NB: Fill attention mask in the correct format [1, 1 ... 1, 0, 0 ... 0, 1]
    std::fill(attention_mask_data, attention_mask_data + m_kvcache_desc.num_stored_tokens - 1u, 1u);
    attention_mask_data[m_kvcache_desc.total_size - 1] = 1u;

    const size_t max_tokens = config.get_max_new_tokens(prompt_len);
    for (int i = 0; i < max_tokens - 1; ++i) {
        input_ids_data[0] = last_token;
        position_ids_data[0] = m_kvcache_desc.num_stored_tokens;
        attention_mask_data[m_kvcache_desc.num_stored_tokens - 1] = 1u;

        m_kvcache_request.infer();
        m_kvcache_desc.num_stored_tokens += 1;

        last_token = utils::argmax(m_kvcache_request.get_tensor("logits"), 0);
        results.tokens[0].push_back(last_token);

        raw_perf_counters.m_new_token_times.emplace_back(std::chrono::steady_clock::now());
        raw_perf_counters.m_batch_sizes.emplace_back(batch_size);
        if (streamer_ptr && streamer_ptr->put(last_token)) {
            break;
        }

        if (last_token == config.eos_token_id && !config.ignore_eos) {
            break;
        }

        // NB: KV-cache is full, further generation is impossible
        if (m_kvcache_desc.num_stored_tokens == m_kvcache_desc.total_size) {
            break;
        }

        // NB: Write KV-cache for the new token to the correct input position for the next iteration
        for (int i = 0; i < kvcache_compiled.outputs().size() - 1; ++i) {
            const auto& input_name = kvcache_compiled.inputs()[kStartInputKVCacheLayers + i].get_any_name();
            auto kvcache_in_tensor = m_kvcache_request.get_tensor(input_name);
            auto kvcache_in_slice = make_tensor_slice(
                kvcache_in_tensor, m_kvcache_desc.dim, m_kvcache_desc.num_stored_tokens - 1, m_kvcache_desc.num_stored_tokens
            );
            const auto& output_name = kvcache_compiled.outputs()[kStartOutputKVCacheLayers + i].get_any_name();
            m_kvcache_request.get_tensor(output_name).copy_to(kvcache_in_slice);
        }
    }
    auto stop_time = std::chrono::steady_clock::now();
    // If is called without tokenization then that stat will not be reported.
    auto& metrics = results.perf_metrics;
    metrics.num_input_tokens = batch_size * input_ids.get_shape().at(1);
    metrics.load_time = this->m_load_time_ms;
    metrics.raw_metrics.generate_durations.emplace_back(PerfMetrics::get_microsec(stop_time - start_time));
    metrics.evaluate_statistics(start_time);
    return results;
}

}  // namespace genai
}  // namespace ov
