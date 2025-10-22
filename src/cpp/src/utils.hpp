// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <type_traits>
#include <optional>
#include <stdexcept>
#include <utility>

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"
#include "openvino/runtime/core.hpp"

#include "openvino/genai/generation_handle.hpp"
#include "openvino/genai/scheduler_config.hpp"
#include "openvino/genai/generation_config.hpp"
#include "visual_language/processor_config.hpp"

#include "openvino/genai/streamer_base.hpp"

namespace ov {
namespace genai {

extern const std::string PA_BACKEND;
extern const std::string SDPA_BACKEND;

struct ModelDesc {
    std::string device;
    ov::genai::SchedulerConfig scheduler_config;
    ov::AnyMap properties;
    ov::genai::GenerationConfig generation_config;
    std::shared_ptr<ov::Model> model = nullptr;
    ov::genai::Tokenizer tokenizer;

    ModelDesc(const std::shared_ptr<ov::Model>& model,
              const ov::genai::Tokenizer& tokenizer,
              const std::string& device = {},
              const ov::AnyMap& properties = {},
              const ov::genai::SchedulerConfig& scheduler_config = {},
              const ov::genai::GenerationConfig& generation_config = {}) :
        model(model),
        tokenizer(tokenizer),
        device(device),
        properties(properties),
        scheduler_config(scheduler_config),
        generation_config(generation_config) {}
    
    ModelDesc() = default;
};
}  // namespace genai
}  // namespace ov

namespace ov {
namespace genai {
namespace utils {

// Variable template that checks if a type has begin() and end() member functions
template<typename, typename = void>
constexpr bool is_container = false;

template<typename T>
constexpr bool is_container<T,
    std::void_t<decltype(std::declval<T>().begin()),
                decltype(std::declval<T>().end())>> = true;

enum class GenerationChatInputsType {
    UNDEF = 0, // Default value, type of inputs is not defined
    STRING = 1, // Type of inputs is StringInputs
    ENCODED_INPUTS = 2, // Type of inputs is EncodedInputs
    CHAT_HISTORY = 3, // Type of inputs is ChatHistory
};

struct GenerationFinishInfo
{
    EncodedResults results;
    GenerationStatus streaming_finish_status;
};

Tensor init_attention_mask(const Tensor& position_ids);

void initialize_position_ids(ov::Tensor& position_ids, const ov::Tensor& attention_mask, int64_t start_pos = 0);

template <typename T> struct OmitOptional { using value = T; };
template <typename T> struct OmitOptional<std::optional<T>> { using value = T; };

template <typename T>
void read_anymap_param(const ov::AnyMap& config_map, const std::string& name, T& param) {
    auto it = config_map.find(name);
    if (it != config_map.end()) {
        if (it->second.empty()) {
            if (ov::genai::utils::is_container<T>)
                param = T{};
            else {
                OPENVINO_THROW("Got empty ov::Any for parameter name: " + name);
            }
        }
        else {
            param = it->second.as<typename OmitOptional<T>::value>();
        }
    }
}

const std::string STREAMER_ARG_NAME = "streamer";
const std::string CONFIG_ARG_NAME = "generation_config";
const std::string DRAFT_MODEL_ARG_NAME = "draft_model";

template<typename Config = ov::genai::GenerationConfig>
Config from_config_json_if_exists(const std::filesystem::path& models_path, const char config_name[] = "generation_config.json") {
    auto config_file_path = models_path / config_name;
    return std::filesystem::exists(config_file_path) ? Config{config_file_path} : Config{};
}

ov::genai::StreamerVariant get_streamer_from_map(const ov::AnyMap& config_map);

ov::genai::OptionalGenerationConfig get_config_from_map(const ov::AnyMap& config_map);

ProcessorConfig from_any_map(
    const ov::AnyMap& config_map,
    const ProcessorConfig& initial
);

ov::genai::ModelDesc get_draft_model_from_config(const ov::AnyMap& config);

ov::genai::ModelDesc extract_draft_model_from_config(ov::AnyMap& config);

bool is_npu_requested(const std::string& device, const ov::AnyMap& properties);

ov::genai::TokenizedInputs subtract_chat_tokenized_inputs(const ov::genai::TokenizedInputs& minuend, const ov::genai::TokenizedInputs& subtrahend);

void apply_slice_before_matmul_transformation(std::shared_ptr<ov::Model> model);

void apply_gather_before_matmul_transformation(std::shared_ptr<ov::Model> model);

ov::Core& singleton_core();

std::pair<ov::AnyMap, bool> extract_gguf_properties(const ov::AnyMap& external_properties);

std::pair<ov::AnyMap, bool> extract_paired_input_props(const ov::AnyMap& external_properties);

std::shared_ptr<ov::Model> read_model(const std::filesystem::path& model_dir,  const ov::AnyMap& config);

void release_core_plugin(const std::string& device);

size_t get_first_history_difference(const ov::Tensor& encoded_history, const std::vector<int64_t> tokenized_history);

struct KVAxesPosition {
    size_t batch;
    size_t seq_len;
};

KVAxesPosition get_kv_axes_pos(std::shared_ptr<const ov::Model> model);

class KVCacheState {
    std::vector<int64_t> state;
public:
    size_t num_tokens_to_trim = 0;
    size_t seq_length_axis = 2;
    bool reset_mem_state = false;

    std::vector<int64_t>& get_state() {
        return state;
    }

    void add_inputs(const ov::Tensor& inputs_ids) {
        OPENVINO_SUPPRESS_DEPRECATED_START
        std::copy_n(inputs_ids.data<int64_t>(), inputs_ids.get_size(), std::back_inserter(state));
        OPENVINO_SUPPRESS_DEPRECATED_END
    }

    void reset_state() {
        reset_mem_state = false;
        num_tokens_to_trim = 0;
        state.clear();
    }
};

void trim_kv_cache(ov::InferRequest request, KVCacheState& kv_cache_state, std::optional<AdapterController> adapter_controller);

ov::Tensor push_front_inputs(const ov::Tensor& base_tensor, int64_t add_to_front);

bool env_setup_for_print_debug_info();

void print_compiled_model_properties(ov::CompiledModel& compiled_Model, const char* model_title);

void print_gguf_debug_info(const std::string& debug_info);

void print_scheduler_config_info(const SchedulerConfig &scheduler_config);

struct KVDesc {
    uint32_t max_prompt_len;
    uint32_t min_response_len;
};

std::pair<ov::CompiledModel, KVDesc> compile_decoder_for_npu(const std::shared_ptr<ov::Model>& model,
                                                             const ov::AnyMap& config,
                                                             const KVAxesPosition& kv_pos);

/// @brief SharedOptional is a wrapper around a reference to an existing object and an optional shared alternative value.
/// The difference from std::optional is that the default state is not empty and contains a reference to an existing object outside the class.
/// Another difference is that the alternative value is shared between all instances of SharedOptional like std::shared_ptr.
/// This class enables copy-on-write behaviour when a potentially expensive to replicate object is modified optionally.
/// @tparam T type of held value
template <typename T>
class SharedOptional {
public:
    using non_const_T = std::remove_const_t<T>;

    // Constructor: Requires a reference to the default value.
    explicit SharedOptional(T* default_value)
        : value_ref(default_value) {}

    // Constructor: In case of reference, the object taken an ownership.
    explicit SharedOptional(T& _alternative)
        : alternative(std::make_shared<non_const_T>(_alternative)) {
        value_ref = &*alternative;
    }

    // The following operators are required to mimic the behavior of std::optional.

    T& operator*() {
        return *value_ref;
    }

    const T& operator*() const {
        return *value_ref;
    }

    operator bool() {
        return alternative;
    }

    T* operator->() {
        return value_ref;
    }

    const T* operator->() const {
        return value_ref;
    }

    /**
     * @brief Creates a fork of the default value if required and returns a reference to it.
     *
     * This function checks if an alternative value exists. If not, it creates an alternative
     * by copying the value referenced by `value_ref` and then updates `value_ref` to point
     * to the new alternative value. Returns a reference to the new alternative value.
     */
    non_const_T& fork() {
        if (!alternative) {
            alternative = std::make_shared<non_const_T>(*value_ref);
            value_ref = alternative.get();
        }
        return *alternative;
    }

private:
    T* value_ref;               // Reference to the default value.
    std::shared_ptr<non_const_T> alternative; // Optional alternative value.
};

template<class... Ts> struct overloaded : Ts... {using Ts::operator()...;};
template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>;
std::shared_ptr<StreamerBase> create_streamer(StreamerVariant streamer, Tokenizer tokenizer);

std::optional<ov::Any> pop_option(ov::AnyMap& config, const std::string& option_name);

template <typename T>
T pop_or_default(ov::AnyMap& config, const std::string& key, const T& default_value) {
    auto anyopt = pop_option(config, key);
    if (anyopt.has_value()) {
        if (anyopt.value().empty()) {
            OPENVINO_THROW("Got empty ov::Any for key: " + key);
        }
        return anyopt.value().as<T>();
    }
    return default_value;
}

const ModelsMap::mapped_type& get_model_weights_pair(const ModelsMap& models_map, const std::string& key);

std::pair<ov::AnyMap, SchedulerConfig> extract_scheduler_config(const ov::AnyMap& properties, std::optional<SchedulerConfig> default_config = std::nullopt);

SchedulerConfig get_latency_oriented_scheduler_config();

bool explicitly_requires_paged_attention(const ov::AnyMap& properties, bool is_npu_requested = false);

std::pair<ov::AnyMap, std::string> extract_attention_backend(const ov::AnyMap& external_properties, bool is_npu_requested = false);

void save_openvino_model(const std::shared_ptr<ov::Model>& model, const std::string& save_path, bool compress_to_fp16);

ov::Tensor merge_text_and_image_embeddings_llava(const ov::Tensor& input_ids, ov::Tensor& text_embeds, const std::vector<ov::Tensor>& image_embeds, int64_t image_token_id);

size_t get_available_gpu_memory(const std::string& device, size_t num_decoder_layers);

/**
 * @brief Extracts and removes blob import/export related properties from the provided map.
 */
std::pair<ov::AnyMap, std::optional<std::filesystem::path>> extract_export_properties(const ov::AnyMap& external_properties);

/**
 * @brief Imports a compiled model from a blob file previously exported using export_model.
 */
ov::CompiledModel import_model(const std::filesystem::path& blob_path,
                               const std::string& device,
                               const ov::AnyMap& properties);

/**
 * @brief Exports a compiled model to a blob file for later use with import_model.
 */
void export_model(ov::CompiledModel& compiled_model, const std::filesystem::path& blob_path);

}  // namespace utils
}  // namespace genai
}  // namespace ov
