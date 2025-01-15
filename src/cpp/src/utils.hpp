// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <type_traits>

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/runtime/core.hpp"

#include "visual_language/processor_config.hpp"

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
};

struct HistoryRemoveManager
{
    size_t num_tokens_to_remove_from_kv_cache = 0;
    size_t trusted_history_length = 0;

    bool does_kv_cache_need_to_update() {
        return (trusted_history_length > 0 || num_tokens_to_remove_from_kv_cache > 0);
    }

    void reset() {
        num_tokens_to_remove_from_kv_cache = 0;
        trusted_history_length = 0;
    }
};

Tensor init_attention_mask(const Tensor& position_ids);

void print_tensor(const ov::Tensor& tensor);

int64_t argmax(const ov::Tensor& logits, const size_t batch_idx);

void initialize_position_ids(ov::Tensor& position_ids, const ov::Tensor& attention_mask, int64_t start_pos = 0);

ov::Tensor extend_attention(ov::Tensor attention_mask);

void update_position_ids(ov::Tensor&& position_ids, const ov::Tensor&& attention_mask);

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


std::pair<ov::AnyMap, SchedulerConfig> split_scheduler_config(const ov::AnyMap& properties);

ov::genai::TokenizedInputs subtract_chat_tokenized_inputs(const ov::genai::TokenizedInputs& minuend, const ov::genai::TokenizedInputs& subtrahend);

void apply_slice_before_matmul_transformation(std::shared_ptr<ov::Model> model);

void apply_gather_before_matmul_transformation(std::shared_ptr<ov::Model> model);

ov::Core singleton_core();

template <typename T>
void read_rt_info(std::shared_ptr<ov::Model>& model, const char* name, T& value);

size_t get_first_history_difference(const ov::Tensor& encoded_history, const std::vector<int64_t> tokenized_history, std::set<int64_t> stop_tokens);

size_t get_seq_len_axis(std::shared_ptr<const ov::Model> model);

void trim_kv_cache(ov::InferRequest request, uint64_t remove_from_end, size_t seq_length_axis, std::optional<AdapterController> adapter_controller);

ov::Tensor push_front_inputs(const ov::Tensor& base_tensor, int64_t add_to_front);

void print_compiled_model_properties(ov::CompiledModel& compiled_Model, const char* model_title);

#include <optional>
#include <stdexcept>
#include <utility>
#include <type_traits>

/// @brief Like std::optional it has two states: default one and set value. The difference from std::optional is that default state is not empty and contains a reference to existing object outside.
/// @tparam T type of held value
template <typename T>
class DefaultOptional {
public:
    // Constructor: Requires a reference to the default value.
    explicit DefaultOptional(T* default_value)
        : value_ref(default_value) {}

    // Constructor: In case of reference, the object taken an ownership.
    explicit DefaultOptional(T& _alternative)
        : alternative(_alternative) {
        value_ref = &*alternative;
    }

    T& value() {
        return *value_ref;
    }

    const T& value() const {
        return *value_ref;
    }

    operator T&() {
        return value();
    }

    operator const T&() const {
        return value();
    }

    /**
     * @brief Creates a fork of the current value.
     *
     * This function checks if an alternative value exists. If not, it creates an alternative
     * by copying the value referenced by `value_ref` and then updates `value_ref` to point
     * to the new alternative value. Returns a reference to the new alternative value.
     */
    std::remove_const_t<T>& fork() {
        if (!alternative) {
            alternative = *value_ref;
            value_ref = &*alternative;
        }
        return *alternative;
    }

private:
    T* value_ref;               // Reference to the default value.
    std::optional<std::remove_const_t<T>> alternative; // Optional alternative value.
};



}  // namespace utils
}  // namespace genai
}  // namespace ov
