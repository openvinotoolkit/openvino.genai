// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/speculative_decoding/perf_metrics.hpp"
#include "speculative_decoding/speculative_decoding_metrics.hpp"
#include "model_desc.hpp"
#include "stateful_pipeline_base.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

struct Gemma4MTPSharedKV {
    ov::Tensor full_key;
    ov::Tensor full_value;
    ov::Tensor sliding_key;
    ov::Tensor sliding_value;
};

struct Gemma4MTPOutput {
    ov::Tensor logits;
    ov::Tensor hidden_states;
    Gemma4MTPSharedKV shared_kv;
};

bool is_gemma4_mtp_model_pair(const std::shared_ptr<ov::Model>& target_model,
                              const std::shared_ptr<ov::Model>& draft_model);

class Gemma4MTPTargetWrapper {
public:
    explicit Gemma4MTPTargetWrapper(const ov::genai::ModelDesc& model_desc);

    void reset_state();
    void release_memory();

    Gemma4MTPOutput infer(const ov::Tensor& input_ids,
                          const ov::Tensor& attention_mask,
                          const ov::Tensor& position_ids);

    ov::Tensor embed_token(int64_t token_id);
    void crop_state_to_length(size_t target_length);

    size_t get_kv_sequence_axis() const {
        return m_kv_axes_pos.seq_len;
    }

    ov::genai::RawPerfMetrics& get_raw_perf_metrics() {
        return m_raw_perf_metrics;
    }

private:
    std::shared_ptr<ov::Model> create_embedding_model(const std::shared_ptr<ov::Model>& model) const;
    uint64_t execute_inference();
    void update_inference_time(uint64_t inference_time_us);

    static constexpr size_t BATCH_SIZE = 1;

    std::string m_device;
    ov::AnyMap m_properties;
    mutable ov::InferRequest m_request;
    mutable ov::InferRequest m_embedding_request;
    ov::genai::utils::KVAxesPosition m_kv_axes_pos;
    ov::genai::utils::CacheTypes m_cache_types;
    ov::genai::RawPerfMetrics m_raw_perf_metrics;
    size_t m_processed_tokens = 0;
};

class Gemma4MTPAssistantWrapper {
public:
    explicit Gemma4MTPAssistantWrapper(const ov::genai::ModelDesc& model_desc);

    void reset_state();
    void release_memory();

    Gemma4MTPOutput infer(const ov::Tensor& inputs_embeds,
                          const ov::Tensor& attention_mask,
                          const ov::Tensor& position_ids,
                          const Gemma4MTPSharedKV& shared_kv);

    ov::genai::RawPerfMetrics& get_raw_perf_metrics() {
        return m_raw_perf_metrics;
    }

private:
    uint64_t execute_inference();
    void update_inference_time(uint64_t inference_time_us);

    std::string m_device;
    ov::AnyMap m_properties;
    mutable ov::InferRequest m_request;
    ov::genai::RawPerfMetrics m_raw_perf_metrics;
};

class StatefulGemma4MTPLLMPipeline : public StatefulSpeculativePipelineBase {
public:
    StatefulGemma4MTPLLMPipeline(const ov::genai::ModelDesc& target_model_desc,
                                 const ov::genai::ModelDesc& draft_model_desc);
    ~StatefulGemma4MTPLLMPipeline();

    ov::genai::SpeculativeDecodingMetrics get_speculative_decoding_metrics() const;

    void finish_chat() override;

protected:
    GenerationConfig resolve_generation_config(OptionalGenerationConfig generation_config) override;

    EncodedResults generate_tokens(const EncodedInputs& inputs,
                                   const GenerationConfig& config,
                                   StreamerVariant streamer) override;

private:
    struct DraftResult {
        std::vector<int64_t> tokens;
    };

    Gemma4MTPSharedKV crop_shared_kv(const Gemma4MTPSharedKV& shared_kv, size_t accepted_length) const;
    ov::Tensor select_hidden_state(const ov::Tensor& hidden_states, size_t position) const;
    ov::Tensor concatenate_embedding_and_hidden(const ov::Tensor& embedding, const ov::Tensor& hidden_state);
    std::vector<int64_t> sample_greedy_tokens(const ov::Tensor& logits, size_t token_count) const;
    bool is_stop_token(int64_t token, const GenerationConfig& config) const;

    DraftResult draft_tokens(const GenerationConfig& config,
                             const Gemma4MTPOutput& previous_target_output,
                             const std::vector<int64_t>& accepted_tokens,
                             size_t n_last_matches,
                             size_t remaining_tokens);

    std::unique_ptr<Gemma4MTPTargetWrapper> m_target;
    std::unique_ptr<Gemma4MTPAssistantWrapper> m_assistant;
    ov::Tensor m_inputs_embeds_buffer;
    size_t m_num_assistant_tokens = DEFAULT_NUM_ASSISTANT_TOKENS;
    size_t m_prompt_length = 0;
};

}  // namespace genai
}  // namespace ov
