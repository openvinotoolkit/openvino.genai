// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <openvino/genai/perf_metrics.hpp>
#include <openvino/genai/speculative_decoding/perf_metrics.hpp>

#include "sampling/sampler.hpp"
#include "sequence_group.hpp"
#include "speculative_decoding/eagle3_model_transforms.hpp"
#include "speculative_decoding/speculative_decoding_metrics.hpp"
#include "stateful_pipeline_base.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

struct DFlashInferenceOutput {
    ov::Tensor logits;
    ov::Tensor hidden_features;
};

struct DFlashInferResult {
    DFlashInferenceOutput output;
    std::vector<int64_t> sampled_tokens;
};

class DFlashHiddenStateProvider {
public:
    void reset();
    void append(const ov::Tensor& hidden_states, size_t token_count);
    void truncate(size_t context_length);
    const ov::Tensor& tensor() const { return m_context; }
    size_t context_length() const;

private:
    ov::Tensor m_context;
};

class DFlashSamplerAdapter {
public:
    explicit DFlashSamplerAdapter(const Tokenizer& tokenizer);
    std::vector<int64_t> sample(SequenceGroup::Ptr sequence_group,
                                const ov::Tensor& logits,
                                size_t input_token_count,
                                size_t sample_count,
                                size_t num_tokens_to_validate = 0,
                                bool validation_mode = false);
    void clear(uint64_t request_id);

private:
    Sampler m_sampler;
};

class DFlashTargetWrapper {
public:
    explicit DFlashTargetWrapper(const ov::genai::ModelDesc& model_desc);
    ~DFlashTargetWrapper() = default;
    void initialize_sequence(const ov::Tensor& input_ids, const GenerationConfig& config);
    void append_tokens(const std::vector<int64_t>& tokens);
    void truncate_sequence(size_t size);
    void trim_kv_cache(size_t tokens_to_remove);
    void reset_state();
    void release_memory();
    DFlashInferenceOutput infer(const ov::Tensor& input_ids,
                                const ov::Tensor& attention_mask,
                                const ov::Tensor& position_ids);
    DFlashInferResult forward(size_t input_token_count, size_t sample_count = 1, size_t num_tokens_to_validate = 0);
    Sequence::Ptr get_current_sequence() const;
    SequenceGroup::Ptr get_sequence_group() const { return m_sequence_group; }
    const std::vector<int64_t>& get_generated_tokens() const;
    size_t get_sequence_length() const;
    ov::genai::RawPerfMetrics& get_raw_perf_metrics() { return m_raw_perf_metrics; }

private:
    void build_model_inputs(size_t input_token_count, ov::Tensor& input_ids, ov::Tensor& attention_mask, ov::Tensor& position_ids);
    ov::Tensor get_logits() const;
    ov::Tensor get_hidden_features() const;
    uint64_t execute_inference();
    void update_inference_time(uint64_t inference_time_us);

    static constexpr size_t BATCH_SIZE = 1;
    std::string m_device;
    ov::AnyMap m_properties;
    Tokenizer m_tokenizer;
    mutable ov::InferRequest m_request;
    ov::genai::utils::KVAxesPosition m_kv_axes_pos;
    ov::genai::utils::CacheTypes m_cache_types;
    SequenceGroup::Ptr m_sequence_group;
    DFlashSamplerAdapter m_sampler;
    ov::genai::RawPerfMetrics m_raw_perf_metrics;
};

class DFlashDraftWrapper {
public:
    DFlashDraftWrapper(const ov::genai::ModelDesc& model_desc,
                       const Tokenizer& tokenizer,
                       const ov::genai::utils::dflash::DFlashRTInfo& rt_info);
    ~DFlashDraftWrapper() = default;
    void initialize_sequence(const ov::Tensor& input_ids, const GenerationConfig& config);
    void append_tokens(const std::vector<int64_t>& tokens);
    void sync_generated_tokens(const std::vector<int64_t>& target_generated_tokens);
    void reset_state();
    void release_memory();
    DFlashInferenceOutput infer(int64_t seed_token, const ov::Tensor& target_hidden);
    std::vector<int64_t> sample_candidates(const ov::Tensor& logits, size_t candidate_count);
    ov::genai::RawPerfMetrics& get_raw_perf_metrics() { return m_raw_perf_metrics; }

private:
    ov::Tensor build_input_ids(int64_t seed_token) const;
    ov::Tensor build_position_ids(size_t context_length) const;
    ov::Tensor get_logits() const;
    uint64_t execute_inference();
    void update_inference_time(uint64_t inference_time_us);

    static constexpr size_t BATCH_SIZE = 1;
    std::string m_device;
    ov::AnyMap m_properties;
    Tokenizer m_tokenizer;
    mutable ov::InferRequest m_request;
    SequenceGroup::Ptr m_sequence_group;
    DFlashSamplerAdapter m_sampler;
    ov::genai::RawPerfMetrics m_raw_perf_metrics;
    size_t m_prompt_length = 0;
    size_t m_block_size = 0;
    int64_t m_mask_token_id = -1;
};

class StatefulDFlashLLMPipeline : public StatefulSpeculativePipelineBase {
public:
    StatefulDFlashLLMPipeline(const ov::genai::ModelDesc& target_model_desc,
                              const ov::genai::ModelDesc& draft_model_desc,
                              const ov::genai::utils::dflash::DFlashRTInfo& rt_info);
    ~StatefulDFlashLLMPipeline();
    ov::genai::SpeculativeDecodingMetrics get_speculative_decoding_metrics() const;
    void finish_chat() override;

protected:
    GenerationConfig resolve_generation_config(OptionalGenerationConfig generation_config) override;
    EncodedResults generate_tokens(const EncodedInputs& inputs,
                                   const GenerationConfig& config,
                                   StreamerVariant streamer) override;

private:
    struct SpeculativeResult {
        size_t accepted_tokens_count = 0;
        bool eos_reached = false;
        std::vector<int64_t> validated_tokens;
    };

    SpeculativeResult run_speculative_iteration(size_t current_generated_tokens,
                                                size_t max_new_tokens,
                                                int64_t eos_token_id);

    std::unique_ptr<DFlashTargetWrapper> m_target;
    std::unique_ptr<DFlashDraftWrapper> m_draft;
    DFlashHiddenStateProvider m_hidden_state_provider;
    ov::genai::utils::dflash::DFlashRTInfo m_rt_info;
    size_t m_prompt_length = 0;
};

}  // namespace genai
}  // namespace ov
