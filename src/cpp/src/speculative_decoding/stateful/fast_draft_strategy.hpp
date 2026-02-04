// Copyright (C) 2025-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "sampling/sampler.hpp"
#include "utils.hpp"
#include "openvino/genai/perf_metrics.hpp"
#include "openvino/genai/speculative_decoding/perf_metrics.hpp"

#include "speculative_decoding/speculative_decoding_metrics.hpp"
#include "stateful_pipeline_base.hpp"

namespace ov {
namespace genai {

class LLMInferWrapper {
public:
    LLMInferWrapper(const ov::genai::ModelDesc& model_desc);

    std::string device() const;

    ov::genai::GenerationConfig get_generation_config() const;

    void set_generation_config(ov::genai::GenerationConfig config);

    int64_t get_kvcache_capacity() const;

    int64_t get_generation_capacity() const;

    int64_t infer_first(const ov::Tensor &input_ids,
                        const ov::Tensor &attention_mask,
                        const ov::Tensor &position_ids);

    bool can_infer(const std::size_t prompt_len = 0);

    int64_t infer_next(int64_t out_token, bool append_perf_stat = false);

    std::vector<int64_t> infer_next_return_all(const std::vector<int64_t>& tokens);

    ov::Tensor get_logits();

    std::size_t get_num_processed_tokens() const;

    void trim_kv_cache(const std::size_t tokens_to_remove);

    void reset_state();

    void release_memory();

public:
    ov::genai::RawPerfMetrics raw_perf_metrics;

private:
    static constexpr std::size_t BATCH_SIZE = 1;

    void set_already_allocated_input_for_1_token();

    std::variant<int64_t, std::vector<int64_t>> sample_tokens(
        const ov::Tensor& logits, std::size_t num_tokens_to_return);

private:
    std::string m_device;
    ov::AnyMap m_properties;
    ov::genai::GenerationConfig m_generation_config;
    ov::genai::Tokenizer m_tokenizer;

    std::size_t m_max_prompt_len = 0u;
    std::size_t m_kvcache_total = 0u;
    std::size_t m_first_prompt_len = 0u;
    std::size_t m_num_processed_tokens = 0u;
    int64_t last_token = -1;
    ov::genai::utils::KVAxesPosition m_kv_pos;
    ov::InferRequest m_request;

    // Data placeholder for 1-token inference:
    int64_t m_new_input_token = -1;
    int64_t m_new_position_id = -1;
    std::vector<int64_t> m_new_atten_mask_data;
};

class StatefulSpeculativeLLMPipeline : public StatefulSpeculativePipelineBase {
public:
    StatefulSpeculativeLLMPipeline(
    const ov::genai::ModelDesc& main_model_desc, 
    const ov::genai::ModelDesc& draft_model_desc
    );

    ~StatefulSpeculativeLLMPipeline();

    ov::genai::SpeculativeDecodingMetrics get_speculative_decoding_metrics() const;

    // Override to reset model states
    void finish_chat() override;

protected:
    // Override base class methods
    GenerationConfig resolve_generation_config(OptionalGenerationConfig generation_config) override;

    EncodedResults generate_tokens(const EncodedInputs& inputs,
                                   const GenerationConfig& config,
                                   StreamerVariant streamer) override;

private:
    void update_candidate_strategy(const std::size_t matches_num);

private:
    std::unique_ptr<LLMInferWrapper> m_draft_request;
    std::unique_ptr<LLMInferWrapper> m_main_request;
    std::size_t m_candidates_num = 5;
    std::size_t m_max_candidates_num = 10;
};

}  // namespace genai
}  // namespace ov
