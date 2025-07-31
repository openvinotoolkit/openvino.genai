// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <openvino/genai/generation_handle.hpp>
#include <openvino/genai/speculative_decoding/perf_metrics.hpp>
#include "llm/pipeline_base.hpp"
#include "sampling/sampler.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {
constexpr size_t BATCH_SIZE = 1;

class LLMInferWrapper {
public:
    LLMInferWrapper::LLMInferWrapper(const ov::genai::ModelDesc& model_desc);
    ov::genai::GenerationConfig get_generation_config() const;
    void set_generation_config(ov::genai::GenerationConfig config);
    int64_t infer_first(const ov::Tensor &input_ids,
                        const ov::Tensor &attention_mask,
                        const ov::Tensor &position_ids);
    bool can_infer();
    int64_t infer_next(const std::vector<int64_t> tokens);
    int64_t infer_next(int64_t out_token);
    std::vector<int64_t> infer_next_return_all(const std::vector<int64_t> tokens);
    ov::Tensor get_logits();
    std::size_t get_num_processed_tokens() const;
    ov::genai::GenerationHandle create_generation_handle();
    void remove_last_generated_tokens(const std::size_t tokens_to_remove); 
    void trimm_kv_cache(const std::size_t tokens_to_remove);
    ov::genai::EncodedResults finalize();
    ov::genai::GenerationStatus get_generation_status() const;
    void reset_state();

private:
    ov::Tensor infer_next_internal(const std::vector<int64_t> tokens);
    void set_already_allocated_input_for_1_token();
    std::variant<int64_t, std::vector<int64_t>> sample_tokens(
        const ov::Tensor& logits, std::size_t num_tokens_to_return);

private:
    ov::AnyMap m_properties;
    ov::genai::GenerationConfig m_generation_config;
    ov::genai::Tokenizer m_tokenizer;

    std::size_t m_num_processed_tokens = 0u;
    uint32_t m_max_prompt_len = 0u;
    uint32_t m_kvcache_total = 0u;
    ov::genai::utils::KVAxesPosition m_kv_pos;
    ov::InferRequest m_request;
    ov::genai::Sampler m_sampler;
    std::shared_ptr<ov::genai::SequenceGroup> m_sequence_group = nullptr;
    GenerationHandle m_handle = nullptr;
    // Separate metrics?

    // Data placeholder for 1-token inference:
    int64_t m_new_input_token = -1;
    int64_t m_new_position_id = -1;
    std::vector<int64_t> m_new_atten_mask_data;
};

struct SpeculativeConfig {
    void update_candidate_strategy(const size_t num_matches);

    std::size_t max_seq_length = SIZE_MAX;
    std::size_t num_pred_tokens = 5;
    const std::size_t max_pred_tokens = 10;
};

class SpeculativeLLMPipelineNPU : public ov::genai::LLMPipelineImplBase {
public:
    SpeculativeLLMPipelineNPU(
    const ov::genai::ModelDesc& main_model_desc, 
    const ov::genai::ModelDesc& draft_model_desc
    );

    DecodedResults generate(
        StringInputs inputs,
        OptionalGenerationConfig generation_config,
        StreamerVariant streamer
    ) override;

    EncodedResults generate(
        const EncodedInputs& inputs,
        OptionalGenerationConfig generation_config,
        StreamerVariant streamer
    ) override;

    void start_chat(const std::string& system_message) override;
    void finish_chat() override;
    ~SpeculativeLLMPipelineNPU();

private:
    int64_t generate_next_token(const std::vector<int64_t> tokens);
    std::vector<int64_t> generate_candidates(int64_t out_token);
    void update_candidate_strategy(const size_t num_matches);
    void update_kv_cache(const size_t seq_length);

private:
    uint32_t m_max_prompt_len = 0u;
    uint32_t m_kvcache_total = 0u;
    std::unique_ptr<LLMInferWrapper> m_draft_request;
    std::unique_ptr<LLMInferWrapper> m_main_request;
    SpeculativeConfig m_speculative_config;
    ov::genai::SDPerModelsPerfMetrics m_perf_metrics;

    bool m_is_chat_conversation = false;
    ChatHistory m_history;
    ov::genai::GenerationStatus m_chat_generation_finish_status = ov::genai::GenerationStatus::RUNNING;
};

}  // namespace genai
}  // namespace ov
