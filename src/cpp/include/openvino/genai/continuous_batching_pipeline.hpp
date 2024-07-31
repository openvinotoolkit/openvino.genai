// Copyright (C) 2023-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "openvino/genai/scheduler_config.hpp"
#include "openvino/genai/generation_handle.hpp"
#include "openvino/genai/visibility.hpp"

#include "openvino/genai/cb_basic_pipeline.hpp"

namespace ov::genai {
class OPENVINO_GENAI_EXPORTS ContinuousBatchingPipeline : public ov::genai::BasicPipeline {
protected:
    class Impl;
    std::shared_ptr<Impl> m_impl;

    // GenerationHandle add_request(uint64_t request_id, ov::Tensor tokenized_prompt, ov::genai::GenerationConfig sampling_params) override;

    std::vector<GenerationHandle> generate_sequences(
        const std::vector<ov::Tensor> prompts, std::vector<ov::genai::GenerationConfig> sampling_params) override;

public:
    ContinuousBatchingPipeline(const std::string& models_path,
                               const SchedulerConfig& scheduler_config,
                               const std::string& device = "CPU",
                               const ov::AnyMap& llm_plugin_config = {},
                               const ov::AnyMap& tokenizer_plugin_config = {});

    /**
    * @brief Constructs a ContinuousBatchingPipeline when ov::genai::Tokenizer is initialized manually using file from the different dirs.
    *
    * @param model_path Path to the dir with model, tokenizer .xml/.bin files, and generation_configs.json
    * @param scheduler_config
    * @param tokenizer manually initialized ov::genai::Tokenizer
    * @param device optional device
    * @param plugin_config optional plugin_config
    */
    ContinuousBatchingPipeline(
        const std::string& model_path,
        const ov::genai::Tokenizer& tokenizer,
        const SchedulerConfig& scheduler_config,
        const std::string& device="CPU",
        const ov::AnyMap& plugin_config={}
    );

    ContinuousBatchingPipeline() = default;

    PipelineMetrics get_metrics() const override;

    void step() override;

    bool has_non_finished_requests() override;

    GenerationHandle add_request(uint64_t request_id, std::string prompt, ov::genai::GenerationConfig sampling_params);
    GenerationHandle add_request(uint64_t request_id, ov::Tensor prompt, ov::genai::GenerationConfig sampling_params);

    struct GeneratedSequence {
        uint64_t request_id, sequence_id;
        std::vector<int64_t> token_ids;
        std::vector<float> log_probs;

        GeneratedSequence(uint64_t req_id, uint64_t seq_id, const  std::vector<int64_t>& generated_token_ids, const std::vector<float>& generated_log_probs) :
            request_id(req_id),
            sequence_id(seq_id),
            token_ids(generated_token_ids),
            log_probs(generated_log_probs) {};
    };

    struct UpdateSeqResult {
        size_t to_insert, to_remove;
        UpdateSeqResult(size_t _to_insert = 0, size_t _to_remove = 0) : to_insert(_to_insert), to_remove(_to_remove) {};
    };

    std::vector<GeneratedSequence> get_generated_sequences();
    UpdateSeqResult update_generated_sequence(const GeneratedSequence& new_sequence);
    void enable_validation_mode();
};
}
