// Copyright (C) 2023-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "fast_draft_strategy.hpp"
#include "openvino/op/constant.hpp"

namespace ov::genai {
class KVUpdateWrapper {
public:
    KVUpdateWrapper() = default;
    explicit KVUpdateWrapper(const ov::genai::ModelDesc& kv_model_desc);
    ~KVUpdateWrapper() = default;
    void infer(const ov::Tensor& block_indices,
                          const ov::Tensor& block_indices_begins,
                          const ov::Tensor& block_update_indices,
                          const ov::Tensor& block_update_indices_begins,
                          const std::vector<ov::Tensor>& key_caches,
                          const std::vector<ov::Tensor>& value_caches) {
                            // set input tensors
                            m_request.set_tensor("block_indices", block_indices);
                            m_request.set_tensor("block_indices_begins", block_indices_begins);
                            m_request.set_tensor("block_update_indices", block_update_indices);
                            m_request.set_tensor("block_update_indices_begins", block_update_indices_begins);
                            for (size_t i = 0; i < key_caches.size(); ++i) {
                                m_request.set_tensor("key_cache." + std::to_string(i), key_caches[i]);
                            }
                            for (size_t i = 0; i < value_caches.size(); ++i) {
                                m_request.set_tensor("value_cache." + std::to_string(i), value_caches[i]);
                            }
                            // infer and get output tensors
                            m_request.infer();
                        }

    const ov::CompiledModel& get_compiled_model() const {
        return m_compiled_model;
    }

private:
    ov::CompiledModel m_compiled_model;
    mutable ov::InferRequest m_request;
};
class ContinuousBatchingPipeline::Eagle3DecodingImpl : public ContinuousBatchingPipeline::SpeculativeDecodingImpl {
public:
    template<class Impl>
    friend std::vector<EncodedGenerationResult> generate_common(
        Impl*,
        const std::vector<ov::Tensor>&,
        const std::vector<GenerationConfig>&,
        const StreamerVariant&,
        std::optional<std::vector<ov::Tensor>>,
        std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>,
        std::optional<std::vector<ov::Tensor>>,
        const std::optional<std::vector<std::unordered_map<std::string, ov::Tensor>>>&,
        GenerateStrategy&);

    Eagle3DecodingImpl(const ov::genai::ModelDesc& main_model_desc, const ov::genai::ModelDesc& draft_model_desc, const std::vector<int>& hidden_layers_to_abstract);

    std::vector<EncodedGenerationResult>
    generate(const std::vector<ov::Tensor>& input_ids,
             const std::vector<GenerationConfig>& sampling_params,
             const StreamerVariant& streamer,
             const std::optional<std::vector<ov::Tensor>>& token_type_ids = std::nullopt,
             const std::optional<std::vector<std::pair<ov::Tensor, std::optional<int64_t>>>>& position_ids = std::nullopt,
             const std::optional<std::vector<ov::Tensor>>& prompt_ids = std::nullopt,
             const std::optional<std::vector<std::unordered_map<std::string, ov::Tensor>>>& lm_extra_inputs_list = std::nullopt) override;

    void step() override;

    GenerationHandle add_request(uint64_t request_id,
                                 const ov::Tensor& input_ids,
                                 const ov::genai::GenerationConfig& sampling_params,
                                 std::optional<ov::Tensor> token_type_ids = std::nullopt,
                                 std::optional<ov::Tensor> prompt_ids = std::nullopt,
                                 std::optional<std::unordered_map<std::string, ov::Tensor>> lm_extra_inputs = std::nullopt) override;

    GenerationHandle add_request(uint64_t request_id,
                                 const std::string& prompt,
                                 const ov::genai::GenerationConfig& sampling_params) override;
protected:
    void update_eagle_pipeline_params(const std::shared_ptr<ov::op::v0::Constant>& d2t_tensor);
    ov::Tensor create_draft_input(const ov::Tensor& original_input);
    // Creates draft model input by removing the first token from the original input sequence.
    ov::Tensor create_draft_input_ids(const ov::Tensor& original_input_ids);
    ov::Tensor create_draft_input_embeddings(const ov::Tensor& original_input_embeddings);
    // the wrapper for executing kv cache update model in eagle3 pipeline
    std::shared_ptr<KVUpdateWrapper> m_kv_update_wrapper;
};
}
