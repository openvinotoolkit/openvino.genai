#pragma once

#include <optional>
#include "openvino/genai/llm_pipeline.hpp"
#include "visual_language/embedding_model.hpp"
#include "sampling/sampler.hpp"

namespace ov {
namespace genai {

ov::genai::utils::GenerationFinishInfo get_lm_encoded_results(ov::InferRequest& m_llm, const ov::Tensor& input_ids, const ov::Tensor& attention_mask,
                                                              const std::shared_ptr<StreamerBase>& streamer_ptr, Sampler& sampler, std::vector<SequenceGroup::Ptr> sequence_groups,
                                                              std::optional<ov::Tensor> position_ids, std::optional<ov::Tensor> token_type_ids, utils::KVCacheState& m_kv_cache_state, EmbeddingsModel::Ptr m_embedding,
                                                              std::optional<int64_t> rope_delta = std::nullopt, const size_t max_kv_cache_size = std::numeric_limits<size_t>::max());


void align_kv_cache_and_history(const ov::Tensor& new_chat_tokens, utils::KVCacheState& kv_cache_state);


TokenizedInputs get_chat_encoded_input(const ov::Tensor& new_chat_tokens, utils::KVCacheState& kv_cache_state);

}
}
