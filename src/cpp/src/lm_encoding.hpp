#pragma once

#include <optional>
#include "openvino/genai/llm_pipeline.hpp"
#include "visual_language/embedding_model.hpp"
#include "sampler.hpp"

namespace ov {
namespace genai {

ov::genai::utils::GenerationFinishInfo get_lm_encoded_results(ov::InferRequest& m_llm, const ov::Tensor& input_ids, const ov::Tensor& attention_mask,
                                                              const std::shared_ptr<StreamerBase>& streamer_ptr, Sampler& sampler, std::vector<SequenceGroup::Ptr> sequence_groups,
                                                              std::optional<ov::Tensor> position_ids, std::optional<EmbeddingsModel> m_embedding, std::optional<int64_t> rope_delta = std::nullopt);

void update_kv_history_manager(ov::genai::utils::HistoryRemoveManager& kv_history_manager, const ov::Tensor& prev_chat_tokens, const std::vector<int64_t> tokenized_history,
                               const std::set<int64_t> stop_tokens, const ov::genai::GenerationStatus finish_status);

TokenizedInputs get_chat_encoded_input(const ov::Tensor& new_chat_tokens, const ov::Tensor& prev_chat_tokens, const std::vector<int64_t> tokenized_history, ov::genai::utils::HistoryRemoveManager kv_history_manager);

void update_tokenized_history(std::vector<int64_t>& tokenized_history, const ov::Tensor& new_chat_tokens);

}
}
