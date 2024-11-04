#pragma once

#include <optional>
#include "openvino/genai/llm_pipeline.hpp"
#include "visual_language/embedding_model.hpp"
#include "sampler.hpp"

namespace ov {
namespace genai {

std::pair<EncodedResults, int32_t> get_lm_encoded_results(ov::InferRequest& m_llm, const ov::Tensor& input_ids, const ov::Tensor& attention_mask,
                                                          const std::shared_ptr<StreamerBase>& streamer_ptr, Sampler& sampler, std::vector<SequenceGroup::Ptr> sequence_groups,
                                                          std::optional<ov::Tensor> position_ids, std::optional<EmbeddingsModel> m_embedding, std::optional<int32_t> selected_beam_idx);

void update_attention_mask_with_beams(ov::Tensor&& attention_mask, std::vector<int32_t> next_beams);

void update_position_ids(ov::Tensor&& position_ids, const ov::Tensor&& attention_mask);

}
}
