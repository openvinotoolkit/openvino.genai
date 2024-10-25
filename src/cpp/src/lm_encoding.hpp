#pragma once

#include <optional>
#include "openvino/genai/llm_pipeline.hpp"
#include "sampler.hpp"

namespace ov {
namespace genai {

ov::genai::EncodedResults get_lm_encoded_results(ov::InferRequest& m_model_runner, const ov::Tensor& input_ids, ov::Tensor attention_mask,
                                                 const std::shared_ptr<StreamerBase>& streamer_ptr, Sampler& sampler, std::vector<SequenceGroup::Ptr> sequence_groups,
                                                 std::optional<ov::Tensor> position_ids, std::optional<ov::InferRequest> m_embedding, std::optional<float> scale_emb, std::optional<int32_t> selected_beam_idx);

}
}