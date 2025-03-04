#pragma once

#include <optional>
#include "openvino/genai/llm_pipeline.hpp"
#include "visual_language/embedding_model.hpp"
#include "sampler.hpp"

namespace ov {
namespace genai {

class KVCacheState {
    std::vector<int64_t> state;
public:
    std::vector<int64_t>& get_state() {
        return state;
    }

    void add_inputs(const ov::Tensor& inputs_ids) {
        std::copy_n(inputs_ids.data<int64_t>(), inputs_ids.get_size(), std::back_inserter(state));
    }

    void reset_state() {
        return state.clear();
    }
};


struct KVCacheTrimManager
{
    size_t num_tokens_to_trim = 0;
    size_t kv_cache_seq_length_axis = 2;

    void reset() {
        num_tokens_to_trim = 0;
    }
};


ov::genai::utils::GenerationFinishInfo get_lm_encoded_results(ov::InferRequest& m_llm, const ov::Tensor& input_ids, const ov::Tensor& attention_mask,
                                                              const std::shared_ptr<StreamerBase>& streamer_ptr, Sampler& sampler, std::vector<SequenceGroup::Ptr> sequence_groups,
                                                              std::optional<ov::Tensor> position_ids, KVCacheState& m_kv_cache_state, std::optional<EmbeddingsModel> m_embedding,
                                                              std::optional<int64_t> rope_delta = std::nullopt, const size_t max_kv_cache_size = std::numeric_limits<size_t>::max());


void align_kv_cache_and_history(ov::genai::KVCacheTrimManager& kv_history_manager, const ov::Tensor& new_chat_tokens, KVCacheState& kv_cache_state);


TokenizedInputs get_chat_encoded_input(const ov::Tensor& new_chat_tokens, KVCacheState& kv_cache_state);

}
}
