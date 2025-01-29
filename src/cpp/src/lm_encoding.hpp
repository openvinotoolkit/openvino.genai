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
    using Ptr = std::shared_ptr<KVCacheState>;

    std::vector<int64_t> get_state() {
        return state;
    }

    void add_input(int64_t input_idx) {
        state.push_back(input_idx);
    }

    void add_inputs(const ov::Tensor& inputs_ids) {
        auto data = inputs_ids.data<int64_t>();
        std::copy(data, data + inputs_ids.get_size(), std::back_inserter(state));
    }

    void trim_until(size_t num_to_keep) {
        state.resize(num_to_keep);
    }

    bool is_state_empty() {
        return state.empty();
    }

    void reset() {
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
                                                              std::optional<ov::Tensor> position_ids, KVCacheState::Ptr m_kv_cache_state, std::optional<EmbeddingsModel> m_embedding, std::optional<int64_t> rope_delta = std::nullopt);


void align_kv_cache_and_history(ov::genai::KVCacheTrimManager& kv_history_manager, const ov::Tensor& new_chat_tokens, KVCacheState::Ptr kv_cache_state);


TokenizedInputs get_chat_encoded_input(const ov::Tensor& new_chat_tokens, KVCacheState::Ptr kv_cache_state);

}
}
