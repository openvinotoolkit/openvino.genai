// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <ostream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "openvino/core/model.hpp"
#include "openvino/genai/expert_predictor.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "utils.hpp"

namespace ov {
namespace genai {

class ExpertPredictorImpl {
public:
    struct Head {
        std::shared_ptr<ov::op::v0::Constant> weight;           // [num_experts, hidden_size], current token
        std::shared_ptr<ov::op::v0::Constant> previous_weight;  // [num_experts, hidden_size], previous token
        std::shared_ptr<ov::op::v0::Constant> selection_bias;   // [num_experts]
        int64_t input_mode = 0;
    };

    explicit ExpertPredictorImpl(const std::filesystem::path& path);

    std::map<size_t, Head> heads;  // key: 0-based index of the MoE decoder layer the head predicts
    size_t hidden_size = 0;
    size_t num_experts = 0;
};

namespace les {

// Names of the tensors GenAI adds to the language model
inline constexpr const char* STATE_NAME = "les_prev_hidden";             // stateful model: [batch, num_heads, hidden]
inline constexpr const char* PREV_HIDDEN_INPUT = "les_prev_hidden";      // PA model: [num_heads, num_sequences, hidden]
inline constexpr const char* LAST_HIDDEN_OUTPUT = "les_last_hidden";     // PA model: [num_heads, num_sequences, hidden]
inline constexpr const char* PREDICTED_EXPERTS_PREFIX = "les_predicted_experts.";  // + 0-based MoE layer index
inline constexpr const char* ROUTER_EXPERTS_PREFIX = "les_router_experts.";        // debug outputs, see stats_enabled()

inline bool is_state_name(const std::string& name) {
    return name == STATE_NAME;
}

// OPENVINO_GENAI_EXPERT_PREDICTOR_STATS=1 adds the router choices as model outputs and prints the recall of the
// predictions at the end of the pipeline lifetime.
bool stats_enabled();

enum class GraphMode {
    STATEFUL,  // SDPA model with ReadValue / Assign states, [batch, seq_len, hidden] hidden states
    PAGED      // model after SDPAToPagedAttention, [total_tokens, 1, hidden] hidden states
};

struct GraphInfo {
    std::vector<size_t> layers;  // MoE layers with a head, ascending
    size_t hidden_size = 0;
    size_t top_k = 0;         // predicted experts per token
    size_t router_top_k = 0;  // experts the router selects per token
    bool use_previous_token = false;
    bool router_outputs = false;
};

// Removes the expert_predictor property from `properties`; returns true and fills `predictor` if it was set.
bool extract_from_properties(utils::SharedOptional<const ov::AnyMap>& properties,
                             std::optional<ExpertPredictor>& predictor);

// GPU plugin property naming the predicted experts outputs (value: their name prefix, PREDICTED_EXPERTS_PREFIX), so
// that the plugin can prefetch the predicted experts of MoE offloading (OFFLOAD_RATIO).
inline constexpr const char* PLUGIN_PROPERTY = "MOE_LES_PREDICTED_EXPERTS";

// Adds PLUGIN_PROPERTY to `properties` if `device` supports it.
void add_plugin_properties(utils::SharedOptional<const ov::AnyMap>& properties, const std::string& device);

// Builds the predictor heads inside `model` and connects them to the router inputs of the MoE layers.
GraphInfo apply(const std::shared_ptr<ov::Model>& model, const ExpertPredictor& predictor, GraphMode mode);

// Recall of the predicted experts against the router choices, needs GraphInfo::router_outputs.
class Stats {
public:
    explicit Stats(GraphInfo info);
    ~Stats();
    // Counts the given rows (tokens) of the outputs of the last inference, all rows if `rows` is empty.
    void accumulate(ov::InferRequest& request, const std::vector<size_t>& rows = {});
    // Stateful model: counts the last inference only if it was a decode step (one token per sequence).
    void accumulate_stateful_decode(ov::InferRequest& request);
    void print(std::ostream& os) const;

private:
    GraphInfo m_info;
    mutable std::mutex m_mutex;
    std::vector<uint64_t> m_hits, m_total;
    uint64_t m_tokens = 0;
};

struct ScheduledSequence {
    uint64_t id;        // Sequence::get_id()
    size_t num_tokens;  // tokens of the sequence in this inference
};

// Keeps the router inputs of the last processed token of every sequence for the PA model: GenAI passes them as
// `les_prev_hidden` and reads the new ones from `les_last_hidden`. Only one token per sequence is kept, so a rollback
// (e.g. rejected speculative tokens) leaves the state of a later token: it only makes the next prediction less
// accurate, never changes the model results.
class PagedState {
public:
    explicit PagedState(GraphInfo info);
    void set_inputs(ov::InferRequest& request, const std::vector<ScheduledSequence>& sequences);
    void update(ov::InferRequest& request,
                const std::vector<ScheduledSequence>& sequences,
                const std::unordered_set<uint64_t>& alive_sequence_ids);

private:
    GraphInfo m_info;
    std::unordered_map<uint64_t, std::vector<float>> m_prev;  // [num_heads * hidden_size] per sequence
    std::unique_ptr<Stats> m_stats;
};

}  // namespace les
}  // namespace genai
}  // namespace ov
