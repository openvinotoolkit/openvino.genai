// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include "model_desc.hpp"
#include "openvino/genai/speculative_decoding/drafter.hpp"
#include "stateful_pipeline_base.hpp"

namespace ov {
namespace genai {

/**
 * @brief Stateful speculative decoding with a user-defined Drafter: NPU main model, tree verification, greedy.
 *
 * Every step the drafter proposes a token tree after the last generated token. The main model verifies the root and
 * the tree in one call with tree attention (NPUW_EAGLE, `eagle_tree_mask`), greedy acceptance keeps the longest
 * matching path plus the main model's next token, and NPUW drops the rejected rows from its KV cache. The prompt and
 * the accepted tokens, with the main model's `last_hidden_state` rows for them (when it has that output), go back to
 * the drafter.
 */
class StatefulDrafterLLMPipeline : public StatefulSpeculativePipelineBase {
public:
    StatefulDrafterLLMPipeline(const ModelDesc& main_model_desc, std::shared_ptr<Drafter> drafter);
    ~StatefulDrafterLLMPipeline() override;

    SpeculativeDecodingMetrics get_speculative_decoding_metrics() const {
        return m_sd_metrics;
    }

protected:
    GenerationConfig resolve_generation_config(OptionalGenerationConfig generation_config) override;

    EncodedResults generate_tokens(const EncodedInputs& inputs,
                                   const GenerationConfig& config,
                                   StreamerVariant streamer) override;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

namespace drafter_utils {

/// Validates a proposal (size, parent links) and returns the depth of every node (children of the root: 1).
std::vector<size_t> tree_depths(const DraftTree& tree, size_t max_nodes);

/// Additive tree-attention mask [1, 1, w, w] for the root and `tree` (w = nodes + 1, row 0 = root, row i + 1 =
/// node i): each row sees itself and its ancestors.
ov::Tensor make_tree_mask(const DraftTree& tree, ov::element::Type type);

/// Greedy acceptance walk. `greedy_token(row)` returns the main model's token after verifier row `row`. Fills
/// `accepted_rows` with the root row 0 followed by the accepted nodes' rows and returns the generated tokens (the
/// accepted nodes' tokens plus the main model's next token); stops early after a token `is_stop` accepts.
template <typename GreedyToken, typename IsStop>
std::vector<int64_t> accept_greedy(const DraftTree& tree,
                                   GreedyToken&& greedy_token,
                                   IsStop&& is_stop,
                                   std::vector<size_t>& accepted_rows) {
    accepted_rows.assign(1, 0);
    std::vector<int64_t> generated;
    size_t row = 0;
    while (true) {
        const int64_t token = greedy_token(row);
        generated.push_back(token);
        if (is_stop(token)) {
            break;
        }
        // Children of `row`: nodes whose parent is that row's node (row 0 = root = parent -1).
        const int64_t parent = static_cast<int64_t>(row) - 1;
        size_t next = 0;
        for (size_t node = row; node < tree.tokens.size(); ++node) {
            if (tree.parents[node] == parent && tree.tokens[node] == token) {
                next = node + 1;
                break;
            }
        }
        if (next == 0) {
            break;
        }
        accepted_rows.push_back(next);
        row = next;
    }
    return generated;
}

}  // namespace drafter_utils

}  // namespace genai
}  // namespace ov
