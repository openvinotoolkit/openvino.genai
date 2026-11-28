// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>

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

/// True when `attention_mask` (i64, shaped like the input ids) keeps every token, or isn't set.
bool attends_to_all_tokens(const ov::Tensor& attention_mask, const ov::Shape& input_shape);

/// Greedy acceptance walk. `greedy_token(row)` returns the main model's token after verifier row `row`. Fills
/// `accepted_rows` with the root row 0 followed by the accepted nodes' rows and returns the generated tokens (the
/// accepted nodes' tokens plus the main model's next token); a path ends after a token `is_stop` accepts. Siblings may
/// share a token, so every matching node is followed and the longest path is kept.
template <typename GreedyToken, typename IsStop>
std::vector<int64_t> accept_greedy(const DraftTree& tree,
                                   GreedyToken&& greedy_token,
                                   IsStop&& is_stop,
                                   std::vector<size_t>& accepted_rows) {
    const size_t rows = tree.tokens.size() + 1;
    // The main model's token after every row an accepted path reaches; the other rows aren't evaluated.
    std::vector<std::optional<int64_t>> greedy(rows);
    std::vector<size_t> depth(rows, 0);
    greedy[0] = greedy_token(0);
    size_t last = 0;  // the last row of the longest path
    // Row i + 1 holds node i, whose parent comes before it, so one pass in row order follows every accepted path.
    for (size_t row = 1; row < rows; ++row) {
        const auto parent = static_cast<size_t>(tree.parents[row - 1] + 1);
        if (!greedy[parent] || is_stop(*greedy[parent]) || tree.tokens[row - 1] != *greedy[parent]) {
            continue;
        }
        greedy[row] = greedy_token(row);
        depth[row] = depth[parent] + 1;
        if (depth[row] > depth[last]) {
            last = row;
        }
    }
    const size_t length = depth[last] + 1;
    accepted_rows.resize(length);
    std::vector<int64_t> generated(length);
    for (size_t i = length, row = last; i-- > 0;) {
        accepted_rows[i] = row;
        generated[i] = *greedy[row];
        if (row > 0) {
            row = static_cast<size_t>(tree.parents[row - 1] + 1);
        }
    }
    return generated;
}

}  // namespace drafter_utils

}  // namespace genai
}  // namespace ov
