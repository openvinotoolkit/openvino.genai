// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "openvino/core/any.hpp"
#include "openvino/genai/generation_config.hpp"
#include "openvino/genai/visibility.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov {
namespace genai {

/**
 * @brief Candidate tokens proposed by a Drafter for one verification step, as a tree.
 *
 * Node i holds token `tokens[i]`; `parents[i]` is -1 when its parent is the root (the last generated token) or the
 * index of an earlier node (`parents[i] < i`). A chain of k guesses is tokens {t1, ..., tk} with parents
 * {-1, 0, ..., k - 2}.
 */
struct OPENVINO_GENAI_EXPORTS DraftTree {
    std::vector<int64_t> tokens;
    std::vector<int64_t> parents;
};

/**
 * @brief User-defined drafter for speculative decoding.
 *
 * At each step the pipeline asks the drafter for a tree of candidate tokens following the last generated token,
 * verifies the whole tree with the main model in one inference (tree attention) and keeps the longest path the main
 * model agrees with (greedy decoding), plus the main model's next token. Every token that enters the main model's KV
 * cache is reported back to the drafter: first the prompt, then the accepted tokens of every step, together with the
 * main model's `last_hidden_state` output rows for them when the model has that output, so a drafter can condition on
 * the main model's features.
 *
 * Supported by LLMPipeline with the main model on NPU, greedy decoding and batch size 1:
 * @code
 * ov::genai::LLMPipeline pipe(models_path, "NPU", ov::genai::drafter(std::make_shared<MyDrafter>()));
 * @endcode
 * GenerationConfig::num_assistant_tokens limits the tree size of every step (at most max_draft_tokens(), which is
 * the default); 0 disables drafting. Beam search, repetition / presence / frequency penalties, min_new_tokens,
 * stop_strings, logprobs, echo, structured output, LoRA adapters, attention masks that exclude prompt tokens and main
 * models with linear attention states aren't supported. Tensors passed to the drafter are only valid during the call.
 */
class OPENVINO_GENAI_EXPORTS Drafter {
public:
    virtual ~Drafter();

    /// @brief Largest tree a proposal may have. The main model is compiled to verify max_draft_tokens() + 1 tokens.
    virtual size_t max_draft_tokens() const = 0;

    /**
     * @brief Called once per generate() call, before the main model processes the prompt (the prompt's tokens and
     * features follow in accept()).
     * @param prompt Prompt token ids.
     * @param config Generation config of this generate() call.
     */
    virtual void start(const std::vector<int64_t>& prompt, const GenerationConfig& config) = 0;

    /**
     * @brief Proposes candidates for the tokens that follow `root`.
     * @param root Last generated token, not yet processed by the main model.
     * @param position Position of `root` in the sequence.
     * @param max_nodes Largest tree the pipeline verifies in this step.
     * @return Candidate tree with at most max_nodes nodes; an empty tree runs a regular decoding step.
     */
    virtual DraftTree propose(int64_t root, int64_t position, size_t max_nodes) = 0;

    /**
     * @brief Reports tokens that entered the main model's KV cache: first the prompt, once the main model processed
     * it, then after every verification step the root followed by the accepted nodes' tokens (the main model's next
     * token, the next root, follows them).
     * @param tokens Token ids.
     * @param positions Positions of `tokens`.
     * @param features Main model `last_hidden_state` rows for `tokens`, [tokens size, hidden size]; an empty tensor
     * when the main model has no such output.
     */
    virtual void accept(const std::vector<int64_t>& tokens,
                        const std::vector<int64_t>& positions,
                        const ov::Tensor& features) = 0;

    /// @brief Called when a generate() call ends (also when it stops early or fails). When the generation fails, the
    /// generation's error is reported and an error from finish() is ignored.
    virtual void finish() {}
};

/**
 * @brief Property to run LLMPipeline speculative decoding with a user-defined drafter.
 */
OPENVINO_GENAI_EXPORTS std::pair<std::string, ov::Any> drafter(std::shared_ptr<Drafter> drafter);

}  // namespace genai
}  // namespace ov
