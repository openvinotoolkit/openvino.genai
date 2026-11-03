// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "openvino/genai/generation_config.hpp"
#include "openvino/runtime/infer_request.hpp"
#include "openvino/runtime/tensor.hpp"
#include "sampling/block_row_source.hpp"
#include "speculative_decoding/dflash_model_transforms.hpp"

namespace ov::genai::dflash_cb {

// True when the logit processors of a draft sampler with this config cannot change which token has the highest logit,
// so a greedy draft can take the best token of every row without the sampler. Follows the transforms that
// LogitProcessor adds for greedy decoding; the log-prob transforms keep the argmax.
inline bool logit_processors_keep_argmax(const GenerationConfig& config) {
    return config.is_greedy_decoding() && config.repetition_penalty == 1.0f && config.presence_penalty == 0.0f &&
           config.frequency_penalty == 0.0f && config.min_new_tokens == 0 && !config.is_structured_output_generation();
}

// The rows that the DFlash draft sampler draws its candidates from: the backbone logits scaled by output_multiplier and
// soft-capped, and, for a draft with a Markov head (DSpark, DFlash-2), rows that make every candidate depend on the one
// drafted before it. The Markov head runs once for the blocks of all requests: the Top-K candidate scorer scores every
// row in one inference, the dense Markov step one position of every block per inference.
//
// With the candidate scorer, a sampled request draws from the K candidates of every lattice row and a greedy request
// whose logit processors keep the argmax walks the lattice greedily. Any other request's sampler draws from the lattice
// rows spread over the vocabulary, and a token outside a row's candidates ends its block. Without a Markov head or with
// the dense Markov step, a greedy request whose logit processors keep the argmax takes the best token of every row.
class CandidateSelector final : public BlockRowSource {
public:
    // rt_info holds the Markov head computation, if any, that read_markov_head() selected.
    CandidateSelector(const utils::dflash::DFlashRTInfo& rt_info,
                      const std::string& device,
                      const ov::AnyMap& properties);

    bool has_markov_head() const {
        return m_candidate_scorer.has_value() || m_markov_step.has_value();
    }

    // A gated Markov head (DFlash-2) also scores the draft's last_hidden_state of every row.
    bool reads_hidden_states() const {
        return m_reads_hidden_states;
    }

    // The draft request whose last_hidden_state holds the hidden state of every logits row.
    void set_draft_request(ov::InferRequest draft_request) {
        m_draft_request = std::move(draft_request);
    }

    void prepare(const std::vector<SampledBlock>& blocks, const ov::Tensor& logits) override;

    bool is_position_major() const override {
        return m_markov_step.has_value();
    }

    void step(size_t position, const std::vector<size_t>& blocks, const std::vector<int64_t>& previous_tokens) override;

    std::optional<std::vector<BlockToken>> draw(size_t block,
                                                const GenerationConfig& config,
                                                std::mt19937& rng) const override;

    std::optional<BlockToken> draw_token(size_t block, size_t position, const GenerationConfig& config) const override;

    bool fill_row(size_t block, size_t position, int64_t previous_token, float* row) const override;

    // Microseconds the Markov head ran since the last call. The head runs during sampling, after the draft's
    // inference timer stops, so the draft adds this to its inference duration.
    float take_inference_duration();

private:
    void score_candidates();
    void infer(ov::InferRequest& request);
    const float* backbone_logits(size_t block, size_t position) const;
    float scaled_logit(float logit) const;
    const float* markov_bias(size_t block) const;

    bool m_reads_hidden_states = false;
    // the backbone logits are scaled by output_multiplier, then soft-capped
    float m_output_multiplier = 1.0f;
    float m_final_logit_softcapping = 0.0f;
    // at most one of them, for a draft with a Markov head
    std::optional<ov::InferRequest> m_candidate_scorer;
    std::optional<ov::InferRequest> m_markov_step;
    std::optional<ov::InferRequest> m_draft_request;

    std::vector<SampledBlock> m_blocks;
    ov::Tensor m_logits;
    size_t m_vocab_size = 0;
    // the candidate scorer's lattice over the rows of all blocks packed back to back
    ov::Tensor m_edge_scores;
    ov::Tensor m_candidate_ids;
    std::vector<size_t> m_lattice_begins;
    // the dense step's Markov bias at the current position, the row of each block that draws a token there
    ov::Tensor m_markov_bias;
    std::vector<size_t> m_markov_bias_rows;
    float m_inference_duration = 0.0f;
};

}  // namespace ov::genai::dflash_cb
