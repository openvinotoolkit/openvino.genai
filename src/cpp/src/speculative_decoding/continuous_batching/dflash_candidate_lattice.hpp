// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <vector>

#include "openvino/core/except.hpp"
#include "openvino/runtime/tensor.hpp"
#include "sampling/block_row_source.hpp"
#include "sampling/draft_proposal.hpp"

namespace ov::genai::dflash_cb {

// The candidate scorer takes the rows of num_blocks draft blocks packed back to back: block n spans rows
// subsequence_begins[n] to subsequence_begins[n + 1]. The graph marks the first row of every block and gives it the
// block's anchor as predecessor, so an empty block would mark a row twice and shift the anchors of later blocks.
inline void validate_subsequence_begins(const ov::Tensor& subsequence_begins, size_t num_blocks, size_t num_rows) {
    OPENVINO_ASSERT(num_blocks > 0, "The candidate scorer needs at least one draft block.");
    OPENVINO_ASSERT(subsequence_begins && subsequence_begins.get_element_type() == ov::element::i64 &&
                        subsequence_begins.get_shape() == ov::Shape{num_blocks + 1},
                    "The candidate scorer subsequence_begins must be an I64 tensor of shape [",
                    num_blocks + 1,
                    "].");
    const auto* begins = subsequence_begins.data<const int64_t>();
    OPENVINO_ASSERT(begins[0] == 0 && begins[num_blocks] == static_cast<int64_t>(num_rows),
                    "The candidate scorer subsequence_begins must start at 0 and end at the number of draft rows ",
                    num_rows,
                    ".");
    for (size_t block = 0; block < num_blocks; ++block) {
        OPENVINO_ASSERT(begins[block] < begins[block + 1],
                        "The candidate scorer subsequence_begins must be strictly increasing: draft block ",
                        block,
                        " is empty.");
    }
}

// The candidate scorer returns a Top-K lattice over the packed rows of all draft blocks:
//   candidate_ids [rows, K]: the backbone's K best tokens at every row;
//   edge_scores [rows, K, K]: edge_scores[s, i, j] scores candidate j at row s after candidate i at row s - 1.
// The first row of a block follows the block's anchor instead, so edge_scores[s, i, :] is the same for every i there.
inline void validate_candidate_lattice(const ov::Tensor& edge_scores,
                                       const ov::Tensor& candidate_ids,
                                       size_t num_positions,
                                       size_t vocab_size) {
    OPENVINO_ASSERT(candidate_ids && candidate_ids.get_element_type() == ov::element::i64 &&
                        candidate_ids.get_shape().size() == 2 && candidate_ids.get_shape()[0] == num_positions &&
                        candidate_ids.get_shape()[1] > 0,
                    "Candidate scorer candidate_ids must be an I64 tensor of shape [",
                    num_positions,
                    ", K].");
    const size_t top_k = candidate_ids.get_shape()[1];
    OPENVINO_ASSERT(edge_scores && edge_scores.get_element_type() == ov::element::f32 &&
                        edge_scores.get_shape() == ov::Shape({num_positions, top_k, top_k}),
                    "Candidate scorer edge_scores must be an FP32 tensor of shape [",
                    num_positions,
                    ", ",
                    top_k,
                    ", ",
                    top_k,
                    "].");
    const auto* ids = candidate_ids.data<const int64_t>();
    OPENVINO_ASSERT(std::all_of(ids,
                                ids + candidate_ids.get_size(),
                                [&](int64_t id) {
                                    return id >= 0 && static_cast<size_t>(id) < vocab_size;
                                }),
                    "Candidate scorer candidate_ids must be draft vocabulary ids.");
}

// Walks the lattice rows [first_row, first_row + num_positions) of one block greedily: at every position, the best
// candidate in the row of the candidate drafted before it. Ties go to the smaller token id, as in a greedy scan over
// the vocabulary. A row's scores depend on that row and its predecessor only, so a walk over the leading positions of
// a block drafts the leading tokens of a walk over the whole block.
inline std::vector<int64_t> greedy_candidate_path(const ov::Tensor& edge_scores,
                                                  const ov::Tensor& candidate_ids,
                                                  size_t first_row,
                                                  size_t num_positions) {
    const size_t top_k = candidate_ids.get_shape()[1];
    OPENVINO_ASSERT(first_row + num_positions <= candidate_ids.get_shape()[0],
                    "The candidate path exceeds the rows of the candidate lattice.");
    const auto* scores = edge_scores.data<const float>();
    std::vector<int64_t> path;
    path.reserve(num_positions);
    size_t previous_index = 0;
    for (size_t position = first_row; position < first_row + num_positions; ++position) {
        const auto* row = scores + (position * top_k + previous_index) * top_k;
        const auto* ids = candidate_ids.data<const int64_t>() + position * top_k;
        size_t best = 0;
        for (size_t j = 1; j < top_k; ++j) {
            if (row[j] > row[best] || (row[j] == row[best] && ids[j] < ids[best])) {
                best = j;
            }
        }
        previous_index = best;
        path.push_back(ids[best]);
    }
    return path;
}

// Tokens outside the candidates are shifted at least this far below the worst candidate.
inline constexpr float CANDIDATE_ROW_GAP = 1.0e4f;

// Spreads the lattice row at `position` that follows candidate previous_index at position - 1 over the vocabulary, so
// the request's logit processors and sampler pick the token as they do from a full logits row. Candidate j gets
// edge_scores[position, previous_index, j]. Every other token keeps its backbone logit, shifted below the candidates
// by at least CANDIDATE_ROW_GAP: a candidate still wins after the usual penalties unless the processors mask all K of
// them, and then the token comes from the remaining backbone logits. A repetition_penalty far below 1 can lift a
// non-candidate above the candidates, as it would with the dense Markov step, and the walk then ends the block at
// that token.
inline void fill_candidate_row(const float* backbone_logits,
                               size_t vocab_size,
                               const ov::Tensor& edge_scores,
                               const ov::Tensor& candidate_ids,
                               size_t position,
                               size_t previous_index,
                               float* row) {
    const size_t top_k = candidate_ids.get_shape()[1];
    const auto* ids = candidate_ids.data<const int64_t>() + position * top_k;
    const auto* scores = edge_scores.data<const float>() + (position * top_k + previous_index) * top_k;
    // The candidates are the backbone's Top-K at this position, so its best logit is among them.
    float best_logit = -std::numeric_limits<float>::infinity();
    float worst_score = std::numeric_limits<float>::infinity();
    for (size_t j = 0; j < top_k; ++j) {
        best_logit = std::max(best_logit, backbone_logits[ids[j]]);
        worst_score = std::min(worst_score, scores[j]);
    }
    const float shift = worst_score - CANDIDATE_ROW_GAP - best_logit;
    std::transform(backbone_logits, backbone_logits + vocab_size, row, [shift](float logit) {
        return logit + shift;
    });
    for (size_t j = 0; j < top_k; ++j) {
        row[ids[j]] = scores[j];
    }
}

inline std::optional<size_t> find_candidate(const ov::Tensor& candidate_ids, size_t position, int64_t token_id) {
    const size_t top_k = candidate_ids.get_shape()[1];
    const auto* ids = candidate_ids.data<const int64_t>() + position * top_k;
    const auto* found = std::find(ids, ids + top_k, token_id);
    if (found == ids + top_k) {
        return std::nullopt;
    }
    return static_cast<size_t>(std::distance(ids, found));
}

// Draws the candidates of the lattice rows [first_row, first_row + num_positions) of one block: every position from
// the softmax, at `temperature`, of the row that follows the candidate drafted before it. Each candidate carries that
// distribution over the row's K candidates, which the target verifies it against.
inline std::vector<BlockToken> sample_candidate_path(const ov::Tensor& edge_scores,
                                                     const ov::Tensor& candidate_ids,
                                                     size_t first_row,
                                                     size_t num_positions,
                                                     float temperature,
                                                     std::mt19937& rng) {
    OPENVINO_ASSERT(temperature > 0.0f, "Sampled DFlash drafts with a Markov head require temperature > 0.");
    OPENVINO_ASSERT(first_row + num_positions <= candidate_ids.get_shape()[0],
                    "The candidate path exceeds the rows of the candidate lattice.");
    const size_t top_k = candidate_ids.get_shape()[1];
    const auto* scores = edge_scores.data<const float>();
    const auto* ids = candidate_ids.data<const int64_t>();
    std::vector<BlockToken> candidates;
    candidates.reserve(num_positions);
    size_t previous_index = 0;
    for (size_t position = first_row; position < first_row + num_positions; ++position) {
        const auto* row = scores + (position * top_k + previous_index) * top_k;
        const float max_score = *std::max_element(row, row + top_k);
        DraftProposal proposal;
        proposal.token_ids.assign(ids + position * top_k, ids + (position + 1) * top_k);
        proposal.probabilities.resize(top_k);
        // In float, each weight of a long tail of low scores would round away against the leading 1.
        double probability_sum = 0.0;
        for (size_t index = 0; index < top_k; ++index) {
            proposal.probabilities[index] = std::exp((row[index] - max_score) / temperature);
            probability_sum += proposal.probabilities[index];
        }
        OPENVINO_ASSERT(probability_sum > 0.0 && std::isfinite(probability_sum),
                        "The DFlash candidate lattice produced an invalid proposal distribution.");
        for (auto& probability : proposal.probabilities) {
            probability = static_cast<float>(probability / probability_sum);
        }
        std::discrete_distribution<size_t> distribution(proposal.probabilities.begin(), proposal.probabilities.end());
        const size_t selected = distribution(rng);
        candidates.push_back({proposal.token_ids[selected],
                              std::log(proposal.probabilities[selected]),
                              std::make_shared<const DraftProposal>(std::move(proposal))});
        previous_index = selected;
    }
    return candidates;
}

}  // namespace ov::genai::dflash_cb
