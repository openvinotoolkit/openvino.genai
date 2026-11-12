// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

#include "openvino/genai/generation_config.hpp"
#include "openvino/runtime/tensor.hpp"

namespace ov::genai {

// One request's block in a block sampling step. The logits hold the gathered rows of all blocks one block after
// another.
struct SampledBlock {
    // the token before the first candidate, the seed
    int64_t anchor_token = 0;
    size_t first_row = 0;
    size_t num_rows = 0;
    // the leading rows the sampler draws tokens from, at most num_rows
    size_t num_candidates = 0;
};

struct BlockToken {
    int64_t token = 0;
    float log_prob = 0.0f;
};

// Gives block sampling the logits row each token of a block is drawn from, so that a draft can make every token
// depend on the token drawn before it (a Markov head). The sampler calls prepare() once per step with the blocks of
// all requests, so the source can run a model once for all of them, and then draws the tokens of each block in order:
// - by request: draw(), or fill_row() for one position after another, in parallel across blocks;
// - by position when is_position_major(): step() for the blocks that draw a token at the position, then draw_token(),
//   or fill_row(), for each of them in parallel.
// The sampler applies the request's logit processors to a filled row and samples it as a logits row.
class BlockRowSource {
public:
    virtual ~BlockRowSource() = default;

    // logits [rows, 1, vocab] hold the rows of every block from block.first_row.
    virtual void prepare(const std::vector<SampledBlock>& blocks, const ov::Tensor& logits) = 0;

    virtual bool is_position_major() const {
        return false;
    }

    // Position-major only: runs before the fill_row() calls of `position`. blocks are the indices into the blocks of
    // prepare() that draw a token at the position, previous_tokens the tokens they drew at the position before.
    virtual void step(size_t position, const std::vector<size_t>& blocks, const std::vector<int64_t>& previous_tokens) {
    }

    // Request-major only: the tokens of a whole block when the request's config allows a faster draw than one row
    // after another; nullopt draws the block with fill_row().
    virtual std::optional<std::vector<BlockToken>> draw(size_t block, const GenerationConfig& config) const {
        return std::nullopt;
    }

    // Position-major only: the token of the block's `position`, after the token step() received for the block, when
    // the request's config allows a faster draw than sampling the row; nullopt draws it from fill_row(). Called
    // concurrently for different blocks.
    virtual std::optional<BlockToken> draw_token(size_t block, size_t position, const GenerationConfig& config) const {
        return std::nullopt;
    }

    // Writes the vocabulary row of the block's `position` that follows previous_token, the anchor at position 0, or
    // returns false to end the block before the position. Called concurrently for different blocks.
    virtual bool fill_row(size_t block, size_t position, int64_t previous_token, float* row) const = 0;
};

}  // namespace ov::genai
