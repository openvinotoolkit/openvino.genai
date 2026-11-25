// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "openvino/core/type/float16.hpp"
#include "openvino/op/parameter.hpp"
#include "speculative_decoding/stateful/drafter_strategy.hpp"

using ov::genai::DraftTree;
namespace drafter_utils = ov::genai::drafter_utils;

namespace {

auto never_stop = [](int64_t) {
    return false;
};

class IdleDrafter : public ov::genai::Drafter {
public:
    size_t max_draft_tokens() const override {
        return 1;
    }
    void start(const std::vector<int64_t>&, const ov::genai::GenerationConfig&) override {}
    DraftTree propose(int64_t, int64_t, size_t) override {
        return {};
    }
    void accept(const std::vector<int64_t>&, const std::vector<int64_t>&, const ov::Tensor&) override {}
};

}  // namespace

TEST(DrafterUtils, TreeDepths) {
    // root -> 10 -> 11 -> 13, root -> 12
    const DraftTree tree{{10, 11, 12, 13}, {-1, 0, -1, 1}};
    EXPECT_EQ(drafter_utils::tree_depths(tree, 4), (std::vector<size_t>{1, 2, 1, 3}));
    EXPECT_TRUE(drafter_utils::tree_depths(DraftTree{}, 0).empty());
}

TEST(DrafterUtils, TreeDepthsRejectsInvalidTrees) {
    EXPECT_THROW(drafter_utils::tree_depths(DraftTree{{1, 2}, {-1}}, 4), ov::Exception);
    EXPECT_THROW(drafter_utils::tree_depths(DraftTree{{1, 2}, {-1, 1}}, 4), ov::Exception);
    EXPECT_THROW(drafter_utils::tree_depths(DraftTree{{1, 2}, {-1, -2}}, 4), ov::Exception);
    EXPECT_THROW(drafter_utils::tree_depths(DraftTree{{1, 2, 3}, {-1, 0, 1}}, 2), ov::Exception);
}

TEST(DrafterUtils, TreeMaskShowsEachRowItsAncestors) {
    // rows: 0 = root, 1 = node 0 (child of root), 2 = node 1 (child of root), 3 = node 2 (child of node 0)
    const DraftTree tree{{1, 2, 3}, {-1, -1, 0}};
    const float hidden = -std::numeric_limits<float>::infinity();
    const std::vector<std::vector<float>> expected = {
        {0, hidden, hidden, hidden},
        {0, 0, hidden, hidden},
        {0, hidden, 0, hidden},
        {0, 0, hidden, 0},
    };

    const ov::Tensor mask = drafter_utils::make_tree_mask(tree, ov::element::f32);
    ASSERT_EQ(mask.get_shape(), (ov::Shape{1, 1, 4, 4}));
    const ov::Tensor mask_f16 = drafter_utils::make_tree_mask(tree, ov::element::f16);
    const float* data = mask.data<const float>();
    const ov::float16* data_f16 = mask_f16.data<const ov::float16>();
    for (size_t row = 0; row < expected.size(); ++row) {
        for (size_t col = 0; col < expected[row].size(); ++col) {
            EXPECT_EQ(data[row * 4 + col], expected[row][col]) << "row " << row << ", column " << col;
            EXPECT_EQ(static_cast<float>(data_f16[row * 4 + col]), expected[row][col])
                << "row " << row << ", column " << col;
        }
    }
}

TEST(DrafterUtils, AcceptGreedyFollowsTheMatchingBranch) {
    // root -> 5 (row 1) -> 7 (row 3); root -> 6 (row 2) -> 8 (row 4)
    const DraftTree tree{{5, 6, 7, 8}, {-1, -1, 0, 1}};
    const std::map<size_t, int64_t> greedy = {{0, 6}, {2, 8}, {4, 42}};
    std::vector<size_t> rows;
    const auto generated = drafter_utils::accept_greedy(
        tree,
        [&](size_t row) {
            return greedy.at(row);
        },
        never_stop,
        rows);
    EXPECT_EQ(generated, (std::vector<int64_t>{6, 8, 42}));
    EXPECT_EQ(rows, (std::vector<size_t>{0, 2, 4}));
}

TEST(DrafterUtils, AcceptGreedyKeepsTheLongestPathAcrossSiblingsWithTheSameToken) {
    // root -> 5 (row 1); root -> 5 (row 2) -> 7 (row 3) -> 8 (row 4)
    const DraftTree tree{{5, 5, 7, 8}, {-1, -1, 1, 2}};
    const std::map<size_t, int64_t> greedy = {{0, 5}, {1, 7}, {2, 7}, {3, 8}, {4, 9}};
    std::vector<size_t> rows;
    const auto generated = drafter_utils::accept_greedy(
        tree,
        [&](size_t row) {
            return greedy.at(row);
        },
        never_stop,
        rows);
    EXPECT_EQ(generated, (std::vector<int64_t>{5, 7, 8, 9}));
    EXPECT_EQ(rows, (std::vector<size_t>{0, 2, 3, 4}));
}

TEST(DrafterUtils, AcceptGreedyRejectsAMismatch) {
    const DraftTree tree{{5, 7}, {-1, 0}};
    std::vector<size_t> rows;
    const auto generated = drafter_utils::accept_greedy(
        tree,
        [](size_t) {
            return int64_t{9};
        },
        never_stop,
        rows);
    EXPECT_EQ(generated, (std::vector<int64_t>{9}));
    EXPECT_EQ(rows, (std::vector<size_t>{0}));
}

TEST(DrafterUtils, AcceptGreedyStopsAtStopToken) {
    const DraftTree tree{{5, 7}, {-1, 0}};
    std::vector<size_t> rows;
    const auto generated = drafter_utils::accept_greedy(
        tree,
        [](size_t row) {
            return row == 0 ? int64_t{5} : int64_t{7};
        },
        [](int64_t token) {
            return token == 5;
        },
        rows);
    EXPECT_EQ(generated, (std::vector<int64_t>{5}));
    EXPECT_EQ(rows, (std::vector<size_t>{0}));
}

TEST(DrafterUtils, AcceptGreedyWithEmptyTree) {
    std::vector<size_t> rows;
    const auto generated = drafter_utils::accept_greedy(
        DraftTree{},
        [](size_t) {
            return int64_t{3};
        },
        never_stop,
        rows);
    EXPECT_EQ(generated, (std::vector<int64_t>{3}));
    EXPECT_EQ(rows, (std::vector<size_t>{0}));
}

TEST(DrafterUtils, AttentionMaskMustKeepEveryToken) {
    const ov::Shape shape{1, 3};
    ov::Tensor ones(ov::element::i64, shape);
    std::fill_n(ones.data<int64_t>(), 3, int64_t{1});
    ov::Tensor padded(ov::element::i64, shape);
    const int64_t padded_values[] = {0, 1, 1};
    std::copy_n(padded_values, 3, padded.data<int64_t>());

    EXPECT_TRUE(drafter_utils::attends_to_all_tokens(ones, shape));
    EXPECT_TRUE(drafter_utils::attends_to_all_tokens(ov::Tensor{}, shape));
    EXPECT_FALSE(drafter_utils::attends_to_all_tokens(padded, shape));
    EXPECT_FALSE(drafter_utils::attends_to_all_tokens(ones, ov::Shape{1, 4}));
}

TEST(DrafterUtils, DrafterPropertyRejectsNull) {
    EXPECT_THROW(ov::genai::drafter(nullptr), ov::Exception);
}

TEST(DrafterPipeline, RejectsMainModelWithLinearAttentionStates) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::Shape{1});
    auto model = std::make_shared<ov::Model>(ov::OutputVector{input}, ov::ParameterVector{input});
    model->set_rt_info(std::vector<std::string>{"recurrent_state"}, "gguf_recurrent_states");
    const ov::genai::ModelDesc main_model(model, ov::genai::Tokenizer{}, "NPU");
    // The check runs before the main model is compiled, so no NPU is needed.
    try {
        ov::genai::StatefulDrafterLLMPipeline pipeline(main_model, std::make_shared<IdleDrafter>());
        FAIL() << "a main model with linear attention states was accepted";
    } catch (const ov::Exception& error) {
        EXPECT_NE(std::string(error.what()).find("linear attention states"), std::string::npos) << error.what();
    }
}
