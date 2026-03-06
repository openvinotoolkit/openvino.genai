// Copyright (C) 2024-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Unit tests for EagleCandidateGraph.
// These tests exercise the standalone tree data structure used by TreeSearcher
// without requiring a live model or KV-cache, making them fast and hermetic.

#include <gtest/gtest.h>
#include "sampling/sampler.hpp"

using namespace ov::genai;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Collect node_ids from a vector of Nodes.
static std::vector<uint64_t> node_ids(const std::vector<EagleCandidateGraph::Node>& nodes) {
    std::vector<uint64_t> ids;
    ids.reserve(nodes.size());
    for (const auto& n : nodes)
        ids.push_back(n.node_id);
    return ids;
}

// ---------------------------------------------------------------------------
// Constructor / root
// ---------------------------------------------------------------------------

TEST(EagleCandidateGraphTest, RootIsAlwaysPresent) {
    EagleCandidateGraph g(42, 0.5f, 4, 3);
    const auto top = g.get_top_k_nodes();
    ASSERT_FALSE(top.empty());
    EXPECT_EQ(top[0].node_id, 0u);
    EXPECT_EQ(top[0].token_id, 42);
    EXPECT_FLOAT_EQ(top[0].score, 0.5f);
    EXPECT_EQ(top[0].tree_layer, 0);
}

TEST(EagleCandidateGraphTest, EmptyBudgetStillReturnsRoot) {
    // max_tokens = 0: budget for non-root nodes is 0, but root must still be returned.
    EagleCandidateGraph g(0, 0.0f, /*max_tokens=*/0, /*max_depth=*/2);
    const auto top = g.get_top_k_nodes();
    ASSERT_EQ(top.size(), 1u);
    EXPECT_EQ(top[0].node_id, 0u);
}

// ---------------------------------------------------------------------------
// add_node
// ---------------------------------------------------------------------------

TEST(EagleCandidateGraphTest, AddNodeReturnsSequentialIds) {
    EagleCandidateGraph g(0, 0.0f, 10, 3);
    const uint64_t id1 = g.add_node(10, -1.0f, /*parent=*/0u);
    const uint64_t id2 = g.add_node(20, -2.0f, /*parent=*/0u);
    EXPECT_EQ(id1, 1u);
    EXPECT_EQ(id2, 2u);
}

TEST(EagleCandidateGraphTest, AddNodeBeyondMaxDepthReturnsZero) {
    // max_depth = 1: children of root are at layer 1 (allowed),
    // grandchildren would be at layer 2 (beyond limit).
    EagleCandidateGraph g(0, 0.0f, 10, /*max_depth=*/1);
    const uint64_t child = g.add_node(1, -1.0f, 0u);
    ASSERT_NE(child, 0u) << "First-level child should be accepted";
    const uint64_t grand_child = g.add_node(2, -2.0f, child);
    EXPECT_EQ(grand_child, 0u) << "Second-level child must be rejected when max_depth=1";
}

TEST(EagleCandidateGraphTest, AddNodeStoresCorrectMetadata) {
    EagleCandidateGraph g(0, 0.0f, 10, 5);
    const uint64_t child_id = g.add_node(99, -0.5f, 0u);
    ASSERT_NE(child_id, 0u);

    // The node should appear in get_top_k_nodes with the expected fields.
    const auto top = g.get_top_k_nodes();
    const auto it = std::find_if(top.begin(), top.end(),
        [child_id](const EagleCandidateGraph::Node& n) { return n.node_id == child_id; });
    ASSERT_NE(it, top.end()) << "Child node not found in top-k result";
    EXPECT_EQ(it->token_id, 99);
    EXPECT_FLOAT_EQ(it->score, -0.5f);
    EXPECT_EQ(it->tree_layer, 1);
}

// ---------------------------------------------------------------------------
// get_top_k_nodes
// ---------------------------------------------------------------------------

TEST(EagleCandidateGraphTest, TopKRespectsBudget) {
    // Build a tree with 5 children under root; budget = 3.
    EagleCandidateGraph g(0, 0.0f, /*max_tokens=*/3, /*max_depth=*/2);
    for (int i = 0; i < 5; ++i)
        g.add_node(i, static_cast<float>(-i), 0u);  // scores: 0, -1, -2, -3, -4

    const auto top = g.get_top_k_nodes();
    // Root + 3 non-root nodes = 4 total
    EXPECT_EQ(top.size(), 4u);

    // Verify that the 3 highest-scoring non-root nodes (scores 0, -1, -2) are selected.
    for (const auto& n : top) {
        if (n.node_id == 0u) continue;  // skip root
        EXPECT_GE(n.score, -2.0f) << "Low-scoring node should have been evicted";
    }
}

TEST(EagleCandidateGraphTest, TopKSortedByLayer) {
    EagleCandidateGraph g(0, 0.0f, 10, 3);
    const uint64_t c1 = g.add_node(1, -0.5f, 0u);   // layer 1
    const uint64_t c2 = g.add_node(2, -1.0f, 0u);   // layer 1
    g.add_node(3, -0.3f, c1);                         // layer 2
    g.add_node(4, -0.4f, c2);                         // layer 2

    const auto top = g.get_top_k_nodes();
    for (size_t i = 1; i < top.size(); ++i)
        EXPECT_LE(top[i - 1].tree_layer, top[i].tree_layer)
            << "Nodes must be sorted by tree_layer";
}

TEST(EagleCandidateGraphTest, TopKExactBudget) {
    // Exactly max_tokens non-root nodes; all should be selected.
    EagleCandidateGraph g(0, 0.0f, /*max_tokens=*/3, /*max_depth=*/2);
    g.add_node(1, -1.0f, 0u);
    g.add_node(2, -2.0f, 0u);
    g.add_node(3, -3.0f, 0u);
    const auto top = g.get_top_k_nodes();
    EXPECT_EQ(top.size(), 4u);  // root + 3
}

// ---------------------------------------------------------------------------
// get_leaf_nodes
// ---------------------------------------------------------------------------

TEST(EagleCandidateGraphTest, LeafNodesOfFlatTree) {
    // Each child of root is a leaf when no grandchildren exist.
    EagleCandidateGraph g(0, 0.0f, 10, 3);
    g.add_node(1, -1.0f, 0u);
    g.add_node(2, -2.0f, 0u);
    g.add_node(3, -3.0f, 0u);

    const auto selected = g.get_top_k_nodes();
    const auto leaves   = g.get_leaf_nodes(selected);

    // Root is not a leaf (it has selected children).
    for (const auto& leaf : leaves)
        EXPECT_NE(leaf.node_id, 0u) << "Root should not be a leaf node";
    EXPECT_EQ(leaves.size(), 3u);
}

TEST(EagleCandidateGraphTest, LeafNodesWithGrandchildren) {
    EagleCandidateGraph g(0, 0.0f, 10, 3);
    const uint64_t c1 = g.add_node(1, -1.0f, 0u);
    const uint64_t c2 = g.add_node(2, -2.0f, 0u);
    const uint64_t g1 = g.add_node(3, -0.5f, c1);  // grandchild: score -0.5 (high)
    g.add_node(4, -0.6f, c2);                        // grandchild

    const auto selected = g.get_top_k_nodes();
    const auto leaves   = g.get_leaf_nodes(selected);

    // c1 and c2 each have a grandchild in selected, so they are not leaves.
    const auto leaf_ids = node_ids(leaves);
    EXPECT_EQ(std::find(leaf_ids.begin(), leaf_ids.end(), c1), leaf_ids.end())
        << "c1 has a selected child and must not be a leaf";
    EXPECT_EQ(std::find(leaf_ids.begin(), leaf_ids.end(), c2), leaf_ids.end())
        << "c2 has a selected child and must not be a leaf";

    // The grandchildren are leaves.
    const bool g1_is_leaf = std::find(leaf_ids.begin(), leaf_ids.end(), g1) != leaf_ids.end();
    EXPECT_TRUE(g1_is_leaf) << "Grandchild g1 should be a leaf node";
}

TEST(EagleCandidateGraphTest, LeafNodesWhenBudgetExcludesGrandchildren) {
    // Budget = 2: only root + 2 children fit; grandchildren are excluded.
    // When grandchildren are not in selected, their parents ARE leaves.
    EagleCandidateGraph g(0, 0.0f, /*max_tokens=*/2, /*max_depth=*/3);
    const uint64_t c1 = g.add_node(1, -1.0f, 0u);
    const uint64_t c2 = g.add_node(2, -2.0f, 0u);
    g.add_node(3, -0.1f, c1);  // grandchild, but budget is full; won't appear in selected

    const auto selected = g.get_top_k_nodes();
    ASSERT_EQ(selected.size(), 3u);  // root + c1 + c2 (or root + grandchild + one of c1/c2)

    const auto leaves = g.get_leaf_nodes(selected);
    const auto leaf_ids = node_ids(leaves);
    // Whatever is in selected without selected children is a leaf.
    // Important: no crash or infinite loop.
    EXPECT_FALSE(leaves.empty());
}

// ---------------------------------------------------------------------------
// get_path_to_node
// ---------------------------------------------------------------------------

TEST(EagleCandidateGraphTest, PathToRootIsSingleElement) {
    EagleCandidateGraph g(0, 0.0f, 10, 3);
    const auto path = g.get_path_to_node(0u);
    ASSERT_EQ(path.size(), 1u);
    EXPECT_EQ(path[0], 0u);
}

TEST(EagleCandidateGraphTest, PathToChildIsCorrect) {
    EagleCandidateGraph g(0, 0.0f, 10, 3);
    const uint64_t c = g.add_node(1, -1.0f, 0u);
    const auto path = g.get_path_to_node(c);
    ASSERT_EQ(path.size(), 2u);
    EXPECT_EQ(path[0], 0u);
    EXPECT_EQ(path[1], c);
}

TEST(EagleCandidateGraphTest, PathToGrandchildIsCorrect) {
    EagleCandidateGraph g(0, 0.0f, 10, 3);
    const uint64_t c  = g.add_node(1, -1.0f, 0u);
    const uint64_t gc = g.add_node(2, -2.0f, c);
    const auto path = g.get_path_to_node(gc);
    ASSERT_EQ(path.size(), 3u);
    EXPECT_EQ(path[0], 0u);
    EXPECT_EQ(path[1], c);
    EXPECT_EQ(path[2], gc);
}

TEST(EagleCandidateGraphTest, PathRootFirstLeafLast) {
    // Build a chain: root -> a -> b -> c
    EagleCandidateGraph g(0, 0.0f, 10, 4);
    const uint64_t a = g.add_node(1, -0.1f, 0u);
    const uint64_t b = g.add_node(2, -0.2f, a);
    const uint64_t c = g.add_node(3, -0.3f, b);
    const auto path = g.get_path_to_node(c);
    ASSERT_EQ(path.size(), 4u);
    EXPECT_EQ(path[0], 0u);
    EXPECT_EQ(path[1], a);
    EXPECT_EQ(path[2], b);
    EXPECT_EQ(path[3], c);
}

// ---------------------------------------------------------------------------
// is_ancestor
// ---------------------------------------------------------------------------

TEST(EagleCandidateGraphTest, RootIsAncestorOfItself) {
    EagleCandidateGraph g(0, 0.0f, 10, 3);
    EXPECT_TRUE(g.is_ancestor(0u, 0u));
}

TEST(EagleCandidateGraphTest, RootIsAncestorOfChild) {
    EagleCandidateGraph g(0, 0.0f, 10, 3);
    const uint64_t c = g.add_node(1, -1.0f, 0u);
    EXPECT_TRUE(g.is_ancestor(0u, c));
    EXPECT_TRUE(g.is_ancestor(c, c));   // node is its own ancestor (self-inclusive)
}

TEST(EagleCandidateGraphTest, ChildIsNotAncestorOfRoot) {
    EagleCandidateGraph g(0, 0.0f, 10, 3);
    const uint64_t c = g.add_node(1, -1.0f, 0u);
    EXPECT_FALSE(g.is_ancestor(c, 0u));
}

TEST(EagleCandidateGraphTest, SiblingIsNotAncestor) {
    EagleCandidateGraph g(0, 0.0f, 10, 3);
    const uint64_t c1 = g.add_node(1, -1.0f, 0u);
    const uint64_t c2 = g.add_node(2, -2.0f, 0u);
    EXPECT_FALSE(g.is_ancestor(c1, c2));
    EXPECT_FALSE(g.is_ancestor(c2, c1));
}

TEST(EagleCandidateGraphTest, AncestorChainCorrect) {
    // root -> a -> b -> c
    EagleCandidateGraph g(0, 0.0f, 10, 4);
    const uint64_t a = g.add_node(1, -0.1f, 0u);
    const uint64_t b = g.add_node(2, -0.2f, a);
    const uint64_t c = g.add_node(3, -0.3f, b);

    // All ancestors of c
    EXPECT_TRUE(g.is_ancestor(0u, c));
    EXPECT_TRUE(g.is_ancestor(a,  c));
    EXPECT_TRUE(g.is_ancestor(b,  c));
    EXPECT_TRUE(g.is_ancestor(c,  c));  // self

    // b is NOT an ancestor of a
    EXPECT_FALSE(g.is_ancestor(b, a));

    // c is NOT an ancestor of a or b
    EXPECT_FALSE(g.is_ancestor(c, a));
    EXPECT_FALSE(g.is_ancestor(c, b));
}

TEST(EagleCandidateGraphTest, AncestorQueryForNonExistentNodeReturnsFalse) {
    EagleCandidateGraph g(0, 0.0f, 10, 3);
    // node_id 99 was never added
    EXPECT_FALSE(g.is_ancestor(0u, 99u));
}

// ---------------------------------------------------------------------------
// Tree mask correctness (integration of get_top_k_nodes + is_ancestor)
// ---------------------------------------------------------------------------
// This reproduces exactly the tree-mask construction in finalize_tree(),
// so a bug there will show up here.

TEST(EagleCandidateGraphTest, TreeMaskDiagonalIsAllOnes) {
    // Every node is an ancestor of itself → all diagonal entries must be 1.
    EagleCandidateGraph g(0, 0.0f, 10, 3);
    const uint64_t c1 = g.add_node(1, -1.0f, 0u);
    const uint64_t c2 = g.add_node(2, -2.0f, 0u);
    g.add_node(3, -0.5f, c1);

    const auto selected = g.get_top_k_nodes();
    for (const auto& node : selected) {
        EXPECT_TRUE(g.is_ancestor(node.node_id, node.node_id))
            << "Node " << node.node_id << " must be its own ancestor";
    }
}

TEST(EagleCandidateGraphTest, TreeMaskLinearChain) {
    // root -> a -> b
    // Expected mask (row i = node i, col j = node j, 1 iff j is ancestor of i):
    //      root  a   b
    // root [ 1,  0,  0 ]
    //    a [ 1,  1,  0 ]
    //    b [ 1,  1,  1 ]
    EagleCandidateGraph g(0, 0.0f, 10, 3);
    const uint64_t a = g.add_node(10, -0.5f, 0u);
    const uint64_t b = g.add_node(20, -0.3f, a);

    const auto selected = g.get_top_k_nodes();
    // get_top_k_nodes sorts by layer, so order should be: root(0), a(1), b(2)
    ASSERT_EQ(selected.size(), 3u);
    EXPECT_EQ(selected[0].node_id, 0u);
    EXPECT_EQ(selected[1].node_id, a);
    EXPECT_EQ(selected[2].node_id, b);

    // Build mask as in finalize_tree
    std::vector<std::vector<uint8_t>> mask;
    mask.reserve(selected.size());
    for (const auto& node : selected) {
        std::vector<uint8_t> row;
        row.reserve(selected.size());
        for (const auto& other : selected)
            row.push_back(g.is_ancestor(other.node_id, node.node_id) ? 1u : 0u);
        mask.push_back(row);
    }

    // row 0 (root): only root itself
    EXPECT_EQ(mask[0][0], 1u);
    EXPECT_EQ(mask[0][1], 0u);
    EXPECT_EQ(mask[0][2], 0u);

    // row 1 (a): root and a
    EXPECT_EQ(mask[1][0], 1u);
    EXPECT_EQ(mask[1][1], 1u);
    EXPECT_EQ(mask[1][2], 0u);

    // row 2 (b): root, a, and b
    EXPECT_EQ(mask[2][0], 1u);
    EXPECT_EQ(mask[2][1], 1u);
    EXPECT_EQ(mask[2][2], 1u);
}

TEST(EagleCandidateGraphTest, TreeMaskTwoSiblings) {
    // root -> c1
    //      -> c2
    // c1 and c2 are siblings, not ancestors of each other.
    EagleCandidateGraph g(0, 0.0f, 10, 2);
    const uint64_t c1 = g.add_node(1, -1.0f, 0u);
    const uint64_t c2 = g.add_node(2, -2.0f, 0u);

    const auto selected = g.get_top_k_nodes();
    ASSERT_EQ(selected.size(), 3u);

    for (const auto& node : selected) {
        for (const auto& other : selected) {
            const bool expected = g.is_ancestor(other.node_id, node.node_id);
            if (node.node_id == c1 && other.node_id == c2)
                EXPECT_FALSE(expected) << "c2 is not an ancestor of c1";
            if (node.node_id == c2 && other.node_id == c1)
                EXPECT_FALSE(expected) << "c1 is not an ancestor of c2";
        }
    }
}

// ---------------------------------------------------------------------------
// retrieve_indices correctness (integration of get_leaf_nodes + get_path_to_node)
// ---------------------------------------------------------------------------
// Reproduces the retrieve_indices construction in finalize_tree().

TEST(EagleCandidateGraphTest, RetrieveIndicesLinearChain) {
    // root -> a -> b: single leaf b, path should map to indices [0, 1, 2]
    EagleCandidateGraph g(0, 0.0f, 10, 3);
    const uint64_t a = g.add_node(10, -0.5f, 0u);
    const uint64_t b = g.add_node(20, -0.3f, a);

    const auto selected = g.get_top_k_nodes();  // [root, a, b]
    std::unordered_map<uint64_t, size_t> nodeid_to_index;
    for (size_t i = 0; i < selected.size(); ++i)
        nodeid_to_index[selected[i].node_id] = i;

    const auto leaves = g.get_leaf_nodes(selected);
    ASSERT_EQ(leaves.size(), 1u);
    EXPECT_EQ(leaves[0].node_id, b);

    const auto raw_path = g.get_path_to_node(b);
    ASSERT_EQ(raw_path.size(), 3u);

    std::vector<int64_t> path;
    for (uint64_t node_id : raw_path)
        path.push_back(static_cast<int64_t>(nodeid_to_index.at(node_id)));

    EXPECT_EQ(path[0], 0);  // root at index 0
    EXPECT_EQ(path[1], 1);  // a at index 1
    EXPECT_EQ(path[2], 2);  // b at index 2
}

TEST(EagleCandidateGraphTest, RetrieveIndicesTwoLeaves) {
    // root -> c1 -> gc1
    //      -> c2
    // Two leaves: gc1 and c2
    EagleCandidateGraph g(0, 0.0f, 10, 3);
    const uint64_t c1  = g.add_node(1, -0.5f, 0u);
    const uint64_t c2  = g.add_node(2, -1.5f, 0u);
    const uint64_t gc1 = g.add_node(3, -0.3f, c1);

    const auto selected = g.get_top_k_nodes();
    std::unordered_map<uint64_t, size_t> nodeid_to_index;
    for (size_t i = 0; i < selected.size(); ++i)
        nodeid_to_index[selected[i].node_id] = i;

    const auto leaves = g.get_leaf_nodes(selected);
    EXPECT_EQ(leaves.size(), 2u);

    for (const auto& leaf : leaves) {
        const auto raw_path = g.get_path_to_node(leaf.node_id);
        // All node IDs in path must exist in nodeid_to_index
        for (uint64_t node_id : raw_path)
            EXPECT_NE(nodeid_to_index.find(node_id), nodeid_to_index.end())
                << "Path node " << node_id << " is not in selected set";
        // First element must be root
        EXPECT_EQ(raw_path.front(), 0u);
        // Last element must be the leaf itself
        EXPECT_EQ(raw_path.back(), leaf.node_id);
    }
}
