// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

// Unit tests for CandidateGraph.
// These tests exercise the standalone tree data structure used by TreeSearcher
// without requiring a live model or KV-cache, making them fast and hermetic.

#include <gtest/gtest.h>
#include "sampling/sampler.hpp"

using ov::genai::CandidateGraph;
using NodePtr = CandidateGraph::NodePtr;

// ---------------------------------------------------------------------------
// Constructor / root
// ---------------------------------------------------------------------------

TEST(CandidateGraphTest, RootIsAlwaysPresent) {
    CandidateGraph g(42, 0.5f, 4, 3);
    const auto top = g.select_candidate_nodes();
    ASSERT_FALSE(top.empty());
    EXPECT_EQ(top[0], g.get_root());
    EXPECT_EQ(top[0]->token_id, 42);
    EXPECT_FLOAT_EQ(top[0]->score, 0.5f);
    EXPECT_EQ(top[0]->tree_layer, 0);
}

TEST(CandidateGraphTest, EmptyBudgetStillReturnsRoot) {
    // max_tokens = 0: budget for non-root nodes is 0, but root must still be returned.
    CandidateGraph g(0, 0.0f, /*max_tokens=*/0, /*max_depth=*/2);
    const auto top = g.select_candidate_nodes();
    ASSERT_EQ(top.size(), 1u);
    EXPECT_EQ(top[0], g.get_root());
}

// ---------------------------------------------------------------------------
// add_node
// ---------------------------------------------------------------------------

TEST(CandidateGraphTest, AddNodeReturnsDistinctNonNullNodes) {
    CandidateGraph g(0, 0.0f, 10, 3);
    const NodePtr n1 = g.add_node(10, -1.0f, g.get_root());
    const NodePtr n2 = g.add_node(20, -2.0f, g.get_root());
    EXPECT_NE(n1, nullptr);
    EXPECT_NE(n2, nullptr);
    EXPECT_NE(n1, n2);
    EXPECT_EQ(n1->token_id, 10);
    EXPECT_EQ(n2->token_id, 20);
}

TEST(CandidateGraphTest, AddNodeStoresCorrectMetadata) {
    CandidateGraph g(0, 0.0f, 10, 5);
    const NodePtr child = g.add_node(99, -0.5f, g.get_root());
    ASSERT_NE(child, nullptr);

    // The node should appear in select_candidate_nodes with the expected fields.
    const auto top = g.select_candidate_nodes();
    const auto it = std::find(top.begin(), top.end(), child);
    ASSERT_NE(it, top.end()) << "Child node not found in top-k result";
    EXPECT_EQ((*it)->token_id, 99);
    EXPECT_FLOAT_EQ((*it)->score, -0.5f);
    EXPECT_EQ((*it)->tree_layer, 1);
}

// ---------------------------------------------------------------------------
// select_candidate_nodes
// ---------------------------------------------------------------------------

TEST(CandidateGraphTest, TopKRespectsBudget) {
    // Build a tree with 5 children under root; budget = 3.
    CandidateGraph g(0, 0.0f, /*max_tokens=*/3, /*max_depth=*/2);
    for (int i = 0; i < 5; ++i)
        g.add_node(i, static_cast<float>(-i), g.get_root());  // scores: 0, -1, -2, -3, -4

    const auto top = g.select_candidate_nodes();
    // Root + 3 non-root nodes = 4 total
    EXPECT_EQ(top.size(), 4u);

    // Verify that the 3 highest-scoring non-root nodes (scores 0, -1, -2) are selected.
    for (const NodePtr& n : top) {
        if (n == g.get_root()) continue;
        EXPECT_GE(n->score, -2.0f) << "Low-scoring node should have been evicted";
    }
}

TEST(CandidateGraphTest, TopKSortedByLayer) {
    CandidateGraph g(0, 0.0f, 10, 3);
    const NodePtr c1 = g.add_node(1, -0.5f, g.get_root());   // layer 1
    const NodePtr c2 = g.add_node(2, -1.0f, g.get_root());   // layer 1
    g.add_node(3, -0.3f, c1);                                  // layer 2
    g.add_node(4, -0.4f, c2);                                  // layer 2

    const auto top = g.select_candidate_nodes();
    for (size_t i = 1; i < top.size(); ++i)
        EXPECT_LE(top[i - 1]->tree_layer, top[i]->tree_layer)
            << "Nodes must be sorted by tree_layer";
}

TEST(CandidateGraphTest, TopKExactBudget) {
    // Exactly max_tokens non-root nodes; all should be selected.
    CandidateGraph g(0, 0.0f, /*max_tokens=*/3, /*max_depth=*/2);
    g.add_node(1, -1.0f, g.get_root());
    g.add_node(2, -2.0f, g.get_root());
    g.add_node(3, -3.0f, g.get_root());
    const auto top = g.select_candidate_nodes();
    EXPECT_EQ(top.size(), 4u);  // root + 3
}

// ---------------------------------------------------------------------------
// get_leaf_nodes
// ---------------------------------------------------------------------------

TEST(CandidateGraphTest, LeafNodesOfFlatTree) {
    // Each child of root is a leaf when no grandchildren exist.
    CandidateGraph g(0, 0.0f, 10, 3);
    g.add_node(1, -1.0f, g.get_root());
    g.add_node(2, -2.0f, g.get_root());
    g.add_node(3, -3.0f, g.get_root());

    const auto selected = g.select_candidate_nodes();
    const auto leaves   = g.get_leaf_nodes(selected);

    // Root is not a leaf (it has selected children).
    for (const NodePtr& leaf : leaves)
        EXPECT_NE(leaf, g.get_root()) << "Root should not be a leaf node";
    EXPECT_EQ(leaves.size(), 3u);
}

TEST(CandidateGraphTest, LeafNodesWithGrandchildren) {
    CandidateGraph g(0, 0.0f, 10, 3);
    const NodePtr c1 = g.add_node(1, -1.0f, g.get_root());
    const NodePtr c2 = g.add_node(2, -2.0f, g.get_root());
    const NodePtr g1 = g.add_node(3, -0.5f, c1);  // grandchild: score -0.5 (high)
    g.add_node(4, -0.6f, c2);                       // grandchild

    const auto selected = g.select_candidate_nodes();
    const auto leaves   = g.get_leaf_nodes(selected);

    // c1 and c2 each have a grandchild in selected, so they are not leaves.
    EXPECT_EQ(std::find(leaves.begin(), leaves.end(), c1), leaves.end())
        << "c1 has a selected child and must not be a leaf";
    EXPECT_EQ(std::find(leaves.begin(), leaves.end(), c2), leaves.end())
        << "c2 has a selected child and must not be a leaf";

    // The grandchild g1 is a leaf.
    EXPECT_NE(std::find(leaves.begin(), leaves.end(), g1), leaves.end())
        << "Grandchild g1 should be a leaf node";
}

TEST(CandidateGraphTest, LeafNodesWhenBudgetExcludesGrandchildren) {
    // Budget = 2: root + c1 + c2 fit; the grandchild (score -3.0) is outscored
    // by c1 (-1.0) and c2 (-2.0) and is excluded.  With no grandchild in the
    // selected set, c1 and c2 are both leaves.
    CandidateGraph g(0, 0.0f, /*max_tokens=*/2, /*max_depth=*/3);
    const NodePtr c1 = g.add_node(1, -1.0f, g.get_root());
    const NodePtr c2 = g.add_node(2, -2.0f, g.get_root());
    g.add_node(3, -3.0f, c1);  // grandchild scores lower than c2; excluded by budget

    const auto selected = g.select_candidate_nodes();
    ASSERT_EQ(selected.size(), 3u);  // root + c1 + c2

    const auto leaves = g.get_leaf_nodes(selected);
    ASSERT_EQ(leaves.size(), 2u);
    EXPECT_NE(std::find(leaves.begin(), leaves.end(), c1), leaves.end()) << "c1 must be a leaf";
    EXPECT_NE(std::find(leaves.begin(), leaves.end(), c2), leaves.end()) << "c2 must be a leaf";
}

// ---------------------------------------------------------------------------
// get_path_to_node
// ---------------------------------------------------------------------------

TEST(CandidateGraphTest, PathToRootIsSingleElement) {
    CandidateGraph g(0, 0.0f, 10, 3);
    const auto path = g.get_path_to_node(g.get_root());
    ASSERT_EQ(path.size(), 1u);
    EXPECT_EQ(path[0], g.get_root());
}

TEST(CandidateGraphTest, PathToChildIsCorrect) {
    CandidateGraph g(0, 0.0f, 10, 3);
    const NodePtr c = g.add_node(1, -1.0f, g.get_root());
    const auto path = g.get_path_to_node(c);
    ASSERT_EQ(path.size(), 2u);
    EXPECT_EQ(path[0], g.get_root());
    EXPECT_EQ(path[1], c);
}

TEST(CandidateGraphTest, PathToGrandchildIsCorrect) {
    CandidateGraph g(0, 0.0f, 10, 3);
    const NodePtr c  = g.add_node(1, -1.0f, g.get_root());
    const NodePtr gc = g.add_node(2, -2.0f, c);
    const auto path = g.get_path_to_node(gc);
    ASSERT_EQ(path.size(), 3u);
    EXPECT_EQ(path[0], g.get_root());
    EXPECT_EQ(path[1], c);
    EXPECT_EQ(path[2], gc);
}

TEST(CandidateGraphTest, PathRootFirstLeafLast) {
    // Build a chain: root -> a -> b -> c
    CandidateGraph g(0, 0.0f, 10, 4);
    const NodePtr a = g.add_node(1, -0.1f, g.get_root());
    const NodePtr b = g.add_node(2, -0.2f, a);
    const NodePtr c = g.add_node(3, -0.3f, b);
    const auto path = g.get_path_to_node(c);
    ASSERT_EQ(path.size(), 4u);
    EXPECT_EQ(path[0], g.get_root());
    EXPECT_EQ(path[1], a);
    EXPECT_EQ(path[2], b);
    EXPECT_EQ(path[3], c);
}

// ---------------------------------------------------------------------------
// is_ancestor
// ---------------------------------------------------------------------------

TEST(CandidateGraphTest, RootIsAncestorOfItself) {
    CandidateGraph g(0, 0.0f, 10, 3);
    EXPECT_TRUE(g.is_ancestor(g.get_root(), g.get_root()));
}

TEST(CandidateGraphTest, RootIsAncestorOfChild) {
    CandidateGraph g(0, 0.0f, 10, 3);
    const NodePtr c = g.add_node(1, -1.0f, g.get_root());
    EXPECT_TRUE(g.is_ancestor(g.get_root(), c));
    EXPECT_TRUE(g.is_ancestor(c, c));   // node is its own ancestor (self-inclusive)
}

TEST(CandidateGraphTest, ChildIsNotAncestorOfRoot) {
    CandidateGraph g(0, 0.0f, 10, 3);
    const NodePtr c = g.add_node(1, -1.0f, g.get_root());
    EXPECT_FALSE(g.is_ancestor(c, g.get_root()));
}

TEST(CandidateGraphTest, SiblingIsNotAncestor) {
    CandidateGraph g(0, 0.0f, 10, 3);
    const NodePtr c1 = g.add_node(1, -1.0f, g.get_root());
    const NodePtr c2 = g.add_node(2, -2.0f, g.get_root());
    EXPECT_FALSE(g.is_ancestor(c1, c2));
    EXPECT_FALSE(g.is_ancestor(c2, c1));
}

TEST(CandidateGraphTest, AncestorChainCorrect) {
    // root -> a -> b -> c
    CandidateGraph g(0, 0.0f, 10, 4);
    const NodePtr a = g.add_node(1, -0.1f, g.get_root());
    const NodePtr b = g.add_node(2, -0.2f, a);
    const NodePtr c = g.add_node(3, -0.3f, b);

    // All ancestors of c
    EXPECT_TRUE(g.is_ancestor(g.get_root(), c));
    EXPECT_TRUE(g.is_ancestor(a, c));
    EXPECT_TRUE(g.is_ancestor(b, c));
    EXPECT_TRUE(g.is_ancestor(c, c));  // self

    // b is NOT an ancestor of a
    EXPECT_FALSE(g.is_ancestor(b, a));

    // c is NOT an ancestor of a or b
    EXPECT_FALSE(g.is_ancestor(c, a));
    EXPECT_FALSE(g.is_ancestor(c, b));
}

TEST(CandidateGraphTest, AncestorQueryWithNullptrReturnsFalse) {
    CandidateGraph g(0, 0.0f, 10, 3);
    EXPECT_FALSE(g.is_ancestor(nullptr, g.get_root()));
    EXPECT_FALSE(g.is_ancestor(g.get_root(), nullptr));
}

// ---------------------------------------------------------------------------
// Tree mask correctness (integration of select_candidate_nodes + is_ancestor)
// ---------------------------------------------------------------------------
// This reproduces exactly the tree-mask construction in finalize_tree(),
// so a bug there will show up here.

TEST(CandidateGraphTest, TreeMaskDiagonalIsAllOnes) {
    // Every node is an ancestor of itself → all diagonal entries must be 1.
    CandidateGraph g(0, 0.0f, 10, 3);
    const NodePtr c1 = g.add_node(1, -1.0f, g.get_root());
    const NodePtr c2 = g.add_node(2, -2.0f, g.get_root());
    g.add_node(3, -0.5f, c1);

    const auto selected = g.select_candidate_nodes();
    for (const NodePtr& node : selected) {
        EXPECT_TRUE(g.is_ancestor(node, node))
            << "Node must be its own ancestor";
    }
}

TEST(CandidateGraphTest, TreeMaskLinearChain) {
    // root -> a -> b
    // Expected mask (row i = node i, col j = node j, 1 iff j is ancestor of i):
    //      root  a   b
    // root [ 1,  0,  0 ]
    //    a [ 1,  1,  0 ]
    //    b [ 1,  1,  1 ]
    CandidateGraph g(0, 0.0f, 10, 3);
    const NodePtr a = g.add_node(10, -0.5f, g.get_root());
    const NodePtr b = g.add_node(20, -0.3f, a);

    const auto selected = g.select_candidate_nodes();
    // select_candidate_nodes sorts by layer, so order should be: root, a, b
    ASSERT_EQ(selected.size(), 3u);
    EXPECT_EQ(selected[0], g.get_root());
    EXPECT_EQ(selected[1], a);
    EXPECT_EQ(selected[2], b);

    // Build mask as in finalize_tree
    std::vector<std::vector<uint8_t>> mask;
    mask.reserve(selected.size());
    for (const NodePtr& node : selected) {
        std::vector<uint8_t> row;
        row.reserve(selected.size());
        for (const NodePtr& other : selected)
            row.push_back(g.is_ancestor(other, node) ? 1u : 0u);
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

TEST(CandidateGraphTest, TreeMaskTwoSiblings) {
    // root -> c1
    //      -> c2
    // c1 and c2 are siblings, not ancestors of each other.
    CandidateGraph g(0, 0.0f, 10, 2);
    const NodePtr c1 = g.add_node(1, -1.0f, g.get_root());
    const NodePtr c2 = g.add_node(2, -2.0f, g.get_root());

    const auto selected = g.select_candidate_nodes();
    ASSERT_EQ(selected.size(), 3u);

    for (const NodePtr& node : selected) {
        for (const NodePtr& other : selected) {
            const bool expected = g.is_ancestor(other, node);
            if (node == c1 && other == c2)
                EXPECT_FALSE(expected) << "c2 is not an ancestor of c1";
            if (node == c2 && other == c1)
                EXPECT_FALSE(expected) << "c1 is not an ancestor of c2";
        }
    }
}

// ---------------------------------------------------------------------------
// retrieve_indices correctness (integration of get_leaf_nodes + get_path_to_node)
// ---------------------------------------------------------------------------
// Reproduces the retrieve_indices construction in finalize_tree().

TEST(CandidateGraphTest, RetrieveIndicesLinearChain) {
    // root -> a -> b: single leaf b, path should map to indices [0, 1, 2]
    CandidateGraph g(0, 0.0f, 10, 3);
    const NodePtr a = g.add_node(10, -0.5f, g.get_root());
    const NodePtr b = g.add_node(20, -0.3f, a);

    const auto selected = g.select_candidate_nodes();  // [root, a, b]
    std::unordered_map<const CandidateGraph::Node*, size_t> node_to_index;
    for (size_t i = 0; i < selected.size(); ++i)
        node_to_index[selected[i].get()] = i;

    const auto leaves = g.get_leaf_nodes(selected);
    ASSERT_EQ(leaves.size(), 1u);
    EXPECT_EQ(leaves[0], b);

    const auto raw_path = g.get_path_to_node(b);
    ASSERT_EQ(raw_path.size(), 3u);

    std::vector<int64_t> path;
    for (const NodePtr& n : raw_path)
        path.push_back(static_cast<int64_t>(node_to_index.at(n.get())));

    EXPECT_EQ(path[0], 0);  // root at index 0
    EXPECT_EQ(path[1], 1);  // a at index 1
    EXPECT_EQ(path[2], 2);  // b at index 2
}

TEST(CandidateGraphTest, RetrieveIndicesTwoLeaves) {
    // root -> c1 -> gc1
    //      -> c2
    // Two leaves: gc1 and c2
    CandidateGraph g(0, 0.0f, 10, 3);
    const NodePtr c1  = g.add_node(1, -0.5f, g.get_root());
    const NodePtr c2  = g.add_node(2, -1.5f, g.get_root());
    const NodePtr gc1 = g.add_node(3, -0.3f, c1);

    const auto selected = g.select_candidate_nodes();
    std::unordered_map<const CandidateGraph::Node*, size_t> node_to_index;
    for (size_t i = 0; i < selected.size(); ++i)
        node_to_index[selected[i].get()] = i;

    const auto leaves = g.get_leaf_nodes(selected);
    EXPECT_EQ(leaves.size(), 2u);

    for (const NodePtr& leaf : leaves) {
        const auto raw_path = g.get_path_to_node(leaf);
        // All nodes in path must exist in node_to_index
        for (const NodePtr& n : raw_path)
            EXPECT_NE(node_to_index.find(n.get()), node_to_index.end())
                << "Path node is not in selected set";
        // First element must be root
        EXPECT_EQ(raw_path.front(), g.get_root());
        // Last element must be the leaf itself
        EXPECT_EQ(raw_path.back(), leaf);
    }
}

// ---------------------------------------------------------------------------
// select_candidate_nodes: cross-layer score eviction
// ---------------------------------------------------------------------------
// A deeper node with a higher score must evict a shallower node with a lower
// score when the budget is tight.

TEST(CandidateGraphTest, CrossLayerEvictionByScore) {
    // Budget = 2, tree:
    //   root -> lo (score -10)  <- should be evicted
    //        -> hi (score -1)   <- should survive
    //              -> gc (score -0.5, layer 2) <- should survive over lo
    CandidateGraph g(0, 0.0f, /*max_tokens=*/2, /*max_depth=*/3);
    const NodePtr lo = g.add_node(1, -10.0f, g.get_root());
    const NodePtr hi = g.add_node(2, -1.0f,  g.get_root());
    const NodePtr gc = g.add_node(3, -0.5f,  hi);

    const auto top = g.select_candidate_nodes();
    ASSERT_EQ(top.size(), 3u);  // root + hi + gc

    EXPECT_NE(std::find(top.begin(), top.end(), hi), top.end()) << "hi must be selected";
    EXPECT_NE(std::find(top.begin(), top.end(), gc), top.end()) << "gc must be selected";
    EXPECT_EQ(std::find(top.begin(), top.end(), lo), top.end()) << "lo must be evicted";
}

// ---------------------------------------------------------------------------
// Tree mask: branching tree with depth
// ---------------------------------------------------------------------------
// Tree:  root -> c1 -> gc1
//             -> c2 -> gc2
//
// Verifies off-diagonal mask entries for a tree with both depth and branches.

TEST(CandidateGraphTest, TreeMaskBranchingWithDepth) {
    CandidateGraph g(0, 0.0f, /*max_tokens=*/10, /*max_depth=*/3);
    const NodePtr c1  = g.add_node(1, -0.5f, g.get_root());
    const NodePtr c2  = g.add_node(2, -0.6f, g.get_root());
    const NodePtr gc1 = g.add_node(3, -0.3f, c1);
    const NodePtr gc2 = g.add_node(4, -0.4f, c2);

    const auto selected = g.select_candidate_nodes();
    ASSERT_EQ(selected.size(), 5u);

    // c1 and c2 are siblings: neither is an ancestor of the other
    EXPECT_FALSE(g.is_ancestor(c1, c2));
    EXPECT_FALSE(g.is_ancestor(c2, c1));

    // gc1 sees root and c1, but not c2 or gc2
    EXPECT_TRUE (g.is_ancestor(g.get_root(), gc1));
    EXPECT_TRUE (g.is_ancestor(c1,           gc1));
    EXPECT_FALSE(g.is_ancestor(c2,           gc1));
    EXPECT_FALSE(g.is_ancestor(gc2,          gc1));

    // gc2 sees root and c2, but not c1 or gc1
    EXPECT_TRUE (g.is_ancestor(g.get_root(), gc2));
    EXPECT_TRUE (g.is_ancestor(c2,           gc2));
    EXPECT_FALSE(g.is_ancestor(c1,           gc2));
    EXPECT_FALSE(g.is_ancestor(gc1,          gc2));
}

// ---------------------------------------------------------------------------
// select_candidate_nodes: fewer nodes than budget
// ---------------------------------------------------------------------------

TEST(CandidateGraphTest, AllNodesReturnedWhenBelowBudget) {
    CandidateGraph g(0, 0.0f, /*max_tokens=*/10, /*max_depth=*/3);
    const NodePtr c1 = g.add_node(1, -1.0f, g.get_root());
    const NodePtr c2 = g.add_node(2, -2.0f, g.get_root());

    const auto top = g.select_candidate_nodes();
    ASSERT_EQ(top.size(), 3u);  // root + c1 + c2

    EXPECT_NE(std::find(top.begin(), top.end(), c1), top.end());
    EXPECT_NE(std::find(top.begin(), top.end(), c2), top.end());
}

// ---------------------------------------------------------------------------
// retrieve_indices: shared prefix
// ---------------------------------------------------------------------------
// Tree: root -> a -> b
//                 -> c
// Both leaves share the prefix [root, a]; verifies path remapping correctness.

TEST(CandidateGraphTest, RetrieveIndicesSharedPrefix) {
    CandidateGraph g(0, 0.0f, /*max_tokens=*/10, /*max_depth=*/3);
    const NodePtr a = g.add_node(10, -0.5f, g.get_root());
    const NodePtr b = g.add_node(20, -0.3f, a);
    const NodePtr c = g.add_node(30, -0.4f, a);

    const auto selected = g.select_candidate_nodes();  // [root, a, b, c] sorted by layer
    ASSERT_EQ(selected.size(), 4u);

    std::unordered_map<const CandidateGraph::Node*, size_t> node_to_index;
    for (size_t i = 0; i < selected.size(); ++i)
        node_to_index[selected[i].get()] = i;

    const auto leaves = g.get_leaf_nodes(selected);
    ASSERT_EQ(leaves.size(), 2u);

    for (const NodePtr& leaf : leaves) {
        const auto raw_path = g.get_path_to_node(leaf);
        ASSERT_EQ(raw_path.size(), 3u);  // root, a, leaf

        EXPECT_EQ(raw_path[0], g.get_root());  // root
        EXPECT_EQ(raw_path[1], a);             // shared prefix node
        EXPECT_TRUE(leaf == b || leaf == c);
        EXPECT_EQ(raw_path[2], leaf);

        // All path nodes must appear in the selected set
        for (const NodePtr& n : raw_path)
            EXPECT_NE(node_to_index.find(n.get()), node_to_index.end())
                << "Path node not in selected set";
    }
}
