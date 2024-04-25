// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>
#include "openvino/runtime/core.hpp"
#include "continuous_batching_pipeline.hpp"
#include "sequence_group.hpp"
#include "scheduler.hpp"
#include "generation_config.hpp"


TEST(TestScheduler, general_test) {
    SchedulerConfig scheduler_config {
        // batch size
        .max_num_batched_tokens = 32,
        // cache params
        .num_kv_blocks = 6,
        .block_size = 4,
        // mode - vLLM or dynamic_split_fuse
        .dynamic_split_fuse = false,
        // vLLM specific params
        .max_num_seqs = 5,
        .max_paddings = 8,
    };
    std::vector<uint64_t> tokens = {0,1,2,3,4,5,6,7};
    SequenceGroup::Ptr sequence_group1 = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                            GenerationConfig::greedy(), scheduler_config.block_size);
    auto idx0 = (*sequence_group1)[0]->get_id();
    SequenceGroup::Ptr sequence_group2 = std::make_shared<SequenceGroup>(1, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                            GenerationConfig::greedy(), scheduler_config.block_size);
    auto idx1 = (*sequence_group2)[0]->get_id();
    std::vector<SequenceGroup::Ptr> requests = {sequence_group1, sequence_group2};
                                                                       
    
    // schedule 2 sequence groups that use 4 kv blocks 
    Scheduler scheduler = Scheduler(scheduler_config);
    auto out1 = scheduler.schedule(requests);

    std::vector<size_t> ref_ids = {0, 1};
    EXPECT_EQ(out1.m_scheduled_sequence_groups_ids, ref_ids);
    EXPECT_EQ(out1.m_block_tables[idx0].size(), 2);
    EXPECT_EQ(out1.m_block_tables[idx1].size(), 2);
    EXPECT_FALSE(out1.m_block_tables[idx0][0]->is_free());
    EXPECT_EQ(out1.m_block_tables[idx0][0]->get_index(), 0);
    EXPECT_FALSE(out1.m_block_tables[idx0][1]->is_free());
    EXPECT_EQ(out1.m_block_tables[idx0][1]->get_index(), 1);
    EXPECT_FALSE(out1.m_block_tables[idx1][0]->is_free());
    EXPECT_EQ(out1.m_block_tables[idx1][0]->get_index(), 2);
    EXPECT_FALSE(out1.m_block_tables[idx1][1]->is_free());
    EXPECT_EQ(out1.m_block_tables[idx1][1]->get_index(), 3);
    // tokens.size() * 2 tokens should be scheduled on promt phase, corresponding to first two sequences 
    EXPECT_EQ(out1.m_total_num_scheduled_tokens, tokens.size() * 2);
    EXPECT_EQ(out1.is_prompt, true);
    for (auto seq: requests) {
        seq->finish_iteration();
    }


    SequenceGroup::Ptr sequence_group3 = std::make_shared<SequenceGroup>(2, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                            GenerationConfig::greedy(), scheduler_config.block_size);
    auto idx2 = (*sequence_group3)[0]->get_id();

    // schedule 1 more sequence group that use 2 kv blocks 
    std::vector<SequenceGroup::Ptr> requests1 = {sequence_group1, sequence_group2, sequence_group3};
    auto out2 = scheduler.schedule(requests1);

    std::vector<size_t> ref_ids1 = {2};
    EXPECT_EQ(out2.m_scheduled_sequence_groups_ids, ref_ids1);
    EXPECT_EQ(out2.m_block_tables[idx2].size(), 2);
    EXPECT_FALSE(out2.m_block_tables[idx2][0]->is_free());
    EXPECT_EQ(out2.m_block_tables[idx2][0]->get_index(), 4);
    EXPECT_FALSE(out2.m_block_tables[idx2][1]->is_free());
    EXPECT_EQ(out2.m_block_tables[idx2][1]->get_index(), 5);
    // tokens.size() tokens should be scheduled on promt phase, corresponding to third sequence
    EXPECT_EQ(out2.m_total_num_scheduled_tokens, tokens.size()); 
    for (auto seq: requests1) {
        seq->finish_iteration();
    }
    // at this point we scheduled all available kv blocks

    // sequence_group3 should be evicted
    auto out3 = scheduler.schedule(requests1);
    for (auto seq: requests1) {
        seq->finish_iteration();
    }

    std::vector<size_t> ref_ids2 = {0, 1};
    EXPECT_EQ(out3.m_scheduled_sequence_groups_ids, ref_ids2);
    EXPECT_FALSE(out3.m_block_tables[idx0][0]->is_free());
    EXPECT_EQ(out3.m_block_tables[idx0][0]->get_index(), 0);
    EXPECT_FALSE(out3.m_block_tables[idx0][1]->is_free());
    EXPECT_EQ(out3.m_block_tables[idx0][1]->get_index(), 1);
    EXPECT_FALSE(out3.m_block_tables[idx0][2]->is_free());
    EXPECT_EQ(out3.m_block_tables[idx0][2]->get_index(), 4);
    EXPECT_FALSE(out3.m_block_tables[idx1][0]->is_free());
    EXPECT_EQ(out3.m_block_tables[idx1][0]->get_index(), 2);
    EXPECT_FALSE(out3.m_block_tables[idx1][1]->is_free());
    EXPECT_EQ(out3.m_block_tables[idx1][1]->get_index(), 3);
    EXPECT_FALSE(out3.m_block_tables[idx1][2]->is_free());
    EXPECT_EQ(out3.m_block_tables[idx1][2]->get_index(), 5);
    // 2 tokens should be scheduled on generate phase for "0" and "1" sequence, "2" sequence should be preempted
    EXPECT_EQ(out3.m_total_num_scheduled_tokens, 2); 

    // check that 1 token was scheduled for "2" sequence (preempted on previous iteraition) 
    auto out4 = scheduler.schedule(requests1);
    // At this point scheduler preempts "1" sequence, as it assumes "0" sequence requires new block, but in fact it doesn't. 
    // This part of test should be updated when preemtion algorithm finished.
    
    EXPECT_FALSE(out4.m_block_tables[idx2][0]->is_free());
    EXPECT_EQ(out4.m_block_tables[idx2][0]->get_index(), 2); // index here should be updated later
}



TEST(TestScheduler, test_case1) {
    SchedulerConfig scheduler_config {
        // batch size
        .max_num_batched_tokens = 32,
        // cache params
        .num_kv_blocks = 5,
        .block_size = 4,
        // mode - vLLM or dynamic_split_fuse
        .dynamic_split_fuse = false,
        // vLLM specific params
        .max_num_seqs = 5,
        .max_paddings = 8,
    };
    std::vector<uint64_t> tokens = {0,1,2,3,4,5,6,7};
    SequenceGroup::Ptr sequence_group1 = std::make_shared<SequenceGroup>(0, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                            GenerationConfig::greedy(), scheduler_config.block_size);
    auto idx0 = (*sequence_group1)[0]->get_id();
    SequenceGroup::Ptr sequence_group2 = std::make_shared<SequenceGroup>(1, ov::Tensor(ov::element::i64, {tokens.size()}, tokens.data()),
                                                                            GenerationConfig::greedy(), scheduler_config.block_size);
    auto idx1 = (*sequence_group2)[0]->get_id();
    std::vector<SequenceGroup::Ptr> requests = {sequence_group1, sequence_group2};
    
                                                                       
    
    Scheduler scheduler = Scheduler(scheduler_config);
    auto out1 = scheduler.schedule(requests);

    std::vector<size_t> ref_ids = {0, 1};
    EXPECT_EQ(out1.m_scheduled_sequence_groups_ids, ref_ids);
    EXPECT_EQ(out1.m_block_tables[idx0].size(), 2);
    EXPECT_EQ(out1.m_block_tables[idx1].size(), 2);
    EXPECT_FALSE(out1.m_block_tables[idx0][0]->is_free());
    EXPECT_EQ(out1.m_block_tables[idx0][0]->get_index(), 0);
    EXPECT_FALSE(out1.m_block_tables[idx0][1]->is_free());
    EXPECT_EQ(out1.m_block_tables[idx0][1]->get_index(), 1);
    EXPECT_FALSE(out1.m_block_tables[idx1][0]->is_free());
    EXPECT_EQ(out1.m_block_tables[idx1][0]->get_index(), 2);
    EXPECT_FALSE(out1.m_block_tables[idx1][1]->is_free());
    EXPECT_EQ(out1.m_block_tables[idx1][1]->get_index(), 3);
    EXPECT_EQ(out1.m_total_num_scheduled_tokens, tokens.size() * 2);
    EXPECT_EQ(out1.is_prompt, true);
    for (auto seq: requests) {
        seq->finish_iteration();
    }

    // at this point we used 4/5 KV blocks. Both sequences requre new KV block, but we have space for only one.
    auto out2 = scheduler.schedule(requests); // fails currently as we check can append slot for each sequence separetely. First sequence schedule 1 new block, second fails on scheduling.
    
}
