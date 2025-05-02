// Copyright (C) 2023-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <filesystem>
#include <vector>

#include "continuous_batching/block_manager.hpp"
#include "sequence_group.hpp"
#include "scheduler.hpp"

namespace ov::genai {
const std::string DEFAULT_POSTFIX = std::string();


/** Class to dump the current state of the KV block cache to disk as a number of text files, to be further parsed
* and visualized by the `cacheviz` tool.
*/
class CacheStateDumper {
public:
    /**
     * Constructs the CacheStateDumper
     * @param run_id Identifier of the cache dumping session. The output .txt files will have this run_id as a
     * postfix in the name.
     */
    CacheStateDumper(const std::string &run_id) : m_run_id(run_id) {}

    std::filesystem::path get_per_layer_folder(size_t layer_idx) {
        auto per_layer_folder = std::filesystem::path("debug") / "cache_dump";
        per_layer_folder /= std::to_string(layer_idx);
        std::filesystem::create_directories(per_layer_folder);
        auto file_path = (per_layer_folder / (m_run_id + ".txt")).string();
        return per_layer_folder;
    }

    /**
     * Dumps the state of the cache described by a given block manager
     * @param block_mgr A block manager owning the caches.
     * @param sequence_groups Sequence groups currently utilizing the cache.
     */
    void dump_cache_state(const std::shared_ptr<BlockManager> block_mgr, const std::vector <SequenceGroup::Ptr> &sequence_groups,
                          size_t dump_count) {
        for (size_t layer_idx = 0; layer_idx < block_mgr->m_num_layers; layer_idx++) {
            auto per_layer_folder = get_per_layer_folder(layer_idx);
            auto file_path = (per_layer_folder / (m_run_id + ".txt")).string();
            std::ofstream out_stream(file_path, std::ios::out);
            OPENVINO_ASSERT(out_stream.is_open());

            out_stream << block_mgr->m_allocator.m_total_num_blocks << std::endl;
            out_stream << sequence_groups.size() << std::endl;
            for (const auto &seq_group_ptr: sequence_groups) {
                out_stream << seq_group_ptr->get_request_id() << ' ';
                for (const auto &seq_ptr: seq_group_ptr->get_sequences()) {
                    out_stream << seq_ptr->get_id() << ' ';
                }
                out_stream << std::endl;
            }
            for (const auto &seq_id_and_blocks: block_mgr->m_block_table) {
                for (const auto &block: seq_id_and_blocks.second[layer_idx]) {
                    const size_t seq_id = seq_id_and_blocks.first;
                    out_stream << seq_id << " " << block->get_index() << " " << block->get_references_count()
                               << std::endl;
                }
            }
            out_stream.flush();

            auto cache_usage_file_path = (per_layer_folder / ("cache_usage.txt")).string();
            std::ofstream out_stream_cache_usage;

            out_stream_cache_usage.open(cache_usage_file_path, std::ios::app);
            out_stream_cache_usage << dump_count << ' ' << block_mgr->get_used_percentage() << std::endl;
            out_stream_cache_usage.flush();
            dump_count++;
        }
    }

    /**
     * Dumps the state of the cache described by a given scheduler.
     * @param schdl A scheduler managing certain sequence groups.
     * @param sequence_groups Sequence groups currently utilizing the cache (managed by the scheduler).
     */
    void dump_cache_state(const Scheduler &schdl, const std::vector <SequenceGroup::Ptr> &sequence_groups,
                          size_t dump_count) {
        dump_cache_state(schdl.m_block_manager, sequence_groups, dump_count);

    }

    /**
     * @param step Current step number during the generation.
     * @param postfix Postfix to the returned string ID.
     * @return A string identifier for the current generation step.
     */
    static std::string get_run_id_for_generation_step(size_t step, const std::string &postfix = DEFAULT_POSTFIX) {
        std::stringstream ss;
        ss << "cache_dump";
        if (!postfix.empty()) {
            ss << "_" << postfix;
        }
        ss << "_step_" << step;
        return ss.str();
    }

private:
    std::string m_run_id;
};
}
