// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/speculative_decoding/perf_metrics.hpp"

namespace {

// Proposes the continuations of earlier occurrences of the latest n-gram (prompt lookup) as a tree: each occurrence
// becomes a branch, branches that start with the same tokens share nodes.
class PromptLookupDrafter : public ov::genai::Drafter {
public:
    size_t max_draft_tokens() const override {
        return 8;
    }

    void start(const std::vector<int64_t>& /*prompt*/, const ov::genai::GenerationConfig& /*config*/) override {
        m_history.clear();
    }

    ov::genai::DraftTree propose(int64_t root, int64_t /*position*/, size_t max_nodes) override {
        std::vector<int64_t> text = m_history;
        text.push_back(root);
        ov::genai::DraftTree tree;
        std::map<std::pair<int64_t, int64_t>, int64_t> node_of;  // (parent, token) -> node
        for (const auto& continuation : continuations(text, max_nodes)) {
            int64_t parent = -1;
            for (const int64_t token : continuation) {
                auto node = node_of.find({parent, token});
                if (node == node_of.end()) {
                    if (tree.tokens.size() == max_nodes) {
                        break;
                    }
                    node =
                        node_of.emplace(std::make_pair(parent, token), static_cast<int64_t>(tree.tokens.size())).first;
                    tree.tokens.push_back(token);
                    tree.parents.push_back(parent);
                }
                parent = node->second;
            }
        }
        return tree;
    }

    void accept(const std::vector<int64_t>& tokens,
                const std::vector<int64_t>& /*positions*/,
                const ov::Tensor& /*features*/) override {
        // Tokens the main model has processed: the prompt, then the root and the accepted candidates of every step
        // (the main model's own next token is the next root).
        m_history.insert(m_history.end(), tokens.begin(), tokens.end());
    }

private:
    // Up to 3 continuations of the longest n-gram suffix (up to 3 tokens) of `text` that also occurs earlier, latest
    // first.
    static std::vector<std::vector<int64_t>> continuations(const std::vector<int64_t>& text, size_t length) {
        constexpr size_t max_ngram_size = 3, max_branches = 3;
        std::vector<std::vector<int64_t>> result;
        for (size_t n = std::min(max_ngram_size, text.size() - 1); n > 0 && result.empty(); --n) {
            const auto suffix = text.end() - n;
            for (size_t begin = text.size() - n; begin-- > 0 && result.size() < max_branches;) {
                if (std::equal(suffix, text.end(), text.begin() + begin)) {
                    const auto from = text.begin() + begin + n;
                    result.emplace_back(from, from + std::min(length, static_cast<size_t>(text.end() - from)));
                }
            }
        }
        return result;
    }

    std::vector<int64_t> m_history;
};

}  // namespace

int main(int argc, char* argv[]) try {
    if (3 != argc) {
        throw std::runtime_error(std::string{"Usage: "} + argv[0] + " <MODEL_DIR> '<PROMPT>'");
    }

    std::string model_path = argv[1];
    std::string prompt = argv[2];

    // Speculative decoding with a custom drafter runs the main model on NPU (greedy decoding, batch size 1).
    std::string device = "NPU";
    ov::genai::LLMPipeline pipe(model_path, device, ov::genai::drafter(std::make_shared<PromptLookupDrafter>()));

    ov::genai::GenerationConfig config;
    config.max_new_tokens = 100;
    // Up to num_assistant_tokens candidates per step (default: the drafter's max_draft_tokens()).
    config.num_assistant_tokens = 8;

    auto streamer = [](std::string subword) {
        std::cout << subword << std::flush;
        return ov::genai::StreamingStatus::RUNNING;
    };

    // Since the streamer is set, the results will
    // be printed each time a new token is generated.
    auto result = pipe.generate(prompt, config, streamer);
    std::cout << std::endl;

    auto sd_perf_metrics = std::dynamic_pointer_cast<ov::genai::SDPerModelsPerfMetrics>(result.extended_perf_metrics);
    if (sd_perf_metrics) {
        std::cout << "Accepted " << sd_perf_metrics->get_num_accepted_tokens() << " of "
                  << sd_perf_metrics->get_num_draft_tokens() << " draft tokens" << std::endl;
    }
} catch (const std::exception& error) {
    try {
        std::cerr << error.what() << '\n';
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
} catch (...) {
    try {
        std::cerr << "Non-exception object thrown\n";
    } catch (const std::ios_base::failure&) {}
    return EXIT_FAILURE;
}
