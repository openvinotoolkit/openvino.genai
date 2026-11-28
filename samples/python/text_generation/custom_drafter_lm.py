#!/usr/bin/env python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import openvino_genai


class PromptLookupDrafter(openvino_genai.Drafter):
    """Proposes the continuations of earlier occurrences of the latest n-gram (prompt lookup) as a tree:
    each occurrence becomes a branch, branches that start with the same tokens share nodes."""

    def __init__(self, max_draft_tokens=8, max_ngram_size=3, max_branches=3):
        super().__init__()
        self.max_tokens = max_draft_tokens
        self.max_ngram_size = max_ngram_size
        self.max_branches = max_branches
        self.history = []

    def max_draft_tokens(self):
        return self.max_tokens

    def start(self, prompt, config):
        self.history = []

    def propose(self, root, position, max_nodes):
        tokens, parents, node_of = [], [], {}
        for continuation in self.continuations(self.history + [root], max_nodes):
            parent = -1
            for token in continuation:
                if (parent, token) not in node_of:
                    if len(tokens) == max_nodes:
                        break
                    node_of[(parent, token)] = len(tokens)
                    tokens.append(token)
                    parents.append(parent)
                parent = node_of[(parent, token)]
        return openvino_genai.DraftTree(tokens, parents)

    def accept(self, tokens, positions, features):
        # Tokens the main model has processed: the prompt, then the root and the accepted candidates of every step
        # (the main model's own next token is the next root).
        self.history.extend(tokens)

    def continuations(self, text, length):
        """Up to max_branches continuations of the longest n-gram suffix of text that also occurs earlier, latest first."""
        for n in range(min(self.max_ngram_size, len(text) - 1), 0, -1):
            suffix = text[-n:]
            found = 0
            for begin in range(len(text) - n - 1, -1, -1):
                if text[begin:begin + n] == suffix:
                    yield text[begin + n:begin + n + length]
                    found += 1
                    if found == self.max_branches:
                        return
            if found:
                return


def streamer(subword):
    print(subword, end='', flush=True)
    # Return flag corresponds whether generation should be stopped.
    return openvino_genai.StreamingStatus.RUNNING


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('prompt')
    args = parser.parse_args()

    # Speculative decoding with a custom drafter runs the main model on NPU (greedy decoding, batch size 1).
    device = 'NPU'
    pipe = openvino_genai.LLMPipeline(args.model_dir, device, drafter=PromptLookupDrafter())

    config = openvino_genai.GenerationConfig()
    config.max_new_tokens = 100
    # Up to num_assistant_tokens candidates per step (default: the drafter's max_draft_tokens()).
    config.num_assistant_tokens = 8

    # Since the streamer is set, the results will be printed
    # every time a new token is generated and put into the streamer queue.
    result = pipe.generate([args.prompt], config, streamer)
    print()

    metrics = result.extended_perf_metrics
    if metrics:
        print(f"Accepted {metrics.get_num_accepted_tokens()} of {metrics.get_num_draft_tokens()} draft tokens")


if '__main__' == __name__:
    main()
