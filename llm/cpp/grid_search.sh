#!/bin/bash
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

set -e  # Exit immediately if a command exits with a non-zero status

for promt in 69 你好; do
for max_new_tokens in {3..5}; do  # 9 makes the detokenizer to generate invalid Unicode val
for n_groups in {2..3}; do  # n_groups=1 requires diversity_penalty to be 0.0, thus start wuth 2
for group_size in {1..2}; do
for no_repeat_ngram_size in {1..2}; do
for diversity_penalty in 1.0 999999; do
for length_penalty in -1.0 0.0 1.5; do
./build/llm/cpp/llm ./tiny-llama-fast-tokenizer/openvino_model.xml tokenizer.xml detokenizer.xml $promt $max_new_tokens $n_groups $group_size early $no_repeat_ngram_size $diversity_penalty $length_penalty -1 > ./pred.txt && python3 ./text_generation/llama/cpp/ref.py ./pred.txt ./tiny-llama-fast-tokenizer/ $promt $max_new_tokens $n_groups $group_size early $no_repeat_ngram_size $diversity_penalty $length_penalty -1 || { echo $promt $max_new_tokens $n_groups $group_size $no_repeat_ngram_size $diversity_penalty $length_penalty && exit 1; };
done
done
done
done
done
done
done
