#!/bin/bash
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

git lfs install
export GIT_LFS_SKIP_SMUDGE=0
echo "Download stable-diffusion-v1-4"
git_clone_stable_diff="git clone https://huggingface.co/Narsil/stable-diffusion-v1-4"
echo ${git_clone_stable_diff}
eval ${git_clone_stable_diff}
wait

original_dir="stable-diffusion-v1-4"
dst_dir="./ov_models/stable-diffusion-v1-4"

convert_model="python ./llm_bench/python/convert.py --model_id ${original_dir} --output_dir ${dst_dir} --precision FP16"
echo ${convert_model}
eval ${convert_model}
wait

bemchmarking="python ./llm_bench/python/benchmark.py -m ${dst_dir}/pytorch/dldt/FP16/ -pf ./llm_bench/python/prompts/stable-diffusion-v2-1.jsonl -d cpu -n 1"
echo ${bemchmarking}
eval ${bemchmarking}

rm -rf ${original_dir}
rm -rf ${dst_dir}