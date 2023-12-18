#!/bin/bash
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
 
git lfs install
export GIT_LFS_SKIP_SMUDGE=0
git_clone_bloomz_560m="git clone https://huggingface.co/bigscience/bloomz-560m"
echo ${git_clone_bloomz_560m}
eval ${git_clone_bloomz_560m}
wait

git_clone_stable_diff_2_1="git clone https://huggingface.co/stabilityai/stable-diffusion-2-1"
echo ${git_clone_stable_diff_2_1}
eval ${git_clone_stable_diff_2_1}
wait

convert_model="python ./llm_bench/python/convert.py --model_id bloomz-560m/ --output_dir ./ov_models/bloomz-560m --precision FP16"
echo ${convert_model}
eval ${convert_model}
wait

bemchmarking="python ./llm_bench/python/benchmark.py -m ./ov_models/bloomz-560m/pytorch/dldt/FP16/ -d cpu -n 1"
echo ${bemchmarking}
eval ${bemchmarking}

convert_model="python ./llm_bench/python/convert.py --model_id stable-diffusion-2-1/ --output_dir ./ov_models/stable-diffusion-v2-1 --precision FP16"
echo ${convert_model}
eval ${convert_model}
wait

bemchmarking="python ./llm_bench/python/benchmark.py -m ./ov_models/stable-diffusion-v2-1/pytorch/dldt/FP16/ -pf ./llm_bench/python/prompts/stable-diffusion-v2-1.jsonl -d cpu -n 1"
echo ${bemchmarking}
eval ${bemchmarking}