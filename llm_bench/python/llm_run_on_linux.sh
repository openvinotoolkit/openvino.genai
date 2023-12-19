#!/bin/bash
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

convert_model="python ./llm_bench/python/convert.py --model_id stable-diffusion-nano-2-1/ --output_dir ./ov_models/stable-diffusion-nano-2-1 --precision FP16"
echo ${convert_model}
eval ${convert_model}
wait

bemchmarking="python ./llm_bench/python/benchmark.py -m ./ov_models/stable-diffusion-nano-2-1/pytorch/dldt/FP16/ -pf ./llm_bench/python/prompts/stable-diffusion-v2-1.jsonl -d cpu -n 1"
echo ${bemchmarking}
eval ${bemchmarking}